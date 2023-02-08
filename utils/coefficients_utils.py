import os
import pickle
import shutil
from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import torch
import transformers
import yaml
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm

from utils.utils_funcs import set_global_seed


def get_coefficients_name(config: Dict):
    if config['coefficients'].get('type', None) is None:
        return 'vanilla'
    if config['coefficients']['type'] not in ('fisher', 'l1_sens'):
        raise ValueError(f'No such coefficients_type as {config["coefficients"]["type"]}')
    return f'{config["coefficients"]["type"]}_{"param" if config["coefficients"]["is_param_wise"] else "row"}'


def get_coefficients(coefficients_name: str, module_name: str, config: Dict, submodule_name: str = None,
                     compute: bool = False, use_cache=True):
    if coefficients_name == 'vanilla':
        return None

    data_folder = config['coefficients']['data_folder']
    if submodule_name:
        coefficients_file_name = os.path.join(data_folder, f'{module_name}_{submodule_name}.pkl')
    else:
        coefficients_file_name = os.path.join(data_folder, f'{module_name}.pkl')

    if os.path.exists(coefficients_file_name):
        with open(coefficients_file_name, 'rb') as f:
            coefficients_dict = pickle.load(f)
        return coefficients_dict[coefficients_name]

    if compute:
        coefficients_dict = compute_coefficients(module_name, config, data_folder, submodule_name=submodule_name,
                                                 use_cache=use_cache,
                                                 short_seq_thresh=config['coefficients']['short_seq_thresh'],
                                                 long_seq_thresh=config['coefficients']['long_seq_thresh'],
                                                 mask_ratio=config['coefficients']['mask_ratio'],
                                                 max_seq_length=config['coefficients']['max_seq_length'])
        with open(coefficients_file_name, 'wb') as f:
            pickle.dump(coefficients_dict, f)
        return coefficients_dict[coefficients_name]

    raise ValueError(f"No precomputed coefficients found in {data_folder}")


text_column_name = "text"


class TokenizerClass:
    def __init__(self, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq = max_seq_len

    def __call__(self, examples: Union[OrderedDict, Dict]):
        '''
        This functions recursively halves the text chunks until reaching 'max_seq' and then
        tokenizes this chunks using class tokenizer.
        '''
        input_texts = [(i.replace("=", "").strip() if (i[:2] == " =" and i[-2:] == "= ") else i)
                       for i in examples[text_column_name] if i != " "]  # removing empty strings and headers tokens
        tokenized_texts = self.tokenizer(input_texts, padding='max_length')
        long_text_indices = [idx for idx, i in enumerate(tokenized_texts['input_ids']) if len(i) > self.max_seq]
        if len(long_text_indices) == 0:
            return tokenized_texts

        long_texts = [input_texts[j].split() for j in long_text_indices]
        shortened_texts = [" ".join(i[: len(i) // 2]) for i in long_texts]

        # recursive calling until reaching max_seq number of tokens for all sentences
        recursive_tokenized_texts = self.__call__({text_column_name: shortened_texts})

        for dx, index in enumerate(long_text_indices):
            for i in recursive_tokenized_texts:
                tokenized_texts[i][index] = recursive_tokenized_texts[i][dx]
        return tokenized_texts


def initialize_model_data(config: Dict, cache_dir: str, module_name: str, submodule_name: str = None, batch_size=128,
                          max_seq_length=256, use_cache=False, verbose=False, seed=42):
    text_corpus_path = config["dataset"]['lm_dataset']['text_frequency_dataset_path']
    model_name = config["dataset"]['lm_dataset']["model_name"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True,
                                                           model_max_length=max_seq_length,
                                                           cache_dir=f'{cache_dir}/tokenizer')
    tokenize_function = TokenizerClass(tokenizer, max_seq_length)
    if verbose:
        print('tokenizer loaded')

    prepared_dataset_path = os.path.join(cache_dir, 'prepared_dataset')
    if use_cache and os.path.exists(prepared_dataset_path):
        tokenized_datasets = load_from_disk(prepared_dataset_path)
    else:
        dataset = load_dataset(text_column_name, data_files={"validation": text_corpus_path}, name='coef_dataset',
                               data_dir='data', cache_dir=f'{cache_dir}/dataset')
        tokenized_datasets = dataset.map(
            tokenize_function,
            remove_columns=dataset["validation"].column_names,
            batched=True,
            batch_size=8192,
            num_proc=None
        )
        tokenized_datasets.save_to_disk(prepared_dataset_path)

    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(tokenized_datasets["validation"],
                                             batch_size=batch_size, shuffle=True, generator=generator)
    if verbose:
        print('dataset loaded')

    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=f'{cache_dir}/model')
    if verbose:
        print('model loaded')

    device_ids = [0]
    device = torch.device("cuda:%s" % (device_ids[0]) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    # freezing all layers except the target ones
    for name, param in model.named_parameters():
        name_split = name.split(".")
        if name_split[-1] == 'bias' or 'LayerNorm' in name_split:
            param.requires_grad = False
            continue
        if name_split[3] == module_name:
            if submodule_name is None:  # word_embeddings case
                param.requires_grad = True
            elif name_split[7] == submodule_name:  # key, value, value case
                param.requires_grad = True
            elif '_'.join(name_split[5:7]) == submodule_name:  # attention_output, output_dense, intermediate_dense case
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    return model, dataloader, tokenizer, device


def compute_coefficients(module_name: str, config: Dict, coefficients_folder: str, submodule_name: str = None,
                         batch_size=128, use_cache=False, chunks_number=6, batches_to_combine_number=111,
                         short_seq_thresh=5, long_seq_thresh=180, mask_ratio=15, max_seq_length=256, seed=42):
    cache_folder = os.path.join(coefficients_folder, 'cache')
    model, dataloader, tokenizer, device = initialize_model_data(config, cache_folder, module_name, seed=seed,
                                                                 batch_size=batch_size, submodule_name=submodule_name,
                                                                 max_seq_length=max_seq_length, use_cache=use_cache)
    mask_token_id = tokenizer.mask_token_id

    if module_name == 'word_embeddings':
        fisher_coefficients_list = []
        l1_sens_coefficients_list = []
    else:
        num_model_layers = len(model.module.bert.encoder.layer)
        fisher_coefficients_list = [[] for _ in range(num_model_layers)]
        l1_sens_coefficients_list = [[] for _ in range(num_model_layers)]

    computations_folder = os.path.join(cache_folder,
                                       f'temp_computations_folder_{module_name if submodule_name is None else f"{module_name}_{submodule_name}"}')

    cur_idx = 0

    for idx, i in enumerate(tqdm(dataloader)):
        batch_attention_mask = i["attention_mask"].to(device)
        batch_input_ids = i["input_ids"].to(device)
        # removing samples with short sequences and also very long ones
        batch_input_ids = batch_input_ids[(batch_attention_mask.sum(axis=1) >= short_seq_thresh) &
                                          (batch_attention_mask.sum(axis=1) <= long_seq_thresh)]
        batch_attention_mask = batch_attention_mask[(batch_attention_mask.sum(axis=1) >= short_seq_thresh) &
                                                    (batch_attention_mask.sum(axis=1) <= long_seq_thresh)]
        # removing unneeded pads
        batch_input_ids = batch_input_ids[:, :batch_attention_mask.sum(axis=0).count_nonzero()]
        batch_attention_mask = batch_attention_mask[:, :batch_attention_mask.sum(axis=0).count_nonzero()]

        if batch_input_ids.shape[0] == 0:  # no sample remained in the batch after filtering
            continue

        # performing random masking
        with torch.no_grad():
            seq_real_length = batch_input_ids.count_nonzero(dim=1)
            masked_token_pos = torch.cat([torch.cat([dx * torch.ones((1, max(i // mask_ratio, 1))),
                                                     torch.randint(low=1, high=i - 1,
                                                                   size=(1, max(i // mask_ratio, 1)))])
                                          for dx, i in enumerate(seq_real_length)], dim=1).long()
            gt_masked_words = batch_input_ids[(masked_token_pos[0], masked_token_pos[1])]
            batch_input_ids[(masked_token_pos[0], masked_token_pos[1])] = mask_token_id

        output = model(**{"input_ids": batch_input_ids, "attention_mask": batch_attention_mask})
        masked_token_logits = output.logits[(masked_token_pos[0], masked_token_pos[1])]
        masked_token_prob = torch.softmax(masked_token_logits, dim=1)
        correct_token_prob = masked_token_prob[(torch.arange(0, len(gt_masked_words)), gt_masked_words)]
        correct_token_prob.mean().backward()

        if module_name == 'word_embeddings':
            fisher_coefficients_list.append(
                torch.square(model.module.bert.embeddings.word_embeddings.weight.grad).detach().cpu())
            l1_sens_coefficients_list.append(
                torch.abs(model.module.bert.embeddings.word_embeddings.weight.grad).detach().cpu())
        else:
            for layer_num in range(num_model_layers):
                if submodule_name == 'key':
                    fisher_coefficients_list[layer_num].append(torch.square(
                        model.module.bert.encoder.layer[layer_num].attention.self.key.weight.grad).detach().cpu())
                    l1_sens_coefficients_list[layer_num].append(
                        torch.abs(
                            model.module.bert.encoder.layer[layer_num].attention.self.key.weight.grad).detach().cpu())
                elif submodule_name == 'value':
                    fisher_coefficients_list[layer_num].append(torch.square(
                        model.module.bert.encoder.layer[layer_num].attention.self.value.weight.grad).detach().cpu())
                    l1_sens_coefficients_list[layer_num].append(
                        torch.abs(
                            model.module.bert.encoder.layer[layer_num].attention.self.value.weight.grad).detach().cpu())
                elif submodule_name == 'query':
                    fisher_coefficients_list[layer_num].append(torch.square(
                        model.module.bert.encoder.layer[layer_num].attention.self.query.weight.grad).detach().cpu())
                    l1_sens_coefficients_list[layer_num].append(
                        torch.abs(
                            model.module.bert.encoder.layer[layer_num].attention.self.query.weight.grad).detach().cpu())
                elif submodule_name == 'attention_output':
                    fisher_coefficients_list[layer_num].append(torch.square(
                        model.module.bert.encoder.layer[layer_num].attention.output.dense.weight.grad).detach().cpu())
                    l1_sens_coefficients_list[layer_num].append(
                        torch.abs(model.module.bert.encoder.layer[
                                      layer_num].attention.output.dense.weight.grad).detach().cpu())
                elif submodule_name == 'output_dense':
                    fisher_coefficients_list[layer_num].append(torch.square(
                        model.module.bert.encoder.layer[layer_num].output.dense.weight.grad).detach().cpu())
                    l1_sens_coefficients_list[layer_num].append(
                        torch.abs(
                            model.module.bert.encoder.layer[layer_num].output.dense.weight.grad).detach().cpu())
                elif submodule_name == 'intermediate_dense':
                    fisher_coefficients_list[layer_num].append(torch.square(
                        model.module.bert.encoder.layer[layer_num].intermediate.dense.weight.grad).detach().cpu())
                    l1_sens_coefficients_list[layer_num].append(
                        torch.abs(
                            model.module.bert.encoder.layer[layer_num].intermediate.dense.weight.grad).detach().cpu())

        if (idx + 1) % batches_to_combine_number == 0:
            os.makedirs(computations_folder, exist_ok=True)
            if module_name == 'word_embeddings':
                fisher_coefficients_list = torch.stack(fisher_coefficients_list).sum(0)
                l1_sens_coefficients_list = torch.stack(l1_sens_coefficients_list).sum(0)
            else:
                fisher_coefficients_list = torch.stack([torch.stack(l).sum(0) for l in fisher_coefficients_list])
                l1_sens_coefficients_list = torch.stack([torch.stack(l).sum(0) for l in l1_sens_coefficients_list])

            torch.save(fisher_coefficients_list, os.path.join(computations_folder, f'fisher_{cur_idx}.pt'))
            torch.save(l1_sens_coefficients_list, os.path.join(computations_folder, f'l1_sens_{cur_idx}.pt'))
            cur_idx += 1

            if module_name == 'word_embeddings':
                fisher_coefficients_list = []
                l1_sens_coefficients_list = []
            else:
                fisher_coefficients_list = [[] for _ in range(num_model_layers)]
                l1_sens_coefficients_list = [[] for _ in range(num_model_layers)]
        model.zero_grad()

    if module_name == 'word_embeddings':
        if fisher_coefficients_list:
            fisher_coefficients_list = torch.stack(fisher_coefficients_list).sum(0)
            l1_sens_coefficients_list = torch.stack(l1_sens_coefficients_list).sum(0)
            torch.save(fisher_coefficients_list, os.path.join(computations_folder, f'fisher_{cur_idx}.pt'))
            torch.save(l1_sens_coefficients_list, os.path.join(computations_folder, f'l1_sens_{cur_idx}.pt'))
    else:
        if fisher_coefficients_list[0]:
            fisher_coefficients_list = torch.stack([torch.stack(l).sum(0) for l in fisher_coefficients_list])
            l1_sens_coefficients_list = torch.stack([torch.stack(l).sum(0) for l in l1_sens_coefficients_list])
            torch.save(fisher_coefficients_list, os.path.join(computations_folder, f'fisher_{cur_idx}.pt'))
            torch.save(l1_sens_coefficients_list, os.path.join(computations_folder, f'l1_sens_{cur_idx}.pt'))

    sub_results_files = os.listdir(computations_folder)
    fisher_coefficients_files = [f for f in sub_results_files if f.startswith('fisher')]
    l1_sens_coefficients_files = [f for f in sub_results_files if f.startswith('l1_sens')]

    def combine_chunks(chunks):
        sub_res = []
        for files_chink in chunks:
            coefficients_list = [torch.load(os.path.join(computations_folder, f))
                                 for f in files_chink]
            coefficients_list = torch.stack(coefficients_list).sum(0)
            sub_res.append(coefficients_list)
        return torch.stack(sub_res).sum(0)

    fisher_coefficients_list = combine_chunks(np.array_split(fisher_coefficients_files, chunks_number))
    l1_sens_coefficients_list = combine_chunks(np.array_split(l1_sens_coefficients_files, chunks_number))

    shutil.rmtree(computations_folder)

    sum_dim = -1
    coefficients_dict_ = {
        "fisher_param": fisher_coefficients_list / (len(dataloader) * batch_size),
        "l1_sens_param": l1_sens_coefficients_list / (len(dataloader) * batch_size),
        "fisher_row": fisher_coefficients_list.sum(sum_dim).sqrt() / (len(dataloader) * batch_size),
        "l1_sens_row": l1_sens_coefficients_list.sum(sum_dim) / (len(dataloader) * batch_size),
    }
    return coefficients_dict_


if __name__ == '__main__':
    with open("../configs/config.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    set_global_seed(config['coefficients']['seed'])

    batch_size = 128
    data_folder = config['coefficients']['data_folder']

    modules = ['word_embeddings', 'layer', 'layer', 'layer', 'layer', 'layer', 'layer']
    sub_modules = [None, 'key', 'query', 'value', 'attention_output', 'output_dense', 'intermediate_dense']

    for module_name, submodule_name in zip(modules, sub_modules):
        coefficients_dict = compute_coefficients(module_name, config, data_folder,
                                                 submodule_name=submodule_name, batch_size=batch_size, use_cache=True,
                                                 short_seq_thresh=config['coefficients']['short_seq_thresh'],
                                                 long_seq_thresh=config['coefficients']['long_seq_thresh'],
                                                 mask_ratio=config['coefficients']['mask_ratio'],
                                                 max_seq_length=config['coefficients']['max_seq_length'],
                                                 seed=config['coefficients']['seed'])
        if submodule_name:
            coefficients_file_name = os.path.join(data_folder, f'{module_name}_{submodule_name}.pkl')
        else:
            coefficients_file_name = os.path.join(data_folder, f'{module_name}.pkl')
        with open(coefficients_file_name, 'wb') as f:
            pickle.dump(coefficients_dict, f)
