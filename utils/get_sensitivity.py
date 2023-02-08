from typing import Dict, Union
import os
import torch
import pickle
import numpy as np
from tqdm.auto import tqdm
from collections import Counter, OrderedDict
from datasets import load_dataset
from collections import defaultdict
import transformers

max_seq_length = 256  # used in tokenization
short_seq_thresh = 5  # used during training
long_seq_thresh = 180  # used during training
mask_ratio = 15  # percentage of masked words for finding token sensitivity
text_column_name = "text"


class TokenizerClass:
    def __init__(self, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq = max_seq_len

    def __call__(self, examples: Union[OrderedDict, Dict]):
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


def initialize_model_data(config: Dict, num_tokenization_workers: int = 4):
    text_corpus_path = config[TEXT_FREQUENCY_PATH]
    model_name = config["model_name"]
    dataset = load_dataset(text_column_name, data_files={"validation": text_corpus_path})
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True,
                                                           model_max_length=max_seq_length)
    tokenize_function = TokenizerClass(tokenizer, max_seq_length)
    column_names = dataset["validation"].column_names
    tokenized_datasets = dataset.map(
        tokenize_function,
        remove_columns=column_names,
        batched=True,
        num_proc=num_tokenization_workers,
    )

    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(tokenized_datasets["validation"],
                                             batch_size=24, shuffle=True, num_workers=6)
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
    device_ids = [0, 1]
    device = torch.device("cuda:%s" % (device_ids[0]) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    # freezing all layers except token embeddings
    for name, param in model.named_parameters():
        if name.split(".")[1] == "cls" or name.split(".")[2] == "encoder":
            param.requires_grad = False
        else:
            assert name.split(".")[2] == "embeddings"
            if "word_embeddings" != name.split(".")[3]:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return model, dataloader, tokenizer, device


def get_sensitivity_grad(config: Dict, save_path: str = "sensitivity_store_list.pkl"):
    model, dataloader, tokenizer, device = initialize_model_data(config)
    mask_token_id = tokenizer.mask_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sensitivity_score_list = []

    for i in tqdm(dataloader):
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

        output, = model(**{"input_ids": batch_input_ids, "attention_mask": batch_attention_mask})
        masked_token_logits = output[(masked_token_pos[0], masked_token_pos[1])]
        masked_token_prob = torch.softmax(masked_token_logits, dim=1)
        correct_token_prob = masked_token_prob[(torch.arange(0, len(gt_masked_words)), gt_masked_words)]
        correct_token_prob.mean().backward()  # weighting different samples??

        seen_tokens = batch_input_ids[(batch_input_ids != pad_token_id) & (batch_input_ids != cls_token_id) &
                                      (batch_input_ids != sep_token_id) & (batch_input_ids != mask_token_id)]

        tokens_counter = Counter(seen_tokens.cpu().numpy())
        unique_seen_tokens = torch.unique(seen_tokens)

        seen_tokens_grad_norm = torch.linalg.norm(model.module.bert.embeddings.word_embeddings.weight.grad[
                                                      unique_seen_tokens], dim=1)
        sensitivity_score_list.append((unique_seen_tokens, tokens_counter, seen_tokens_grad_norm))

        model.zero_grad()

    # Saving the captured gradient norm during the epoch
    if save_path is not None:
        with open(save_path, "wb") as f1:
            pickle.dump(sensitivity_score_list, f1)

    return sensitivity_score_list, len(tokenizer.vocab)


def compute_agg_sensitivity_scores(config: Dict,
                                   save_path: str = "sensitivity_store_list.pkl",
                                   data_weights_path: str = "data_weights_linear_cdf.pkl"):
    if not os.path.isfile(save_path):
        sensitivity_score_list, vocab_size = get_sensitivity_grad(config, save_path=save_path)
    else:
        with open(save_path, "rb") as f:
            sensitivity_score_list = pickle.load(f)
            model_name = config["model_name"]
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
            vocab_size = len(tokenizer.vocab)

    token_sensitivity_dict = defaultdict(list)
    for i in tqdm(sensitivity_score_list):
        tokens_freq = torch.tensor([i[1][j.item()] for j in i[0]]).cuda()
        tokens_grad = (i[2] / tokens_freq).log()
        grad_dist_mean = tokens_grad.mean()
        grad_dist_std = tokens_grad.std()
        norm_grad_norm = (tokens_grad - grad_dist_mean) / grad_dist_std
        batch_sensitivity_dict = dict(zip(i[0].cpu().numpy(), norm_grad_norm.cpu().numpy()))
        for j in batch_sensitivity_dict.items():
            token_sensitivity_dict[j[0]].append(j[1])

    final_sensitivity_dict = {i[0]: np.array(i[1]).mean() if len(i[1]) > 0 else 0
                              for i in token_sensitivity_dict.items()}
    # CDF-based weighting for tokens based on their aggregate sensitivity
    transformed_sensitivity_dict = {i: 2*idx/len(tokenizer) for idx, i in
                                    enumerate(np.argsort([final_sensitivity_dict[i] for i in range(len(tokenizer))]))}

    data_weights = [transformed_sensitivity_dict[i] for i in range(len(tokenizer))]
    with open(data_weights_path, 'wb') as f:
        pickle.dump(data_weights, f)
