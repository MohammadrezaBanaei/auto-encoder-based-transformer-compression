import math
import os
from typing import Tuple, Iterable, List, Dict

import numpy as np
import random
import torch
import copy
import torch.nn as nn
import transformers
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset

from evaluation import eval_substitution_module
from utils.zeroshot_mlm_utils import get_transformer_mlm_trainer


def set_global_seed(seed_value: int):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def find_quadratic_solution(a, b, c):
    delta = b * b - 4 * a * c
    assert delta >= 0, "Delta < 0, complex roots"
    sqrt_d = math.sqrt(delta)

    if delta > 0:
        x1 = (-b + sqrt_d) / (2 * a)
        x2 = (-b - sqrt_d) / (2 * a)
        if x1 > 0 and x2 > 0:
            raise ValueError(f'Both roots are > 0 ({x1}, {x2}), please check this case.')
        x = max(x1, x2)
    else:
        x = -b / (2 * a)
    return int(x)


def create_dirs_if_needed(dirs: List):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)


def get_mlm_trainer(config: Dict, seed: int) -> transformers.trainer.Trainer:
    text_download_folder = config["text_download_folder"]
    model_name = config["model_name"]
    text_dataset_path = config["LM_text_dataset_path"]

    if not (os.path.isfile(text_dataset_path)):
        os.makedirs(text_download_folder, exist_ok=True)
        assert os.system("wget %s -P %s" % (config['wiki_text_103_url'], text_download_folder)) == 0, \
            "Downloading text dataset  failed"
        assert os.system("unzip %s -d %s" % (os.path.join(text_download_folder, "wikitext-103-raw-v1.zip"),
                                             text_download_folder)) == 0, "unzip of text dataset failed"
        text_dataset_path = os.path.join(text_download_folder, "wikitext-103-raw", "wiki.test.raw")

    mlm_trainer = get_transformer_mlm_trainer(eval_data_path=text_dataset_path, seed=seed, model_name=model_name)
    mlm_trainer.initial_state = copy.deepcopy(mlm_trainer.model.state_dict())
    return mlm_trainer


def get_bpe_tokens_frequency(config: dict) -> Counter:
    text_corpus_path = config['text_frequency_dataset_path']
    model_name = config["model_name"]
    dataset = load_dataset("text", data_files={"validation": text_corpus_path})
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    text_column_name = "text"

    column_names = dataset["validation"].column_names

    def tokenize_function(examples):
        # preprocessing input text and removing noise sentences
        input_texts = [text for text in examples[text_column_name] if text != " " and
                       (text[:2] != " =" or text[-2:] != "= ")]
        if len(input_texts) == 0:
            return {}
        return tokenizer(input_texts, return_special_tokens_mask=True)

    tokenized_datasets = dataset.map(
        tokenize_function,
        remove_columns=column_names,
        batched=True,
        num_proc=4
    )

    bpe_frequency = Counter()
    temp_words = []
    for i in tqdm(tokenized_datasets["validation"]):
        if len(i["input_ids"]) > 2:  # if it's not an empty string (i.e. [CLS][SEP])
            temp_words += i["input_ids"]
        if len(temp_words) > 1e7:  # limiting RAM usage
            bpe_frequency += Counter(temp_words)
            temp_words = []
    bpe_frequency += Counter(temp_words)

    return bpe_frequency


def row_wise_quantize(input_matrix: torch.tensor, n_bits: int):
    row_wise_scale = (input_matrix.max(dim=1).values - input_matrix.min(dim=1).values) / (2**n_bits)
    row_min = input_matrix.min(dim=1).values
    quantized_values = torch.round((input_matrix - row_min.unsqueeze(-1)) / row_wise_scale.unsqueeze(-1))
    quantized_matrix = quantized_values * row_wise_scale.unsqueeze(-1) + row_min.unsqueeze(-1)
    return quantized_matrix


def run_svd(input_matrix: np.ndarray, latent_dim: int, n_iter: int, random_state: int,
            rec_coef: np.ndarray = None) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=latent_dim, n_iter=n_iter, random_state=random_state)
    if rec_coef is None:
        svd.fit(input_matrix)
        reduced_matrix = svd.transform(input_matrix)
    else:
        grad_weights_matrix = np.diag(rec_coef)
        inverse_grad_weights_matrix = np.diag(1.0 / rec_coef)
        svd.fit(grad_weights_matrix @ input_matrix)
        reduced_matrix = inverse_grad_weights_matrix @ svd.transform(grad_weights_matrix @ input_matrix)
    return reduced_matrix, svd


def get_svd_prunning_mask(w: torch.tensor, zeros_frac: float) -> torch.tensor:
    mask = torch.ones_like(w)
    k = int(zeros_frac * mask.numel())
    th = torch.topk(abs(w).flatten(), k=k, largest=False).values.max()
    mask[abs(w) <= th] = 0.0
    return mask


def get_linear_rec_svd(input_matrix: np.ndarray, latent_dim: int, n_iter: int,
                       random_state: int, target_cr: float = None, rec_coef: np.ndarray = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd(input_matrix, latent_dim, n_iter, random_state, rec_coef=rec_coef)

    zeros_frac_svd = 0.0
    if target_cr is not None:
        zeros_frac_svd = get_zeros_frac_for_svd(reduced_matrix=reduced_matrix,
                                                svd_components=svd.components_,
                                                original_params_num=input_matrix.size,
                                                target_cr=target_cr)

        if zeros_frac_svd > 0:
            reduced_matrix_mask = get_svd_prunning_mask(torch.tensor(reduced_matrix), zeros_frac_svd)
            reduced_matrix = reduced_matrix * np.array(reduced_matrix_mask)

            svd_comp_mask = get_svd_prunning_mask(torch.tensor(svd.components_), zeros_frac_svd)
            svd.components_ = svd.components_ * np.array(svd_comp_mask)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_, zeros_frac_svd


def get_lin_rec_latent_dim(compression_ratio: float, input_matrix: np.ndarray) -> int:
    original_params_num = input_matrix.shape[0] * input_matrix.shape[1]
    return int(original_params_num / (compression_ratio * (input_matrix.shape[0] + input_matrix.shape[1])))


def get_zeros_frac_for_svd(reduced_matrix: np.ndarray, svd_components: np.ndarray,
                           original_params_num: int, target_cr: float) -> float:
    new_params_num = reduced_matrix.size + svd_components.size
    target_params_num = math.ceil(original_params_num / target_cr)

    if target_params_num > new_params_num:
        return 0.0

    zeros_frac = (new_params_num - target_params_num) / new_params_num

    return zeros_frac


def save_lin_rec_stats(module_name: str,
                       writer: SummaryWriter,
                       original_matrix: torch.tensor,
                       latent_dim: int,
                       iters: Iterable,
                       mlm_trainer: transformers.trainer.Trainer,
                       eval_metric: str,
                       seed: int,
                       device: torch.device,
                       rec_coef: torch.tensor,
                       target_compression_ratio: float = None) -> Tuple[torch.tensor, int, float, int]:
    input_matrix = original_matrix.cpu().numpy()
    best_run_metric = math.inf
    if rec_coef is not None:
        rec_coef = rec_coef.cpu().numpy()

    for i in iters:
        lin_rec, reduced_matrix, svd_comp, zeros_frac_svd = get_linear_rec_svd(input_matrix=input_matrix,
                                                                               latent_dim=latent_dim,
                                                                               n_iter=i,
                                                                               random_state=42,
                                                                               target_cr=target_compression_ratio,
                                                                               rec_coef=rec_coef)
        lin_rec_tensor = torch.tensor(lin_rec, dtype=original_matrix.dtype, device=original_matrix.device)

        result_metrics = eval_substitution_module(module_name=module_name, original_module=original_matrix,
                                                  reconstructed_module=lin_rec_tensor, writer=writer, log_to_tb=True,
                                                  mlm_trainer=mlm_trainer, global_step=i,
                                                  seed=seed, device=device)

        if result_metrics[eval_metric] < best_run_metric:
            best_run_metric = result_metrics[eval_metric]
            best_reconstruction = lin_rec_tensor
            best_iter = i

    svd_cr, svd_cr_with_zeros = get_svd_compression_stats(original_params_num=input_matrix.size,
                                                          reduced_matrix_numel=reduced_matrix.size,
                                                          second_matrix_numel=svd_comp.size,
                                                          zeros_frac=zeros_frac_svd)

    return best_reconstruction, best_iter, svd_cr, reduced_matrix.size + svd_comp.size


def get_ae_sub_compression_stats(original_params_num: int,
                                 embedding: torch.Tensor,
                                 decoder: nn.Module) -> Tuple[float, float, float, float, float]:
    """
    Returns compresion_ratio, compresion_ratio_with_zeros, zeros_fraction, emb_zeros_fraction, dec_zeros_fraction
    """
    embedding_params_num = embedding.numel()
    emb_zeros_cntr = (embedding == 0.0).sum().item()

    dec_params_num, dec_zeros_cntr = decoder.get_params_num()

    zeros_cntr = emb_zeros_cntr + dec_zeros_cntr
    current_params_num = embedding_params_num + dec_params_num

    zeros_frac = zeros_cntr / current_params_num

    cr = original_params_num / (current_params_num - zeros_cntr)
    cr_with_zeros = original_params_num / current_params_num

    return cr, cr_with_zeros, zeros_frac, emb_zeros_cntr / embedding_params_num, dec_zeros_cntr / dec_params_num


def get_svd_compression_stats(original_params_num: int, reduced_matrix_numel: int, second_matrix_numel: int,
                              zeros_frac: float = 0.0) -> Tuple[float, float]:
    reduced_matrix_zeros_cntr = int(zeros_frac * reduced_matrix_numel)
    second_comp_zeros_cntr = int(zeros_frac * second_matrix_numel)

    cr = original_params_num / (
            reduced_matrix_numel + second_matrix_numel - reduced_matrix_zeros_cntr - second_comp_zeros_cntr)
    cr_with_zeros = original_params_num / (reduced_matrix_numel + second_matrix_numel)

    return cr, cr_with_zeros


def init_model_with_svd(input_matrix: np.ndarray, model: nn.Module, num_iters: int, device: torch.device) -> nn.Module:
    _, svd = run_svd(input_matrix=input_matrix,
                     latent_dim=model.latent_size,
                     n_iter=num_iters,
                     random_state=42)

    dtype = model.encoder.enc.weight.dtype

    model.encoder.enc.weight.data = torch.tensor(svd.components_, device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.zeros_(model.encoder.enc.bias.data)
    model.decoder.dec.weight.data = torch.tensor(svd.components_.T, device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.zeros_(model.decoder.dec.bias.data)

    return model

