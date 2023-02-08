# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
import ast
import json
import logging
import os
import random
import re
import sys
import math
import torch
import pickle
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
import yaml
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from utils.coefficients_utils import get_coefficients_name, get_coefficients
from evaluation import compute_perplexity
from transformer_experiments.substitution_modules import (WordEmbeddingModule, MFWordEmbeddingModule,
                                                          SubstitutionWeightModule, MFWeightModule)
from main import init_ae_model
from utils.utils_funcs import get_mlm_trainer

check_min_version("4.5.0.dev0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    train_frac: float = field(
        default=1.0,
        metadata={"help": "What percentage of data to train on."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    wiki_text_103_url: str = field(
        default=None, metadata={"help": "Wiki text file/url for masked language model trainer"}
    )
    text_download_folder: str = field(
        default=None, metadata={"help": "Download folder for masked language model trainer"}
    )
    lm_text_dataset_path: str = field(
        default=None, metadata={"help": "Text dataset path for masked language model trainer"}
    )
    text_frequency_dataset_path: str = field(
        default=None, metadata={"help": "Text frequency dataset path for masked language model trainer"}
    )
    model_name: str = field(
        default=None, metadata={"help": "Model name for masked language model trainer"}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pretrained_ae_dir: str = field(
        default=None, metadata={"help": "Paths to pretrained AutoEncoder models"}
    )
    svd: bool = field(
        default=False, metadata={"help": "Whether to use SVD reduction"}
    )
    svd_config_dir: Optional[str] = field(
        default=None, metadata={"help": "SVD config dir"}
    )
    svd_compression_mode: str = field(
        default="separated", metadata={"help": "Mode for matrices in SVD compression"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    save_model: bool = field(
        default=True,
        metadata={
            "help": "Save the final model."
        },
    )
    ae_random_reinit: bool = field(
        default=False,
        metadata={
            "help": "Use if you want to randomly initialize AE weights"
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        raise ValueError("Please specify a GLUE task by setting the 'task_name' argument")

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        raise ValueError("Please specify a GLUE task by setting the 'task_name' argument")

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm_data_dict = {
        "wiki_text_103_url": data_args.wiki_text_103_url,
        "text_download_folder": data_args.text_download_folder,
        "LM_text_dataset_path": data_args.lm_text_dataset_path,
        "text_frequency_dataset_path": data_args.text_frequency_dataset_path,
        "model_name": data_args.model_name
    }
    mlm_trainer = get_mlm_trainer(lm_data_dict, seed=training_args.seed)
    reconstructed_modules_dict = {}

    if model_args.pretrained_ae_dir is not None:
        pretrained_ae_dirs = [x for x in model_args.pretrained_ae_dir.split(',')]
        for pretrained_ae_dir in pretrained_ae_dirs:
            logger.info('Using AE compression...')
            with open(os.path.join(pretrained_ae_dir, 'config.yml'), 'r') as stream:
                ae_config = yaml.load(stream, Loader=yaml.FullLoader)
            with open(os.path.join(pretrained_ae_dir, 'checkpoints', 'transformer_bias_data.pkl'), 'rb') as f:
                transformer_bias_data = pickle.load(f)

            # SUBSTITUTE WORD EMBEDDINGS
            if ae_config["dataset"]["to_compress"]['word_embeddings']:
                assert os.path.exists(os.path.join(pretrained_ae_dir,
                                                   'checkpoints',
                                                   "word_embeddings")), "Pretrained AE embedding folder is missing"
                original_emb_param = model.bert.embeddings.word_embeddings.weight
                original_emb_params_num = original_emb_param.numel()

                emb_ae_model = init_ae_model(enc_type=ae_config['model']['ae']['encoder_type'],
                                             dec_type=ae_config['model']['ae']['decoder_type'],
                                             original_module=original_emb_param,
                                             symmetric_ae_scale=ae_config['model']['ae']['symmetric_ae_scale'],
                                             latent_dim_relative=ae_config['model']['latent_dim']['token_emb'],
                                             weights_path=ae_config['model']['weights_path'],
                                             enc_layer_sizes=ae_config['model']['ae']['enc_layer_sizes'],
                                             dec_layer_sizes=ae_config['model']['ae']['dec_layer_sizes'],
                                             cr=ae_config['model']['cr'],
                                             device=device,
                                             enforce_norm=ae_config['model']['ae']['enforce_norm'])

                emb_ae_model.load_state_dict(torch.load(os.path.join(pretrained_ae_dir, 'checkpoints',
                                                                     'word_embeddings', 'model.pt')))
                emb_weights = torch.load(os.path.join(pretrained_ae_dir, 'checkpoints',
                                                      'word_embeddings', 'latents.pt'))

                if model_args.ae_random_reinit:
                    if type(emb_ae_model.decoder.dec) == torch.nn.Linear:
                        emb_ae_model.decoder.dec.reset_parameters()
                    else:
                        for l in emb_ae_model.decoder.dec:
                            if type(l) == torch.nn.Linear:
                                l.reset_parameters()
                    torch.nn.init.kaiming_uniform_(emb_weights, a=math.sqrt(5))

                reconstructed_modules_dict['word_embeddings'] = {}
                reconstructed_modules_dict['word_embeddings']['reconstructed_module'] = emb_ae_model.decoder(
                    emb_weights, range(emb_weights.shape[0]))
                reconstructed_modules_dict['word_embeddings']['original_module'] = original_emb_param

                for p in emb_ae_model.encoder.parameters():
                    p.requires_grad = False

                word_emb_module = WordEmbeddingModule(decoder_module=emb_ae_model.decoder,
                                                      emb_weights=emb_weights,
                                                      pad_token_id=config.pad_token_id,
                                                      original_params_num=original_emb_params_num)
                model.bert.embeddings.word_embeddings = word_emb_module

            # SUBSTITUTE MATRICES
            temp_name_mapping = {"key": "all_keys",
                                 "value": "all_values",
                                 "query": "all_queries",
                                 "attention_output": "all_attention_outputs",
                                 "output_dense": "all_output_denses",
                                 "intermediate_dense": "all_intermediate_denses"}
            for proj_name in ["key", "value", "query", "attention_output", "output_dense", "intermediate_dense"]:
                latent_dim = ae_config['model']['latent_dim'][proj_name]
                if ae_config["dataset"]["to_compress"][temp_name_mapping[proj_name]]:
                    if ae_config['modules_training_setting'] == 'separated':
                        for layer_idx in range(model.bert.config.num_hidden_layers):
                            module_name = f'layer_{layer_idx}_{proj_name}'
                            reconstructed_modules_dict[module_name] = {}
                            if proj_name == 'key':
                                reconstructed_modules_dict[module_name]['original_module'] = model.bert.encoder.layer[
                                    layer_idx].attention.self.key.weight
                            elif proj_name == 'query':
                                reconstructed_modules_dict[module_name]['original_module'] = model.bert.encoder.layer[
                                    layer_idx].attention.self.query.weight
                            elif proj_name == 'value':
                                reconstructed_modules_dict[module_name]['original_module'] = model.bert.encoder.layer[
                                    layer_idx].attention.self.value.weight
                            elif proj_name == 'attention_output':
                                reconstructed_modules_dict[module_name]['original_module'] = model.bert.encoder.layer[
                                    layer_idx].attention.output.dense.weight
                            elif proj_name == 'output_dense':
                                reconstructed_modules_dict[module_name]['original_module'] = model.bert.encoder.layer[
                                    layer_idx].output.dense.weight
                            elif proj_name == 'intermediate_dense':
                                reconstructed_modules_dict[module_name]['original_module'] = model.bert.encoder.layer[
                                    layer_idx].intermediate.dense.weight

                            ae_path = os.path.join(pretrained_ae_dir,
                                                   'checkpoints',
                                                   f'layer_{layer_idx}_{proj_name}')
                            ae_model = init_ae_model(enc_type=ae_config['model']['ae']['encoder_type'],
                                                     dec_type=ae_config['model']['ae']['decoder_type'],
                                                     original_module=reconstructed_modules_dict[module_name][
                                                         'original_module'],
                                                     symmetric_ae_scale=ae_config['model']['ae']['symmetric_ae_scale'],
                                                     latent_dim_relative=latent_dim,
                                                     weights_path=ae_config['model']['weights_path'],
                                                     enc_layer_sizes=ae_config['model']['ae']['enc_layer_sizes'],
                                                     dec_layer_sizes=ae_config['model']['ae']['dec_layer_sizes'],
                                                     cr=ae_config['model']['cr'],
                                                     device=device,
                                                     enforce_norm=ae_config['model']['ae']['enforce_norm'])

                            ae_model.load_state_dict(torch.load(os.path.join(ae_path, 'model.pt')))
                            latent_weights = torch.load(os.path.join(ae_path, 'latents.pt'))

                            if model_args.ae_random_reinit:
                                for l in ae_model.decoder.dec:
                                    if type(l) == torch.nn.Linear:
                                        l.reset_parameters()
                                torch.nn.init.kaiming_uniform_(latent_weights, a=math.sqrt(5))

                            reconstructed_modules_dict[module_name]['reconstructed_module'] = ae_model.decoder(
                                latent_weights, range(len(latent_weights)))

                            for p in ae_model.encoder.parameters():
                                p.requires_grad = False

                            sub_module = SubstitutionWeightModule(decoder_module=ae_model.decoder,
                                                                  latent_weights=latent_weights,
                                                                  original_params_num=
                                                                  reconstructed_modules_dict[module_name][
                                                                      'original_module'].numel(),
                                                                  module_bias=transformer_bias_data[f'layer_{layer_idx}']
                                                                  [proj_name])

                            if proj_name == 'attention_output':
                                setattr(model.bert.encoder.layer[layer_idx].attention, 'dense', sub_module)
                            elif proj_name == 'output_dense':
                                setattr(model.bert.encoder.layer[layer_idx].output, 'dense', sub_module)
                            elif proj_name == 'intermediate_dense':
                                setattr(model.bert.encoder.layer[layer_idx].intermediate, 'dense', sub_module)
                            else:
                                setattr(model.bert.encoder.layer[layer_idx].attention.self, proj_name, sub_module)
                    else:  # concatenated
                        if proj_name == 'key':
                            module_name = 'keys'
                            original_module = torch.cat(
                                [model.bert.encoder.layer[layer_idx].attention.self.key.weight for layer_idx in
                                 range(model.bert.config.num_hidden_layers)])
                            hidden_size = model.bert.encoder.layer[0].attention.self.key.weight.shape[0]
                            original_params_num = model.bert.encoder.layer[0].attention.self.key.weight.numel()
                        elif proj_name == 'query':
                            module_name = 'queries'
                            original_module = torch.cat(
                                [model.bert.encoder.layer[layer_idx].attention.self.query.weight for layer_idx in
                                 range(model.bert.config.num_hidden_layers)])
                            hidden_size = model.bert.encoder.layer[0].attention.self.query.weight.shape[0]
                            original_params_num = model.bert.encoder.layer[0].attention.self.query.weight.numel()
                        elif proj_name == 'value':
                            module_name = 'values'
                            original_module = torch.cat(
                                [model.bert.encoder.layer[layer_idx].attention.self.value.weight for layer_idx in
                                 range(model.bert.config.num_hidden_layers)])
                            hidden_size = model.bert.encoder.layer[0].attention.self.value.weight.shape[0]
                            original_params_num = model.bert.encoder.layer[0].attention.self.value.weight.numel()
                        elif proj_name == 'attention_output':
                            module_name = 'attention_outputs'
                            original_module = torch.cat(
                                [model.bert.encoder.layer[layer_idx].attention.output.dense.weight for layer_idx in
                                 range(model.bert.config.num_hidden_layers)])
                            hidden_size = model.bert.encoder.layer[0].attention.output.dense.weight.shape[0]
                            original_params_num = model.bert.encoder.layer[0].attention.output.dense.weight.numel()
                        elif proj_name == 'output_dense':
                            module_name = 'output_denses'
                            original_module = torch.cat(
                                [model.bert.encoder.layer[layer_idx].output.dense.weight for layer_idx in
                                 range(model.bert.config.num_hidden_layers)])
                            hidden_size = model.bert.encoder.layer[0].output.dense.weight.shape[0]
                            original_params_num = model.bert.encoder.layer[0].output.dense.weight.numel()
                        elif proj_name == 'intermediate_dense':
                            module_name = 'intermediate_denses'
                            original_module = torch.cat(
                                [model.bert.encoder.layer[layer_idx].intermediate.dense.weight for layer_idx in
                                 range(model.bert.config.num_hidden_layers)])
                            hidden_size = model.bert.encoder.layer[0].intermediate.dense.weight.shape[0]
                            original_params_num = model.bert.encoder.layer[0].intermediate.dense.weight.numel()

                        ae_path = os.path.join(pretrained_ae_dir, 'checkpoints', module_name)

                        ae_model = init_ae_model(enc_type=ae_config['model']['ae']['encoder_type'],
                                                 dec_type=ae_config['model']['ae']['decoder_type'],
                                                 original_module=original_module,
                                                 symmetric_ae_scale=ae_config['model']['ae']['symmetric_ae_scale'],
                                                 latent_dim_relative=latent_dim,
                                                 weights_path=ae_config['model']['weights_path'],
                                                 enc_layer_sizes=ae_config['model']['ae']['enc_layer_sizes'],
                                                 dec_layer_sizes=ae_config['model']['ae']['dec_layer_sizes'],
                                                 cr=ae_config['model']['cr'],
                                                 device=device,
                                                 enforce_norm=ae_config['model']['ae']['enforce_norm'])

                        ae_model.load_state_dict(torch.load(os.path.join(ae_path, 'model.pt')))
                        shared_dec_module = ae_model.decoder
                        all_latent_weights = torch.load(os.path.join(ae_path, 'latents.pt'))

                        if model_args.ae_random_reinit:
                            for l in ae_model.decoder.dec:
                                if type(l) == torch.nn.Linear:
                                    l.reset_parameters()
                            torch.nn.init.kaiming_uniform_(all_latent_weights, a=math.sqrt(5))

                        reconstructed_modules_dict[module_name] = {}
                        reconstructed_modules_dict[module_name]['reconstructed_module'] = ae_model.decoder(
                            all_latent_weights, range(len(all_latent_weights)))
                        reconstructed_modules_dict[module_name]['original_module'] = original_module

                        for layer_idx in range(model.bert.config.num_hidden_layers):
                            latent_weights = all_latent_weights[hidden_size * layer_idx:hidden_size * (layer_idx + 1)]
                            for p in ae_model.encoder.parameters():
                                p.requires_grad = False
                            sub_module = SubstitutionWeightModule(decoder_module=shared_dec_module,
                                                                  latent_weights=latent_weights,
                                                                  original_params_num=original_params_num,
                                                                  module_bias=
                                                                  transformer_bias_data[f'layer_{layer_idx}']
                                                                  [proj_name])

                            if proj_name == 'attention_output':
                                setattr(model.bert.encoder.layer[layer_idx].attention, 'dense', sub_module)
                            elif proj_name == 'output_dense':
                                setattr(model.bert.encoder.layer[layer_idx].output, 'dense', sub_module)
                            elif proj_name == 'intermediate_dense':
                                setattr(model.bert.encoder.layer[layer_idx].intermediate, 'dense', sub_module)
                            else:
                                setattr(model.bert.encoder.layer[layer_idx].attention.self, proj_name, sub_module)
    elif model_args.svd == True:
        raise NotImplementedError('Please use run_glue.py for SVD (for both single and multiple modules)')

    global_perplexity = compute_perplexity(mlm_trainer, reconstructed_modules_dict, training_args.seed)

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        raise ValueError("Please specify a GLUE task by setting the 'task_name' argument")

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        if data_args.train_frac > 0 and data_args.train_frac < 1.0:
            train_dataset = train_dataset.select(random.sample(range(1, len(train_dataset)),
                                                               int(data_args.train_frac * len(train_dataset))))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        if model_args.save_model:
            trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            import time
            start = time.time()
            for i in range(100):
                metrics = trainer.evaluate(eval_dataset=eval_dataset)
            end = time.time()
            print('Mean time for evaluation', (end - start)/100)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            metrics["eval_initial_perplexity"] = global_perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
