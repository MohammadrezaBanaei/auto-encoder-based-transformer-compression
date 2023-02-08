import os
import shutil
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
import sys
import time

from torch.utils.data import ConcatDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithRefDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils.mlm_training_dataclass import ModelArguments, DataTrainingArguments, get_dataset

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def get_transformer_mlm_trainer(eval_data_path: str, seed: int, model_name: str) -> transformers.trainer.Trainer:
    ts = time.time()
    tmp_file_name = f"dummy_{ts}"
    sys.argv = [" ", "--output_dir", tmp_file_name]

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.eval_data_file = eval_data_path
    training_args.seed = seed
    model_args.model_name_or_path = model_name
    training_args.do_eval = True
    training_args.per_device_eval_batch_size = 4
    data_args.mlm = True
    data_args.whole_word_mask = True
    data_args.line_by_line = True
    training_args.dataloader_num_workers = 4
    training_args.prediction_loss_only = True

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    data_args.block_size = tokenizer.model_max_length
    # Our input block size will be the max possible for the model

    # Get datasets
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir)

    if data_args.mlm and data_args.whole_word_mask:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=eval_dataset
    )
    # removing the folder created by trainer
    shutil.rmtree(tmp_file_name)

    return trainer
