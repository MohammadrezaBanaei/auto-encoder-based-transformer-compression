import os
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import ConcatDataset
from glob import glob
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

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TOKENIZERS_PARALLELISM"] = "True"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    chinese_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input ref data file for whole word mask in Chinees."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={"help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool = False,
        cache_dir: Optional[str] = None,
):
    def _dataset(file_path):
        if args.line_by_line:
            if args.chinese_ref_file is not None:
                if not args.whole_word_mask or not args.mlm:
                    raise ValueError("You need to set world whole masking and mlm to True for Chinese Whole Word Mask")
                return LineByLineWithRefDataset(
                    tokenizer=tokenizer,
                    file_path=file_path,
                    block_size=args.block_size,
                    ref_path=args.chinese_ref_file,
                )

            temp_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
            equal_id = tokenizer.convert_tokens_to_ids("=")
            # filtering out short texts + removing "=" from wiki texts (used for headers)
            temp_dataset.examples = [{j[0]: j[1][j[1] != equal_id] for j in i.items()}
                                     for i in temp_dataset.examples if
                                     len(i["input_ids"][i["input_ids"] != equal_id]) >= 20]
            return temp_dataset

        else:
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=cache_dir,
            )

    if evaluate:
        return _dataset(args.eval_data_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)
