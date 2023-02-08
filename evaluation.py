import math
from collections import defaultdict
from typing import Dict

import torch
import transformers
from tensorboardX import SummaryWriter
from transformers import set_seed


def compute_perplexity(mlm_trainer: transformers.trainer.Trainer,
                       reconstructed_modules_dict: Dict,
                       seed: int):
    mlm_trainer.model.load_state_dict(mlm_trainer.initial_state)  # getting initial model weights

    for module_name in reconstructed_modules_dict:
        reconstructed_module = reconstructed_modules_dict[module_name]['reconstructed_module']
        reconstructed_module = reconstructed_module.to(mlm_trainer.model.device)
        if module_name == "word_embeddings":
            mlm_trainer.model.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(reconstructed_module)
        elif "layer_" in module_name:  # separated mode
            matrix_type = module_name.split("_")[2]  # key/query/value/attention_output/intermediate_dense/output_dense
            module_layer_num = int(module_name.split("_")[1])
            if 'key' in module_name or 'query' in module_name or 'value' in module_name:
                eval(f'''mlm_trainer.model.bert.encoder.layer[{module_layer_num}].attention.self.{
                matrix_type}''').weight = torch.nn.Parameter(reconstructed_module)
            elif 'attention_output' in module_name:
                eval(f'''mlm_trainer.model.bert.encoder.layer[{module_layer_num}].attention.output.dense''').weight = torch.nn.Parameter(reconstructed_module)
            elif 'intermediate_dense' in module_name:
                eval(f'''mlm_trainer.model.bert.encoder.layer[{module_layer_num}].intermediate.dense''').weight = torch.nn.Parameter(reconstructed_module)
            elif 'output_dense' in module_name:
                eval(f'''mlm_trainer.model.bert.encoder.layer[{module_layer_num}].output.dense''').weight = torch.nn.Parameter(reconstructed_module)
            else:
                raise ValueError(f'No such name as {module_name} for separated mode')
        elif module_name in ['keys', 'queries', 'values', 'attention_outputs', 'output_denses', 'intermediate_denses']: # concatenation mode
            num_model_layers = len(mlm_trainer.model.bert.encoder.layer)
            split_size = int(reconstructed_module.shape[0] / num_model_layers)

            for layer_num in range(num_model_layers):
                if module_name == 'keys':
                    mlm_trainer.model.bert.encoder.layer[layer_num].attention.self.key.weight = torch.nn.Parameter(
                        reconstructed_module.split(split_size)[layer_num])
                elif module_name == 'queries':
                    mlm_trainer.model.bert.encoder.layer[layer_num].attention.self.query.weight = torch.nn.Parameter(
                        reconstructed_module.split(split_size)[layer_num])
                elif module_name == 'values':
                    mlm_trainer.model.bert.encoder.layer[layer_num].attention.self.value.weight = torch.nn.Parameter(
                        reconstructed_module.split(split_size)[layer_num])
                elif module_name == 'attention_outputs':
                    mlm_trainer.model.bert.encoder.layer[layer_num].attention.output.dense.weight = torch.nn.Parameter(
                        reconstructed_module.split(split_size)[layer_num])
                elif module_name == 'output_denses':
                    mlm_trainer.model.bert.encoder.layer[layer_num].output.dense.weight = torch.nn.Parameter(
                        reconstructed_module.split(split_size)[layer_num])
                elif module_name == 'intermediate_denses':
                    mlm_trainer.model.bert.encoder.layer[layer_num].intermediate.dense.weight = torch.nn.Parameter(
                        reconstructed_module.split(split_size)[layer_num])
        else:
            raise ValueError(f'No such name as {module_name} for perplexity calculation')

    set_seed(seed)
    eval_output = mlm_trainer.evaluate()
    perplexity = math.exp(eval_output["eval_loss"])
    return perplexity


def eval_substitution_module(module_name: str,
                             original_module: torch.tensor,
                             reconstructed_module: torch.tensor,
                             writer: SummaryWriter,
                             log_to_tb: bool,
                             global_step: int,
                             mlm_trainer: transformers.trainer.Trainer,
                             seed: int,
                             device: torch.device) -> Dict:
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    cosine_distance = (1.0 - cos(reconstructed_module, original_module)).mean()
    rmse = torch.sqrt(torch.pow((original_module - reconstructed_module), 2).mean()).cpu()
    mae = (original_module - reconstructed_module).abs().mean().cpu()
    powered_mae = torch.pow(torch.abs(original_module - reconstructed_module) + 1e-10, 0.6).mean()
    perplexity = compute_perplexity(mlm_trainer,
                                    {module_name: {'reconstructed_module': reconstructed_module.to(device)}},
                                    seed)

    if log_to_tb:
        writer.add_scalar(tag=f'{module_name}/metrics/cosine_distance', scalar_value=cosine_distance,
                          global_step=global_step)
        writer.add_scalar(tag=f'{module_name}/metrics/rmse', scalar_value=rmse, global_step=global_step)
        writer.add_scalar(tag=f'{module_name}/metrics/mae', scalar_value=mae, global_step=global_step)
        writer.add_scalar(tag=f'{module_name}/metrics/powered_mae', scalar_value=powered_mae, global_step=global_step)
        writer.add_scalar(tag=f'{module_name}/metrics/perplexity', scalar_value=perplexity, global_step=global_step)

    return {'cosine_distance': cosine_distance,
            'rmse': rmse,
            'mae': mae,
            'powered_mae': powered_mae,
            'perplexity': perplexity}


def eval_sub_modules(reconstructed_modules_dict: dict,
                     writer: SummaryWriter,
                     global_step: int,
                     mlm_trainer: transformers.trainer.Trainer,
                     seed: int,
                     device: torch.device) -> Dict:
    """
    reconstructed_modules_dict:
    {
        module_name: {
            original_module : tensor
            reconstructed_module: tensor
        },
        module_name2: {
            original_module : tensor
            reconstructed_module: tensor
        },
        ...
    }
    """
    all_result_metrics = {}
    for module_name in reconstructed_modules_dict:
        original_module = reconstructed_modules_dict[module_name]['original_module']
        reconstructed_module = reconstructed_modules_dict[module_name]['reconstructed_module']

        all_result_metrics[module_name] = eval_substitution_module(module_name=module_name,
                                                                   original_module=original_module,
                                                                   reconstructed_module=reconstructed_module.detach().cpu(),
                                                                   writer=writer,
                                                                   log_to_tb=False,
                                                                   global_step=global_step,
                                                                   mlm_trainer=mlm_trainer,
                                                                   seed=seed,
                                                                   device=device)

    global_perplexity = compute_perplexity(mlm_trainer, reconstructed_modules_dict, seed)
    writer.add_scalar(tag=f'global/perplexity', scalar_value=global_perplexity, global_step=global_step)

    global_metrics = defaultdict(int)
    for module_name in all_result_metrics:
        for metric_name in ['cosine_distance', 'rmse', 'mae', 'powered_mae']:
            global_metrics[metric_name] += all_result_metrics[module_name][metric_name]

    for metric_name in global_metrics:
        global_metrics[metric_name] = global_metrics[metric_name] / len(all_result_metrics)
        writer.add_scalar(tag=f'global/{metric_name}', scalar_value=global_metrics[metric_name],
                          global_step=global_step)




