import os
from functools import partial
from typing import List, Dict, Tuple

import yaml
import torch
import pickle
import math
import copy
from tensorboard import summary
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import transformers
from utils import transformers_data_utils
from utils.ae_train import train_model
from utils.coefficients_utils import get_coefficients_name, get_coefficients
from utils.custom_dataset import MatrixDataset, HiddenStateDataset
from losses.cosine_distance_loss import cosine_distance_loss
from losses.l1_norm_loss import l1_norm_loss
from losses.l2_norm_loss import l2_norm_loss
from models.auto_encoder import AutoEncoder
from models.decoder import LinearDecoder, NonLinearDecoder
from models.encoder import LinearEncoder, NonLinearEncoder
from models.kronecker_model import KroneckerProductModel
from models.tucker_model import TuckerDecompositionModel
from utils.utils_funcs import init_model_with_svd, get_mlm_trainer, save_lin_rec_stats, \
    create_dirs_if_needed, set_global_seed, row_wise_quantize, find_quadratic_solution
from evaluation import eval_sub_modules


def init_ae_model(enc_type: str, dec_type: str, original_module: torch.tensor, symmetric_ae_scale: bool,
                  latent_dim_relative: int, weights_path: str, enc_layer_sizes: List, dec_layer_sizes: List,
                  cr: float, device: torch.device, enforce_norm: bool, finetune_org_norm=True):
    n, m = original_module.shape[0], original_module.shape[1]

    if enforce_norm:
        org_norm = torch.linalg.norm(original_module, dim=-1).to(device)  # 2-norm
    else:
        org_norm = None

    if dec_type == 'linear':
        if enforce_norm:
            ld = int(((n * m / cr) - m - n) / (n + m))
        else:
            ld = int(((n * m / cr) - m) / (n + m))
        decoder = LinearDecoder(out_dim=m, latent_dim=ld, org_norm=org_norm, finetune_org_norm=finetune_org_norm)
    elif dec_type == 'non_linear':
        compressed_numel = (n * m) / cr
        a, b, c = 0, 0, 0  # ax^2 + bx + c
        b += (n * latent_dim_relative + latent_dim_relative)  # latent space and bias

        a += (latent_dim_relative * dec_layer_sizes[0])  # First layer
        b += dec_layer_sizes[0]  # First layer bias
        for i in range(len(dec_layer_sizes) - 1):
            a += (dec_layer_sizes[i] * dec_layer_sizes[i + 1])
            b += dec_layer_sizes[i + 1]
        b += (dec_layer_sizes[-1] * m)
        c += m
        c -= compressed_numel
        if enforce_norm:
            c += n

        x = find_quadratic_solution(a, b, c)

        ld = x * latent_dim_relative
        dec_layer_sizes = [el * x for el in dec_layer_sizes]
        decoder = NonLinearDecoder(out_dim=m, latent_dim=ld,
                                   dec_layer_sizes=dec_layer_sizes, org_norm=org_norm,
                                   finetune_org_norm=finetune_org_norm)
    else:
        raise ValueError(f'No such encoder type as {dec_type}')

    if enc_type == 'linear':
        encoder = LinearEncoder(input_dim=m, latent_dim=ld)
    elif enc_type == 'non_linear':
        if symmetric_ae_scale:
            enc_layer_sizes = [el * x for el in enc_layer_sizes]
        encoder = NonLinearEncoder(input_dim=m, latent_dim=ld, enc_layer_sizes=enc_layer_sizes)
    else:
        raise ValueError(f'No such encoder type as {enc_type}')

    model = AutoEncoder(encoder=encoder, decoder=decoder, weights_path=weights_path)
    model.to(device)
    print(model)

    return model


def init_kronecker_model(r: int, original_matrix_shape: int, device: torch.device):
    model = KroneckerProductModel(r=r, original_matrix_shape=original_matrix_shape)
    model.to(device)
    return model


def init_tucker_model(rank: list, input_matrix: torch.tensor, module_name: str, device: torch.device,
                      num_layers: int, num_heads: int):
    model = TuckerDecompositionModel(tucker_rank=rank, input_matrix=input_matrix, module_name=module_name,
                                     num_layers=num_layers, num_heads=num_heads)
    model.to(device)
    return model


def get_pruned_module(original_matrix: torch.tensor,
                      device: torch.device,
                      compression_ratio: float,
                      pruning_method="magnitude_pruning"):
    if pruning_method != "magnitude_pruning":
        raise ValueError('Only magnitude pruning is currently supported for pruning baseline.')
    matrix_num_elements = original_matrix.numel()
    non_zero_values = round(matrix_num_elements / compression_ratio)  # number of non-zero elements after pruning
    smallest_abs_value = original_matrix.abs().reshape(-1).sort().values[-non_zero_values]
    pruned_matrix = copy.deepcopy(original_matrix)
    pruned_matrix[pruned_matrix.abs() < smallest_abs_value] = 0
    pruned_matrix = pruned_matrix.to(device)
    return pruned_matrix


def main(config: Dict):
    main_dir = config["paths"]["main_dir"]
    exp_dir = os.path.join(main_dir, config["paths"]["experiment_name"])
    log_dir = os.path.join(exp_dir, 'logs')
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    create_dirs_if_needed([main_dir, exp_dir, log_dir, checkpoints_dir])

    seed_value = config["global"]["seed"]
    set_global_seed(seed_value=seed_value)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlm_trainer = get_mlm_trainer(config['dataset']['lm_dataset'], seed=config["global"]["seed"])

    # We assume the lower metric is the better it is.
    early_stopping_metric = config['training']['early_stopping_metric']

    transformer_data, transformer_bias_data = transformers_data_utils.get_model_weight_dict()

    coefficients_name = get_coefficients_name(config)

    training_data = {}
    training_data_coefficients = {}

    if config['dataset']['to_compress']['word_embeddings']:
        training_data['word_embeddings'] = transformer_data['embedding']['word_embeddings']
        training_data_coefficients['word_embeddings'] = get_coefficients(coefficients_name, 'word_embeddings',
                                                                         config, compute=True)

    modules_training_setting_name = config['modules_training_setting']
    if config['dataset']['to_compress']['all_keys']:
        keys_coefficients = get_coefficients(coefficients_name, 'layer', config, submodule_name='key', compute=True)
    if config['dataset']['to_compress']['all_queries']:
        queries_coefficients = get_coefficients(coefficients_name, 'layer', config, submodule_name='query',
                                                compute=True)
    if config['dataset']['to_compress']['all_values']:
        values_coefficients = get_coefficients(coefficients_name, 'layer', config, submodule_name='value', compute=True)
    if config['dataset']['to_compress']['all_attention_outputs']:
        attention_output_coefficients = get_coefficients(coefficients_name, 'layer', config,
                                                         submodule_name='attention_output', compute=True)
    if config['dataset']['to_compress']['all_output_denses']:
        output_dense_coefficients = get_coefficients(coefficients_name, 'layer', config, submodule_name='output_dense',
                                                     compute=True)
    if config['dataset']['to_compress']['all_intermediate_denses']:
        intermediate_dense_coefficients = get_coefficients(coefficients_name, 'layer', config,
                                                           submodule_name='intermediate_dense', compute=True)

    if modules_training_setting_name == 'separated':
        for module_name in transformer_data:
            if 'layer' in module_name:
                if config['dataset']['to_compress']['all_keys']:
                    training_data[f'{module_name}_key'] = transformer_data[module_name]['key']
                    training_data_coefficients[f'{module_name}_key'] = \
                        None if keys_coefficients is None else keys_coefficients[int(module_name.split('_')[1])]
                if config['dataset']['to_compress']['all_queries']:
                    training_data[f'{module_name}_query'] = transformer_data[module_name]['query']
                    training_data_coefficients[f'{module_name}_query'] = \
                        queries_coefficients[
                            int(module_name.split('_')[1])] if queries_coefficients is not None else None
                if config['dataset']['to_compress']['all_values']:
                    training_data[f'{module_name}_value'] = transformer_data[module_name]['value']
                    training_data_coefficients[f'{module_name}_value'] = \
                        values_coefficients[int(module_name.split('_')[1])] if values_coefficients is not None else None
                if config['dataset']['to_compress']['all_attention_outputs']:
                    training_data[f'{module_name}_attention_output'] = transformer_data[module_name]['attention_output']
                    training_data_coefficients[f'{module_name}_attention_output'] = \
                        attention_output_coefficients[
                            int(module_name.split('_')[1])] if attention_output_coefficients is not None else None
                if config['dataset']['to_compress']['all_intermediate_denses']:
                    training_data[f'{module_name}_intermediate_dense'] = transformer_data[module_name][
                        'intermediate_dense']
                    training_data_coefficients[f'{module_name}_intermediate_dense'] = \
                        intermediate_dense_coefficients[
                            int(module_name.split('_')[1])] if intermediate_dense_coefficients is not None else None
                if config['dataset']['to_compress']['all_output_denses']:
                    training_data[f'{module_name}_output_dense'] = transformer_data[module_name]['output_dense']
                    training_data_coefficients[f'{module_name}_output_dense'] = \
                        output_dense_coefficients[
                            int(module_name.split('_')[1])] if output_dense_coefficients is not None else None
    elif modules_training_setting_name == 'concatenated':
        if config['dataset']['to_compress']['all_keys']:
            training_data['keys'] = torch.cat([transformer_data[f"layer_{layer_num}"]['key']
                                               for layer_num in range(len(mlm_trainer.model.bert.encoder.layer))])
            training_data_coefficients['keys'] = None if keys_coefficients is None \
                else keys_coefficients.flatten(0, 1)
        if config['dataset']['to_compress']['all_queries']:
            training_data['queries'] = torch.cat([transformer_data[f"layer_{layer_num}"]['query']
                                                  for layer_num in range(len(mlm_trainer.model.bert.encoder.layer))])
            training_data_coefficients['queries'] = None if queries_coefficients is None \
                else queries_coefficients.flatten(0, 1)
        if config['dataset']['to_compress']['all_values']:
            training_data['values'] = torch.cat([transformer_data[f"layer_{layer_num}"]['value']
                                                 for layer_num in range(len(mlm_trainer.model.bert.encoder.layer))])
            training_data_coefficients['values'] = None if values_coefficients is None \
                else values_coefficients.flatten(0, 1)
        if config['dataset']['to_compress']['all_attention_outputs']:
            training_data['attention_outputs'] = torch.cat([transformer_data[f"layer_{layer_num}"]['attention_output']
                                                            for layer_num in
                                                            range(len(mlm_trainer.model.bert.encoder.layer))])
            training_data_coefficients['attention_outputs'] = None if attention_output_coefficients is None \
                else attention_output_coefficients.flatten(0, 1)
        if config['dataset']['to_compress']['all_output_denses']:
            training_data['output_denses'] = torch.cat(
                [transformer_data[f"layer_{layer_num}"]['output_dense']
                 for layer_num in
                 range(len(mlm_trainer.model.bert.encoder.layer))])
            training_data_coefficients['output_denses'] = None if output_dense_coefficients is None \
                else output_dense_coefficients.flatten(0, 1)
        if config['dataset']['to_compress']['all_intermediate_denses']:
            training_data['intermediate_denses'] = torch.cat(
                [transformer_data[f"layer_{layer_num}"]['intermediate_dense']
                 for layer_num in
                 range(len(mlm_trainer.model.bert.encoder.layer))])
            training_data_coefficients['intermediate_denses'] = None if intermediate_dense_coefficients is None \
                else intermediate_dense_coefficients.flatten(0, 1)
    else:
        raise ValueError(f'No such modules_training_setting as {modules_training_setting_name}.')

    writer = SummaryWriter(os.path.join(log_dir, config['model']['type']))
    writer.add_text('config', str(config))

    compressed_modules = {}
    for module_name in training_data:
        module_dataset = MatrixDataset(matrix=training_data[module_name],
                                       row_ids=[i for i in range(training_data[module_name].shape[0])],
                                       coefficients=training_data_coefficients[module_name])

        mlm_trainer = get_mlm_trainer(config['dataset']['lm_dataset'], seed=config["global"]["seed"])
        original_module = module_dataset.matrix

        bs = config['training']['batch_size']
        if bs == -1:
            bs = original_module.shape[0]

        loader = DataLoader(module_dataset, batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=False)

        if "key" in module_name:
            latent_dim_relative = config['model']['latent_dim']['key']
            module_type = 'key'
        elif "value" in module_name:
            latent_dim_relative = config['model']['latent_dim']['value']
            module_type = 'value'
        elif "query" in module_name or module_name == "queries":
            latent_dim_relative = config['model']['latent_dim']['query']
            module_type = 'query'
        elif module_name == "word_embeddings":
            latent_dim_relative = config['model']['latent_dim']['token_emb']
            module_type = 'word_embeddings'
        elif 'attention_output' in module_name or module_name == 'attention_outputs':
            latent_dim_relative = config['model']['latent_dim']['attention_output']
            module_type = 'attention_output'
        elif 'intermediate_dense' in module_name:
            latent_dim_relative = config['model']['latent_dim']['intermediate_dense']
            module_type = 'intermediate_dense'
        elif 'output_dense' in module_name:
            latent_dim_relative = config['model']['latent_dim']['output_dense']
            module_type = 'output_dense'
        else:
            raise Exception(f"Unexpected module name '{module_name}' in run-time")

        losses_cfg = config['training']['loss']
        losses = {}
        if losses_cfg['cos_dist']['coeff'] > 0.0:
            losses['cos_dist'] = (losses_cfg['cos_dist']['coeff'], partial(cosine_distance_loss, dim=-1))
        if losses_cfg['l2_norm']['coeff'] > 0.0:
            losses['l2_norm'] = (losses_cfg['l2_norm']['coeff'], l2_norm_loss)
        if losses_cfg['l1_norm']['coeff'] > 0.0:
            losses['l1_norm'] = (losses_cfg['l1_norm']['coeff'], partial(l1_norm_loss,
                                                                         config=losses_cfg['l1_norm'],
                                                                         changing_epochs_num=config['training'][
                                                                             'epochs']))

        compressed_modules[module_name] = {}
        input_shape = module_dataset.matrix.shape
        if config['model']['type'] == 'svd':
            latent_dim = int(
                (input_shape[0] * input_shape[1]) / (config['model']['cr'] * (input_shape[0] + input_shape[1])))
            recon_module, best_it, cr, subs_module_size = save_lin_rec_stats(module_name=module_name,
                                                                             writer=writer,
                                                                             original_matrix=original_module,
                                                                             rec_coef=training_data_coefficients[
                                                                                 module_name],
                                                                             latent_dim=latent_dim,
                                                                             iters=range(1, config['training'][
                                                                                 'svd_max_iters']),
                                                                             mlm_trainer=mlm_trainer,
                                                                             eval_metric=early_stopping_metric,
                                                                             seed=config["global"]["seed"],
                                                                             device=device)
            if 'svd_best_it' not in config['training']:
                config['training']['svd_best_it'] = {}
                config['training']['svd_cr'] = {}
            config['training']['svd_best_it'][module_name] = best_it
            config['training']['svd_cr'][module_name] = cr
        elif config['model']['type'] == 'ae':
            model = init_ae_model(enc_type=config['model']['ae']['encoder_type'],
                                  dec_type=config['model']['ae']['decoder_type'],
                                  original_module=original_module,
                                  symmetric_ae_scale=config['model']['ae']['symmetric_ae_scale'],
                                  latent_dim_relative=latent_dim_relative,
                                  weights_path=config['model']['weights_path'],
                                  enc_layer_sizes=config['model']['ae']['enc_layer_sizes'],
                                  dec_layer_sizes=config['model']['ae']['dec_layer_sizes'],
                                  cr=config['model']['cr'],
                                  device=device,
                                  enforce_norm=config['model']['ae']['enforce_norm'],
                                  finetune_org_norm=False)
            if config['model']['svd_model_init']['enabled'] == True:
                model = init_model_with_svd(input_matrix=training_data[module_name].cpu().detach().numpy(), model=model,
                                            num_iters=config['model']['svd_model_init']['svd_iters'], device=device)

            latent_dim = model.get_latent_dim()
            subs_module_size = model.get_substitution_module_size() + latent_dim * input_shape[0]
            cr = original_module.numel() / subs_module_size
        elif config['model']['type'] == 'pruning':
            recon_module = get_pruned_module(original_matrix=original_module,
                                             device=device,
                                             compression_ratio=config['model']['cr'])
            subs_module_size = recon_module[recon_module.abs() > 0].numel()
            cr = original_module.numel() / subs_module_size
        elif config['model']['type'] == 'quantization':
            quant_bits = config['model']['quantization']['n_bits']
            recon_module = row_wise_quantize(input_matrix=original_module, n_bits=quant_bits).to(device)

            subs_module_size = (2 * original_module.shape[0]) + \
                original_module.numel() * quant_bits / (original_module.element_size() * 8)
            cr = original_module.numel() / subs_module_size
        elif config['model']['type'] == 'kronecker':
            kronecker_cfg = config['model']['kronecker']
            model = init_kronecker_model(r=kronecker_cfg['r'], original_matrix_shape=original_module.shape,
                                         device=device)
            subs_module_size = model.get_substitution_module_size()
            cr = original_module.numel() / subs_module_size
        elif config['model']['type'] == 'tucker':
            tucker_cfg = config['model']['tucker']
            num_layers = mlm_trainer.model.config.num_hidden_layers
            num_heads = mlm_trainer.model.config.num_attention_heads
            model = init_tucker_model(rank=tucker_cfg['rank'][module_type],
                                      module_name=module_type,
                                      input_matrix=training_data[module_name],
                                      device=device,
                                      num_layers=num_layers,
                                      num_heads=num_heads)
            subs_module_size = model.get_substitution_module_size()
            cr = original_module.numel() / subs_module_size
        else:
            raise ValueError('Wrong model type given.')

        if config['model']['type'] not in ['svd', 'pruning', 'quantization']:  # TODO: check if others should be included
            optimizer = Adam(model.parameters(), lr=config['training']['lr'])
            scheduler = StepLR(optimizer,
                               step_size=config['training']['step_lr_scheduler']['step_size'],
                               gamma=config['training']['step_lr_scheduler']['gamma'])
            recon_module = train_model(module_name=module_name,
                                       early_stopping_metric=early_stopping_metric,
                                       checkpoints_dir=checkpoints_dir,
                                       epochs=config['training']['epochs'],
                                       additional_epochs=config['training']['additional_epochs'],
                                       model=model,
                                       data_loader=loader,
                                       validation_dataset=module_dataset,
                                       writer=writer,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       losses=losses,
                                       mlm_trainer=mlm_trainer,
                                       seed=config["global"]["seed"],
                                       device=device,
                                       coef_normalization=config['coefficients']['norm'],
                                       module_training_setting=modules_training_setting_name)
            writer.add_text('model', str(model), 0)
        else:
            # per module evaluation for methods that don't need training
            temp_compressed_modules = {module_name: {}}
            temp_compressed_modules[module_name]['reconstructed_module'] = recon_module
            temp_compressed_modules[module_name]['original_module'] = original_module

            eval_sub_modules(reconstructed_modules_dict=temp_compressed_modules,
                             writer=writer,
                             global_step=0,
                             mlm_trainer=mlm_trainer,
                             seed=config["global"]["seed"],
                             device=device)

        compressed_modules[module_name]['reconstructed_module'] = recon_module
        compressed_modules[module_name]['original_module'] = original_module

        writer.add_scalar(f'{module_name}/training_info/compression_ratio', cr, 0)
        writer.add_scalar(f'{module_name}/training_info/substitution_module_size', subs_module_size, 0)

    eval_sub_modules(reconstructed_modules_dict=compressed_modules,
                     writer=writer,
                     global_step=0,
                     mlm_trainer=mlm_trainer,
                     seed=config["global"]["seed"],
                     device=device)

    writer.close()

    with open(os.path.join(checkpoints_dir, 'transformer_bias_data.pkl'), 'wb') as f:
        pickle.dump(transformer_bias_data, f)

    with open(os.path.join(exp_dir, 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


if __name__ == '__main__':
    with open("configs/config.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    main(config)
