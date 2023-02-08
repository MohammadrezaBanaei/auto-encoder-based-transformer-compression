import os
from typing import Dict, List

import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation import eval_substitution_module
from utils.utils_funcs import create_dirs_if_needed


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_epoch_info(module_name: str, writer: SummaryWriter, optimizer: _LRScheduler, epoch: int):
    writer.add_scalar(f'{module_name}/training_info/lr', get_lr(optimizer), epoch)


def log_iteration(module_name: str, writer: SummaryWriter, multi_obj_losses: Dict, iter_: int):
    for loss_name, (coeff, loss_value) in multi_obj_losses.items():
        writer.add_scalar(f'{module_name}/raw_losses/{loss_name}', loss_value, iter_)
        writer.add_scalar(f'{module_name}/losses_coeffs/{loss_name}', coeff, iter_)
        writer.add_scalar(f'{module_name}/loss_times_coeff/{loss_name}', loss_value * coeff, iter_)


def save_state(module_name: str, checkpoints_dir: str, model: nn.Module, optimizer: Optimizer,
               embedding_latents: torch.tensor):
    module_checkpoint_dir = os.path.join(checkpoints_dir, module_name)
    create_dirs_if_needed([module_checkpoint_dir])

    torch.save(model.state_dict(), os.path.join(module_checkpoint_dir, 'model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(module_checkpoint_dir, 'optimizer.pt'))
    if embedding_latents is not None:
        torch.save(embedding_latents, os.path.join(module_checkpoint_dir, 'latents.pt'))


def train_model(module_name: str, early_stopping_metric: str, checkpoints_dir: str, epochs: int, additional_epochs: int,
                model: nn.Module, data_loader: DataLoader, validation_dataset: Dataset, writer: SummaryWriter,
                optimizer: Optimizer, scheduler: _LRScheduler, losses: Dict, mlm_trainer: transformers.trainer.Trainer,
                seed: int, device: torch.device, module_training_setting: str, coef_normalization='none'):
    best_early_stopping_obj_score = None
    iter_ = 0
    model.train()

    for epoch in range(0, epochs + additional_epochs):
        log_to_tb = ((epoch == 0) or ((epoch + 1) % 2 == 0))
        if log_to_tb:
            log_epoch_info(module_name, writer, optimizer, epoch)
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                log_to_tb_iter = (iter_ % len(data_loader) == 0) or ((iter_ + 1) % 50 == 0)
                if data_loader.dataset.coefficients is None:
                    batch, ids = batch_data
                    rec_coef = None
                else:
                    batch, ids, rec_coef = batch_data
                    rec_coef = rec_coef.to(device, non_blocking=True)
                    if coef_normalization == 'sum':
                        rec_coef = rec_coef / rec_coef.sum()
                    if coef_normalization.startswith('sum_log'):
                        temp = float(coef_normalization[7:])
                        rec_coef = torch.log(rec_coef) + temp
                        assert rec_coef.min() > 0.0,  rec_coef.min()
                        rec_coef = rec_coef / rec_coef.sum()
                    if coef_normalization.startswith('pow'):
                        alpha = float(coef_normalization[3:])
                        rec_coef = torch.pow(rec_coef, alpha)
                    if coef_normalization.startswith('sum_pow'):
                        alpha = float(coef_normalization[7:])
                        rec_coef = torch.pow(rec_coef, alpha)
                        rec_coef = rec_coef / rec_coef.sum()

                batch = batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                out, _ = model(batch, ids)

                multi_obj_losses = {loss_name: (coeff, loss_func(output=out, target=batch, epoch=epoch, rec_coef=rec_coef))
                                    for loss_name, (coeff, loss_func) in losses.items()}
                final_loss = sum([coeff * loss_value for loss_name, (coeff, loss_value) in multi_obj_losses.items()])

                if log_to_tb_iter:
                    log_iteration(module_name, writer, multi_obj_losses, iter_)
                    writer.add_scalar(f'{module_name}/final_loss', final_loss, iter_)

                final_loss.backward()
                optimizer.step()

                iter_ += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            reconstructed_module, module_latents = model(validation_dataset.matrix.to(device),
                                                         validation_dataset.row_ids)
            result_metrics = eval_substitution_module(module_name=module_name,
                                                      original_module=validation_dataset.matrix,
                                                      reconstructed_module=reconstructed_module.detach().cpu(),
                                                      writer=writer, log_to_tb=log_to_tb, global_step=epoch,
                                                      mlm_trainer=mlm_trainer, seed=seed, device=device)
            early_stopping_obj_score = result_metrics[early_stopping_metric]
        model.train()

        if best_early_stopping_obj_score is None or early_stopping_obj_score < best_early_stopping_obj_score:
            best_early_stopping_obj_score = early_stopping_obj_score
            save_state(module_name=module_name, checkpoints_dir=checkpoints_dir, model=model, optimizer=optimizer,
                       embedding_latents=module_latents)
            best_reconstructed_module = reconstructed_module

    return best_reconstructed_module
