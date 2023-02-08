import os
import argparse
import json

import yaml


def glue_main(config):
    pre_trained_models_dir_list = config['pre_trained_models_dir']
    if config["pre_trained_models_root"] is not None:
        for root, dirs, files in os.walk(config["pre_trained_models_root"]):
            for dir in dirs:
                if "run_" in dir:
                    pre_trained_models_dir_list.append(os.path.join(root, dir))

    task_name = config['task_name']
    save_dir = config['save_dir']
    run_name = config['run_name']
    run_type = config['run_type']
    save_model = config['save_model']
    save_strategy = config['save_strategy']
    ae_random_reinit = config['ae_random_reinit']
    train_frac = config['train_frac'] if 'train_frac' in config else 1.0
    for pretrained_model_dir in pre_trained_models_dir_list:
        with open(os.path.join(pretrained_model_dir, 'config.yml'), 'r') as stream:
            pretrained_config = yaml.load(stream, Loader=yaml.FullLoader)

        seed = pretrained_config['global']['seed']
        print('Setting seed as in pretrained model. Seed from config will be ignored!')

        additional_args = ""
        if run_type == 'ae':
            run_name = f"{config['run_name']}_{pretrained_model_dir.split('/')[-1]}" if run_name is not None else None
            additional_args = f"--pretrained_ae_dir {pretrained_model_dir} "
        elif run_type == 'svd':
            target_cr_info = str(config['svd_info']['svd_cr'])
            run_name = f'''{config['run_name']}_{target_cr_info}''' if run_name is not None else None
            additional_args = f"--svd --svd_config_dir {pretrained_model_dir}"
        if run_name is not None:
            output_dir = os.path.join(save_dir, task_name, f'{run_name}_trainfrac_{train_frac}_seed_{seed}')
        else:
            output_dir = os.path.join(save_dir, f'{task_name}', f'trainfrac_{train_frac}_seed_{seed}')
        print(f'Processing {output_dir}...')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.system(f"python -m transformer_experiments.GLUE_experiments.run_glue \
                               --output_dir {output_dir} \
                               --task_name {task_name} \
                               --model_name_or_path=bert-base-uncased \
                               --evaluation_strategy=steps \
                               --logging_dir {output_dir} \
                               --max_seq_length 128 \
                               --do_train \
                               --do_predict \
                               --save_strategy epoch \
                               --fp16 \
                               --do_eval \
                               --eval_steps {config['eval_steps']} \
                               --logging_steps 10 \
                               --per_device_train_batch_size 32 \
                               --per_device_eval_batch_size 32 \
                               --learning_rate 2e-5 \
                               --gradient_accumulation_steps 2 \
                               --num_train_epochs {config['epochs']} \
                               --seed={seed} {additional_args} \
                               --train_frac {train_frac} \
                               --save_strategy {save_strategy} \
                               --ae_random_reinit {ae_random_reinit} \
                               --wiki_text_103_url {config['lm_dataset']['wiki_text_103_url']} \
                               --text_download_folder {config['lm_dataset']['text_download_folder']} \
                               --lm_text_dataset_path {config['lm_dataset']['lm_text_dataset_path']} \
                               --text_frequency_dataset_path {config['lm_dataset']['text_frequency_dataset_path']} \
                               --model_name {config['lm_dataset']['model_name']} \
                               --save_model {save_model}")

            with open(os.path.join(output_dir, 'exp_cfg.json'), 'w') as json_file:
                json.dump(config, json_file)
        else:
            print(f'Skipping {output_dir}... already exists')

        if run_type == 'svd':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    glue_main(config)
