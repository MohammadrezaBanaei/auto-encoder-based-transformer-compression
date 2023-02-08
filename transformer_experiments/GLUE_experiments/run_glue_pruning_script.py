import os
import argparse
import json

import yaml


def glue_main(config, input_task_name_):

    save_dir = config['save_dir']
    run_name = None
    save_model = config['save_model']
    seed = config['seed']
    save_strategy = 'no'
    for task_name in [input_task_name_]:
        for compression_ratio in [3, 10, 25]:
            for pruning_module in ["word_embeddings", "key", "output_dense"]:

                additional_args = f"--compression_ratio {compression_ratio} --pruning_module {pruning_module}"
                if run_name is not None:
                    output_dir = os.path.join(save_dir, task_name, f'{run_name}_seed_{seed}')
                else:
                    output_dir = os.path.join(save_dir, task_name,
                                              f'{pruning_module}_cr_{compression_ratio}_seed_{seed}')
                print(f'Processing {output_dir}...')

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    os.system(f"python -m transformer_experiments.GLUE_experiments.run_glue_pruning \
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
                                       --save_strategy {save_strategy} \
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--task_name', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    glue_main(config, args.task_name)
