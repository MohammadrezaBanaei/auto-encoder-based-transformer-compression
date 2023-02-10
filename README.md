# Revisiting Offline Compression: Going Beyond Factorization-based Methods for Transformer Language Models

Code for the paper: "[Revisiting Offline Compression: Going Beyond Factorization-based Methods for Transformer Language Models](https://arxiv.org/abs/2302.04045)" [to appear in EACL 2023]


## Abstract
Recent transformer language models achieve outstanding results in many natural language processing (NLP) tasks. However, their enormous size often makes them impractical on memory-constrained devices, requiring practitioners to compress them to smaller networks.
In this paper, we explore offline compression methods, meaning computationally-cheap approaches that do not require further fine-tuning of the compressed model. 
We challenge the classical matrix factorization methods by proposing a novel, better-performing autoencoder-based framework. We perform a comprehensive ablation study of our approach, examining its different aspects over a diverse set of evaluation settings. Moreover, we show that enabling collaboration between modules across layers by compressing certain modules together positively impacts the final model performance. 
Experiments on various NLP tasks demonstrate that our approach significantly outperforms commonly used factorization-based offline compression methods.

## Requirements

Create your new conda environment by:

`conda env create -f environment.yaml`

and then activate it by:

`conda activate ae-compression-env`.

Alternatively, you may install needed packages by running:

`pip install -r requirements.txt`. 

## Experiments

### Compressed modules generation
Please use the following command in order to generate compressed modules.
```
CUDA_VISIBLE_DEVICES=0 python main.py
```
The aforementioned command will read the config file located at `configs/config.yaml`.
Please update the output paths, model type `('svd', 'ae', 'kronecker', 'tucker', 'pruning')`,
and hyper-parameters that you would like to use by updating this configuration file.

### GLUE experiments

To use a specific experiments containing your compressed modules that you want to apply when training GLUE tasks 
fill the `pre_trained_models_dir` field with absolute paths to your experiments. For example:

`pre_trained_models_dir = ["main_dir/run_1", "main_dir/run_2"]`.

Alternatively, you may provide a root directory containing your all experiments
(experiments need to be in a directory starting with a `run_` prefix):

`pre_trained_models_root = "main_dir"`.

#### One module compressed by SVD or AE or multiple modules compressed by SVD:

Run from `transformer_experiments/GLUE_experiments` directory with the following command:

```
PYTHONPATH=.:repo_root_directory CUDA_VISIBLE_DEVICES=0 python run_glue_script.py  --config glue_cfg.json
```

Please update the paths, `task_name` and your intended hyper-parameters in `glue_cfg.json` in the same directory.

#### Multiple modules compressed by AE:

Run from `transformer_experiments/GLUE_experiments` directory with the following command:

```
PYTHONPATH=.:repo_root_directory CUDA_VISIBLE_DEVICES=0 python run_glue_script_multiple_modules.py  --config glue_cfg.json
```

Please update the paths, `task_name` and your intended hyper-parameters in `glue_cfg.json` in the same directory.


#### Modules compressed using pruning:

Run from `transformer_experiments/GLUE_experiments` directory with the following command:

```
PYTHONPATH=.:repo_root_directory CUDA_VISIBLE_DEVICES=0 python run_glue_pruning_script.py  --config glue_cfg.json --task_name GLUE_TASK_NAME
```

Please update the paths, `task_name`, `run_type` and your intended hyper-parameters in `glue_cfg.json` in the same directory.

## Reference

If you found the provided code useful, please consider citing our work.

```
@article{banaeirevisitingofflinecompression,
  doi = {10.48550/ARXIV.2302.04045},
  url = {https://arxiv.org/abs/2302.04045},
  author = {Banaei, Mohammadreza and Bałazy, Klaudia and Kasymov, Artur and Lebret, Rémi and Tabor, Jacek and Aberer, Karl},
  title = {Revisiting Offline Compression: Going Beyond Factorization-based Methods for Transformer Language Models},
  publisher = {arXiv},
  year = {2023}
}
```
