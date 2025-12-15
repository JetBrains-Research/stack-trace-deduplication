[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)

# 🐑 Stack Trace Deduplication 🐑

This repository provides an overview and instructions for replicating experiments on stack trace deduplication from our paper ["_Stack Trace Deduplication: Faster, More Accurately, and in More Realistic Scenarios_"](https://arxiv.org/abs/2412.14802), including details on code structure, setup, and execution steps. Below, you will find a breakdown of the key directories and scripts essential for the experiments.

## 🏗️ Repository structure

The directory [ea/sim/main/methods/neural/encoders/](ea/sim/main/methods/neural/encoders/) contains the implementation of the neural encoders used in the experiments:
- our [embedding model](ea/sim/main/methods/neural/encoders/texts/rnn.py) presented in the paper,
- our implementation of the [DeepCrash model](ea/sim/main/methods/neural/encoders/tokens/skip_gram_BOW.py).

The directory [ea/sim/main/methods/neural/cross_encoders/](ea/sim/main/methods/neural/cross_encoders/) contains the implementation of the models that involve interaction between stack traces when computing similarity scores:
- our [cross-encoder](ea/sim/main/methods/neural/cross_encoders/rnn.py) presented in the paper,
- [S3M](ea/sim/main/methods/neural/cross_encoders/s3m.py),
- [Lerch](ea/sim/main/methods/neural/cross_encoders/lerch.py).

The implementation of the FaST model is located [here](ea/sim/main/methods/classic/fast.py).

The training scripts are located in the directory [ea/sim/dev/scripts/training/training/](ea/sim/dev/scripts/training/training/).

The evaluation scripts are located in the directory [ea/sim/dev/scripts/training/evaluating/](ea/sim/dev/scripts/training/evaluating/).

## 🗃️ Data for experiments

To train and evaluate the models, you need a dataset of stack traces. In our paper, we present a novel industrial dataset
and also use established open-source ones.

**SlowOps**, our new dataset of _Slow Operation Assertion_ stack traces from IntelliJ-based products, can be found [here](https://doi.org/10.5281/zenodo.14364857).

Open-source datasets, namely **Ubuntu**, **Eclipse**, **NetBeans**, and **Gnome**, can be found [here](https://doi.org/10.5281/zenodo.5746044).

_Note_: to run our models on open-source datasets, you need to transform them into the right format. The script for doing that is available [here](helpers/dataset_converter.py).

## 🏃 Running the code

### 1. Install the required packages
```bash
poetry install
``` 

### 2. Setup

To run experiments for a specific dataset, create a designated directory `ARTIFACTS_DIR` for the dataset.
Inside this directiry, there should be a `config.json` file with the following structure:
```json
{
    "reports_dir": "path/to/dataset/reports",
    "labels_dir": "path/to/dataset/labels",
    "data_name": "dataset_name",
    "scope": "dataset_scope (same as data_name if not specified)",
    "train_start": "days from the first report to start training",
    "train_longitude": "longitude of the training period in days",
    "val_start": "days from the first report to start validation",
    "val_longitude": "longitude of the validation period in days",
    "test_start": "days from the first report to start testing",
    "test_longitude": "longitude of the testing period in days",
    "forget_days": "days to use for report attaching",
    "dup_attach": "whether to attach duplicates"
}
```

In the `reports_dir` directory, all reports should be located. Each report should be a separate file with the following
name format: `report_id.json`.

In the `labels_dir` directory, there should be a CSV file with the following structure:
```
timestamp,rid,iid
...
```
where `timestamp` is the timestamp of the report, `rid` is the report ID, and `iid` is the category ID.

An  example of a config can be found in the [NetBeans_config_example.json](NetBeans_config_example.json) file.

### 3. Run the experiments

#### Generating the training dataset

Before training an embedding model (`embedding_model`, `cross_encoder`, `deep_crash`, `s3m`), the training dataset should be generated from the reports and labels. Scripts for generating the training dataset are located in the directory  [ea/sim/dev/scripts/data/dataset/](ea/sim/dev/scripts/data/dataset/). Here is an example of how to generate the training dataset for the NetBeans dataset:

```bash
python ea/sim/dev/scripts/data/dataset/nb/main.py --reports_dir=path/to/dataset/NetBeans/ --state_path=path/to/dataset/NetBeans/state.csv --save_dir=path/to/save/netbeans/
```

The generated dataset should be passed to training scripts as a `dataset_dir` argument.

#### Training the models

Training scripts are located in the directory [ea/sim/dev/scripts/training/training](ea/sim/dev/scripts/training/training).
To run the script, `ARTIFACTS_DIR` should be specified as an environment variable.

```bash
export ARTIFACTS_DIR=artifacts_dir; python ea/sim/dev/scripts/training/training/<script_name>.py  
```

Here are the available scripts for training:
- Embedding model
    ```bash
    python ea/sim/dev/scripts/training/training/train_model.py --path_to_save='path/to/save/model/embedding_model.pth'
    ```
- Cross Encoder
    ```bash
    python ea/sim/dev/scripts/training/training/train_model.py --path_to_save='path/to/save/model/cross_encoder.pth'
    ```

- DeepCrash
    ```bash
    python ea/sim/dev/scripts/training/training/train_model.py --path_to_save='path/to/save/model/deep_crash.pth'
    ```

- S3M
    ```bash
    python ea/sim/dev/scripts/training/training/train_s3m.py --path_to_save='path/to/save/model/s3m.pth'
    ```

#### Evaluating the models

Evaluation scripts are located in the directory [ea/sim/dev/scripts/training/evaluating](ea/sim/dev/scripts/training/evaluating).
To run the script, `ARTIFACTS_DIR` should be specified as an environment variable.

```bash
export ARTIFACTS_DIR=artifacts_dir; python ea/sim/dev/scripts/training/evaluating/<script_name>.py  
```

Here are the available scripts for evaluation:
- Embedding model
    ```bash
    python ea/sim/dev/scripts/training/evaluating/retrieval_stage.py --model_ckpt_path='path/to/model/embedding_model.pth'
    ```

- Cross Encoder
    ```bash
    python ea/sim/dev/scripts/training/evaluating/scoring_stage.py --cross_encoder_path='path/to/model/cross_encoder.pth'
    ```

- DeepCrash
    ```bash
    python ea/sim/dev/scripts/training/evaluating/retrieval_stage.py --model_ckpt_path='path/to/model/deep_crash.pth'
    ```

- S3M
    ```bash
    python ea/sim/dev/scripts/training/evaluating/eval_s3m.py --model_ckpt_path='path/to/model/s3m.pth'
    ```

- FaST
    ```bash
    python ea/sim/dev/scripts/training/evaluating/eval_fast.py 
    ```

- Lerch
    ```bash
    python ea/sim/dev/scripts/training/evaluating/eval_lerch.py 
    ```

- OpenAI embedding model

  First, precompute the embeddings using [ea/sim/dev/scripts/training/training/embeddings/main.py](ea/sim/dev/scripts/training/training/embeddings/main.py).
  Then, run the following script:
    ```bash
    python ea/sim/dev/scripts/training/evaluating/openai/run.py 
    ```

The results of the evaluation will be saved in the `ARTIFACTS_DIR` directory.

## 👩🏻‍🔬 Citing

If you want to find more details about the models or the evaluation, please refer to our [SANER paper](https://arxiv.org/abs/2412.14802). If you use the code in your work, please consider citing us:

```angular2html
@article{shibaev2024stack,
  title={Stack Trace Deduplication: Faster, More Accurately, and in More Realistic Scenarios},
  author={Shibaev, Egor and Sushentsev, Denis and Golubev, Yaroslav and Khvorov, Aleksandr},
  journal={arXiv preprint arXiv:2412.14802},
  year={2024}
}
```
