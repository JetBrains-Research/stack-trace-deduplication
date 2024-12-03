# Stack Trace Similarity

This replication package presents the code of our approach and an instruction for running the experiments.

Since we cannot provide the newly collected industrial dataset due to anonymity restrictions, instead, we append the existing 
open-source NetBeans dataset in the same format for clarity (it can be found in the `NetBeans` directory). 
The structure of the dataset is described below in section "Setup". Upon acceptance, we will release the new 
industrial dataset to the research community.

## 1. Install the required packages
```bash
poetry install
``` 

## 2. Setup

To run experiments for a specific dataset, create a designated folder `ARTIFACTS_DIR` for the dataset.
Inside this folder, there should be a `config.json` file with the following structure:
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

In the `reports_dir` folder, there should be a folder with all reports. Each report should be a separate file with the following
name format: `report_id.json`.

In the `labels_dir` folder, there should be a CSV file with the following structure:
```
timestamp,rid,iid
...
```
where `timestamp` is the timestamp of the report, `rid` is the report ID, and `iid` is the category ID.

The example of a config file can be found in the [NetBeans_config_example.json](NetBeans_config_example.json) file.

## 3. Run the experiments

### Training the models

Training scripts are located in the folder `ea/sim/dev/scripts/training/training`.

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

### Evaluating the models

Evaluation scripts are located in the folder `ea/sim/dev/scripts/training/evaluating`.

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

- OpenAI embedding model:
First, precompute the embeddings using `ea/sim/dev/scripts/training/training/embeddings/main.py`.
Then, run the following script:
```bash
python ea/sim/dev/scripts/training/evaluating/openai/run.py 
```

The results of the evaluation will be saved in the `ARTIFACTS_DIR` folder.
