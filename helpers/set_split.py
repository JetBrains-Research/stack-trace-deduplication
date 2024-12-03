import argparse
import pandas as pd
import json
from datetime import datetime
import numpy as np


def split_and_update_config(state_path, config_path):
    """
    Split the state file into train, val, and test sets and update the config file with start and longitude values.

    :param state_path: Path to the state.csv file.
    :param config_path: Path to the configuration JSON file.
    """
    state = pd.read_csv(state_path)
    config = json.load(open(config_path))

    # 70% train, 10% val, 20% test
    state_train, state_val, state_test = np.split(state, [int(.7 * len(state)), int(.8 * len(state))])

    print(f"Train size: {len(state_train)}")
    print(f"Validation size: {len(state_val)}")
    print(f"Test size: {len(state_test)}")

    train_start = state_train.iloc[0]['timestamp']
    train_longitude = state_train.iloc[-1]['timestamp'] - train_start
    train_longitude_days = int(train_longitude / (1000 * 60 * 60 * 24))
    train_start_days = 0

    val_start = state_val.iloc[0]['timestamp']
    val_longitude = state_val.iloc[-1]['timestamp'] - val_start
    val_start_days = int((val_start - train_start) / (1000 * 60 * 60 * 24))
    val_longitude_days = int(val_longitude / (1000 * 60 * 60 * 24))

    test_start = state_test.iloc[0]['timestamp']
    test_longitude = state_test.iloc[-1]['timestamp'] - test_start
    test_start_days = int((test_start - train_start) / (1000 * 60 * 60 * 24))
    test_longitude_days = int(test_longitude / (1000 * 60 * 60 * 24))

    print(f"train_start: {train_start_days}, train_longitude: {train_longitude_days}")
    print(f"val_start: {val_start_days}, val_longitude: {val_longitude_days}")
    print(f"test_start: {test_start_days}, test_longitude: {test_longitude_days}")

    print(config)

    config['train_start'] = train_start_days
    config['train_longitude'] = train_longitude_days
    config['val_start'] = val_start_days
    config['val_longitude'] = val_longitude_days
    config['test_start'] = test_start_days
    config['test_longitude'] = test_longitude_days

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print("Updated config:")
    print(json.dumps(config, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split state data and update config file.")
    parser.add_argument("--state_path", type=str, required=True, help="Path to the state.csv file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config.json file.")

    args = parser.parse_args()

    split_and_update_config(state_path=args.state_path, config_path=args.config_path)
