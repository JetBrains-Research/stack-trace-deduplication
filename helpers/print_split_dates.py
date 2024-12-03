import argparse
import pandas as pd
import json
from datetime import datetime


def calculate_intervals(state_path, config_path):
    """
    Calculate and print training, validation, and testing intervals based on the given configuration.

    :param state_path: Path to the state.csv file.
    :param config_path: Path to the configuration JSON file.
    """
    state = pd.read_csv(state_path)
    config = json.load(open(config_path))

    print(config)

    train_start = config["train_start"]
    train_longitude = config["train_longitude"]

    start_date = state.iloc[0]['timestamp']
    start_date = datetime.fromtimestamp(start_date / 1000)

    print(f"Start date: {start_date.strftime('%m/%d/%Y')}")

    train_start_date = start_date + pd.DateOffset(days=train_start)
    train_end_date = train_start_date + pd.DateOffset(days=train_longitude)

    print(f'Train interval: {train_start_date.strftime("%m/%d/%Y")} - {train_end_date.strftime("%m/%d/%Y")}')

    val_start = config["val_start"]
    test_start = config["test_start"]
    val_longitude = test_start - val_start

    val_start_date = start_date + pd.DateOffset(days=val_start)
    val_end_date = val_start_date + pd.DateOffset(days=val_longitude)
    print(f'Validation interval: {val_start_date.strftime("%m/%d/%Y")} - {val_end_date.strftime("%m/%d/%Y")}')

    test_longitude = config["test_longitude"]
    test_start_date = start_date + pd.DateOffset(days=test_start)
    test_end_date = test_start_date + pd.DateOffset(days=test_longitude)
    print(f'Test interval: {test_start_date.strftime("%m/%d/%Y")} - {test_end_date.strftime("%m/%d/%Y")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset intervals.")
    parser.add_argument("--state_path", type=str, required=True, help="Path to the state.csv file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config.json file.")

    args = parser.parse_args()

    state_path = args.state_path
    config_path = args.config_path

    calculate_intervals(state_path=state_path, config_path=config_path)
