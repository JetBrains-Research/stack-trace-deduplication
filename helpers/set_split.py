import pandas as pd
import json
from datetime import datetime
import numpy as np

# state_path = "/home/ec2-user/ea-ml-data/similarity/Campbell/state.csv"
# state_path = "/home/ec2-user/ea-ml-data/similarity/Eclipse/state.csv"
# state_path = "/home/ec2-user/ea-ml-data/similarity/NetBeans/state.csv"
# state_path = "/home/ec2-user/ea-ml-data/similarity/Gnome/state.csv"
# state_path = "/home/ec2-user/ea-ml-data/similarity/SlowOps/030324/processed/state.csv"
# state_path = "/home/ec2-user/ea-ml-data/similarity/SlowOps_cleaned/state.csv"
state_path = "/home/ec2-user/ea-ml-data/similarity/NetBeans/state.csv"


# config_path = "/home/ec2-user/similarity_artifacts/campbell/config.json"
# config_path = "/home/ec2-user/similarity_artifacts/eclipse/config.json"
# config_path = "/home/ec2-user/similarity_artifacts/NETBEANS/config.json"
# config_path = "/home/ec2-user/similarity_artifacts/gnome/config.json"
# config_path = "/home/ec2-user/similarity_artifacts/slowops/config.json"
# config_path = "/home/ec2-user/similarity_artifacts/slowops_cleaned/config.json"
config_path = "/home/ec2-user/similarity_artifacts/netbeans/config.json"

state = pd.read_csv(state_path)
config = json.load(open(config_path))

# 70 % train, 10% val, 20% test 

state_train, state_val, state_test = np.split(state, [int(.7*len(state)), int(.8*len(state))])

print(f"train size {len(state_train)}")
print(f"val size {len(state_val)}")
print(f"test size {len(state_test)}")

train_start = state_train.iloc[0]['timestamp']
train_longitude = state_train.iloc[-1]['timestamp'] - train_start
train_longitude_days = train_longitude / (1000 * 60 * 60 * 24)
train_longitude_days = int(train_longitude_days)
train_start_days = 0

val_start = state_val.iloc[0]['timestamp']
val_longitude = state_val.iloc[-1]['timestamp'] - val_start
val_longitude_days = val_longitude / (1000 * 60 * 60 * 24)
val_start = val_start - train_start
val_start_days = val_start / (1000 * 60 * 60 * 24)
val_start_days = int(val_start_days)
val_longitude_days = int(val_longitude_days)

test_start = state_test.iloc[0]['timestamp']
test_longitude = state_test.iloc[-1]['timestamp'] - test_start
test_longitude_days = test_longitude / (1000 * 60 * 60 * 24)
test_start = test_start - train_start
test_start_days = test_start / (1000 * 60 * 60 * 24)
test_start_days = int(test_start_days)
test_longitude_days = int(test_longitude_days)

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

print(config)
