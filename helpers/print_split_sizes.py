import pandas as pd
import json
from datetime import datetime

index = 4

state_path = [
    "/home/ec2-user/ea-ml-data/similarity/Eclipse/state.csv",
    "/home/ec2-user/ea-ml-data/similarity/NetBeans/state.csv",
    "/home/ec2-user/ea-ml-data/similarity/Campbell/state.csv",
    "/home/ec2-user/ea-ml-data/similarity/Gnome/state.csv",
    "/home/ec2-user/ea-ml-data/similarity/SlowOps_cleaned/state.csv"
][index]


config_path = [
    "/home/ec2-user/similarity_artifacts/eclipse/config.json",
    "/home/ec2-user/similarity_artifacts/netbeans/config.json",
    "/home/ec2-user/similarity_artifacts/campbell/config.json",
    "/home/ec2-user/similarity_artifacts/gnome/config.json",
    "/home/ec2-user/similarity_artifacts/slowops_cleaned/config.json"
][index]

state = pd.read_csv(state_path)
config = json.load(open(config_path))

print(config)

train_start = config["train_start"]
train_longitude = config["train_longitude"]

start_date = state.iloc[0]['timestamp']
start_date = datetime.fromtimestamp(start_date / 1000)

print(f"start date: {start_date.strftime('%m/%d/%Y')}")

train_start_date = start_date + pd.DateOffset(days=train_start)
train_end_date = train_start_date + pd.DateOffset(days=train_longitude)

# print dates without time like 7/29/2003
print(f'train interval: {train_start_date.strftime("%m/%d/%Y")} - {train_end_date.strftime("%m/%d/%Y")}')

val_start = config["val_start"]
test_start = config["test_start"]
val_longitude = test_start - val_start
val_start_date = start_date + pd.DateOffset(days=val_start)
val_end_date = val_start_date + pd.DateOffset(days=val_longitude)
print(f'val interval: {val_start_date.strftime("%m/%d/%Y")} - {val_end_date.strftime("%m/%d/%Y")}')

test_start = config["test_start"]
test_longitude = config["test_longitude"]
test_start_date = start_date + pd.DateOffset(days=test_start)
test_end_date = test_start_date + pd.DateOffset(days=test_longitude)
print(f'test interval: {test_start_date.strftime("%m/%d/%Y")} - {test_end_date.strftime("%m/%d/%Y")}')














# state = pd.read_csv(state_path)

# issues = {} # dict[int, list]

# for index, row in state.iterrows():
#     issue_id = row["iid"]
#     report_id = row["rid"]

#     if issue_id not in issues:
#         issues[issue_id] = []

#     issues[issue_id].append(report_id)

# print(f"dataset name: {state_path.split('/')[-2]}")
# print(f"Number of issues: {len(issues)}")
# print(f"Number of reports: {len(state)}")
# print(f"Average number of reports per issue: {len(state) / len(issues)}")