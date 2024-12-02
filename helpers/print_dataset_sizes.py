from pathlib import Path

from tqdm import tqdm
from ea.sim.dev.scripts.training.training.common import create_bucket_data
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.utils import Scope, StackId, ARTIFACTS_DIR
import pandas as pd

# data_name = 'campbell'
# data_name = 'eclipse'
# data_name = 'netbeans'
# data_name = 'gnome'
data_name = 'slowops_cleaned'


data_name_to_dir = {
    'campbell': 'Campbell',
    'eclipse': 'Eclipse',
    'netbeans': 'NetBeans',
    'gnome': 'Gnome',
    'slowops_cleaned': 'SlowOps_cleaned'
}
state_path = Path(f"/home/ec2-user/ea-ml-data/similarity/{data_name_to_dir[data_name]}/state.csv")

config_path = f"/home/ec2-user/similarity_artifacts/{data_name}/config.json"

forget_days = 365

data_name_to_scope = {
    'campbell': Scope.Campbell,
    'eclipse': Scope.Eclipse,
    'netbeans': Scope.NetBeans,
    'gnome': Scope.Gnome,
    'slowops_cleaned': Scope.SlowOpsCleaned
}


def print_stats():
    # print number of reports, number of issues, number of unique reports and average number of unique reports per issue
    print(f"Data name: {data_name}")

    state = pd.read_csv(state_path) # has columns: timestamp, rid, iid
    ARTIFACTS_DIR = Path(config_path).parent

    data = create_bucket_data(
        data_name=data_name,
        scope=data_name_to_scope[data_name],
        reports_dir=state_path.parent / 'reports',
        actions_path=True,
        state_path=state_path,
        forget_days=forget_days,
    )

    HashStorage.initialize(data.stack_loader)
    hash_storage = HashStorage.get_instance()

    number_of_reports = len(state)

    print(f"Number of reports: {number_of_reports}")

    issue_to_reports: dict[int, list[int]] = {} # iid -> [hashes of reports]
    hash_to_count = {}  # hash -> count

    for _, row in tqdm(list(state.iterrows()), desc='Processing state'):
        iid = row['iid']
        rid = row['rid']
        if iid not in issue_to_reports:
            issue_to_reports[iid] = []
        hash = hash_storage.hash(rid)
        issue_to_reports[iid].append(hash)

        if hash not in hash_to_count:
            hash_to_count[hash] = 0
        hash_to_count[hash] += 1
    
    number_of_issues = len(issue_to_reports)
    print(f"Number of issues: {number_of_issues}")

    number_of_unique_reports = len(set([hash for hashes in issue_to_reports.values() for hash in hashes]))

    print(f"Number of unique reports: {number_of_unique_reports}")

    average_number_of_unique_reports_per_issue = sum([len(set(hashes)) for hashes in issue_to_reports.values()]) / number_of_issues

    print(f"Average number of unique reports per issue: {average_number_of_unique_reports_per_issue}")

    print(f"Average number of reports per issue: {number_of_reports / number_of_issues}")

    counts = list(hash_to_count.values())

    # save histogram of counts

    import matplotlib.pyplot as plt

    plt.hist(counts, bins=100)
    plt.yscale('log')
    plt.xlabel('Number of repetitions of report')
    plt.title(f'Histogram of report repetitions for {data_name}')

    plt.savefig(f'{data_name}_histogram.png')


if __name__ == '__main__':
    print_stats()
