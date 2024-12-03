import argparse
from pathlib import Path
from tqdm import tqdm
from ea.sim.dev.scripts.training.training.common import create_bucket_data
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.utils import Scope
import pandas as pd
import matplotlib.pyplot as plt


def print_stats(data_name: str, state_path: Path, forget_days: int = 365):
    """
    Print statistics for a given dataset.

    :param data_name: Name of the dataset.
    :param state_path: Path to the state.csv file.
    :param config_path: Path to the config.json file.
    :param forget_days: Number of days for forgetting reports.
    """
    print(f"Data name: {data_name}")

    state = pd.read_csv(state_path)  # has columns: timestamp, rid, iid

    # Map data_name to scope
    data_name_to_scope = {
        'campbell': Scope.Campbell,
        'eclipse': Scope.Eclipse,
        'netbeans': Scope.NetBeans,
        'gnome': Scope.Gnome,
        'slowops_cleaned': Scope.SlowOpsCleaned
    }

    scope = data_name_to_scope[data_name]

    data = create_bucket_data(
        data_name=data_name,
        scope=scope,
        reports_dir=state_path.parent / 'reports',
        actions_path=True,
        state_path=state_path,
        forget_days=forget_days,
    )

    HashStorage.initialize(data.stack_loader)
    hash_storage = HashStorage.get_instance()

    number_of_reports = len(state)
    print(f"Number of reports: {number_of_reports}")

    issue_to_reports: dict[int, list[int]] = {}  # iid -> [hashes of reports]
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

    average_number_of_unique_reports_per_issue = sum(
        [len(set(hashes)) for hashes in issue_to_reports.values()]
    ) / number_of_issues
    print(f"Average number of unique reports per issue: {average_number_of_unique_reports_per_issue}")

    print(f"Average number of reports per issue: {number_of_reports / number_of_issues}")

    counts = list(hash_to_count.values())

    # Save histogram of counts
    plt.hist(counts, bins=100)
    plt.yscale('log')
    plt.xlabel('Number of repetitions of report')
    plt.title(f'Histogram of report repetitions for {data_name}')
    plt.savefig(f'{data_name}_histogram.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate statistics for a dataset.")
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--state_path', type=str, required=True, help='Path to the state.csv file.')
    parser.add_argument('--forget_days', type=int, default=365, help='Number of days for forgetting reports.')

    args = parser.parse_args()

    data_name = args.data_name
    state_path = Path(args.state_path)
    forget_days = args.forget_days

    print_stats(data_name=data_name, state_path=state_path, forget_days=forget_days)
