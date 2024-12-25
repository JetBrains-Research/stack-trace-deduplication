import pandas as pd
from tqdm import tqdm
import json 
import argparse
import os

def parse_stacktraces(path_to_stacktraces):
    content = open(path_to_stacktraces).read()
    content = json.loads(content) 

    stack_traces = []

    state_data = []

    for entry in tqdm(content):
        ts = entry['creation_ts']
        rid = entry['bug_id']
        iid = entry['dup_id']
        if iid is None:
            iid = rid
        errors = []
        elements = []

        for frame in entry['stacktrace']['frames']:
            # print keys
            # pprint(frame)
            name = frame.get('function', None)
            if name is None:
                name = ""
            filename = frame.get('file_name', None)
            if filename is None:
                filename = ""
            line_number = None
            subsystem = None

            elements.append({
                'name': name,
                'file_name': filename,
                'line_number': line_number,
                'subsystem': subsystem
            })
        
        stack_traces.append({
            'id': iid,
            'timestamp': ts,
            'errors': errors,
            'elements': elements
        })

        state_data.append({
            'timestamp': ts,
            'rid': rid,
            'iid': iid
        })

    state = pd.DataFrame(state_data)

    return stack_traces, state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_stacktraces', type=str, default='campbell_dataset/campbell_stacktraces.json')
    parser.add_argument('--path_to_save', type=str, default='parsed_dataset')

    args = parser.parse_args()
    path_to_stacktraces = args.path_to_stacktraces
    path_to_save = args.path_to_save

    stack_traces, state = parse_stacktraces(path_to_stacktraces)
    
    os.makedirs(f'{path_to_save}/reports/', exist_ok=True)

    for entry in stack_traces:
        iid = entry['id']
        with open(f'{path_to_save}/reports/{iid}.json', 'w') as f:
            json.dump(entry, f)

    state.to_csv(f'{path_to_save}/state.csv', index=False)
