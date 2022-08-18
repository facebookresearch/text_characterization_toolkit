# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
This script collects task-level text features from the Natural Instructions dataset (https://github.com/allenai/natural-instructions)

Usage:
python tools/text_extraction/natural_instructions.py <PATH_TO_YOUR_NI_REPO> examples/ni_explanations/ni_text_features.jsonl
"""

import json
import glob
import sys

def parse_ni_file(path):
    with open(path) as f:
        data = json.load(f)
        
    return {
        "instruction": data["Definition"][0],
        "examples": data["Positive Examples"]
    }

def read_ni_tasks(root_dir):
    tasks = {}
    task_paths = glob.glob(root_dir + "/tasks/task*")
    num_tasks = len(task_paths)
    print(f"Processing {num_tasks} tasks...")
    for path in task_paths:
        task_id = path.split("/")[-1].split(".json")[0]
        tasks[task_id] = parse_ni_file(path)
        if len(tasks) % 100 == 0:
            print(f"Processed {len(tasks)} tasks")
    return tasks

ni_root = sys.argv[1]
output_file = sys.argv[2]

ni_tasks = read_ni_tasks(ni_root)

with open(output_file, "w") as f:
    print(f"Writing resulting text features to {output_file}")
    for id, task_data in ni_tasks.items():
        extracted_text_features = {
            "id": id,
            "instruction": task_data["instruction"],
            "explanations": "\n".join([sample["explanation"] for sample in task_data["examples"]]),
        }
        f.write(json.dumps(extracted_text_features) + "\n")
