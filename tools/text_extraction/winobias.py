# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script collects text from WinoBias dataset (https://github.com/uclanlp/corefBias). For each example, we extract the occupation and the pronoun token.
Usage:
python tools/text_extraction/winobias.py <PATH_TO_YOUR_WinoBias_DATA> examples/coref_bias/winobias_text_features.jsonl
"""

import json
import glob
import sys
import re
import os

wino_root = sys.argv[1]
output_file = sys.argv[2]

pro_path = os.path.join(wino_root, "pro_stereotyped_type1.txt.dev")
anti_path = os.path.join(wino_root, "anti_stereotyped_type1.txt.dev")

with open(output_file.replace(".jsonl", "_pro.jsonl"), "w") as f:
    with open(pro_path) as f_in:
        lines = f_in.readlines()
    for idx, line in enumerate(lines):
        match = re.search(r'.*\[(.*)\].*\[(.*)\].*', line)
        extracted_text_features = {
            "id": idx,
            "token1": match.group(1),
            "token2": match.group(2),
        }
        print(extracted_text_features)
        f.write(json.dumps(extracted_text_features) + "\n")

with open(output_file.replace(".jsonl", "_anti.jsonl"), "w") as f:
    with open(anti_path) as f_in:
        lines = f_in.readlines()
    for idx, line in enumerate(lines):
        match = re.search(r'.*\[(.*)\].*\[(.*)\].*', line)
        extracted_text_features = {
            "id": idx,
            "token1": match.group(1),
            "token2": match.group(2),
        }
        f.write(json.dumps(extracted_text_features) + "\n")
