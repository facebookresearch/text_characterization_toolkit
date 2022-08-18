# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import csv
from collections import OrderedDict


import pandas as pd

from pandarallel import pandarallel

from text_characterization.metrics import MetricProcessor
from text_characterization.parser_backends import (
    BasicSpacyBackend,
    FullSpacyBackend,
)
from text_characterization.utils import parse_filename_macros


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    default="<CONFIG_DIR>/default.json",
    help="Config File"
)

parser.add_argument(
    "-i",
    "--input_file",
    help="File containting the paragraphs to be evaluated.",
    default=None,
    type=str,
)
parser.add_argument(
    "-o", "--output_file", help="Output file for batch mode", default=None, type=str
)
parser.add_argument("-t", "--text", help="Text to be evaulated", default=None, type=str)
parser.add_argument(
    "-a",
    "--annotate",
    help="Provide the column names that should be fetched from the tsv file (separated by a comma if there are multiple)",
    default="sentence",
    type=str,
)

parser.add_argument(
    "--basic_backend",
    help="Only run a cheap backend that does not involve complex NLP methods like POS tagging",
    action="store_true",
)
args = parser.parse_args()

assert  (
    (args.input_file is None) != (args.text is None)
), "Input must be specified either as a file or directly as a text in the argument"

with open(parse_filename_macros(args.config)) as config_file:
    config = json.load(config_file)

# Basic SpaCy backend does not run any expensive NLP pipelines, suitable for simple metrics at large scale.
backend = BasicSpacyBackend if args.basic_backend else FullSpacyBackend
mp = MetricProcessor(config=config, backend=backend)

print("Loading spaCy..\n")
backend.load_resources()

print("Loading resource files..\n")
mp.load_resources()

if args.text is not None:
    results = mp.compute_metrics(args.text)
    metrics, values = zip(*results.items())
    results_df = pd.DataFrame({"Metric": metrics, "Value": values})
    print(results_df.to_string(index=False))
else:
    assert (args.input_file is not None) and (args.output_file is not None)

    inputs = []

    extension = args.input_file.split(".")[-1]

    if extension == 'jsonl':
        with open(args.input_file, "r") as fin:
            for line in fin:
                line_dict = json.loads(line[:-1])
                id = line_dict.pop("id")
                for key, text in line_dict.items():
                    inputs.append({"id": id, "text_key": key, "text": text})

    elif extension == "tsv":
        cols = args.annotate.split(",")
        df = pd.read_csv(args.input_file, sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False)
        for index, row in df.iterrows():
            for col in cols:
                inputs.append({"id": id, "text_key": col, "text": row[col]})
    else:
        raise NotImplementedError(f"Unsupported input format: {extension}")

    input_df = pd.DataFrame(inputs)

    metrics_df = pd.DataFrame()

    if len(input_df) > 10:
        pandarallel.initialize()
        # Note parallel_apply(compute_metrics) does not parallelize!
        metrics_df = input_df["text"].parallel_apply(lambda text: mp.compute_metrics(text)).apply(pd.Series)
    else:
        metrics_df = input_df["text"].apply(lambda x: mp.compute_metrics(x, log_time=True)).apply(pd.Series)

    metrics_df["id"] = input_df["id"]
    metrics_df["text_key"] = input_df["text_key"]
    metrics_df = metrics_df.set_index("id")

    metrics_df.to_csv(args.output_file, sep="\t", index=True)