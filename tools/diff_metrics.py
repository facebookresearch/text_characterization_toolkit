# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import pandas as pd

from text_characterization.utils import load_text_metrics


parser = argparse.ArgumentParser()
parser.add_argument(
    "--old",
    type=str
)
parser.add_argument(
    "--new",
    type=str
)
args = parser.parse_args()

print(f"\nLoading old version: {args.old}")
old = load_text_metrics(args.old)

print(f"\nLoading new version: {args.new}")
new = load_text_metrics(args.new)

print("\nRunning checks...")

assert set(old.index) == set(new.index), "Input seems to be different, something is not right!"

new_columns = set(new.columns) - set(old.columns)
if len(new_columns) > 0:
    print("[WARNING] New characteristics introduced!\n  " + "\n  ".join(new_columns) + "\n")

missing_columns = set(old.columns) - set(new.columns)
if len(missing_columns) > 0:
    print("[WARNING] Found missing characteristics!\n  " + "\n  ".join(missing_columns) + "\n")

if set(new.columns) == set(old.columns):
    print("[OK] No characteristics added or missing.")
    
    
ok = True
for col in set(new.columns).intersection(set(old.columns)):
    
    if col in {"id", "text_key"}:
        continue

    value_changed = ~np.isclose(new[col], old[col], rtol=0.1)
    value_changed_pct = sum(value_changed) * 1.0 / len(value_changed)
    if value_changed_pct > 0:
        print(f"[WARNING] Differences found ({value_changed_pct*100:.2f}% of data points) in metric {col}!")
        ok = False
if ok:
    print("[OK] All values are unchanged for existing metrics.")