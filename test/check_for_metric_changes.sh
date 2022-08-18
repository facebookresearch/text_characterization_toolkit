#! /bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

input_text="test/ni_instructions.jsonl"
old_output="test/ni_instructions_characteristics.csv"
new_output="test/ni_instructions_characteristics.csv.new"

if [ -f "$new_output" ] ; then
    rm "$new_output"
fi

echo "Computing characteristics on test input..."
python tools/compute.py -i $input_text -o $new_output

echo "Checking output..."
python tools/diff_metrics.py --old $old_output --new $new_output

# Remove this line if you want to debug output or update the reference numbers
rm "$new_output"