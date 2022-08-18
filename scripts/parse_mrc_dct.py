# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import pandas as pd
import sys

class MRCRecord:
    def __init__(self, line):
        # Offsets based on mrc2.doc
        self.props = {}
        self.props["fam"] = int(line[25:28])
        self.props["conc"] = int(line[28:31])
        self.props["imag"] = int(line[31:34])
        self.props["meanc"] = int(line[34:37])
        self.props["meanp"] = int(line[37:40])
        self.props["aoa"] = int(line[40:43])
        #self.props["word_orig"] = line[51:].split("|")[0]
        self.props["word"] = line[51:].split("|")[0].lower()


def parse_mrc_file(filepath):
    data = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            props = MRCRecord(line).props
            keep_row = any("word" not in k and v > 0 for k, v in props.items())
            if keep_row:
                for key, val in MRCRecord(line).props.items():
                    data[key].append(
                        val
                        if type(val) is str or int(val) > 0
                        else None
                    )
    return pd.DataFrame(data).groupby("word", as_index=False).first()

df = parse_mrc_file(sys.argv[1])
df.to_csv(sys.argv[2], sep=",", index=False)