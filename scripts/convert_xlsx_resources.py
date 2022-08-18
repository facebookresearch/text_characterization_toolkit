# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import sys

print(sys.argv[1])
print(sys.argv[2])

df = pd.read_excel(sys.argv[1])
df.to_csv(sys.argv[2])