#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
#
# This file is part of ISAAC Toolkit.
# See https://github.com/tum-ei-eda/isaac-toolkit.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pandas as pd
from tqdm import tqdm

input_file = "/workspaces/isaac-toolkit/examples/standalone/coremark/log"

dfs = []
# 'with' context works in pandas >= 1.4 for TextFileReader
with pd.read_csv(
    input_file,
    sep="@",
    header=None,
    chunksize=2**22,  # ~4 million lines per chunk
    # chunksize=2**20,  # ~4 million lines per chunk
    engine="python",
) as reader:
    for df in tqdm(reader, disable=False):
        # print("A", df.head())
        # df is a DataFrame chunk
        # e.g., extract PCs here
        pcs_chunk = df[0].str.extract(r"\[(?:[^/]+/){1}([^/]+)/")[0].apply(lambda x: int(x, 16))
        pcs_chunk = pcs_chunk.astype("category")

        # print("B", pcs_chunk.head(), len(pcs_chunk), pcs_chunk.dtypes, pcs_chunk.memory_usage())
        # process pcs_chunk or append to list
        dfs.append(pcs_chunk)

full_df = pd.concat(dfs)
full_df = full_df.astype("category")

# print("FULL", full_df.head(), len(full_df), full_df.dtypes, full_df.memory_usage())
