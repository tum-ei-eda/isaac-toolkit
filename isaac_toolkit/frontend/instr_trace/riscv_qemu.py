import pandas as pd
from tqdm import tqdm

input_file = "/workspaces/isaac-toolkit/examples/standalone/coremark/log"

dfs = []
# 'with' context works in pandas >= 1.4 for TextFileReader
with pd.read_csv(input_file, sep="@",
                 header=None,
                 chunksize=2**22,  # ~4 million lines per chunk
                 # chunksize=2**20,  # ~4 million lines per chunk
                 engine="python") as reader:
    for df in tqdm(reader, disable=False):
        # print("A", df.head())
        # df is a DataFrame chunk
        # e.g., extract PCs here
        pcs_chunk = df[0].str.extract(r"\[(?:[^/]+/){1}([^/]+)/")[0].apply(lambda x: int(x, 16))
        pcs_chunk = pcs_chunk.astype("category")
        
        print("B", pcs_chunk.head(), len(pcs_chunk), pcs_chunk.dtypes, pcs_chunk.memory_usage())
        # process pcs_chunk or append to list
        dfs.append(pcs_chunk)

full_df = pd.concat(dfs)
full_df = full_df.astype("category")

print("FULL", full_df.head(), len(full_df), full_df.dtypes, full_df.memory_usage())