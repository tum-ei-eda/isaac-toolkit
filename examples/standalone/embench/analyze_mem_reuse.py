import sys
from collections import defaultdict

import pandas as pd

# --- Load trace ---
assert len(sys.argv) == 3
df = pd.read_pickle(sys.argv[1])

# For now few PC -> ignore all other rows (conflicts with unrolling?)
pcs = list(map(lambda x: int(x, 0), sys.argv[2].split(",")))
hex_pcs = list(map(hex, pcs))
print("pcs", pcs, hex_pcs)

df = df[df["idx"] != 0]
print("df", df)

pc_df = df[df["pc"].isin(pcs)]
print("pc_df", pc_df)

# prev_idx = None
# prev_addrs = None
idx2pc = {}
idx2row = {}
idx2mode = {}
# idx2addrs = {}
idx2addrs_bytes = {}


overlap_counts = defaultdict(int)

filter_modes = []
# filter_modes = ["R->R", "R->W"]
# filter_modes = ["W->R"]
# filter_max_idx_distance = 10000
filter_max_idx_distance = 1000

row = 0
for idx, pc_df_ in pc_df.groupby("idx"):
    # print("idx", idx)
    # print("pc_df_", pc_df_)
    unique_pcs = pc_df_["pc"].unique()
    assert len(unique_pcs) == 1
    pc = unique_pcs[0]
    unique_modes = pc_df_["mode"].unique()
    assert len(unique_modes) == 1
    mode = unique_modes[0]
    # addrs = set(map(int, set(pc_df_["addr"].unique())))
    # print("addrs", addrs)
    addrs_bytes = pc_df_[["addr", "bytes"]].value_counts().reset_index(name="count")[["addr", "bytes"]].values.tolist()
    addrs_bytes = set(tuple(x) for x in addrs_bytes)
    # print("addrs_bytes", addrs_bytes)
    # input(">")
    # TODO: include bytes in PK
    # TODO: include mode in PK?
    # if prev_idx is not None:
    #     addrs_overlap = addrs & prev_addrs
    #     print("addrs_overlap", addrs_overlap, len(addrs_overlap))
    #     if len(addrs_overlap) > 0:
    #         input("%")
    # prev_idx = idx
    # prev_addrs = addrs
    # for idx_, addrs_ in reversed(idx2addrs.items()):
    for idx_, addrs_bytes_ in reversed(idx2addrs_bytes.items()):
        # print("idx_,len(addrs_)", idx_, len(addrs_))
        pc_ = idx2pc[idx_]
        row_ = idx2row[idx_]
        mode_ = idx2mode[idx_]
        addrs_overlap = addrs_bytes & addrs_bytes_
        if len(addrs_overlap) > 0:
            # print("addrs_overlap", addrs_overlap, len(addrs_overlap))
            overlap_size = len(addrs_overlap)
            overlap_size_rel = overlap_size / max(len(addrs_bytes_), len(addrs_bytes))
            rows_distance = row - row_
            # print("rows_distance", rows_distance)
            idx_distance = idx - idx_
            same_pc = pc == pc_
            PRINT = True
            mode_str = f"{mode_.upper()}->{mode.upper()}"
            if filter_modes and mode_str not in filter_modes:
                continue
            if filter_max_idx_distance and idx_distance > filter_max_idx_distance:
                # continue
                break
            if PRINT:
                idxs_str = f"IDXs {idx_} -> {idx} [Distance: {idx_distance}]"
                rows_str = f"Rows {row_} -> {row} [Distance: {rows_distance}]"
                if same_pc:
                    temp_df = df[(df["idx"] > idx_) & (df["idx"] <= idx) & (df["pc"] == pc)]
                    # print("temp_df", temp_df)
                    execs = list(temp_df["idx"].value_counts().index)
                    # print("execs", execs, len(execs))
                    pc_execs = len(execs)
                    pcs_str = f"PC {hex(pc)} [Execs: {pc_execs}]"
                else:
                    pcs_str = f"PCs {hex(pc_)} -> {hex(pc)}"
                to_print = (
                    f"Found {mode_str} overlap of size {overlap_size} "
                    f"({overlap_size_rel*100:.1f}%) for {pcs_str} @ {idxs_str}, {rows_str}"
                )
                print(to_print)
                if overlap_size_rel != 1.0:
                    input("%")
            key = (pc, overlap_size, idx_distance)
            overlap_counts[key] += 1
            break  # only the closest overlap is considered
    idx2pc[idx] = pc
    idx2row[idx] = row
    idx2mode[idx] = mode
    # idx2addrs[idx] = addrs
    idx2addrs_bytes[idx] = addrs_bytes
    row += 1

print("overlap_counts", overlap_counts)
