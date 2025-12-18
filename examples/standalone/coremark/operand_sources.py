import pandas as pd

trace_df = pd.read_pickle("sess/instr_trace/etiss_instrs.log.pkl")
trace_df.drop(columns=["bytecode", "pc", "size"], inplace=True, errors="ignore")
operands_df = pd.read_pickle("sess/instr_trace/etiss_instrs.log.pkl")
df_operands_full = pd.concat([trace_df, operands_df], axis=1)
df_operands_full.drop(columns=["csr", "shamt"], inplace=True, errors="ignore")

# df_short = df.iloc[:10000000]
# df = df_short

df_operands_full["instr"] = df_operands_full["instr"].astype(str)
df_operands_full["pseudo_instr"] = df_operands_full["instr"]  # start with actual instruction
df_operands_full.loc[
    (df_operands_full["instr"] == "addi") & (df_operands_full["rs1"] == 0) & (df_operands_full["imm"] != 0),
    "pseudo_instr",
] = "li"
df_operands_full.loc[
    (df_operands_full["instr"] == "addi") & (df_operands_full["imm"] == 0) & (df_operands_full["rs1"] != 0),
    "pseudo_instr",
] = "mv"

df_operands_full.drop(columns=["imm"], inplace=True, errors="ignore")

df_operands_full["instr"] = df_operands_full["instr"].astype("category")
df_operands_full["pseudo_instr"] = df_operands_full["pseudo_instr"].astype("category")

last_written_instr = {}
rs1_writers = []
rs2_writers = []

for idx, row in df_operands_full.iterrows():
    rs1 = row["rs1"]
    rs2 = row["rs2"]
    rs1_writers.append(last_written_instr.get(rs1, None) if rs1 is not None else None)
    rs2_writers.append(last_written_instr.get(rs2, None) if rs2 is not None else None)
    rd = row["rd"]
    if rd is not None:
        last_written_instr[rd] = row["pseudo_instr"]  # use pseudo name now

df_operands_full["rs1_src"] = rs1_writers
df_operands_full["rs2_src"] = rs2_writers
df_operands_full.rs1_src.value_counts()

df_operands_full["rs1_src_flt"] = df_operands_full["rs1_src"].apply(lambda x: x if x in ["mv", "li"] else "other")
df_operands_full["rs2_src_flt"] = df_operands_full["rs2_src"].apply(lambda x: x if x in ["mv", "li"] else "other")

for instr, instr_df in df_operands_full.groupby("pseudo_instr"):
    num = len(instr_df)
    print(">>>", instr, f"[count={num}]", "<<<")
    rs1_src_counts = instr_df.rs1_src_flt.value_counts()
    rs2_src_counts = instr_df.rs2_src_flt.value_counts()
    rs1_src_counts_rel = rs1_src_counts / num
    rs2_src_counts_rel = rs2_src_counts / num
    if len(rs1_src_counts) > 1:
        print("rs1:", rs1_src_counts_rel.to_dict())
    else:
        print("rs1: -")
    if len(rs2_src_counts) > 1:
        print("rs2:", rs2_src_counts_rel.to_dict())
    else:
        print("rs2: -")
