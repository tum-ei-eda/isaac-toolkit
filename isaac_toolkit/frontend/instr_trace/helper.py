#
# Copyright (c) 2025 TUM Department of Electrical and Computer Engineering.
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

# import time
import sys
import time
import pandas as pd
import argparse
from typing import List, Union
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from tqdm import tqdm
from capstone import Cs, CS_ARCH_RISCV, CS_MODE_RISCV64, CS_MODE_RISCV32, CS_MODE_RISCVC
from elftools.elf.elffile import ELFFile
from elftools.elf.constants import SH_FLAGS

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


# from pandarallel import pandarallel

# pandarallel.initialize()


# def process_df(df: pd.DataFrame, fetcher, md):
# def process_df(df: pd.DataFrame):
def process_df(df):
    # print("process_df", len(df), df.memory_usage(deep=True).sum())
    # return None

    assert len(df.columns) == 4, "Excpected 4 columns"
    df = df.rename(columns={0: "pc", 1: "?", 2: "is_branch", 3: "size"})
    # df["is_branch"] = df["is_branch"].astype("category")
    # print("A", time.time())

    # df["pc"] = df["pc"].parallel_apply(lambda x: int(x, 0))
    df["pc"] = df["pc"].apply(lambda x: int(x, 0))
    # df["pc"] = df["pc"].apply(lambda x: int(x, 0))
    # df["pc"] = df["pc"].apply(lambda x: int(x, 0))
    # df["pc"] = df["pc"].map(lambda x: int(x, 0))
    # df["pc"] = df["pc"].astype(int, base=0)
    # df["pc"] = pd.to_numeric(df["pc"])
    df["size"] = df["size"].astype(int)
    df["size"] = df["size"].astype("category")
    df.drop(columns=["?"], inplace=True)
    # df.drop(columns=["is_branch"], inplace=True)

    # df["bytecode"] = df[["pc", "size"]].apply(lambda x: lookup_bytecode(x["pc"], x["size"]), axis=1)
    # df["instr"] = df[["pc", "bytecode", "size"]].apply(
    #     lambda x: lookup_name(x["bytecode"], pc=x["pc"], size=x["size"]), axis=1
    # )
    # df["instr"] = df["instr"].astype("category")
    # print("A")
    # unique_pc_size = df[["pc", "size"]].drop_duplicates()
    # unique_pc_size["bytecode"] = unique_pc_size.apply(
    #     lambda x: fetcher.read_word_at_pc(x["pc"], size=x["size"]), axis=1
    # )
    # print("B")

    # def disassemble_row(row):
    #     return disassemble_word(md, int(row["pc"]), int(row["bytecode"]), size=int(row["size"]), operands=False)

    # unique_pc_size["instr"] = unique_pc_size.apply(disassemble_row, axis=1)
    # print("unique_pc_size", unique_pc_size)
    # input(">>")
    # print("C")
    # df = df.merge(unique_pc_size, on=["pc", "size"], how="left")
    # df["bytecode"] = pd.to_numeric(df["bytecode"], downcast="unsigned")
    # df["instr"] = df["instr"].astype("category")
    # print("D")

    # print("pc2bytecode", len(pc2bytecode))
    # print("df", df.head(), df.columns, df.dtypes, df.memory_usage())
    # print("ret")
    return df
