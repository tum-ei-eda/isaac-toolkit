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

# from pandarallel import pandarallel

# pandarallel.initialize()

import io
import itertools
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from tqdm import tqdm
import pandas as pd


DEFAULT_CHUNK_SIZE = 2**22


def parse_and_process(chunk_bytes, process_func, sep: str = ",", names="infer", operands: bool = False, header=None):
    df = pd.read_csv(io.BytesIO(chunk_bytes), header=header, sep=sep, names=names)
    return process_func(df, operands=operands)


def chunk_iter(path, chunk_size=2**22):
    f = open(path, "rb")

    with f:
        buf = b""
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            buf += data
            last_nl = buf.rfind(b"\n")
            if last_nl == -1:
                continue
            yield buf[: last_nl + 1]
            buf = buf[last_nl + 1 :]
        if buf:
            yield buf


def parse_instr_trace(
    input_file: Path,
    process_func: callable,
    num_workers: Optional[int] = None,
    progress: bool = False,
    executor: str = "process_pool",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    sep: str = ",",
    names = "infer",
    header = None,
    operands: bool = False,
):
    dfs = []
    executor_map = {
        "thread_pool": ThreadPoolExecutor,
        "process_pool": ProcessPoolExecutor,
    }
    executor_cls = executor_map.get(executor)
    assert executor_cls is not None, f"Unsupported Executor: {executor}"
    with executor_cls(max_workers=num_workers) as executor:  # tune workers
        dfs_ = list(
            tqdm(
                executor.map(
                    parse_and_process,
                    chunk_iter(input_file, chunk_size=chunk_size),
                    itertools.repeat(process_func),
                    itertools.repeat(sep),
                    itertools.repeat(names),
                    itertools.repeat(operands),
                    itertools.repeat(header)
                ),
                disable=not progress,
            )
        )
        dfs += dfs_
    df = pd.concat(dfs, axis=0)
    if "instr" in df.columns:
        df["instr"] = df["instr"].astype("category")
    if "size" in df.columns:
        df["size"] = df["size"].astype("category")
    if "pc" in df.columns:
        df["pc"] = pd.to_numeric(df["pc"], downcast="unsigned")
    if "bytecode" in df.columns:
        df["bytecode"] = pd.to_numeric(df["bytecode"], downcast="unsigned")
    df.reset_index(drop=True, inplace=True)
    return df
