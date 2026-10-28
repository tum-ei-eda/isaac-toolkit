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

# import time
import io
import re
import sys
import itertools
import pandas as pd
import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact, TraceArtifact
from isaac_toolkit.logging import get_logger, set_log_level
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = get_logger()


# TODO: logger


RECORD = re.compile(r"^core\s+(\d+):\s+(0x[0-9a-fA-F]+)\s+\((0x[0-9a-fA-F]+)\)\s+(\S+)")
COLUMNS = ["core", "pc", "bytecode", "instr", "size"]


def process_df(df, operands=False):
    rows = []
    for line in df.iloc[:, 0].dropna():
        match = RECORD.match(str(line))
        if match is None:
            continue
        core, pc, code, instr = match.groups()
        code = int(code, 16)
        # Instruction bits also handle unknown compressed mnemonics correctly.
        if code & 3 != 3:
            size = 2
        elif code & 0x1f != 0x1f:
            size = 4
        elif code & 0x3f == 0x1f:
            size = 6
        elif code & 0x7f == 0x3f:
            size = 8
        else:
            raise ValueError(f"Unsupported instruction length: {code:#x}")
        rows.append((int(core), int(pc, 16), code, instr, size))
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_and_process(chunk_bytes, operands=False):
    # Disassembly is not CSV and can contain commas and quotes.
    return process_df(pd.DataFrame({0: chunk_bytes.decode("utf-8", errors="replace").splitlines()}))


def chunk_iter(path, chunk_size=2**22):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
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


def load_instr_trace(
    sess: Session,
    input_file: Path,
    force: bool = False,
    operands: bool = False,
    progress: bool = False,
    num_workers: Optional[int] = None,
    executor: str = "process_pool",
    chunk_size: int = 2**22,
    proxy_kernel: Optional[bool] = None,
):
    logger.info("Loading Spike instruction trace...")
    assert input_file.is_file(), f"File not found: {input_file}"
    name = input_file.name
    if proxy_kernel is None:
        target = name.split("_instrs.", 1)[0]
        proxy_kernel = target in {"spike", "spike_pk", "spike_rv32", "spike_rv64", "spike_rv32_min"}
    # df = pd.read_csv(input_file, sep=":", names=["pc", "rest"])
    dfs = []
    if operands:
        logger.warning("Skipping optional operands trace for Spike")
    executor_map = {"thread_pool": ThreadPoolExecutor, "process_pool": ProcessPoolExecutor}
    executor_cls = executor_map.get(executor)
    if executor_cls is None:
        raise ValueError(f"Unsupported Executor: {executor}")
    with executor_cls(max_workers=num_workers) as pool:
        dfs = list(tqdm(pool.map(parse_and_process, chunk_iter(input_file, chunk_size=chunk_size)),
                        disable=not progress))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=COLUMNS)
    df["instr"] = df["instr"].astype("category")
    df["size"] = df["size"].astype("category")
    df["core"] = df["core"].astype("category")
    df["pc"] = pd.to_numeric(df["pc"], downcast="unsigned")
    if proxy_kernel:
        # PK occupies high addresses; HTIF programs can execute there too.
        df["is_trap"] = df["pc"] >= 0x80000000
    df["bytecode"] = df["bytecode"].astype("category")
    df.reset_index(drop=True, inplace=True)

    attrs = {
        "simulator": "spike",
        "proxy_kernel": proxy_kernel,
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.instr_trace.spike",
    }
    artifact = InstrTraceArtifact(name, df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    input_file = Path(args.file)
    load_instr_trace(
        sess,
        input_file,
        force=args.force,
        operands=args.operands,
        progress=args.progress,
        proxy_kernel=args.proxy_kernel,
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--operands", action="store_true")
    parser.add_argument("--progress", action="store_true")
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--proxy-kernel", dest="proxy_kernel", action="store_true", default=None)
    runtime.add_argument("--no-proxy-kernel", dest="proxy_kernel", action="store_false")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
