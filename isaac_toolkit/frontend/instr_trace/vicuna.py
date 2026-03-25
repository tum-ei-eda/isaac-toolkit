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
import sys
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact, TraceArtifact
from isaac_toolkit.logging import get_logger, set_log_level
from .utils import parse_instr_trace, DEFAULT_CHUNK_SIZE

logger = get_logger()


def process_vicuna_trace_df(df, operands: bool = False):
    df["pc"] = df["pc"].apply(lambda x: int(x, 0))
    df["pc"] = pd.to_numeric(df["pc"])
    df["bytecode"] = df["bytecode"].apply(lambda x: int(x, 0))
    df["bytecode"] = pd.to_numeric(df["bytecode"])

    df.drop(columns=["cycle"], inplace=True)
    return df


def load_instr_trace(
    sess: Session,
    input_file: Path,
    force: bool = False,
    operands: bool = False,
    progress: bool = False,
    num_workers: Optional[int] = None,
    executor: str = "process_pool",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    logger.info("Loading Vicuna intruction trace...")
    assert input_file.is_file(), f"File not found: {input_file}"
    name = input_file.name

    df = parse_instr_trace(
        input_file,
        process_vicuna_trace_df,
        num_workers=num_workers,
        progress=progress,
        chunk_size=chunk_size,
        executor=executor,
        sep=",",
        names=["cycle", "pc", "bytecode"],
        operands=operands,
        header=0,
    )

    attrs = {
        "simulator": "vicuna",
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.instr_trace.vicuna",
    }
    if operands:
        operands_trace_df = df[["instr", "operands"]]
        df.drop(columns=["operands"], inplace=True)
        operands_artifact = TraceArtifact("operands_trace", operands_trace_df, attrs=attrs)
        sess.add_artifact(operands_artifact, override=force)
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
        executor=args.executor,
        num_workers=args.parallel,
        chunk_size=args.chunk_size,
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
    parser.add_argument("--executor", type=str, choices=["process_pool", "thread_pool"], default="process_pool")
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
