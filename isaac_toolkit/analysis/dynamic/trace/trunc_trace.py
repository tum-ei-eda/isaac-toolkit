#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
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
import sys
import argparse
import posixpath
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import (
    ArtifactFlag,
    TableArtifact,
    filter_artifacts,
    InstrTraceArtifact,
)
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def trunc_trace(
    sess: Session,
    start_pc: Optional[int] = None,
    end_pc: Optional[str] = None,
    start_func: Optional[str] = None,
    end_func: Optional[str] = None,
    force: bool = False,
):
    logger.info("Truncating trace...")
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    # print("elf_artifacts", elf_artifacts)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    trace_df = trace_artifact.df
    assert force

    func2pc_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "func2pc")
    if len(func2pc_artifacts) > 0:
        assert len(func2pc_artifacts) == 1
        func2pc_artifact = func2pc_artifacts[0]
        func2pc_df = func2pc_artifact.df
    else:
        func2pc_df = None

    # if start_pc is not None:
    #     assert start_func is None
    # if end_pc is not None:
    #     assert end_func is None

    # TODO: allow start at 0 and/or end at -1

    def lookup_func_pc(func2pc_df: pd.DataFrame, func_name: str):
        assert func2pc_df is not None
        match_df = func2pc_df[func2pc_df["func"] == func_name]
        # print("match_df", match_df)
        assert len(match_df) > 0
        assert len(match_df) == 1
        pc_range = match_df["pc_range"].values[0]
        # print("pc_range", pc_range)
        start_pc = pc_range[0]
        assert start_pc > 0
        return start_pc

    if start_pc is None:
        if start_func is not None:
            start_pc = lookup_func_pc(func2pc_df, start_func)
    if end_pc is None:
        if end_func is not None:
            end_pc = lookup_func_pc(func2pc_df, end_func)

    # print("start_pc", start_pc)
    # print("end_pc", end_pc)
    # TODO: handle multiple calls to start/end func
    def do_trunc(df, start, end):
        if start is not None:
            start_rows = df[df["pc"] == start]
        else:
            start_rows = df.iloc[0:1]
        # print("start_rows", start_rows)
        start_pos = start_rows.index[0]
        # print("start_pos", start_pos)
        # print("temp1", temp1)
        # print("temp1.", temp1.iloc[0])
        # print("temp1.i", temp1.iloc[0].index)
        # print("temp1.i2", temp1.index[0])
        if end is not None:
            end_rows = df[df["pc"] == end]
        else:
            end_rows = df.iloc[-1:-1]
        # print("end_rows", end_rows)
        if len(end_rows) == 0:
            end_pos = df.index[-1]
        else:
            end_pos = end_rows.index[0]
        # print("end_pos", end_pos)
        # print("temp2", temp2)
        # print("temp2.", temp2.iloc[0])
        # print("temp2.i", temp2.iloc[0].index)
        # print("temp2.i2", temp2.index[0])
        # print("AAA", df.iloc[start_pos:end_pos])
        # print("BBB", df.loc[start_pos:end_pos])
        # return df.iloc[start_pos:end_pos]
        return df.loc[start_pos:end_pos]

    trunc_df = do_trunc(trace_df, start_pc, end_pc)
    assert len(trunc_df) > 0
    # print("trunc_df", len(trunc_df))

    # attrs = {
    #     "trace": trace_artifact.name,
    #     "kind": "mapping",
    #     "by": __name__,
    # }
    attrs = trace_artifact.attrs.copy()
    attrs["truncated"] = True
    artifact = InstrTraceArtifact(trace_artifact.name, trunc_df, attrs=attrs)
    # input("555")

    # pc2bb_artifact = TableArtifact(f"pc2bb", pc2bb, attrs=attrs)
    # sess.add_artifact(pc2bb_artifact, override=force)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    trunc_trace(
        sess,
        start_pc=args.start_pc,
        end_pc=args.end_pc,
        start_func=args.start_func,
        end_func=args.end_func,
        force=args.force,
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--start-pc", type=int, default=None)
    parser.add_argument("--end-pc", type=int, default=None)
    parser.add_argument("--start-func", type=str, default=None)
    parser.add_argument("--end-func", type=str, default=None)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
