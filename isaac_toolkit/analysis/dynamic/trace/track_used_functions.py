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
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def get_effective_footprint_df(trace_df, func2pc_df, footprint_df):
    df = footprint_df.copy()
    trace_df_unique = trace_df[["pc"]].drop_duplicates()
    df["Used"] = False
    func2pc_df[["start", "end"]] = func2pc_df["pc_range"].apply(pd.Series)
    # print("footprint_df", footprint_df)
    for index, row in footprint_df.iterrows():
        # print("index", index)
        # print("row", row)
        func_name = row["func"]
        matches = func2pc_df.where(lambda x: x["func"] == func_name).dropna()
        # print("matches", matches)
        # assert len(matches) == 1
        assert len(matches) > 0
        for _, m in matches.iterrows():
            pc_range = m["pc_range"]
            # print("pc_range", pc_range)
            start_pc, end_pc = pc_range
            if end_pc < 0:
                continue
            matches = trace_df_unique.where(lambda x: x["pc"] >= start_pc).dropna()
            matches = matches.where(lambda x: x["pc"] < end_pc).dropna()
            if len(matches) > 0:
                # print("matches", matches)
                df.loc[index, "Used"] = True
    bytes_before = df["bytes"].sum()
    df = df[df["Used"]]
    bytes_after = df["bytes"].sum()
    df["eff_rel_bytes"] = df["bytes"] / bytes_after
    # rel = bytes_after / bytes_before
    # print("bytes", bytes_before, bytes_after, rel)
    # print("df", df)
    # input("*")
    return df


def track_unused_functions(sess: Session, force: bool = False):
    logger.info("Tracking unused functions...")
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    func2pc_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "func2pc")
    assert len(func2pc_artifacts) == 1
    func2pc_artifact = func2pc_artifacts[0]
    mem_footprint_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "mem_footprint"
    )
    assert len(mem_footprint_artifacts) == 1
    mem_footprint_artifact = mem_footprint_artifacts[0]

    effective_footprint_df = get_effective_footprint_df(
        trace_artifact.df, func2pc_artifact.df, mem_footprint_artifact.df
    )
    # print("effective_footprint_df", effective_footprint_df)
    # input("@")

    attrs = {
        "trace": trace_artifact.name,
        # "elf": ?
        "kind": "footprint",
        "by": __name__,
    }

    effective_mem_footprint_artifact = TableArtifact("effective_mem_footprint", effective_footprint_df, attrs=attrs)
    sess.add_artifact(effective_mem_footprint_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    track_unused_functions(sess, force=args.force)
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
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
