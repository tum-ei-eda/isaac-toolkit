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
import sys
import logging
import argparse
from typing import List
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts

logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def helper(df_operands_full: pd.DataFrame, limit: List[str]):
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

    for _, row in df_operands_full.iterrows():
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

    if len(limit) > 0:
        df_operands_full["rs1_src_flt"] = df_operands_full["rs1_src"].apply(lambda x: x if x in limit else "other")
        df_operands_full["rs2_src_flt"] = df_operands_full["rs2_src"].apply(lambda x: x if x in limit else "other")

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


def analyze_operands_sources(
    sess: Session,
    force: bool = False,
    limit: List[str] = ["lui", "mv"],
    # filter_instrs: Optional[str] = None,
    # filter_operands: Optional[str] = None,
):
    artifacts = sess.artifacts
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    operands_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "operands")
    assert len(operands_artifacts) == 1
    operands_artifact = operands_artifacts[0]

    assert len(trace_artifact.df) == len(operands_artifact.df)
    operands_df = pd.concat([trace_artifact.df[["instr"]], operands_artifact.df], axis=1).copy()
    del trace_artifact
    del operands_artifact

    helper(operands_df, limit)

    # TODO: gen artifacts
    # operands_hist_artifact = TableArtifact("instr_operands_hist", operands_hist_df, attrs=attrs2)
    # sess.add_artifact(operands_hist_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_operands_sources(
        sess,
        limit=args.limit.split(","),
        force=args.force,
        # filter_instrs=args.filter_instrs,
        # filter_operands=args.filter_operands,
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
    parser.add_argument("--limit", type=str, default="li,mv")
    # parser.add_argument("--filter-instrs", type=str, default=None)
    # parser.add_argument("--filter-operands", type=str, default=None)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
