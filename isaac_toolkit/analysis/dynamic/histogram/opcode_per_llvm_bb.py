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
import argparse
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from .opcode import decode_opcode
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def create_opcode_per_llvm_bb_hist(sess: Session, force: bool = False):
    logger.info("Analyzing opcodes per LLVM BB...")
    artifacts = sess.artifacts
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    trace_df = trace_artifact.df

    llvm_bbs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs"
    )  # TODO: optional or different pass
    assert len(llvm_bbs_artifacts) == 1
    llvm_bbs_artifact = llvm_bbs_artifacts[0]
    llvm_bbs_df = llvm_bbs_artifact.df.copy()
    llvm_bbs_df[["start", "end"]] = llvm_bbs_df["pcs"].apply(pd.Series)
    # print("llvm_bbs_df", llvm_bbs_df)

    # TODO: use pc_hist
    pc_counts = trace_df["pc"].value_counts().rename("count").reset_index()
    pc_counts.columns = ["pc", "count"]
    total_count = pc_counts["count"].sum()

    # extract table of unique (pc, bytecode)
    unique_pc_df = trace_df.drop_duplicates(subset=["pc"])[["pc", "bytecode"]]

    # apply detect_opcode once per PC
    unique_pc_df["opcode"] = unique_pc_df["bytecode"].apply(decode_opcode)

    # keep only (pc, opcode)
    pc_to_opcode = unique_pc_df[["pc", "opcode"]]

    pc_opcode_df = (
        pc_counts.merge(pc_to_opcode, on="pc", how="left")
                 .sort_values("pc")
                 .reset_index(drop=True)
    )

    dfs = []
    for _, row in llvm_bbs_df.iterrows():
        start = row["start"]
        end   = row["end"]

        # Efficient filter: this DataFrame is tiny compared to trace_df
        slice_df = pc_opcode_df.loc[
            (pc_opcode_df["pc"] >= start) & (pc_opcode_df["pc"] < end)
        ]

        if slice_df.empty:
            continue

        # aggregate counts by opcode
        op_hist = (
            slice_df.groupby("opcode")["count"]
                    .sum()
                    .reset_index()
        )

        op_hist.insert(0, "func_name", row["func_name"])
        op_hist.insert(1, "bb_name", row["bb_name"])

        dfs.append(op_hist)

    df = pd.concat(dfs)
    df["rel_count"] = df["count"] / total_count

    attrs = {
        "kind": "histogram",
        "by": __name__,
    }

    opcodes_artifact = TableArtifact("opcodes_per_llvm_bb_hist", df, attrs=attrs)
    sess.add_artifact(opcodes_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    create_opcode_per_llvm_bb_hist(sess, force=args.force)
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
