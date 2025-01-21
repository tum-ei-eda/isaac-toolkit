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
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def map_llvm_bbs_new(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    # print("trace_artifact", trace_artifact)
    trace_df = trace_artifact.df
    llvm_bbs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs"
    )  # TODO: optional or different pass
    assert len(llvm_bbs_artifacts) == 1
    llvm_bbs_artifact = llvm_bbs_artifacts[0]
    llvm_bbs_df = llvm_bbs_artifact.df.copy()
    llvm_bbs_df[["start", "end"]] = llvm_bbs_df["pcs"].apply(pd.Series)
    # print("llvm_bbs_df", llvm_bbs_df)
    total_weight = 0
    for index, row in llvm_bbs_df.sort_values("start").iterrows():
        # print("row", row)
        start = row["start"]
        end = row["end"]
        size = row["size"]
        num_instrs = row["num_instrs"]
        if size == 0:
            continue
        assert num_instrs != 0
        assert size > 0, "Encountered basic block with negative size"
        # num_instrs = row["num_instrs"]

        def get_bb_freq_weight(df, start, end, num_instrs):
            # print("get_bb_freq", start, end, num_instrs)
            if num_instrs == 0:
                return 0, 0.0
            assert num_instrs > 0
            matches = df.where(lambda x: x["pc"] >= start).dropna()
            # print("m1", matches)
            matches = matches.where(lambda x: x["pc"] < end).dropna()
            # print("m2", matches)
            # print("matches", matches)
            count = len(matches)
            # print("count", count)
            bb_count = count // num_instrs
            # print("bb_count", bb_count)
            bb_weight = bb_count * num_instrs
            # print("bb_weight", bb_weight)
            return bb_count, bb_weight

        freq, weight = get_bb_freq_weight(trace_df, start, end, num_instrs)
        llvm_bbs_df.loc[index, "freq"] = freq
        llvm_bbs_df.loc[index, "weight"] = weight
        llvm_bbs_df.loc[index, "num_instrs"] = num_instrs
        total_weight += weight
    trace_length = len(trace_df)
    # coverage = total_weight / trace_length
    # print("trace_length", trace_length)
    # print("total_weight", total_weight)
    # print("coverage", coverage)
    llvm_bbs_df.sort_values("freq", inplace=True, ascending=False)
    llvm_bbs_df["rel_weight"] = llvm_bbs_df["weight"] / trace_length
    # input(">")

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mapping",
        "by": __name__,
    }

    artifact = TableArtifact("llvm_bbs_new", llvm_bbs_df, attrs=attrs)
    # print("artifact", artifact)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    map_llvm_bbs_new(sess, force=args.force)
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
