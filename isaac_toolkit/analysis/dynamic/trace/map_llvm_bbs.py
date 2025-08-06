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
import io
import sys
import leb128
import argparse
import posixpath
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def map_llvm_bbs(sess: Session, force: bool = False):
    logger.info("Mapping LLVM BBs (old)...")
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]
    trace_pc2bb_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb"
    )  # TODO: optional or different pass
    assert len(trace_pc2bb_artifacts) == 1
    trace_pc2bb_artifact = trace_pc2bb_artifacts[0]
    trace_pc2bb_df = trace_pc2bb_artifact.df
    # print("trace_pc2bb_df", trace_pc2bb_df)
    llvm_bbs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs"
    )  # TODO: optional or different pass
    assert len(llvm_bbs_artifacts) == 1
    llvm_bbs_artifact = llvm_bbs_artifacts[0]
    llvm_bbs_df = llvm_bbs_artifact.df.copy()
    llvm_bbs_df[["start", "end"]] = llvm_bbs_df["pcs"].apply(pd.Series)
    # print("llvm_bbs_df", llvm_bbs_df)
    for index, row in llvm_bbs_df.sort_values("start").iterrows():
        start = row["start"]
        end = row["end"]

        def find_matching_bb(pc):
            # print("find_matching_bb", pc)
            matches = trace_pc2bb_df.where(lambda x: x["start"] < pc).dropna()
            # print("m1", matches)
            matches = matches.where(lambda x: pc < x["end"]).dropna()
            # print("m2", matches)
            # print("matches", matches)
            if len(matches) == 0:
                return None
            assert len(matches) == 1
            return matches.iloc[0]

        matching_row_start = [find_matching_bb(start)]

        def split_trace_bb(idx, row, start=None, end=None):
            # print("split_trace_bb", idx, row, start, end)
            # idx = row.index[0]
            if start is not None:
                # print(f"SPLIT {row.index} @ {start} (start)")
                pass
                orig_start = row["start"]
                # input(">")
            if end is not None:
                pass
                # print(f"SPLIT {row.index} @ {end} (end)")
                # input(">")

        # print("matching_row_start", matching_row_start)
        for row in matching_row_start:
            if row is None:
                continue
            # if row["start"] == start:
            #     continue
            split_trace_bb(index, row, start=start)
        matching_row_end = [find_matching_bb(end)]
        # print("matching_row_end", matching_row_end)
        for row in matching_row_end:
            if row is None:
                continue
            # if row["end"] == end:
            #     continue
            split_trace_bb(index, row, end=end)

    # input("!")
    def helper(x):
        # print("x", x)
        ret = set()
        func_names = x["func_name"]
        start = x["start"]
        end = x["end"]
        for func_name in func_names:
            func_matches = llvm_bbs_df[llvm_bbs_df["func_name"] == func_name]
            # print("func_matches", func_matches)
            start_matches = func_matches.where(lambda x: x["start"] <= start).dropna()
            # print("start_matches", start_matches)
            end_matches = start_matches.where(lambda x: x["end"] >= end).dropna()
            # print("end_matches", end_matches)
            for bb_name in end_matches["bb_name"]:
                ret.add(bb_name)
        # input("b")
        # x["test"] = 42
        return ret

    trace_pc2bb_df["llvm_bbs"] = trace_pc2bb_df[["func_name", "start", "end"]].apply(
        lambda x: helper(x), axis=1
    )
    # print("trace_pc2bb_df new", trace_pc2bb_df)
    remain = trace_pc2bb_df[trace_pc2bb_df["llvm_bbs"].map(len) == 0]

    # print("remain", remain)
    def helper2(x):
        # print("x", x)
        ret = set()
        func_names = x["func_name"]
        start = x["start"]
        end = x["end"]
        for func_name in func_names:
            func_matches = llvm_bbs_df[llvm_bbs_df["func_name"] == func_name]
            # print("func_matches", func_matches)
            start_matches = func_matches.where(lambda x: x["start"] <= start).dropna()
            # print("start_matches", start_matches)
            end_matches = start_matches.where(lambda x: x["end"] >= end).dropna()
            # print("end_matches", end_matches)
            for bb_name in end_matches["bb_name"]:
                ret.add(bb_name)
        # input("b")
        # x["test"] = 42
        return ret

    remain["llvm_bbs"] = remain[["func_name", "start", "end"]].apply(
        lambda x: helper2(x), axis=1
    )
    # print("remain new", remain)
    # input("a1")

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mapping",
        "by": __name__,
    }

    artifact = TableArtifact(f"pc2bb_llvm", trace_pc2bb_df, attrs=attrs)
    # print("artifact", artifact)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    map_llvm_bbs(sess, force=args.force)
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
