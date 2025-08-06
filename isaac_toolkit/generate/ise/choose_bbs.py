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
from typing import Optional
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def lookup_files(file2funcs_df, func_name):
    matches = set()
    for _, row in file2funcs_df.iterrows():
        func_names = row["func_names"]
        if func_name in func_names:
            file = Path(row["file"])
            matches.add(file.resolve())
    return list(matches)


def choose_bbs(
    sess: Session,
    threshold: float = 0.9,
    min_weight: float = 0.01,
    min_supported_weight: Optional[float] = None,
    min_instrs: Optional[int] = 2,
    max_num: Optional[int] = None,
    force: bool = False,
):
    logger.info("Choosing BBs...")
    artifacts = sess.artifacts
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]
    llvm_bbs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs_new"
    )
    assert len(llvm_bbs_artifacts) == 1
    llvm_bbs_artifact = llvm_bbs_artifacts[0]
    llvm_bbs_df = llvm_bbs_artifact.df.copy()
    ise_potential_per_llvm_bb_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "ise_potential_per_llvm_bb"
    )
    ise_potential_per_llvm_bb_df = None
    if len(ise_potential_per_llvm_bb_artifacts) > 0:
        assert len(ise_potential_per_llvm_bb_artifacts) == 1
        ise_potential_per_llvm_bb_artifact = ise_potential_per_llvm_bb_artifacts[0]
        ise_potential_per_llvm_bb_df = ise_potential_per_llvm_bb_artifact.df.copy()

    file2funcs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "file2funcs"
    )
    if len(file2funcs_artifacts) > 0:
        assert len(file2funcs_artifacts) == 1
        file2funcs_artifact = file2funcs_artifacts[0]
        file2funcs_df = file2funcs_artifact.df.copy()
        # print("llvm_bbs_df", llvm_bbs_df)
    else:
        file2funcs_df = None
    sum_weights = 0.0
    choices = []
    for index, row in llvm_bbs_df.sort_values("rel_weight", ascending=False).iterrows():
        func_name = row["func_name"]
        bb_name = row["bb_name"]
        rel_weight = row["rel_weight"]
        num_instrs = row["num_instrs"]
        if pd.isna(rel_weight):
            continue
        if rel_weight < min_weight:
            continue
        if min_supported_weight is not None:
            assert (
                ise_potential_per_llvm_bb_df is not None
            ), "Run isaac_toolkit.generate.ise.check_ise_potential_per_llvm_bb first!"

            def lookup_supported(df, func_name, bb_name):
                filtered = df[(df["func_name"] == func_name) & (df["bb_name"] == bb_name)]
                assert len(filtered) == 1
                rel_supported_count = filtered["supported_rel_count"].iloc[0]
                return rel_supported_count

            rel_supported_count = lookup_supported(ise_potential_per_llvm_bb_df, func_name, bb_name)
            rel_supported_weight = rel_supported_count * rel_weight
            if rel_supported_weight < min_supported_weight:
                continue
        if min_instrs is not None:
            if num_instrs < min_instrs:
                continue
        func_name = row["func_name"]
        bb_name = row["bb_name"]
        freq = row["freq"]
        if file2funcs_df is not None:
            files = lookup_files(file2funcs_df, func_name)
            if len(files) == 0:
                # func not conatined in file2funcs?
                file = None
            else:
                assert len(files) == 1
                file = files[0]
        else:
            file = None
        choice = {
            "func_name": func_name,
            "file": file,
            "bb_name": bb_name,
            "rel_weight": rel_weight,
            "num_instrs": num_instrs,
            "freq": freq,
        }
        choices.append(choice)
        sum_weights += rel_weight
        if sum_weights >= threshold:
            break
        if max_num is not None:
            if len(choices) >= max_num:
                break
    # print("sum_weights", sum_weights)
    # print("choices", choices, len(choices))
    choices_df = pd.DataFrame(choices)
    # print("choices_df", choices_df)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "list",
        "by": __name__,
    }

    artifact = TableArtifact("choices", choices_df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    choose_bbs(
        sess,
        threshold=args.threshold,
        min_weight=args.min_weight,
        min_supported_weight=args.min_supported_weight,
        max_num=args.max_num,
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
    parser.add_argument("--min-weight", type=float, default=0.01)
    parser.add_argument("--min-supported-weight", type=float, default=None)
    parser.add_argument("--max-num", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
