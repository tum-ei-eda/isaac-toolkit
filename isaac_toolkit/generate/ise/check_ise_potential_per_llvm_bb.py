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
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import TableArtifact, filter_artifacts
from .check_ise_potential import get_unsupported_opcodes, get_ise_potential_df
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def check_ise_potential_per_llvm_bb(
    sess: Session,
    min_supported: float = 0.15,
    allow_mem: bool = False,
    allow_loads: bool = False,
    allow_stores: bool = False,
    allow_branches: bool = False,
    allow_compressed: bool = True,
    allow_custom: bool = True,
    allow_fp: bool = False,
    allow_system: bool = False,
    force: bool = False,
):
    logger.info("Checking ISE potential per LLVM BB...")
    artifacts = sess.artifacts
    opcodes_hist_artifacts = filter_artifacts(
        artifacts, lambda x: x.name == "opcodes_per_llvm_bb_hist"
    )
    assert len(opcodes_hist_artifacts) == 1
    opcodes_hist_artifact = opcodes_hist_artifacts[0]

    opcodes_per_llvm_bb_hist_df = opcodes_hist_artifact.df
    unsupported_opcodes = get_unsupported_opcodes(
        allow_mem=allow_mem,
        allow_loads=allow_loads,
        allow_stores=allow_stores,
        allow_branches=allow_branches,
        allow_compressed=allow_compressed,
        allow_custom=allow_custom,
        allow_fp=allow_fp,
        allow_system=allow_system,
    )

    dfs = []
    for group, opcodes_hist_df in opcodes_per_llvm_bb_hist_df.groupby(
        ["func_name", "bb_name"]
    ):
        func_name, bb_name = group
        # print("func_name", func_name)
        # print("bb_name", bb_name)
        # print("opcodes_hist_df")
        print(opcodes_hist_df)
        ise_potential_df = get_ise_potential_df(
            opcodes_hist_df, unsupported_opcodes, min_supported
        )
        ise_potential_df.insert(0, "func_name", func_name)
        ise_potential_df.insert(1, "bb_name", bb_name)
        # print("ise_potential_df")
        print(ise_potential_df)
        dfs.append(ise_potential_df)
    ise_potential_per_llvm_bb_df = pd.concat(dfs)
    # print(ise_potential_per_llvm_bb_df)
    # input(">")

    attrs = {
        "kind": "table",
        "by": __name__,
    }

    artifact = TableArtifact(
        "ise_potential_per_llvm_bb", ise_potential_per_llvm_bb_df, attrs=attrs
    )
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    check_ise_potential_per_llvm_bb(
        sess,
        allow_mem=args.allow_mem,
        allow_loads=args.allow_loads,
        allow_stores=args.allow_stores,
        allow_branches=args.allow_branches,
        allow_compressed=args.allow_compressed,
        allow_custom=args.allow_custom,
        allow_fp=args.allow_fp,
        allow_system=args.allow_system,
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
    # TODO: default to allowed?
    parser.add_argument("--min-supported", type=float, default=0.15)
    parser.add_argument("--allow-mem", action="store_true")
    parser.add_argument("--allow-loads", action="store_true")
    parser.add_argument("--allow-stores", action="store_true")
    parser.add_argument("--allow-branches", action="store_true")
    parser.add_argument("--allow-compressed", action="store_true")
    parser.add_argument("--allow-custom", action="store_true")
    parser.add_argument("--allow-fp", action="store_true")
    parser.add_argument("--allow-system", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
