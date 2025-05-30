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
from isaac_toolkit.session.artifact import TableArtifact, filter_artifacts


logger = logging.getLogger("check_ise_potential")


def get_unsupported_opcodes(
    allow_mem: bool = False,
    allow_loads: bool = False,
    allow_stores: bool = False,
    allow_branches: bool = False,
    allow_compressed: bool = True,
    allow_custom: bool = True,
    allow_fp: bool = False,
    allow_system: bool = False,
):
    unsupported_opcodes = set()

    allow_lui = False
    # allow_lui = True
    if not allow_lui:
        unsupported_opcodes.add("LUI")

    if not allow_compressed:
        unsupported_opcodes.add("LOAD (Compressed)")
        unsupported_opcodes.add("STORE (Compressed)")
        unsupported_opcodes.add("OP-IMM (Compressed)")
        unsupported_opcodes.add("BRANCH (Compressed)")
        unsupported_opcodes.add("MISC-ALU (Compressed)")
        # unsupported_opcodes.add("JAL (Compressed)")
        unsupported_opcodes.add("JALR (Compressed)")
    if not allow_loads or not allow_mem:
        unsupported_opcodes.add("LOAD")
        unsupported_opcodes.add("LOAD-FP")
        unsupported_opcodes.add("LOAD (Compressed)")
    if not allow_stores or not allow_mem:
        unsupported_opcodes.add("STORE")
        unsupported_opcodes.add("STORE-FP")
        unsupported_opcodes.add("STORE (Compressed)")
    if not allow_mem:
        unsupported_opcodes.add("MISC-MEM")
    if not allow_branches:
        unsupported_opcodes.add("BRANCH")
        unsupported_opcodes.add("JAL")
        unsupported_opcodes.add("JALR")
        unsupported_opcodes.add("BRANCH (Compressed)")
        # unsupported_opcodes.add("JAL (Compressed)")
        unsupported_opcodes.add("JALR (Compressed)")
    if not allow_custom:
        unsupported_opcodes.add("custom-0")
        unsupported_opcodes.add("custom-1")
        unsupported_opcodes.add("custom-2/rv128")
        unsupported_opcodes.add("custom-3/rv128")
    if not allow_fp:
        unsupported_opcodes.add("OP-FP")
        unsupported_opcodes.add("LOAD-FP")
        unsupported_opcodes.add("STORE-FP")
    if not allow_system:
        unsupported_opcodes.add("SYSTEM")
    return unsupported_opcodes


def get_ise_potential_df(opcodes_hist_df, unsupported_opcodes, min_supported):
    # print("opcodes_hist_df")
    # print(opcodes_hist_df)
    supported_opcodes_hist_df = opcodes_hist_df[~opcodes_hist_df["opcode"].isin(unsupported_opcodes)]
    # print("supported_opcodes_hist_df")
    # print(supported_opcodes_hist_df)
    supported_rel_count = supported_opcodes_hist_df["rel_count"].sum()
    # print("supported_rel_count", supported_rel_count)
    unsupported_rel_count = 1 - supported_rel_count
    has_potential = supported_rel_count >= min_supported
    ise_potential_data = {
        "supported_rel_count": supported_rel_count,
        "unsupported_rel_count": unsupported_rel_count,
        "has_potential": has_potential,
    }
    ise_potential_df = pd.DataFrame([ise_potential_data])
    return ise_potential_df


def check_ise_potential(
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
    artifacts = sess.artifacts
    opcodes_hist_artifacts = filter_artifacts(artifacts, lambda x: x.name == "opcodes_hist")
    assert len(opcodes_hist_artifacts) == 1
    opcodes_hist_artifact = opcodes_hist_artifacts[0]

    opcodes_hist_df = opcodes_hist_artifact.df

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

    ise_potential_df = get_ise_potential_df(opcodes_hist_df, unsupported_opcodes, min_supported)

    attrs = {
        "kind": "table",
        "by": __name__,
    }

    artifact = TableArtifact("ise_potential", ise_potential_df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    check_ise_potential(
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
