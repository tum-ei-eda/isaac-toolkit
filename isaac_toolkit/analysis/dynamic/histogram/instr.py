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

# from collections import defaultdict

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def collect_instructions(trace_df):
    instrs = trace_df["instr"].value_counts().to_dict()
    instrs_data = []
    for instr_name, instr_count in instrs.items():
        instr_data = {"instr": instr_name, "count": instr_count}
        instrs_data.append(instr_data)
    instrs_df = pd.DataFrame(instrs_data)
    total_count = instrs_df["count"].sum()
    instrs_df["rel_count"] = instrs_df["count"] / total_count
    instrs_df.sort_values("count", ascending=False, inplace=True)

    return instrs_df


def create_instr_hist(sess: Session, force: bool = False):
    logger.info("Creating instrution histogram...")
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE
    )
    # print("elf_artifacts", elf_artifacts)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]

    instrs_df = collect_instructions(trace_artifact.df)
    # print("operands_df", operands_df)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "histogram",
        "by": __name__,
    }

    instrs_artifact = TableArtifact("instrs_hist", instrs_df, attrs=attrs)
    sess.add_artifact(instrs_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    create_instr_hist(sess, force=args.force)
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
