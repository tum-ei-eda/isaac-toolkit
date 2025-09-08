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

# from collections import defaultdict

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def collect_pcs(trace_df):
    pcs_df = (
        trace_df["pc"]
        .value_counts()
        .rename_axis("pc")       # makes 'pc' a column
        .reset_index(name="count")  # turns counts into a column
    )
    total_count = pcs_df["count"].sum()
    pcs_df["rel_count"] = pcs_df["count"] / total_count
    pcs_df.sort_values("count", ascending=False, inplace=True)
    return pcs_df


def create_pc_hist(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    # print("elf_artifacts", elf_artifacts)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]

    pcs_df = collect_pcs(trace_artifact.df)
    # print("operands_df", operands_df)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "histogram",
        "by": __name__,
    }

    pcs_artifact = TableArtifact("pcs_hist", pcs_df, attrs=attrs)
    sess.add_artifact(pcs_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    create_pc_hist(sess, force=args.force)
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
