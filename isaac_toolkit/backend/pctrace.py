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
from typing import Optional, Union


import sys
import logging
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def export_pctrace(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    force: bool = False,
    compress: bool = False,
):
    del force
    artifacts = sess.artifacts

    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    trace_df = trace_artifact.df
    print("trace_df", trace_df.head())
    if "cost" not in trace_df.columns:
        trace_df["cost"] = 1
    COLS = ["pc", "size", "is_branch", "cost"]
    pctrace_df = trace_df[COLS].copy()
    pctrace_df["pc"] = pctrace_df["pc"].map(lambda x: hex(x))
    print("pctrace_df", pctrace_df.head())
    # TODO: handle --force!

    if output is None:
        profile_dir = sess.directory / "outputs"
        profile_dir.mkdir(exist_ok=True)
        out_name = "pctrace.trc"
        output = profile_dir / out_name
    if compress:
        import lz4.frame

        with lz4.frame.open(output, mode="wt") as f:
            pctrace_df.to_csv(f, index=False, header=False)
    else:
        pctrace_df.to_csv(output, index=False, header=False)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    export_pctrace(
        sess,
        output=args.output,
        force=args.force,
        compress=args.compress,
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
    parser.add_argument("--output", default=None)
    parser.add_argument("--compress", action="store_true", help="Write LZ4 compressed file")
    parser.add_argument("--force", "-f", action="store_true")
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
