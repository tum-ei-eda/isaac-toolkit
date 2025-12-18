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

# import time
import sys
import argparse
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import TraceArtifact


# TODO: logger


def load_perf_trace(sess: Session, input_files: List[Path], force: bool = False):
    assert len(input_files) > 0
    name = input_files[0].name
    # sort input files by name
    sorted_files = sorted(input_files, key=lambda x: x.name)
    dfs = list(
        map(
            lambda x: pd.read_csv(x, sep=r"\s*\|\s*", engine="python")
            .dropna(axis=1)
            .replace("------------------", np.nan)
            # .map(lambda y: int(y, 0) if not pd.isna(y) else y),  # Python 3.10
            .applymap(lambda y: int(y, 0) if not pd.isna(y) else y),  # Python 3.8
            sorted_files,
        )
        # TODO: rewrite to be less hacky and more efficient
    )
    df = pd.concat(dfs, axis=0)
    df.reset_index(inplace=True, drop=True)
    # df["instr"] = df["instr"].astype("category")
    # df["size"] = df["size"].astype("category")

    attrs = {
        "simulator": "etiss",
        "cpu_arch": "unknown",
        "kind": "perf_trace",
        "by": "isaac_toolkit.frontend.perf_trace.etiss_perf",
    }
    artifact = TraceArtifact(name, df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_files = list(map(Path, args.files))
    load_perf_trace(sess, input_files, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
