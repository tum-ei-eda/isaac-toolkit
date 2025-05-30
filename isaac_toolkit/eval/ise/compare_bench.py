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
import io
import sys
import leb128
import logging
import argparse
import posixpath
from typing import Optional, Union
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("compare_bench")


def compare_bench(
    sess: Session,
    report: Union[str, Path] = None,
    mem_report: Optional[Union[str, Path]] = None,
    force: bool = False,
):
    COLS = ["Model", "Arch", "Run Instructions", "Run Instructions (rel.)"]
    MEM_COLS = ["Model", "Arch", "Total ROM", "Total RAM", "ROM code", "ROM code (rel.)"]
    COMMON_COLS = list(set(COLS) & set(MEM_COLS))

    assert report is not None

    report_file = Path(report)
    assert report_file.is_file()
    report_df = pd.read_csv(report_file)[COLS]

    if mem_report:
        mem_report_file = Path(mem_report)
        assert mem_report_file.is_file()
        mem_report_df = pd.read_csv(mem_report_file)[MEM_COLS]
        report_df = report_df.merge(mem_report_df, on=COMMON_COLS)

    compare_df = report_df

    attrs = {}

    artifact = TableArtifact("compare_bench", compare_df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    compare_bench(
        sess,
        report=args.report,
        mem_report=args.mem_report,
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
    parser.add_argument("--report", required=True, help="Report CSV file")
    parser.add_argument("--mem-report", default=None, help="Memory report CSV file")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
