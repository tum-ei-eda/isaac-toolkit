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
import argparse
from typing import Optional
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session

# from isaac_toolkit.logging import get_logger, set_log_level

from .stage.load import load_artifacts
from .stage.normalize import normalize_artifacts
from .stage.analyze import analyze_artifacts
from .stage.visualize import visualize_artifacts
from .stage.report import generate_reports
from .stage.profile import generate_profile

# logger = get_logger()
import logging

logger = logging.getLogger()


def run_full_flow(
    sess: Session,
    elf_file: Optional[Path] = None,
    linker_map_file: Optional[Path] = None,
    instr_trace_file: Optional[Path] = None,
    disass_file: Optional[Path] = None,
    unmangle: bool = False,
    force: bool = False,
    progress: bool = False,
    report_fmt="md",
    report_detailed=False,
    report_portable=False,
    report_style=False,
    report_topk=10,
):
    logger.info("Running full RVF flow...")
    load_artifacts(
        sess,
        elf_file=elf_file,
        linker_map_file=linker_map_file,
        instr_trace_file=instr_trace_file,
        disass_file=disass_file,
        force=force,
    )
    normalize_artifacts(sess, force=force)
    analyze_artifacts(sess, force=force)
    visualize_artifacts(sess, force=force)
    generate_reports(
        sess,
        fmt=report_fmt,
        detailed=report_detailed,
        portable=report_portable,
        style=report_style,
        topk=report_topk,
        force=force,
    )
    generate_profile(sess, force=force, unmangle=unmangle)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)

    # set_log_level(console_level=args.log, file_level=args.log)
    def path_helper(x):
        if x is None:
            return x
        return Path(x)

    run_full_flow(
        sess,
        elf_file=path_helper(args.elf),
        linker_map_file=path_helper(args.linker_map),
        instr_trace_file=path_helper(args.instr_trace),
        disass_file=path_helper(args.disass),
        force=args.force,
        unmangle=args.unmangle,
        report_fmt=args.report_fmt,
        report_detailed=args.report_detailed,
        report_portable=args.report_portable,
        report_style=args.report_style,
        report_topk=args.report_topk,
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
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--elf", default=None)
    parser.add_argument("--linker-map", default=None)
    parser.add_argument("--instr-trace", default=None)
    parser.add_argument("--disass", default=None)
    parser.add_argument("--unmangle", action="store_true")
    parser.add_argument("--report-fmt", choices=["md", "txt", "html", "pdf"], default="md", help="Report format")
    parser.add_argument("--report-detailed", action="store_true", help="Include detailed breakdowns")
    parser.add_argument("--report-portable", action="store_true", help="Embed plots as base64 (only for HTML output)")
    parser.add_argument("--report-style", action="store_true", help="Use custom CSS for HTML output")
    parser.add_argument("--report-topk", type=int, default=10, help="Limit number of table rows")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
