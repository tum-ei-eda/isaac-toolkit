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
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.report.report_runtime import generate_runtime_report

# logger = get_logger()
import logging
logger = logging.getLogger()


def generate_reports(sess, output=None, fmt="md", detailed=False, portable=False, style=False, topk=10, force=False):
    logger.info("Reporting RVF Demo results...")
    generate_runtime_report(
        sess,
        output=output,
        fmt=fmt,
        detailed=detailed,
        portable=portable,
        style=style,
        topk=topk,
        force=force,
    )


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    # set_log_level(console_level=args.log, file_level=args.log)
    generate_reports(
        sess,
        output=args.out,
        fmt=args.fmt,
        detailed=args.detailed,
        portable=args.portable,
        style=args.style,
        topk=args.topk,
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
    parser.add_argument("--out", help="Custom output directory (default: SESSION/reports)")
    parser.add_argument("--fmt", choices=["md", "txt", "html", "pdf"], default="md", help="Report format")
    parser.add_argument("--detailed", action="store_true", help="Include detailed breakdowns")
    parser.add_argument("--portable", action="store_true", help="Embed plots as base64 (only for HTML output)")
    parser.add_argument("--style", action="store_true", help="Use custom CSS for HTML output")
    parser.add_argument("--topk", type=int, default=10, help="Limit number of table rows")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
