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

from . import Session

logger = logging.getLogger("session")


def create(session_dir: Path, force: bool = False, interactive: bool = False):
    if session_dir.is_dir():
        logger.info("Re-initializing existing session: %s", session_dir)
        if interactive:
            raise NotImplementedError("Interactive mode")
        sess = Session.from_dir(session_dir)
    else:
        logger.info("Initializing new session: %s", session_dir)
        sess = Session.create(session_dir)
    return sess


def handle_create(args):
    assert args.session is not None
    session_dir = Path(args.session)
    force = args.force
    interactive = not force
    create(session_dir, force=force, interactive=interactive)


def get_parser():
    parser = argparse.ArgumentParser()
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
    handle_create(args)


if __name__ == "__main__":
    main(sys.argv[1:])
