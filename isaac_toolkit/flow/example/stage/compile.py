#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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
from typing import Optional
from pathlib import Path

from isaac_toolkit.session import Session

# from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.builder.standalone import STANDALONE_BUILDERS

# from isaac_toolkit.frontend.compile_commands.json import load_compile_commands_json

# logger = get_logger()

logger = logging.getLogger()


def compile_program(
    sess: Session,
    program: str,
    simulator: str,
    force: bool = False,
    toolchain: Optional[str] = None,
    optimize: Optional[str] = None,
    overrides: Optional[dict] = None,
    load: bool = True,
    progress: bool = False,
    verbose: bool = False,
):
    logger.info("Compile Example Demo program...")
    # TODO: builder registry
    # TODO: program registry
    builder_cls = STANDALONE_BUILDERS.get(simulator)
    assert builder_cls is not None, f"Builder lookup failed for simulator: {simulator}"

    label = None
    if label is None:
        # TODO: add timestamp?
        label = "{program}-build"

    dest_dir = None
    if dest_dir is not None:
        dest_dir = Path(dest_dir)
    else:
        assert sess is not None
        sess_dir = sess.directory
        dest_dir = sess_dir / "temp"

    builder = builder_cls(program, toolchain=toolchain, optimize=optimize)
    builder.build(dest_dir, verbose=verbose)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)

    compile_program(
        sess,
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
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
