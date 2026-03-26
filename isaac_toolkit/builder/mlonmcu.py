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
import shutil
import argparse
from typing import Optional, Union, List
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import FileArtifact
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()

from .builder import ISAACBuilder


class MLonMCUBuilder(ISAACBuilder):
    def __init__(
        self,
        make_dir: Union[str, Path],
        target: str,
        toolchain: str = "gcc",
        platform: str = "mlif",
        optimize: Optional[str] = None,
    ):
        raise NotImplementedError

    def build(self, dest_dir: Union[str, Path], verbose: bool = False):
        raise NotImplementedError


def invoke_mlonmcu_builder(
    sess: Session,
    simulator: str,
    toolchain: str = "gcc",
    optimize: Optional[str] = None,
    dest_dir: Optional[str] = None,
    label: Optional[str] = None,
    verbose: bool = False,
    force: bool = False,
    load: bool = False,
    cleanup: bool = False,
):
    # TODO: docker support
    # TODO: allow debug?
    logger.info("Building MLonMCU example...")

    builder = MLonMCUBuilder(simulator, toolchain=toolchain, optimize=optimize)

    if label is None:
        # TODO: add timestamp?
        label = "coremark-build"

    if dest_dir is not None:
        dest_dir = Path(dest_dir)
    else:
        assert sess is not None
        sess_dir = sess.directory
        dest_dir = sess_dir / "temp"
    if dest_dir.is_dir():
        assert force, "Destination directory already exists. Use --force to override."
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    builder.build(dest_dir=dest_dir, verbose=verbose)

    if load:
        raise NotImplementedError("load")
    if cleanup:
        raise NotImplementedError("cleanup")

    # attrs = {
    #     "by": __name__,
    #     "kind": "",
    # }
    # initializer_artifact = FileArtifact(name, input_file, attrs=attrs)
    # sess.add_artifact(initializer_artifact, override=force)


def handle(args):
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    invoke_mlonmcu_builder(
        sess,
        args.simulator,
        toolchain=args.toolchain,
        optimize=args.optimize,
        label=args.label,
        dest_dir=args.dest,
        verbose=args.verbose,
        load=args.load,
        cleanup=args.cleanup,
        force=args.force,
    )
    sess.save()


def add_common_args(parser):
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str)
    parser.add_argument("--simulator", type=str, required=True)
    parser.add_argument("--toolchain", type=str, default=None)
    parser.add_argument("--optimize", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--dest", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--load", action="store_true")
    # TODO: no-load?
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")


def get_parser():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
