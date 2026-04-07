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
import argparse
import shutil

# import argparse
from typing import Optional, Union
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.cli.utils import parse_override_args

# from isaac_toolkit.session.artifact import FileArtifact
from isaac_toolkit.logging import get_logger, set_log_level

# from .standalone import StandaloneBuilder

from .riscv import RISCVStandaloneBuilder, add_riscv_args, parse_riscv_args
from .common import load_build_artifacts

# from .cli import add_common_args, add_prog_args

logger = get_logger()


class VicunaStandaloneBuilder(RISCVStandaloneBuilder):

    def __init__(
        self,
        make_dir: Union[str, Path],
        name: str = "vicuna",
        # jit: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(make_dir, name, **kwargs)
        # self.jit = jit


def invoke_vicuna_builder(
    sess: Session,
    make_dir: Union[str, Path],
    program: str = "unknown",
    # jit: Optional[str] = None,
    arch: Optional[str] = None,
    abi: Optional[str] = None,
    xlen: Optional[int] = None,
    gnu_prefix: Optional[Union[str, Path]] = None,
    gnu_name: Optional[Union[str, Path]] = None,
    toolchain: str = "gcc",
    optimize: Optional[str] = None,
    overrides: Optional[dict] = None,
    dest_dir: Optional[str] = None,
    label: Optional[str] = None,
    verbose: bool = False,
    force: bool = False,
    load: bool = False,
    cleanup: bool = False,
):
    # TODO: docker support
    # TODO: allow debug?
    logger.info("Building Vicuna program...")

    builder = VicunaStandaloneBuilder(
        make_dir,
        arch=arch,
        abi=abi,
        xlen=xlen,
        gnu_prefix=gnu_prefix,
        gnu_name=gnu_name,
        toolchain=toolchain,
        optimize=optimize,
    )

    if label is None:
        # TODO: add timestamp?
        label = f"{program}-vicuna-build"

    if dest_dir is not None:
        dest_dir = Path(dest_dir)
    else:
        assert sess is not None
        sess_dir = sess.directory
        dest_dir = sess_dir / "temp" / label
    if dest_dir.is_dir():
        assert force, "Destination directory already exists. Use --force to override."
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)

    builder.build(dest_dir=dest_dir, verbose=verbose, overrides=overrides)

    if load:
        load_build_artifacts(sess, dest_dir, program, force=force)

    # attrs = {
    #     "by": __name__,
    #     "kind": "",
    # }
    # initializer_artifact = FileArtifact(name, input_file, attrs=attrs)
    # sess.add_artifact(initializer_artifact, override=force)

    sess.save()
    if cleanup:
        logger.info("Cleaning up files...")
        shutil.rmtree(dest_dir / "build")
        # shutil.rmtree(dest_dir / "out")


def add_vicuna_args(parser):
    vicuna_group = parser.add_argument_group("vicuna options")
    del vicuna_group


def parse_vicuna_args(args):
    # ret = {"jit": args.jit}
    ret = parse_riscv_args(args)
    return ret


def get_parser():
    from .cli import add_common_args, add_prog_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(simulator="vicuna")
    add_riscv_args(parser)
    add_vicuna_args(parser)
    add_prog_args(parser)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
