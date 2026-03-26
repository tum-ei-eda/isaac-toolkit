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

from .standalone import StandaloneBuilder

# from .cli import add_common_args, add_prog_args

logger = get_logger()


class RISCVStandaloneBuilder(StandaloneBuilder):

    def __init__(
        self,
        make_dir: Union[str, Path],
        simulator: str,
        arch: Optional[str] = None,
        abi: Optional[str] = None,
        xlen: Optional[int] = None,
        gnu_prefix: Optional[Union[str, Path]] = None,
        gnu_name: Optional[Union[str, Path]] = None,
        toolchain: str = "gcc",
        optimize: Optional[str] = None,
    ):
        super().__init__(make_dir, simulator, toolchain=toolchain, optimize=optimize)
        if arch:
            self.defaults["RISCV_ARCH"] = arch
        if abi:
            self.defaults["RISCV_ABI"] = abi
        if xlen:
            self.defaults["RISCV_XLEN"] = xlen
        if gnu_prefix:
            self.defaults["RISCV_PREFIX"] = gnu_prefix
        if gnu_name:
            self.defaults["RISCV_NAME"] = gnu_name


def add_riscv_args(parser):
    riscv_group = parser.add_argument_group("riscv options")
    riscv_group.add_argument("--arch", type=str, default=None)
    riscv_group.add_argument("--abi", type=str, default=None)
    riscv_group.add_argument("--xlen", type=int, default=None)
    riscv_group.add_argument("--gnu-prefix", type=str, default=None)
    riscv_group.add_argument("--gnu-name", type=str, default=None)


def parse_riscv_args(args):
    ret = {
        "arch": args.arch,
        "abi": args.abi,
        "xlen": args.xlen,
        "gnu_prefix": args.gnu_prefix,
        "gnu_name": args.gnu_name,
    }
    return ret
