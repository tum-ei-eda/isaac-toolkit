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

# import argparse
from typing import Union
from pathlib import Path

# from isaac_toolkit.session import Session
# from isaac_toolkit.cli.utils import parse_override_args

# from isaac_toolkit.session.artifact import FileArtifact
from isaac_toolkit.logging import get_logger

from isaac_toolkit.runner.standalone.standalone import StandaloneRunner

# from .cli import add_common_args, add_prog_args

logger = get_logger()


class RISCVStandaloneRunner(StandaloneRunner):

    def __init__(
        self,
        make_dir: Union[str, Path],
        simulator: str,
    ):
        super().__init__(make_dir, simulator)


def add_riscv_args(parser):
    riscv_group = parser.add_argument_group("riscv options")
    del riscv_group


def parse_riscv_args(args):
    ret = {}
    return ret
