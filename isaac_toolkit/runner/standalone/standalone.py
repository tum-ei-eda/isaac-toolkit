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
import subprocess
from typing import Optional, Union, List
from pathlib import Path

from isaac_toolkit.logging import get_logger

from ..runner import ISAACRunner

logger = get_logger()


class StandaloneRunner(ISAACRunner):
    def __init__(
        self,
        make_dir: Union[str, Path],
        simulator: str,
    ):
        super().__init__()
        self.make_dir = Path(make_dir)
        assert self.make_dir.is_dir()
        self.defaults = {}
        self.defaults["SIMULATOR"] = simulator
        self.simulator = simulator

    def run(
        self,
        dest_dir: Union[str, Path],
        elf_file: Optional[Union[str, Path]] = None,
        make_target: str = "run",
        extra_args: Optional[List[str]] = None,
        overrides: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        assert dest_dir is not None
        args = ["make", "-C", self.make_dir, make_target]
        args.append(f"DEST={dest_dir}")
        if elf_file:
            elf_file = Path(elf_file)
            assert elf_file.is_file()
            args.append(f"ELF={elf_file}")
        # extra_args = self.get_extra_args()
        # extra_args = s
        if extra_args:
            args += extra_args
        defaults = self.defaults
        if overrides:
            defaults.update(overrides)
        if defaults:
            args += [f"{k}={v}" for k, v in defaults.items()]
        if verbose:
            command = " ".join(map(str, args))
            logger.info("Executing: %s", command)

        if verbose:
            subprocess.run(args, check=True)
        else:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        metrics = {}
        return metrics

    def trace(
        self,
        dest_dir: Union[str, Path],
        elf_file: Optional[Union[str, Path]] = None,
        make_target: str = "run",
        extra_args: Optional[List[str]] = None,
        overrides: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        return self.run(
            dest_dir,
            elf_file=elf_file,
            make_target="trace",
            extra_args=extra_args,
            overrides=overrides,
            verbose=verbose,
        )
