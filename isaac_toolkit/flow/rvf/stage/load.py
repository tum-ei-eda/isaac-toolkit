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
from isaac_toolkit.session.artifact import ArtifactFlag, ElfArtifact
# from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.frontend.elf.riscv import load_elf
from isaac_toolkit.frontend.linker_map import load_linker_map
from isaac_toolkit.frontend.instr_trace.tgc import (
    load_instr_trace as load_tgc_instr_trace,
)
from isaac_toolkit.frontend.instr_trace.etiss import (
    load_instr_trace as load_etiss_instr_trace,
)
from isaac_toolkit.frontend.instr_trace.etiss_new import (
    load_instr_trace as load_etiss_perf_instr_trace,
)
from isaac_toolkit.frontend.instr_trace.spike import (
    load_instr_trace as load_spike_instr_trace,
)
from isaac_toolkit.frontend.disass.objdump import load_disass
# from isaac_toolkit.frontend.compile_commands.json import load_compile_commands_json

# logger = get_logger()
import logging
logger = logging.getLogger()


def load_artifacts(
    sess: Session, elf_file: Optional[Path] = None, linker_map_file: Optional[Path] = None, instr_trace_file: Optional[Path] = None, disass_file: Optional[Path] = None, force: bool = False, progress: bool = False
):
    logger.info("Loading RVF Demo artifacts...")
    if elf_file:
        load_elf(sess, elf_file, force=force)
    if linker_map_file:
        load_linker_map(sess, linker_map_file, force=force)
    if instr_trace_file:
        assert "_instrs." in instr_trace_file.name
        simulator = instr_trace_file.name.split("_instrs.", 1)[0]
        instr_trace_frontends = {
            "tgc": load_tgc_instr_trace,
            "etiss": load_etiss_instr_trace,
            "etiss_perf": load_etiss_perf_instr_trace,
            "spike": load_spike_instr_trace,
            "spike_bm": load_spike_instr_trace,
        }
        load_instr_trace = instr_trace_frontends.get(simulator)
        operands = False  # TODO: store operands in extra artifact!
        load_instr_trace(
            # sess, instr_trace_file, force=force, progress=progress, operands=operands
            sess, instr_trace_file, force=force, operands=operands
        )
    if disass_file:
        load_disass(sess, disass_file, force=force)
    # load_compile_commands_json(sess, compile_commands_file, force=force)



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
    load_artifacts(sess, elf_file=path_helper(args.elf), linker_map_file=path_helper(args.linker_map), instr_trace_file=path_helper(args.instr_trace), disass_file=path_helper(args.disass), force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elf", default=None)
    parser.add_argument("--linker-map", default=None)
    parser.add_argument("--instr-trace", default=None)
    parser.add_argument("--disass", default=None)
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
