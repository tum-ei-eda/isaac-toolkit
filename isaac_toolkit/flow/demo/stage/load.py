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

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, ElfArtifact
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.frontend.elf.riscv import load_elf
from isaac_toolkit.frontend.linker_map import load_linker_map
from isaac_toolkit.frontend.instr_trace.etiss import (
    load_instr_trace as load_etiss_instr_trace,
)
from isaac_toolkit.frontend.instr_trace.spike import (
    load_instr_trace as load_spike_instr_trace,
)
from isaac_toolkit.frontend.disass.objdump import load_disass
from isaac_toolkit.frontend.compile_commands.json import load_compile_commands_json

logger = get_logger()


def load_mlonmcu_exported_run(sess: Session, run_dir: Path, force: bool = False, progress: bool = False):
    logger.info("Loading MLonMCU exported run...")
    assert run_dir.is_dir(), f"MLonMCU run dir does not exist: {run_dir}"
    elf_file = run_dir / "generic_mlonmcu"
    load_elf(sess, elf_file, force=force)
    linker_map_file = run_dir / "mlif" / "generic" / "linker.map"  # TODO: move to real artifacts?
    load_linker_map(sess, linker_map_file, force=force)
    # TODO: load report via frontend?
    # TODO: load initializer?
    report_file = run_dir / "report.csv"
    assert report_file.is_file()
    report_df = pd.read_csv(report_file)
    assert len(report_df) == 1
    assert "Target" in report_df.columns
    target = report_df["Target"].iloc[0]
    instr_trace_frontends = {
        "etiss": load_etiss_instr_trace,
        "spike": load_spike_instr_trace,
        "spike_rv32": load_spike_instr_trace,
        "spike_rv64": load_spike_instr_trace,
    }
    load_instr_trace = instr_trace_frontends.get(target)
    assert load_instr_trace is not None, f"Frontend lookup failed for target: {target}"
    instr_trace_file = run_dir / f"{target}_instrs.log"
    operands = True  # TODO: store operands in extra artifact!
    load_instr_trace(sess, instr_trace_file, force=force, progress=progress, operands=operands)
    dump_file = run_dir / "generic_mlonmcu.dump"
    load_disass(sess, dump_file, force=force)
    compile_commands_file = run_dir / "mlif" / "compile_commands.json"
    load_compile_commands_json(sess, compile_commands_file, force=force)


def load_artifacts(sess: Session, run_dir: Path, force: bool = False, progress: bool = False):
    logger.info("Loading ISAAC Demo artifacts...")
    load_mlonmcu_exported_run(sess, run_dir, force=force, progress=progress)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    run_dir = Path(args.exported_run)
    load_artifacts(sess, run_dir, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("exported_run")
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
