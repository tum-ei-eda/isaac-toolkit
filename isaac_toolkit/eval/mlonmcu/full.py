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
from typing import Optional
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import filter_artifacts
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
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


# TODO: move to own frontend and share code
def load_mlonmcu_exported_run(sess: Session, run_dir: Path, force: bool = False, progress: bool = False):
    logger.info("Loading MLonMCU exported run...")
    assert run_dir.is_dir(), f"MLonMCU run dir does not exist: {run_dir}"
    elf_file = run_dir / "generic_mlonmcu"
    assert elf_file.is_file(), f"ELF file not found: {elf_file}"
    load_elf(sess, elf_file, force=force)
    linker_map_file = run_dir / "mlif" / "generic" / "linker.map"  # TODO: move to real artifacts?
    if linker_map_file.is_file():
        load_linker_map(sess, linker_map_file, force=force)
    else:
        logger.warning("Skipping loading non-existing linker_map: %s", linker_map_file)
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
    if instr_trace_file.is_file():
        operands = True  # TODO: store operands in extra artifact!
        load_instr_trace(sess, instr_trace_file, force=force, progress=progress, operands=operands)
    else:
        logger.warning("Skipping loading non-existing instr_trace: %s", instr_trace_file)
    dump_file = run_dir / "generic_mlonmcu.dump"
    if dump_file.is_file():
        load_disass(sess, dump_file, force=force)
    else:
        logger.warning("Skipping loading non-existing disass: %s", dump_file)
    compile_commands_file = run_dir / "mlif" / "compile_commands.json"
    if compile_commands_file.is_file():
        load_compile_commands_json(sess, compile_commands_file, force=force)
    else:
        logger.warning("Skipping loading non-existing compile_commands: %s", compile_commands_file)


def load_mlonmcu_artifacts(sess: Session, run_dir: Path, force: bool = False, progress: bool = False):
    load_mlonmcu_exported_run(sess, run_dir, force=force, progress=progress)


def run_mlonmcu_from_initializer(
    sess: Session,
    home: Optional[Path] = None,
    workdir: Optional[Path] = None,
    until: Optional[str] = None,
    force: bool = False,
    trace_instrs: bool = False,
    progress: bool = False,
):
    logger.info("Running MLonMCU with Initializer...")
    artifacts = sess.artifacts
    initializer_artifacts = filter_artifacts(artifacts, lambda x: x.attrs.get("kind") == "mlonmcu_session_initializer")
    print("initializer_artifacts", initializer_artifacts)
    assert len(initializer_artifacts) == 1
    initializer_artifact = initializer_artifacts[0]
    print("initializer_artifact", initializer_artifact)
    initializer_file = initializer_artifact.path
    print("initializer_file", initializer_file)
    # TODO: docker mode
    # TODO: cmdline mode
    # TODO: check if mlonmcu is installed
    import mlonmcu
    import mlonmcu.context
    from mlonmcu.session.run import RunStage, RunInitializer

    if until is None:
        until = RunStage.RUN
    else:
        until = RunStage[until]

    initializers = RunInitializer.from_file(initializer_file)
    if len(initializers) != 1:
        raise ValueError(f"Expected exactly one MLonMCU run initializer, found {len(initializers)}")
    initializer = initializers[0]
    print("initializer", initializer)
    print("initializer.config", initializer.config)
    if trace_instrs:
        if "log_instrs" not in initializer.feature_names:
            initializer.feature_names.append("log_instrs")
        initializer.config["log_instrs.to_file"] = True
    print("initializer.config2", initializer.config)
    # print("initializer.runs", initializer.runs)
    # if len(initializer.runs) > 1:
    #     raise NotImplementedError("Multiple runs not supported")
    if workdir is None:
        workdir = sess.directory / "work" / "local" / "mlonmcu"
        workdir.mkdir(parents=True, exist_ok=True)
    sess_cfg = {
        "session.executor": "process_pool",
    }
    with mlonmcu.context.MlonMcuContext(home, deps_lock="read") as context:
        with context.create_session(dest=workdir, config=sess_cfg) as session:
            session.add_run(initializer, ignore_idx=True)
            assert session.process_runs(
                until=until, context=context, export=True
            ), "Error while processing MLonMCU runs"
            report = session.get_reports()
            print("report", report)
            print("report.df", report.df)
            run_dir = session.results[0].dir
            print("run_dir", run_dir)

    load_mlonmcu_exported_run(sess, run_dir, force=force, progress=progress)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    run_mlonmcu_from_initializer(
        sess,
        force=args.force,
        home=args.home,
        workdir=args.workdir,
        until=args.until,
        trace_instrs=args.trace_instrs,
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
    parser.add_argument("--home", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--until", type=str, default=None)
    parser.add_argument("--trace-instrs", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
