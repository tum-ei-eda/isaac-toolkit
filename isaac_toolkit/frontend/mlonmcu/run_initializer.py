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
"""Run MLonMCU from a run initializer and import its artifacts."""

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from isaac_toolkit.frontend.compile_commands.json import load_compile_commands_json
from isaac_toolkit.frontend.disass.objdump import load_disass
from isaac_toolkit.frontend.elf.riscv import load_elf
from isaac_toolkit.frontend.instr_trace.etiss_new import load_instr_trace as load_etiss_instr_trace
from isaac_toolkit.frontend.linker_map import load_linker_map
from isaac_toolkit.frontend.mem_trace.etiss import load_mem_trace
from isaac_toolkit.frontend.perf_trace.etiss_perf import load_perf_trace
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.session import Session

logger = get_logger()


def _load_initializer(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"MLonMCU run initializer not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    runs = data.get("runs", []) if isinstance(data, dict) else []
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one run in initializer, found {len(runs)}")
    return data


def _enable_tracing(data: dict) -> dict:
    """Return a traced copy of an MLonMCU run initializer."""
    data = copy.deepcopy(data)
    run = data["runs"][0]
    target = run.get("target_name")
    features = run.setdefault("feature_names", [])
    config = run.setdefault("config", {})

    if target and target.startswith("etiss"):
        for feature in ("log_instrs", "trace"):
            if feature not in features:
                features.append(feature)
        config["log_instrs.to_file"] = True
        config["etiss.experimental_print_to_file"] = True
    else:
        raise ValueError(f"--trace is not implemented for MLonMCU target '{target}'")
    return data


def _resolve_model_paths(data: dict, base_dir: Path) -> dict:
    """Return a copy with filesystem-backed model names made absolute."""
    data = copy.deepcopy(data)
    base_dir = Path(base_dir).resolve()
    for run in data.get("runs", []):
        model_name = run.get("model_name")
        if not isinstance(model_name, str):
            continue
        model_path = Path(model_name).expanduser()
        if model_path.is_absolute():
            run["model_name"] = str(model_path.resolve())
            continue
        candidate = base_dir / model_path
        if candidate.exists():
            run["model_name"] = str(candidate.resolve())
    return data


def _find_run_directory(output_dir: Path) -> Path:
    candidates = sorted(
        (path for path in (output_dir / "runs").glob("*") if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one MLonMCU run directory in {output_dir}, found {len(candidates)}")
    return candidates[0]


def _required_file(run_dir: Path, *names: str) -> Path:
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    raise RuntimeError(f"MLonMCU did not produce required artifact: {' or '.join(names)}")


def _import_static_artifacts(sess: Session, run_dir: Path, force: bool) -> None:
    load_elf(sess, _required_file(run_dir, "generic_mlonmcu"), force=force)
    load_disass(sess, _required_file(run_dir, "generic_mlonmcu.dump"), force=force)
    load_compile_commands_json(sess, _required_file(run_dir, "compile_commands.json"), force=force)
    load_linker_map(sess, _required_file(run_dir, "generic_mlonmcu.map"), force=force)


def _import_trace_artifacts(sess: Session, run_dir: Path, target: Optional[str], force: bool) -> None:
    if target and target.startswith("etiss"):
        instr_files = sorted(run_dir.glob("instr_trace*.csv"))
        if not instr_files:
            instr_files = sorted(run_dir.glob("asm_trace_*.txt"))
        if not instr_files:
            instr_files = sorted(run_dir.glob("*_instrs.log"))
        if instr_files:
            load_etiss_instr_trace(sess, instr_files, force=force)
        else:
            logger.warning("MLonMCU produced no ETISS instruction trace")

        mem_file = run_dir / "dBusAccess.csv"
        if not mem_file.is_file():
            exported_mem_files = sorted(run_dir.glob("*_mem.log"))
            mem_file = exported_mem_files[0] if exported_mem_files else mem_file
        if mem_file.is_file():
            load_mem_trace(sess, mem_file, force=force)
        else:
            logger.warning("MLonMCU produced no ETISS memory trace")

        perf_files = sorted(run_dir.glob("perf_trace*.csv"))
        if perf_files:
            load_perf_trace(sess, perf_files, force=force)


def run_mlonmcu_initializer(
    sess: Session,
    initializer_file: Path,
    trace: bool = False,
    force: bool = False,
    mlonmcu_home: Optional[Path] = None,
) -> None:
    """Execute a single-run MLonMCU initializer and import its outputs."""
    initializer_file = Path(initializer_file).resolve()
    data = _load_initializer(initializer_file)
    data = _resolve_model_paths(data, Path.cwd())
    if trace:
        data = _enable_tracing(data)
    target = data["runs"][0].get("target_name")

    with tempfile.TemporaryDirectory(prefix="isaac-mlonmcu-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        effective_initializer = temp_dir / "initializer.yml"
        with open(effective_initializer, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        output_dir = temp_dir / "session"
        command = [
            sys.executable,
            "-m",
            "mlonmcu.cli.main",
            "flow",
            "run",
            "--initializer",
            str(effective_initializer),
            "--dest",
            str(output_dir),
        ]
        if mlonmcu_home is not None:
            command.extend(["--home", str(Path(mlonmcu_home).resolve())])
        logger.info("Running MLonMCU initializer: %s", initializer_file)
        subprocess.run(command, check=True, cwd=Path.cwd())

        run_dir = _find_run_directory(output_dir)
        _import_static_artifacts(sess, run_dir, force=force)
        if trace:
            _import_trace_artifacts(sess, run_dir, target, force=force)
        # File-backed artifacts must be copied before the temporary MLonMCU
        # session is removed.
        sess.save()


def handle(args) -> None:
    session_dir = Path(args.session).resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"ISAAC Toolkit session does not exist: {session_dir}")
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    run_mlonmcu_initializer(
        sess,
        Path(args.file),
        trace=args.trace,
        force=args.force,
        mlonmcu_home=Path(args.mlonmcu_home) if args.mlonmcu_home else None,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="MLonMCU run initializer YAML")
    parser.add_argument("--session", "--sess", "-s", required=True, help="ISAAC Toolkit session directory")
    parser.add_argument("--trace", action="store_true", help="Enable and import simulator tracing")
    parser.add_argument("--mlonmcu-home", help="Override the MLONMCU_HOME environment")
    parser.add_argument("--force", "-f", action="store_true", help="Replace existing artifacts")
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return parser


def main(argv=None) -> None:
    args = get_parser().parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main()
