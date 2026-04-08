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
import time
import shutil
import subprocess
import argparse
from typing import Optional, Union, List
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session

# from isaac_toolkit.session.artifact import FileArtifact
from isaac_toolkit.session.artifact import MetricsArtifact
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.frontend.elf.riscv import load_elf
from isaac_toolkit.frontend.linker_map import load_linker_map
from isaac_toolkit.frontend.disass.objdump import load_disass
from isaac_toolkit.frontend.compile_commands.json import load_compile_commands_json
from .builder import ISAACBuilder

logger = get_logger()


class MLonMCUBuilder(ISAACBuilder):
    def __init__(
        self,
        target: str,
        toolchain: str = "gcc",
        platform: str = "mlif",
        optimize: Optional[str] = None,
    ):
        super().__init__()
        self.common_args = []
        self.common_args += ["--target", target]
        if toolchain is not None:
            assert platform in ["mlif"]
            self.common_args += ["-c", f"{platform}.toolchain={toolchain}"]
        if optimize is not None:
            assert platform in ["mlif"]
            self.common_args += ["-c", f"{platform}.optimize={optimize}"]

    def parse_mlonmcu_metrics(self, dest_dir, until: str = "compile") -> dict:
        # TODO: measure simulation time with subprocess?
        metrics = {"until": until}
        dest_dir = Path(dest_dir)
        assert dest_dir.is_dir()
        report_file = dest_dir / "report.csv"
        assert report_file.is_file()
        report_df = pd.read_csv(report_file)
        assert len(report_df) == 1
        # print("report_df", report_df, report_df.columns)
        report_row = report_df.iloc[0]
        # print("report_row", report_row)
        simulator = report_row["Target"]
        rom_total = report_row["Total ROM"]
        rom_code = report_row["ROM code"]
        rom_rodata = report_row["ROM read-only"]
        rom_misc = report_row["ROM misc"]
        ram_total = report_row["Total RAM"]
        ram_data = report_row["RAM data"]
        ram_bss = report_row["RAM zero-init data"]
        metrics.update(
            {
                "simulator": simulator,
                "rom_total": int(rom_total),
                "rom_code": int(rom_code),
                "rom_rodata": int(rom_rodata),
                "rom_misc": int(rom_misc),
                "ram_total": int(ram_total),
                "ram_data": int(ram_data),
                "ram_bss": int(ram_bss),
            }
        )
        # ram_heap = report_row["RAM heap"]
        # ram_heap = report_row["RAM stack"]
        return metrics

    def build(
        self,
        benchmark: str,
        dest_dir: Union[str, Path],
        until: str = "compile",
        label: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> dict:
        assert dest_dir is not None
        # TODO: MLONMCU_HOME
        args = ["python3", "-m", "mlonmcu.cli.main", "flow", until, benchmark, "--dest", dest_dir]
        args += self.common_args
        if label is not None:
            args += ["--label", label]
        if extra_args:
            args += extra_args
        if verbose:
            command = " ".join(map(str, args))
            logger.info("Executing: %s", command)

        t0 = time.time()
        if verbose:
            subprocess.run(args, check=True)
        else:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t1 = time.time()
        time_s = t1 - t0
        compile_metrics = {"time_s": time_s}
        compile_metrics_ = self.parse_mlonmcu_metrics(dest_dir, until=until)
        compile_metrics.update(compile_metrics_)
        if compile_metrics:
            self.add_metrics(compile_metrics)
        return compile_metrics


def load_build_artifacts(
    sess: Session,
    dest_dir: Union[str, Path],
    program: str,
    force: bool = False,
    # progress: bool = False,
):
    # print("load_build_artifacts", dest_dir)
    dest_dir = Path(dest_dir)
    assert dest_dir.is_dir(), f"Missing: {dest_dir}"
    runs_dir = dest_dir / "runs"
    assert runs_dir.is_dir(), f"Missing: {runs_dir}"
    run_idx = 0
    run_dir = runs_dir / str(run_idx)
    assert run_dir.is_dir(), f"Missing: {run_dir}"

    elf_file = run_dir / "generic_mlonmcu"
    linker_map_file = run_dir / "generic_mlonmcu.map"
    disass_file = run_dir / "generic_mlonmcu.dump"
    compile_commands_file = run_dir / "compile_commands.json"

    if elf_file.is_file():
        load_elf(sess, elf_file, force=force)
    else:
        logger.warning("Skipping missing ELF: %s", elf_file)
    if linker_map_file:
        load_linker_map(sess, linker_map_file, force=force)
    else:
        logger.warning("Skipping missing linker map file: %s", linker_map_file)
    if disass_file:
        load_disass(sess, disass_file, force=force)
    else:
        logger.warning("Skipping missing disass file: %s", disass_file)
    if compile_commands_file:
        load_compile_commands_json(sess, compile_commands_file, force=force)
    else:
        logger.warning("Skipping missing disass file: %s", disass_file)


def load_compile_metrics(
    sess: Session,
    compile_metrics: Union[dict, List[dict]],
    program: str,
    simulator: str,
    force: bool = False,
    # progress: bool = False,
):
    # print("load_compile_metrics", compile_metrics)
    if isinstance(compile_metrics, dict):
        metrics_df = pd.DataFrame([compile_metrics])
    else:
        assert isinstance(compile_metrics, list)
        assert len(compile_metrics) > 0
        assert isinstance(compile_metrics[0], dict)
        metrics_df = pd.DataFrame(compile_metrics)

    attrs = {
        "simulator": simulator,
        "program": program,
        "kind": "compile_metrics",
        "by": __name__,
    }
    metrics_artifact = MetricsArtifact("compile_metrics", metrics_df, attrs=attrs)
    # print("metrics_artifact", metrics_artifact)
    sess.add_artifact(metrics_artifact, override=force)


def invoke_mlonmcu_builder(
    sess: Session,
    benchmark: str,
    simulator: str,
    toolchain: str = "gcc",
    optimize: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    dest_dir: Optional[str] = None,
    until: str = "compile",
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
        label = f"mlonmcu-{until}"

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
    # print("dest_dir", dest_dir)

    compile_metrics = builder.build(
        benchmark, dest_dir=dest_dir, extra_args=extra_args, until=until, label=label, verbose=verbose
    )

    if load:
        load_build_artifacts(sess, dest_dir, benchmark, force=force)
        # compile_metrics = builder.get_metrics(latest=True)
        compile_metrics["program"] = benchmark
        # if simulator is not None:
        #     compile_metrics["simulator"] = simulator
        if toolchain is not None:
            compile_metrics["toolchain"] = toolchain
        if optimize is not None:
            compile_metrics["optimize"] = optimize
        load_compile_metrics(sess, compile_metrics, program=benchmark, simulator=simulator, force=force)
    sess.save()
    if cleanup:
        logger.info("Cleaning up files...")
        shutil.rmtree(dest_dir)

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
        args.benchmark,
        args.simulator,
        toolchain=args.toolchain,
        optimize=args.optimize,
        extra_args=args.extra_args,
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
    parser.add_argument("benchmark", type=str)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, type=str)


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
