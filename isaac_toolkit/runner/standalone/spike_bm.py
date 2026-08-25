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
from typing import Optional, Union, List
from pathlib import Path

from isaac_toolkit.session import Session

# from isaac_toolkit.cli.utils import parse_override_args
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts

# from isaac_toolkit.session.artifact import FileArtifact
from isaac_toolkit.logging import get_logger

from isaac_toolkit.runner.standalone.riscv import RISCVStandaloneRunner, add_riscv_args, parse_riscv_args
from isaac_toolkit.runner.standalone.common import load_trace_artifacts, load_sim_metrics

# from .cli import add_common_args, add_prog_args

logger = get_logger()


class SpikeBMStandaloneRunner(RISCVStandaloneRunner):

    def __init__(
        self,
        make_dir: Union[str, Path],
        name: str = "spike_bm",
        isa: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(make_dir, name, **kwargs)
        if isa is not None:
            self.defaults["SPIKE_ISA"] = isa

    def parse_sim_metrics(self, dest_dir, make_target: str = "run") -> dict:
        # TODO: measure simulation time with subprocess?
        metrics = {"make_target": make_target}
        dest_dir = Path(dest_dir)
        assert dest_dir.is_dir()
        out_dir = dest_dir / "out"
        assert out_dir.is_dir()
        outputs_file = out_dir / f"{self.simulator}_out.log"
        assert outputs_file.is_file()
        # TODO: use_stats_file?
        # with open(outputs_file, "r") as f:
        #     out = f.read()
        # print("out", out)
        # TODO: find a way to get metrics with spike_bm?
        # sim_insns_match = re.search(r"(\d+) instructions", out)
        # if sim_insns_match:
        #     sim_insns_str = sim_insns_match.group(1)
        #     sim_insns = int(float(sim_insns_str))
        #     metrics["instructions"] = sim_insns
        #     metrics["cycles"] = sim_insns
        # mips_match = re.search(r"MIPS \(estimated\): (.*)", out)
        # if mips_match:
        #     mips_str = mips_match.group(1)
        #     mips = float(mips_str)
        #     metrics["mips"] = mips
        cpi = 1.0
        metrics["cpi"] = cpi
        return metrics

    def run(
        self,
        dest_dir: Union[str, Path],
        elf_file: Optional[Union[str, Path]] = None,
        make_target: str = "run",
        extra_args: Optional[List[str]] = None,
        overrides: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        sim_metrics = super().run(
            dest_dir,
            elf_file=elf_file,
            make_target=make_target,
            extra_args=extra_args,
            overrides=overrides,
            verbose=verbose,
        )
        if make_target == "run":
            sim_metrics_ = self.parse_sim_metrics(dest_dir, make_target=make_target)
            sim_metrics.update(sim_metrics_)
        if sim_metrics:
            self.add_metrics(sim_metrics)
        # TODO: store output for trace too to check trace mips
        # TODO: also add build metrics (code size?)
        return sim_metrics


def invoke_spike_bm_runner(
    sess: Session,
    make_dir: Union[str, Path],
    program: str = "unknown",
    isa: Optional[str] = None,
    overrides: Optional[dict] = None,
    dest_dir: Optional[str] = None,
    label: Optional[str] = None,
    verbose: bool = False,
    force: bool = False,
    resume: bool = False,
    trace: bool = False,
    load: bool = False,
    cleanup: bool = False,
    make_target: Optional[str] = None,
):
    # TODO: docker support
    # TODO: allow debug?
    logger.info("Running SpikeBM program...")

    if make_target is None:
        make_target = "trace" if trace else "run"

    runner = SpikeBMStandaloneRunner(
        make_dir,
        isa=isa,
    )

    if label is None:
        # TODO: add timestamp?
        label = f"{program}-spike_bm-{make_target}"

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

    if resume:
        artifacts = sess.artifacts
        elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
        assert len(elf_artifacts) == 1
        elf_artifact = elf_artifacts[0]
        elf_file = elf_artifact.path
    else:
        elf_file = None
    if make_target == "run":
        sim_metrics = runner.run(dest_dir=dest_dir, elf_file=elf_file, verbose=verbose, overrides=overrides)
        if load:
            # TODO: refactor to outp frontend and parser generating sim_metrics.json?
            # sim_metrics = runner.get_metrics(latest=True)
            if isa:
                sim_metrics["isa"] = isa
            load_sim_metrics(sess, sim_metrics, program=program, simulator="spike_bm", force=force)
            # load_run_artifacts(sess, dest_dir, program, force=force)
    elif make_target == "trace":
        _ = runner.trace(dest_dir=dest_dir, elf_file=elf_file, verbose=verbose, overrides=overrides)
        if load:
            # load_run_artifacts(sess, dest_dir, program, force=force)
            load_trace_artifacts(sess, dest_dir, program=program, simulator="spike_bm", force=force)
    else:
        assert False, f"Unsupported make target: {make_target}"

    # attrs = {
    #     "by": __name__,
    #     "kind": "",
    # }
    # initializer_artifact = FileArtifact(name, input_file, attrs=attrs)
    # sess.add_artifact(initializer_artifact, override=force)

    sess.save()
    if cleanup:
        logger.info("Cleaning up files...")
        build_dir = dest_dir / "build"
        out_dir = dest_dir / "out"
        if build_dir.is_dir():
            shutil.rmtree(build_dir)
        if out_dir.is_dir():
            shutil.rmtree(out_dir)


def add_spike_bm_args(parser):
    spike_bm_group = parser.add_argument_group("spike_bm options")
    spike_bm_group.add_argument("--isa", type=str, default=None)


def parse_spike_bm_args(args):
    ret = {"isa": args.isa}
    ret.update(parse_riscv_args(args))
    return ret


def get_parser():
    from .cli import add_common_args, add_prog_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(simulator="spike_bm")
    add_riscv_args(parser)
    add_spike_bm_args(parser)
    add_prog_args(parser)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
