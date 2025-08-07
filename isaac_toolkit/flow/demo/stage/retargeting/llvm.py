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
from typing import Optional, List
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.retargeting.llvm.seal5 import retarget_seal5_llvm

logger = get_logger()


def retarget_llvm_auto(
    sess: Session,
    cfg_files: List[str],
    label: str = "",
    workdir: Optional[str] = None,
    splitted: bool = False,
    force: bool = False,
    progress: bool = False,
):
    logger.info("Retargeting LLVM with ISAAC instructions...")
    cfg_files = [Path(f) for f in cfg_files]
    assert all(f.is_file() for f in cfg_files)
    # input(">>>")
    tool = "seal5"  # TODO: read from config
    assert tool == "seal5"
    tool_retarget_llvm = {
        "seal5": retarget_seal5_llvm,
    }
    retarget_llvm = tool_retarget_llvm.get(tool)
    assert retarget_llvm is not None
    if workdir is None:
        workdir = sess.directory / "work"  # expose via cmd
    else:
        workdir = Path(workdir)
        assert workdir.is_dir()
    config = sess.config
    flow_config = config.flow
    assert flow_config is not None
    demo_config = flow_config.demo
    assert demo_config is not None
    docker_config = demo_config.docker
    assert docker_config is not None
    use_docker = docker_config.enable
    riscv_config = demo_config.riscv
    assert riscv_config is not None
    coredsl_config = demo_config.coredsl
    assert coredsl_config is not None
    mount_dir = None  # Expose to CLI
    set_name = coredsl_config.set_name
    xlen = riscv_config.xlen
    seal5_sets = []
    if splitted:
        raise NotImplementedError("Splitted")
    else:
        seal5_sets.append(set_name)
    if use_docker:
        docker_image = docker_config.seal5_image
        if mount_dir is None:
            mount_dir = Path.cwd()
    else:
        raise NotImplementedError("Non-docker mode")
    retarget_llvm(
        sess,
        workdir=workdir,
        mount_dir=mount_dir,
        docker_image=docker_image,
        seal5_sets=seal5_sets,
        xlen=xlen,
        cfg_files=cfg_files,
        label=label,
        force=force,
    )


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    retarget_llvm_auto(
        sess, args.cfg_file, label=args.label, workdir=args.workdir, splitted=args.splitted, force=args.force
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", nargs="+")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--splitted", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
