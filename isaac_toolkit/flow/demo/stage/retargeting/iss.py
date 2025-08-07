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

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.retargeting.iss.etiss import retarget_etiss_iss

logger = get_logger()


def retarget_iss_auto(
    sess: Session, label: str = "", workdir: Optional[str] = None, force: bool = False, progress: bool = False
):
    logger.info("Retargeting ISS with ISAAC instructions...")
    # input(">>>")
    sim = "etiss"  # TODO: read from config
    assert sim == "etiss"
    sim_retarget_iss = {
        "etiss": retarget_etiss_iss,
    }
    retarget_iss = sim_retarget_iss.get(sim)
    assert retarget_iss is not None
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
    etiss_config = demo_config.etiss
    assert etiss_config is not None
    etiss_core = etiss_config.core_name
    mount_dir = None  # Expose to CLI
    if use_docker:
        docker_image = docker_config.etiss_image
        if mount_dir is None:
            mount_dir = Path.cwd()
    else:
        raise NotImplementedError("Non-docker mode")
    retarget_iss(
        sess,
        workdir=workdir,
        mount_dir=mount_dir,
        docker_image=docker_image,
        etiss_core=etiss_core,
        label=label,
        force=force,
    )


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    retarget_iss_auto(sess, label=args.label, workdir=args.workdir, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
