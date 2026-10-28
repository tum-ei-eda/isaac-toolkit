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

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.retargeting.iss.etiss import retarget_etiss_iss
from isaac_toolkit.retargeting.iss.spike import DEFAULT_DOCKER_IMAGE as DEFAULT_SPIKE_DOCKER_IMAGE
from isaac_toolkit.retargeting.iss.spike import retarget_spike_iss

logger = get_logger()


def retarget_iss_auto(
    sess: Session,
    label: str = "",
    workdir: Optional[str] = None,
    force: bool = False,
    progress: bool = False,
    docker_override: Optional[bool] = None,
    service_override: Optional[bool] = None,
    verbose: bool = False,  # TODO: add to other stages, too
    cleanup: bool = False,
    iss: Optional[str] = None,
):
    logger.info("Retargeting ISS with ISAAC instructions...")
    # input(">>>")
    config = sess.config
    flow_config = config.flow
    sim = "etiss"
    if flow_config:
        demo_config = flow_config.demo
        if demo_config:
            sim_config = demo_config.sim
            if sim_config:
                sim_ = sim_config.iss
                if sim_:
                    sim = sim_
    sim = iss or sim
    if sim.startswith("spike"):
        if workdir is None:
            workdir = sess.directory / "work"
        else:
            workdir = Path(workdir)
            assert workdir.is_dir()
        docker_config = demo_config.docker
        service_config = demo_config.service
        assert docker_config is not None and service_config is not None
        # Spike retargeting has historically compiled locally against the active
        # MLonMCU installation. Only an explicit --docker opts into a container.
        use_docker = bool(docker_override)
        use_service = service_override if service_override is not None else service_config.enable
        if use_docker and use_service:
            raise ValueError("Spike retargeting cannot use Docker and the service at the same time")
        service_addr = None
        service_tag = None
        if use_service:
            service_addr = f"{service_config.hostname}:{service_config.port}"
            service_tag = (service_config.tags or {}).get("spike")
            if service_tag is None:
                raise ValueError("demo.service.tags.spike must be set when Spike service retargeting is enabled")
        image = getattr(docker_config, "spike_image", None) or DEFAULT_SPIKE_DOCKER_IMAGE
        if not use_docker:
            image = None
        return retarget_spike_iss(
            sess, workdir=workdir, label=label, force=force, verbose=verbose,
            spike_core=demo_config.etiss.core_name, docker_image=image,
            service_addr=service_addr, service_tag=service_tag,
        )
    if sim != "etiss":
        raise ValueError(f"Unsupported ISS: {sim}")
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
    service_config = demo_config.service
    assert service_config is not None
    use_docker = docker_override if docker_override is not None else docker_config.enable
    use_service = service_override if service_override is not None else service_config.enable
    etiss_config = demo_config.etiss
    assert etiss_config is not None
    etiss_core = etiss_config.core_name
    mount_dir = None  # Expose to CLI
    if use_docker:
        assert not use_service
        docker_image = docker_config.etiss_image
        if mount_dir is None:
            mount_dir = Path.cwd()
    else:
        docker_image = None
    if use_service:
        assert not use_docker
        service_host = service_config.hostname
        service_port = service_config.port
        service_tags = service_config.tags
        if service_tags is not None:
            service_tag = service_tags.get(sim)
        else:
            service_tag = None
        service_addr = f"{service_host}:{service_port}"
        assert service_tag is not None
    else:
        service_addr = None
        service_tag = None
    # else:
    #     raise NotImplementedError("Non-docker mode")
    retarget_iss(
        sess,
        workdir=workdir,
        mount_dir=mount_dir,
        docker_image=docker_image,
        service_addr=service_addr,
        service_tag=service_tag,
        etiss_core=etiss_core,
        label=label,
        force=force,
        verbose=verbose,
        cleanup=cleanup,
    )


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    docker_override = None
    service_override = False
    if args.docker:
        assert not (args.local or args.service)
        docker_override = True
    if args.local:
        assert not (args.docker or args.service)
        docker_override = False
    if args.service:
        assert not (args.local or args.docker)
        docker_override = False
        service_override = True
    # TODO: expsoe cleanup via config?
    retarget_iss_auto(
        sess,
        label=args.label,
        workdir=args.workdir,
        force=args.force,
        docker_override=docker_override,
        service_override=service_override,
        verbose=args.verbose,
        cleanup=args.cleanup,
        iss=args.iss,
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
    parser.add_argument("--label", default="")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--service", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--iss", choices=["etiss", "spike"])
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
