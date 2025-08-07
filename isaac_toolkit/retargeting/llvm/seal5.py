#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
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
# # import os
import sys
import shutil
import subprocess

# import yaml
import argparse
from typing import Optional, Union, List
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()

DEFAULT_DOCKER_IMAGE = "isaac-quickstart-seal5:latest"


def retarget_seal5_llvm(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    mount_dir: Optional[Union[str, Path]] = None,
    docker_image: Optional[str] = None,
    seal5_sets: Optional[List[str]] = None,
    label: Optional[str] = None,
    cfg_files: Optional[List[Union[str, Path]]] = None,
    splitted: bool = False,
    xlen: Optional[int] = 32,
    force: bool = False,
):
    logger.info("Retargeting Seal5 LLVM...")
    assert xlen == 32
    assert workdir is not None
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert workdir.is_dir()
    if seal5_sets is None:
        seal5_sets = ["XIsaac"]
    if label is None:
        label = "default"
    assert cfg_files is not None
    assert len(cfg_files) > 0
    use_docker = docker_image is not None
    subdir = "docker" if use_docker else "local"
    base_dir = workdir / subdir
    seal5_dir = base_dir / "seal5"
    output_dir = seal5_dir / label
    if output_dir.is_dir():
        assert force, f"Directory already exists: {output_dir}. Use --force or different --label."
        logger.info("Cleaning up old output dir: %s (--force)", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    gen_dir = workdir / "gen" / label
    cdsl_files = [
        gen_dir / f"{set_name}.splitted.core_desl" if splitted else gen_dir / f"{set_name}.core_desc"
        for set_name in seal5_sets
    ]
    # print("cdsl_files", cdsl_files)
    # input("?")
    if use_docker:
        command = "docker run -it --rm"
        if mount_dir is not None:
            command += f" -v {mount_dir}:{mount_dir}"
        command += f" {docker_image}"
        command += f" {output_dir}"
        command += " "
        command += " ".join(map(lambda x: str(Path(x).resolve()), cdsl_files))
        command += " "
        command += " ".join(map(lambda x: str(Path(x).resolve()), cfg_files))

        print("$$$", command)
        subprocess.run(command, check=True, shell=True)
    else:
        raise NotImplementedError


def handle(args):
    # assert args.session is not None
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    retarget_seal5_llvm(
        sess,
        force=args.force,
        workdir=args.workdir,
        docker_image=args.docker,
    )
    if sess is not None:
        sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    # parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--session", "--sess", "-s", type=str, required=False)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--docker", type=str, default=None, const=DEFAULT_DOCKER_IMAGE, nargs="?")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--dse", action="store_true")
    # label: Optional[str] = None,
    # etiss_core: Optional[str] = None,

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
