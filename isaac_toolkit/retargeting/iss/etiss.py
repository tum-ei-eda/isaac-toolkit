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
import os
import sys
import shutil
import subprocess

# import yaml
import argparse
from typing import Optional, Union
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()

DEFAULT_DOCKER_IMAGE = "isaac-quickstart-etiss:latest"


def retarget_etiss_iss(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    mount_dir: Optional[Union[str, Path]] = None,
    docker_image: Optional[str] = None,
    etiss_core: Optional[str] = None,
    label: Optional[str] = None,
    force: bool = False,
    verbose: bool = False,
    cleanup: bool = False,
):
    assert workdir is not None
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert workdir.is_dir()
    if etiss_core is None:
        etiss_core = "XIsaacCore"
    if label is None:
        label = "default"
    use_docker = docker_image is not None
    subdir = "docker" if use_docker else "local"
    base_dir = workdir / subdir
    # etiss_dir = base_dir / "etiss"
    # output_dir = etiss_dir / label
    output_dir = (base_dir / "etiss") if label == "" else (base_dir / f"etiss_{label}")
    if output_dir.is_dir():
        assert (
            force
        ), f"Directory already exists: {output_dir}. Use --force or different --label."
        logger.info("Cleaning up old output dir: %s (--force)", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    # gen_dir = workdir / "gen" / label
    gen_dir = (workdir / "gen") if label == "" else (workdir / f"gen_{label}")
    top_file = gen_dir / f"{etiss_core}.core_desc"
    kwargs = {}
    # print("verbose", verbose)
    if not verbose:
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
        # kwargs.setdefault("text", True)
    if use_docker:
        command = "docker run -it --rm"
        if mount_dir is not None:
            command += f" -v {mount_dir}:{mount_dir}"
        command += f" -e CLEANUP={int(cleanup)}"
        command += f" {docker_image}"
        command += f" {output_dir}"
        command += f" {top_file}"
        # print("$$$", command)

        # input("!")
        try:
            # print("command", command)
            # input(">")
            subprocess.run(command, check=True, shell=True, **kwargs)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed with return code {e.returncode}")
            if e.stdout:
                print("--- STDOUT ---")
                print(e.stdout.decode())
            if e.stderr:
                print("--- STDERR ---")
                print(e.stderr.decode())
            raise  # Re-raise if you want the caller to handle it too
    else:

        temp_dir = base_dir / "temp"
        etiss_home = temp_dir / "etiss"
        env = os.environ.copy()
        env["ETISS_HOME"] = etiss_home
        env["CLEANUP"] = str(int(cleanup))
        # ccache already exported implicitly?
        # TODO: explicit ccache?
        # env["CCACHE"]
        # env["CCACHE_DIR"]
        etiss_script = os.environ.get("ETISS_SCRIPT_LOCAL", None)
        assert etiss_script is not None, "ETISS_SCRIPT_LOCAL undefined"
        assert Path(etiss_script).is_file(), f"Not found: {etiss_script}"
        # TODO: ship with isaac?
        etiss_script_args = [output_dir, top_file]
        try:
            subprocess.run([etiss_script, *etiss_script_args], check=True, **kwargs, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed with return code {e.returncode}")
            if e.stdout:
                print("--- STDOUT ---")
                print(e.stdout.decode())
            if e.stderr:
                print("--- STDERR ---")
                print(e.stderr.decode())
            raise  # Re-raise if you want the caller to handle it too


def handle(args):
    # assert args.session is not None
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    retarget_etiss_iss(
        sess,
        force=args.force,
        workdir=args.workdir,
        docker_image=args.docker,
        verbose=args.verbose,
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
    parser.add_argument(
        "--docker", type=str, default=None, const=DEFAULT_DOCKER_IMAGE, nargs="?"
    )
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    # label: Optional[str] = None,
    # etiss_core: Optional[str] = None,

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
