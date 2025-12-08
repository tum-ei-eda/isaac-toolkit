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
    name: Optional[str] = None,
    label: Optional[str] = None,
    cfg_files: Optional[List[Union[str, Path]]] = None,
    splitted: bool = False,
    xlen: Optional[int] = 32,
    force: bool = False,
    verbose: bool = False,
    cleanup: bool = False,
    progress: bool = False,
):
    logger.info("Retargeting Seal5 LLVM...")
    assert xlen == 32
    assert workdir is not None
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert workdir.is_dir()
    if seal5_sets is None:
        seal5_sets = ["XIsaac"]
    print(f"label '{label}'")
    if label is None:
        # label = "splitted" if splitted else ""
        label = ""
    if name is None:
        name = "seal5_splitted" if splitted else "seal5"
    assert cfg_files is not None
    assert len(cfg_files) > 0
    use_docker = docker_image is not None
    subdir = "docker" if use_docker else "local"
    base_dir = workdir / subdir
    # seal5_dir = base_dir / "seal5"
    # output_dir = seal5_dir / label
    output_dir = (base_dir / name) if label == "" else (base_dir / f"{name}_{label}")
    if output_dir.is_dir():
        assert force, f"Directory already exists: {output_dir}. Use --force or different --label."
        logger.info("Cleaning up old output dir: %s (--force)", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    gen_dir = workdir / "gen"
    # gen_dir = workdir / "gen" / label
    gen_dir = (workdir / "gen") if label == "" else (workdir / f"gen_{label}")
    cdsl_files = [
        (gen_dir / f"{set_name}.splitted.core_desc" if splitted else gen_dir / f"{set_name}.core_desc")
        for set_name in seal5_sets
    ]
    print("cdsl_files", cdsl_files)
    print("cfg_files", cfg_files)
    # input("?")
    # TODO: PROGRESS
    if use_docker:
        command = "docker run -i --rm"
        if mount_dir is not None:
            command += f" -v {mount_dir}:{mount_dir}"
        command += f" -e CLEANUP={int(cleanup)}"
        command += f" {docker_image}"
        command += f" {output_dir}"
        command += " "
        command += " ".join(map(lambda x: str(Path(x).resolve()), cdsl_files))
        command += " "
        command += " ".join(map(lambda x: str(Path(x).resolve()), cfg_files))

        # print("$$$", command)
        kwargs = {}
        # print("verbose", verbose)
        if not verbose:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
            kwargs.setdefault("text", True)
        try:
            subprocess.run(command, check=True, shell=True, **kwargs)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed with return code {e.returncode}")
            if e.stdout:
                print("--- STDOUT ---")
                print(e.stdout)
            if e.stderr:
                print("--- STDERR ---")
                print(e.stderr)
            raise  # Re-raise if you want the caller to handle it too
    else:
        seal5_script_local = os.environ.get("SEAL5_SCRIPT_LOCAL", None)
        assert seal5_script_local is not None, "Undefined: SEAL5_SCRIPT_LOCAL"
        args = [seal5_script_local]
        env = os.environ.copy()
        env["CLEANUP"] = str(int(cleanup))
        env["MGCLIENT_ROOT"] = env["MGCLIENT_INSTALL_DIR"]
        enable_cdfg_pass = True
        env["ENABLE_CDFG_PASS"] = str(int(enable_cdfg_pass))
        temp_dir = base_dir / "temp"
        temp_seal5_home = temp_dir / "seal5_llvm"
        env["SEAL5_HOME"] = temp_seal5_home
        ccache = True  # TODO: expose
        env["CCACHE"] = str(int(ccache))
        if ccache:
            ccache_dir = env.get("CCACHE_DIR", None)
            assert ccache_dir is not None, "Undefined: CCACHE_DIR"
            env["CCACHE_DIR"] = ccache_dir
        seal5_cfg_dir = env.get("SEAL5_CFG_DIR", None)
        assert seal5_cfg_dir is not None, "Undefined: SEAL5_CFG_DIR"
        env["SEAL5_CFG_DIR"] = seal5_cfg_dir
        seal5_dir = env.get("SEAL5_DIR", None)
        assert seal5_dir is not None, "Undefined: SEAL5_DIR"
        env["SEAL5_DIR"] = seal5_dir
        llvm_dir = env.get("LLVM_DIR", None)
        assert llvm_dir is not None, "Undefined: LLVM_DIR"
        print("llvm_dir", llvm_dir)
        env["LLVM_REPO"] = llvm_dir
        llvm_ref = "isaacnew-base-3"  # TODO: expose
        env["LLVM_REF"] = llvm_ref
        clone_depth = 2  # TODO: expose?
        env["CLONE_DEPTH"] = str(clone_depth)
        args += [output_dir]
        args += list(map(lambda x: str(Path(x).resolve()), cdsl_files))
        args += list(map(lambda x: str(Path(x).resolve()), cfg_files))
        # print("env", env)
        # print("$$$", args)
        kwargs = {}
        # print("verbose", verbose)
        if not verbose:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
            kwargs.setdefault("text", True)
        try:
            subprocess.run(args, check=True, env=env, **kwargs)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed with return code {e.returncode}")
            if e.stdout:
                print("--- STDOUT ---")
                print(e.stdout)
            if e.stderr:
                print("--- STDERR ---")
                print(e.stderr)
            raise  # Re-raise if you want the caller to handle it too


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
    parser.add_argument("--docker", type=str, default=None, const=DEFAULT_DOCKER_IMAGE, nargs="?")
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
