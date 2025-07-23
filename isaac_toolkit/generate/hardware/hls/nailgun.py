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
import logging
import argparse
from typing import Optional, Union, List
from pathlib import Path

from isaac_toolkit.session import Session


DEFAULT_DOCKER_IMAGE = "isax-tools-integration-env:latest"
SUPPORTED_CORES = []


logger = logging.getLogger("nailgun")


def run_nailgun_hls(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    index_files: Optional[List[Union[str, Path]]] = None,
    tools_dir: Union[str, Path] = None,
    mount_dir: Union[str, Path] = None,
    docker_image: Optional[str] = None,
    fix_permissions: bool = True,  # Only for docker mode
    # gurubi_license_file: TODO
    isaxes: Optional[List[str]] = None,
    label: Optional[str] = None,
    core_name: Optional[str] = None,
    copy_files: bool = True,
    ilp_solver: str = "gurobi",
    cell_library: Optional[str] = None,
    # cell_library = "$(pwd)/cfg/longnail/library.yaml",
    clock_ns: int = 100,
    schedule_timeout: int = 10,
    refine_timeout: int = 10,
    sched_algo_ms: bool = False,
    sched_algo_pa: bool = False,
    sched_algo_ra: bool = False,
    sched_algo_mi: bool = False,
    share_resources: bool = False,
    mlir_file: Optional[Union[str, Path]] = None,
    force: bool = False,
):
    assert workdir is not None
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert workdir.is_dir()
    if not isinstance(tools_dir, Path):
        tools_dir = Path(tools_dir)
    assert tools_dir.is_dir()
    assert core_name is not None
    assert core_name in SUPPORTED_CORES
    use_docker = docker_image is not None
    subdir = "docker" if use_docker else "local"
    base_dir = workdir / subdir
    hls_dir = base_dir / "hls"
    # syn_dir = base_dir / "syn"
    hls_dir.mkdir(exist_ok=True, parents=True)
    if isaxes is None:
        isaxes = []
    if label is None:
        label = "baseline" if len(isaxes) == 0 else ("shared" if share_resources else "default")
    output_dir = hls_dir / label / "output"
    if output_dir.is_dir():
        assert force, f"Directory already exists: {output_dir}. Use --force or different --label."
        logger.info("Cleaning up old output dir: %s (--force)", output_dir)
        shutil.rmtree(output_dir)
    else:
        output_dir.mkdir(parents=True)
    nailgun_env = {
        "OUTPUT_PATH": output_dir,
        "CONFIG_PATH": output_dir / ".config",
        "SIM_EN": "n",
        "SKIP_AWESOME_LLVM": "y",
        "CORE": core_name,
    }
    use_ol2 = False
    if use_ol2:
        assert mount_dir is not None
        assert Path(mount_dir).is_dir()
        ol2_config_template = mount_dir / "/cfg/openlane/minimal_config_fast.json"
        ol2_until_step = "OpenROAD.Floorplan"
        ol2_target_freq = 20
        ol2_target_util = 20
        nailgun_env.update(
            {
                "OL2_ENABLE": "y",
                "OL2_CONFIG_TEMPLATE": ol2_config_template,
                "OL2_UNTIL_STEP": ol2_until_step,
                "OL2_TARGET_FREQ": ol2_target_freq,
                "OL2_TARGET_UTIL": ol2_target_util,
            }
        )

    def bool_helper(x):
        assert isinstance(x, bool)
        return "y" if x else "n"

    if share_resources:
        assert mlir_file is not None
        assert Path(mlir_file).is_file()
        # TODO: sed -e "s/lil.enc_immediates/lil.sharing_group = 1, lil.enc_immediates/g" $WORK/docker/hls/default/output/mlir/ISAX_ISAAC_EN.mlir > $WORK/docker/hls/ISAX_ISAAC_EN_shared.mlir
        nailgun_env["MLIR_ENTRY_POINT_PATH"] = mlir_file

    elif len(isaxes) == 0:
        nailgun_env["NO_ISAX"] = "y"
    else:
        nailgun_env.update(
            {
                "ISAXES": ",".join(map(lambda x: x.upper(), isaxes)),
            }
        )
    if len(isaxes) > 0:
        nailgun_env.update(
            {
                "LN_ILP_SOLVER": ilp_solver.upper(),
                "CLOCK_TIME": clock_ns,
                "SCHEDULE_TIMEOUT": schedule_timeout,
                "REFINE_TIMEOUT": refine_timeout,
                "SCHED_ALGO_MS": bool_helper(sched_algo_ms),
                "SCHED_ALGO_PA": bool_helper(sched_algo_pa),
                "SCHED_ALGO_RA": bool_helper(sched_algo_ra),
                "SCHED_ALGO_MI": bool_helper(sched_algo_mi),
            }
        )
        if cell_library is not None:
            nailgun_env["CELL_LIBRARY"] = cell_library
    nailgun_command = "make gen_config ci"
    if use_docker:
        command = f"docker run -it --rm -v {tools_dir}:/isax_tools"
        if mount_dir is not None:
            command += f" -v {mount_dir}:{mount_dir}"
        command += f" {docker_image}"
        prepare = "cd /isax-tools/nailgun"
        nailgun_command_env = " ".join([f"{key}={val}" for key, val in nailgun_env.items()])
        command += f' "{prepare} && {nailgun_command_env} {nailgun_command}"'
        print("command", command)
        subprocess.run(command, shell=True)
    else:
        env = os.environ.copy()
        env.update(nailgun_env)
        command = nailgun_command
        cwd = tools_dir
        print("command", command)
        subprocess.run(command, cwd=cwd, env=env, shell=True)
    if copy_files:
        raise NotImplementedError


def handle(args):
    # assert args.session is not None
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    run_nailgun_hls(
        sess,
        force=args.force,
        workdir=args.workdir,
        tools_dir=args.tools_dir,
        docker_image=args.docker,
        isaxes=args.isaxes.split(";") if args.isaxes is not None else None,
        index_files=args.index.split(";") if args.index is not None else None,
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
    parser.add_argument("--tools-dir", type=str, required=True)  # TODO: define via settings?
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--isaxes", type=str, default=None)
    # parser.add_argument("--index", type=str, default=None)
    # gurubi_license_file: TODO
    # isaxes: Optional[List[str]] = None,
    # label: Optional[str] = None,
    # core_name: Optional[str] = None,
    # copy_files: bool = True,
    # ilp_solver: str = "gurobi",
    # cell_library: Optional[str] = None,
    # # cell_library = "$(pwd)/cfg/longnail/library.yaml",
    # clock_ns: int = 100,
    # schedule_timeout: int = 10,
    # refine_timeout: int = 10,
    # sched_algo_ms: bool = False,
    # sched_algo_pa: bool = False,
    # sched_algo_ra: bool = False,
    # sched_algo_mi: bool = False,
    # share_resources: bool = False,
    # mlir_file: Optional[Union[str, Path]] = None,

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
