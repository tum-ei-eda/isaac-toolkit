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
import logging
import argparse
from typing import Optional, Union
from pathlib import Path

from isaac_toolkit.session import Session


DEFAULT_CORE = "CVA5"
SUPPORTED_CORES = ["CVA5", "VEX"]

DEFAULT_PDK = "nangate45"
SUPPORTED_PDKS = ["nangate45"]

DEFAULT_CLK_PERIOD = 50.0
PER_CORE_CLK_PERIOD = {"CVA5": 50.0}


logger = logging.getLogger("synopsys")


def run_synopsys_asip_syn(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    label: Optional[str] = None,
    core_name: Optional[str] = None,
    pdk: Optional[str] = None,
    clk_period: Optional[float] = None,
    script_path: Optional[str, Path] = None,
    constraints_file: Optional[str, Path] = None,
    dse: bool = False,
    force: bool = False,
):
    if dse:
        raise NotImplementedError
    assert workdir is not None
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert workdir.is_dir()
    if core_name is None:
        core_name = DEFAULT_CORE
    assert core_name in SUPPORTED_CORES
    assert label is not None
    assert script_path is not None
    if pdk is None:
        pdk = DEFAULT_PDK
    assert pdk in SUPPORTED_PDKS
    if clk_period is None:
        clk_period = PER_CORE_CLK_PERIOD.get(core_name, DEFAULT_CLK_PERIOD)
    assert constraints_file is not None
    assert Path(constraints_file).is_file()

    possible_subdirs = ["docker", "local"]
    for possible_subdir in possible_subdirs:
        base_dir = workdir / possible_subdir
        hls_dir = base_dir / "hls" / label
        if hls_dir.is_dir():
            break
    assert hls_dir.is_dir()
    output_dir = base_dir / "asip_syn" / label
    if output_dir.is_dir():
        assert (
            force
        ), f"Directory already exists: {output_dir}. Use --force or different --label."
        logger.info("Cleaning up old output dir: %s (--force)", output_dir)
        shutil.rmtree(output_dir)
    else:
        output_dir.mkdir(parents=True)
    hls_rtl_dir = hls_dir / "rtl"
    rtl_dir = output_dir / "rtl"
    shutil.copytree(hls_rtl_dir, rtl_dir)
    script_args = [
        script_path,
        output_dir,
        rtl_dir,
        core_name,
        pdk,
        clk_period,
        constraints_file,
    ]
    subprocess.run(script_args, check=True)


def handle(args):
    # assert args.session is not None
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    run_synopsys_asip_syn(
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
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--dse", action="store_true")
    # label: Optional[str] = None,
    # core_name: Optional[str] = None,
    # pdk: Optional[str] = None,
    # clk_period: Optional[float] = None,
    # script_path: Optional[str, Path] = None,
    # constraints_file: Optional[str, Path] = None,

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
