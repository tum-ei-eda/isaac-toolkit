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
import sys
import yaml
import logging
import argparse
import subprocess
from typing import Optional, Union
from pathlib import Path
from collections import defaultdict

import pandas as pd
from neo4j import GraphDatabase, Query
import networkx as nx
import networkx.algorithms.isomorphism as iso


from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts, TableArtifact
from isaac_toolkit.utils.graph_utils import memgraph_to_nx
from isaac_toolkit.algorithm.ise.identification.maxmiso import maxmiso_algo
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def generate_cdsl(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    index_file: Optional[Union[str, Path]] = None,
    gen_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
):
    logger.info("Generating CDSL...")
    combined_index_file = (
        workdir / "combined_index.yml" if index_file is None else Path(index_file)
    )
    assert combined_index_file.is_file()
    # with open(combined_index_file, "r") as f:
    #     index_data = yaml.safe_load(f)

    gen_dir = Path(gen_dir) if gen_dir is not None else workdir / "gen"
    gen_dir.mkdir(exist_ok=True)
    generate_args = [
        combined_index_file,
        "--output",
        gen_dir,
        "--split",
        "--split-files",
        "--progress",
        "--inplace",  # TODO use gen/index.yml instead!
    ]
    generate_cdsl_args = [
        "python3",
        "-m",
        "tool.gen.cdsl",
        *generate_args,
    ]
    generate_flat_args = [
        "python3",
        "-m",
        "tool.gen.flat",
        *generate_args,
    ]
    generate_fuse_cdsl_args = [
        "python3",
        "-m",
        "tool.gen.fuse_cdsl",
        *generate_args,
    ]
    # print("generate_cdsl_args", generate_cdsl_args)
    # print("generate_flat_args", generate_flat_args)
    # print("generate_fuse_cdsl_args", generate_fuse_cdsl_args)
    subprocess.run(generate_cdsl_args, check=True)
    subprocess.run(generate_flat_args, check=True)
    subprocess.run(generate_fuse_cdsl_args, check=True)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    generate_cdsl(
        sess,
        workdir=args.workdir,
        gen_dir=args.gen_dir,
        index_file=args.index,
        force=args.force,
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
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--gen-dir", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
