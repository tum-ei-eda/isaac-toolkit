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
import configparser
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import TableArtifact


def load_ini(sess: Session, input_file: Path, force: bool = False):
    assert input_file.is_file()
    # name = input_file.name
    attrs = {
        "simulator": "etiss",
        "by": "isaac_toolkit.frontend.ini.etiss_mem_layout",
    }
    config = configparser.ConfigParser(strict=False)
    config.read(input_file)

    if "IntConfigurations" not in config:
        raise RuntimeError("Section [IntConfigurations] does not exist in config file " + input_file)

    cfg = config["IntConfigurations"]

    rom_start = int(cfg["simple_mem_system.memseg_origin_00"], 0)
    rom_size = int(cfg["simple_mem_system.memseg_length_00"], 0)
    ram_start = int(cfg["simple_mem_system.memseg_origin_01"], 0)
    ram_size = int(cfg["simple_mem_system.memseg_length_01"], 0)
    data = [
        {"idx": 0, "segment": "rom", "start": rom_start, "size": rom_size},
        {"idx": 1, "segment": "ram", "start": ram_start, "size": ram_size},
    ]

    df = pd.DataFrame(data)
    artifact = TableArtifact("mem_layout", df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_file = Path(args.file)
    load_ini(sess, input_file, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
