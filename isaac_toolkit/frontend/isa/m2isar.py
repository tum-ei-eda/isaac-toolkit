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
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, M2ISARArtifact
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    logger.info("Loading M2-ISA-R metamodel...")
    override = args.force
    input_file = Path(args.file)
    assert input_file.is_file()
    name = input_file.stem
    attrs = {
        "kind": "full",  # TODO: 32/64?
        "top_level": str(input_file.resolve()),
        "by": "isaac_toolkit.frontend.isa.m2isar",
    }
    # TODO: check for validity
    import pickle

    with open(input_file, "rb") as f:
        model = pickle.load(f)
    artifact = M2ISARArtifact(name, model, attrs=attrs)
    sess.add_artifact(artifact, override=override)
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
