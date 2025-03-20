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
import logging
import argparse
import subprocess
from typing import Optional, Union, List
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


logger = logging.getLogger(__name__)


def get_command_for_file(compile_commands_df: pd.DataFrame, file: str):
    # print("compile_commands_df", compile_commands_df)
    # print("files", compile_commands_df["file"].unique())
    matches = compile_commands_df[compile_commands_df["file_resolved"] == file]
    # print("matches", matches)
    assert len(matches) == 1
    command = matches["command"].values[0]
    directory = matches["directory"].values[0]
    return command, directory


def generate_memgraph_cdfg_via_compile_commands(
    sess: Session,
    label: str,
    stage: int = 32,
    force: bool = False,
):
    artifacts = sess.artifacts
    choices_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "choices")
    assert len(choices_artifacts) == 1
    choices_artifact = choices_artifacts[0]
    choices_df = choices_artifact.df
    compile_commands_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.attrs.get("kind") == "compile_commands"
    )
    assert len(compile_commands_artifacts) == 1
    compile_commands_artifact = compile_commands_artifacts[0]
    compile_commands_df = compile_commands_artifact.df
    compile_commands_df["file_resolved"] = compile_commands_df["file"].apply(lambda x: Path(x).resolve())

    files = choices_df["file"].unique()

    extra_args = f"-mllvm -cdfg-enable=1 -mllvm -cdfg-memgraph-session={label} -mllvm -cdfg-memgraph-purge=0 -mllvm -cdfg-stage-mask={stage}"

    for file in files:
        # print("file", file)
        orig_command, directory = get_command_for_file(compile_commands_df, file)
        # print("orig_command", orig_command)
        # print("directory", directory)
        new_command = f"{orig_command} {extra_args}"
        # print("new_command", new_command)
        # new_args = new_command.split(" ")
        # print("args", new_command)
        # print("cwd", directory)
        # input(">")
        subprocess.run(new_command, check=True, shell=True, cwd=directory)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_memgraph_cdfg_via_compile_commands(sess, label=args.label, stage=args.stage, force=args.force)
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
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--stage", type=int, default=8)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
