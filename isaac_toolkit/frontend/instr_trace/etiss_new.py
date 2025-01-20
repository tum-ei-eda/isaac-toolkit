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
import time
import sys
import pandas as pd
import argparse
from typing import List
from pathlib import Path

from tqdm import tqdm

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact


# TODO: logger


def load_instr_trace(
    sess: Session, input_files: List[Path], force: bool = False, operands: bool = False
):
    assert len(input_files) > 0
    name = input_files[0].name
    # sort input files by name
    sorted_files = sorted(input_files, key=lambda x: x.name)
    # df = pd.read_csv(input_file, sep=":", names=["pc", "rest"])
    dfs = []
    for input_file in sorted_files:
        assert input_file.is_file()
        print("file", input_file)
        with pd.read_csv(input_file, sep=";", chunksize=2**22, header=0) as reader:
            for df in tqdm(reader, disable=False):
                df = df.rename(columns=lambda x: x.strip())
                print("df", df)
                # print("A", time.time())
                df["pc"] = df["pc"].apply(lambda x: int(x, 0))
                df["pc"] = pd.to_numeric(df["pc"])
                # print("B", time.time())
                # TODO: normalize instr names
                df[["instr", "rest"]] = df["assembly"].str.split(
                    " # ", n=1, expand=True
                )
                df["instr"] = df["instr"].apply(lambda x: x.strip())
                df["instr"] = df["instr"].astype("category")
                # print("C", time.time())
                # print("D", time.time())
                df[["bytecode", "operands"]] = df["rest"].str.split(
                    " ", n=1, expand=True
                )
                # print("E", time.time())

                def detect_size(bytecode):
                    if bytecode[:2] == "0x":
                        return len(bytecode[2:]) // 2
                    elif bytecode[:2] == "0b":
                        return len(bytecode[2:]) // 8
                    else:
                        assert len(set(bytecode)) == 2
                        return len(bytecode) // 8

                df["size"] = df["bytecode"].apply(detect_size)
                df["size"] = df["size"].astype("category")
                # print("F", time.time())
                df["bytecode"] = df["bytecode"].apply(
                    lambda x: (
                        int(x, 16)
                        if "0x" in x
                        else (int(x, 2) if "0b" in x else int(x, 2))
                    )
                )
                df["bytecode"] = pd.to_numeric(df["bytecode"])
                # print("H", time.time())

                def convert(x):
                    ret = {}
                    for y in x:
                        if len(y.strip()) == 0:
                            continue
                        assert "=" in y
                        k, v = y.split("=", 1)
                        assert k not in ret
                        ret[k] = int(v)
                    return ret

                if operands:
                    df["operands"] = df["operands"].apply(
                        lambda x: convert(x[1:-1].split(" | "))
                    )
                else:
                    df.drop(columns=["operands"], inplace=True)
                df.drop(columns=["rest"], inplace=True)
                df.drop(columns=["assembly"], inplace=True)
                # print("I", time.time())
                dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df["instr"] = df["instr"].astype("category")
    df["size"] = df["size"].astype("category")

    attrs = {
        "simulator": "etiss",
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.instr_trace.etiss",
    }
    artifact = InstrTraceArtifact(name, df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_files = list(map(Path, args.files))
    load_instr_trace(sess, input_files, force=args.force, operands=args.operands)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--operands", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
