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

# import time
import io
import sys
import itertools
import pandas as pd
import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# TODO: logger


def process_df(df, operands: bool = False):
    df = df[df[0].str.contains(r" \(0x")]
    df = df[~df[0].str.contains("exception")]
    df = df[~df[0].str.contains("tval")]
    df[["core", "rest"]] = df[0].str.split(":", n=1, expand=True)
    df.drop(columns=[0], inplace=True)
    df["core"] = df["core"].apply(lambda x: int(x.split(" ", 1)[-1].strip()))
    df[["pc", "bytecode", "instr", "args"]] = df["rest"].apply(lambda x: x.strip()).str.split(" ", n=3, expand=True)
    df.drop(columns=["rest"], inplace=True)
    if operands:
        raise NotImplementedError
    else:
        df.drop(columns=["args"], inplace=True)
    df["bytecode"] = df["bytecode"].apply(lambda x: x[1:-1])
    df["bytecode"] = df["bytecode"].apply(lambda x: int(x, 0))
    df["pc"] = df["pc"].apply(lambda x: int(x, 0))

    df["pc"] = pd.to_numeric(df["pc"])
    df["instr"] = df["instr"].astype("category")

    def detect_size(instr):
        # if bytecode[:2] == "0x":
        #     return len(bytecode[2:]) // 2
        # elif bytecode[:2] == "0b":
        #     return len(bytecode[2:]) // 8
        # else:
        #     assert len(set(bytecode)) == 2
        #     return len(bytecode) // 8

        # Spike does not use shorter byte codes for compressed instrs
        # Alternatives: check LSBs or instr prefix
        if instr.startswith("c."):
            return 2
        return 4

    df["size"] = df["instr"].apply(detect_size)
    df["size"] = df["size"].astype("category")

    df["bytecode"] = pd.to_numeric(df["bytecode"])
    return df


def parse_and_process(chunk_bytes, operands: bool = False):
    # df = pd.read_csv(input_file, sep="@", header=None, chunksize=2**22) as reader:
    df = pd.read_csv(io.BytesIO(chunk_bytes), header=None, sep="@")
    return process_df(df, operands=operands)


def chunk_iter(path, chunk_size=2**22):
    f = open(path, "rb")

    with f:
        buf = b""
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            buf += data
            last_nl = buf.rfind(b"\n")
            if last_nl == -1:
                continue
            yield buf[: last_nl + 1]
            buf = buf[last_nl + 1 :]
        if buf:
            yield buf


def load_instr_trace(
    sess: Session,
    input_file: Path,
    force: bool = False,
    operands: bool = False,
    num_workers: Optional[int] = None,
    executor: str = "process_pool",
    chunk_size: int = 2**22,
):
    assert input_file.is_file()
    name = input_file.name
    # df = pd.read_csv(input_file, sep=":", names=["pc", "rest"])
    dfs = []
    if True:
        executor_map = {
            "thread_pool": ThreadPoolExecutor,
            "process_pool": ProcessPoolExecutor,
        }
        executor_cls = executor_map.get(executor)
        assert executor_cls is not None, f"Unsupported Executor: {executor}"
        with executor_cls(max_workers=num_workers) as executor:  # tune workers
            dfs_ = list(
                tqdm(
                    executor.map(
                        # lambda x: parse_and_process(x, operands=operands), chunk_iter(input_file, chunk_size=chunk_size)
                        parse_and_process,
                        chunk_iter(input_file, chunk_size=chunk_size),
                        itertools.repeat(operands),
                    ),
                    disable=False,
                )
            )
            dfs += dfs_
    else:
        with pd.read_csv(input_file, sep="@", header=None, chunksize=2**22) as reader:
            for df in tqdm(reader, disable=False):
                df = df[df[0].str.contains(r" \(0x")]
                df[["core", "rest"]] = df[0].str.split(":", n=1, expand=True)
                df.drop(columns=[0], inplace=True)
                df["core"] = df["core"].apply(lambda x: int(x.split(" ", 1)[-1].strip()))
                df[["pc", "bytecode", "instr", "args"]] = (
                    df["rest"].apply(lambda x: x.strip()).str.split(" ", n=3, expand=True)
                )
                df.drop(columns=["rest"], inplace=True)
                if operands:
                    raise NotImplementedError
                else:
                    df.drop(columns=["args"], inplace=True)
                df["bytecode"] = df["bytecode"].apply(lambda x: x[1:-1])
                df["bytecode"] = df["bytecode"].apply(lambda x: int(x, 0))
                df["pc"] = df["pc"].apply(lambda x: int(x, 0))

                df["pc"] = pd.to_numeric(df["pc"])
                df["instr"] = df["instr"].astype("category")

                def detect_size(instr):
                    # if bytecode[:2] == "0x":
                    #     return len(bytecode[2:]) // 2
                    # elif bytecode[:2] == "0b":
                    #     return len(bytecode[2:]) // 8
                    # else:
                    #     assert len(set(bytecode)) == 2
                    #     return len(bytecode) // 8

                    # Spike does not use shorter byte codes for compressed instrs
                    # Alternatives: check LSBs or instr prefix
                    if instr.startswith("c."):
                        return 2
                    return 4

                df["size"] = df["instr"].apply(detect_size)
                df["size"] = df["size"].astype("category")

                df["bytecode"] = pd.to_numeric(df["bytecode"])

                dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df["instr"] = df["instr"].astype("category")
    df["size"] = df["size"].astype("category")
    df["core"] = df["core"].astype("category")
    df["pc"] = pd.to_numeric(df["pc"], downcast="unsigned")
    df["bytecode"] = df["bytecode"].astype("category")
    df.reset_index(drop=True, inplace=True)

    attrs = {
        "simulator": "spike",
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.instr_trace.spike",
    }
    artifact = InstrTraceArtifact(name, df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_file = Path(args.file)
    load_instr_trace(sess, input_file, force=args.force, operands=args.operands)
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
    parser.add_argument("--operands", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
