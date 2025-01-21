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
import sys
import pandas as pd
import argparse
from typing import Optional
from pathlib import Path

from tqdm import tqdm

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import TableArtifact


# TODO: logger

ETISS_CPU_TIME_DEFAULT = 0.000000031250


def load_mem_trace(
    sess: Session,
    input_file: Path,
    force: bool = False,
    cpu_time: Optional[float] = None,
):
    assert input_file.is_file()
    name = "mem_trace"
    dfs = []
    with pd.read_csv(
        input_file,
        sep=";",
        names=["time_ps", "pc", "mode", "addr", "bytes"],
        chunksize=2**22,
    ) as reader:
        for df in tqdm(reader, disable=False):
            # df["pc"] = df["pc"].apply(lambda x: int(x, 0))  # TODO: add 0x prefix
            df["pc"] = df["pc"].apply(lambda x: int(x, 16))
            df["pc"] = pd.to_numeric(df["pc"])
            df["addr"] = df["addr"].apply(lambda x: int(x, 16))
            df["addr"] = pd.to_numeric(df["addr"])
            if cpu_time is not None:
                df["idx"] = round(df["time_ps"] * 1e-12 / cpu_time).astype(int)
            # print("B", time.time())
            # TODO: normalize instr names
            df["mode"] = df["mode"].astype("category")

            def chunker(seq, size):
                return (seq[pos : pos + size] for pos in range(0, len(seq), size))

            # for row_df in chunker(df, 16):
            #     print("df.bytes", row_df["bytes"])
            #     print("df.bytes2", row_df["bytes"].astype(int))
            #     print("df.bytes3", row_df["bytes"].apply(lambda x: int(x)))
            #     print("df.bytes4", pd.to_numeric(row_df["bytes"]))
            # print("df.bytes", df["bytes"])
            # print("df.bytes2", df["bytes"].astype(int))
            df["bytes"] = df["bytes"].apply(lambda x: int(str(x), 16))
            # print("df.bytes4", pd.to_numeric(df["bytes"]))
            # df["bytes"] = pd.to_numeric(df["bytes"]).astype("category")  # TODO: does this work?
            df["bytes"] = df["bytes"].astype("category")  # TODO: does this work?
            dfs.append(df)
    df = pd.concat(dfs, axis=0)

    attrs = {
        "simulator": "etiss",
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.mem_trace.etiss",
    }
    artifact = TableArtifact(name, df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_file = Path(args.file)
    load_mem_trace(sess, input_file, force=args.force, cpu_time=args.cpu_time)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="dBusAccess.csv file generated by ETISS.")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--cpu-time", type=float, default=ETISS_CPU_TIME_DEFAULT)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
