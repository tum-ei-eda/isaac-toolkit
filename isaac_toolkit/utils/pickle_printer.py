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
import pickle
import argparse

import pandas as pd


# TODO: argparse (--mem, ...)


def print_memory_footprint(df):
    print("================")
    print("Memory Footprint")
    print("================")
    mem = pd.concat([df.dtypes, df.memory_usage(deep=True)], axis=1).sort_values(1, ascending=False)
    mem.rename(columns={0: "dtype", 1: "mem"}, inplace=True)
    total = mem["mem"].sum()
    print(mem)
    print("TOTAL:", total)


def handle(args):
    with open(args.file, "rb") as f:
        data = pickle.load(f)
    if not args.skip_print:
        print("Unpickled Data:")
        with pd.option_context(
            "display.max_rows",
            args.max_rows,
            "display.max_columns",
            args.max_columns,
            "display.width",
            None,
            "max_colwidth",
            150,
        ):
            print(data)
            print(f"len={len(data)}")

    if args.memory:
        assert isinstance(data, (pd.DataFrame, pd.Series)), "Memory footprint only available for pandas types"
        print_memory_footprint(data)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--skip-print", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-columns", type=int, default=None)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
