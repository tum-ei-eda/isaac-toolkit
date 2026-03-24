#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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

import pandas as pd


def calc_agg_counts(count_files, output=None):

    dfs = []

    for count_file in count_files:
        df_ = pd.read_pickle(count_file)
        dfs.append(df_)

    df = pd.concat(dfs)
    # print("df", df)

    df = df.groupby("instr").sum().reset_index()
    # df = df.sort_values("util_score", ascending=False)
    df = df.sort_values("estimated_reduction_rel", ascending=False)
    # print("df", df)

    if output is None:
        print(df)
    else:
        df.to_csv(output, index=False)


def handle(args):
    calc_agg_counts(args.index, output=args.output)


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("count_files", nargs="+", help="Pickle files")
    parser.add_argument("-o", "--output", default=None, help="Output CSV file")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
