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

import yaml
import pandas as pd


def names_helper(index, output=None):
    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
        # TODO: index and cdsl should use the same instruction names?
        names = [
            candidate["properties"].get("InstrName", f"CUSTOM{i}")
            for i, candidate in enumerate(combined_index_data["candidates"])
        ]
        print("names", names)
        # num_candidates = len(names)
        names_df = pd.DataFrame({"instr": names})
        names_df["instr_lower"] = names_df["instr"].apply(lambda x: x.lower())
        names_df["idx"] = names_df["instr"].apply(lambda x: names.index(x))

    if output is None:
        print(names_df)
    else:
        names_df.to_csv(output, index=False)


def handle(args):
    names_helper(args.index, output=args.output)


def get_parser():
    parser = argparse.ArgumentParser(description="Extract instr names from index YAML and write to CSV")
    parser.add_argument("index", help="Index yaml file")
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
