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
import yaml


def annotate_util_score(index, util_score_csv, output=None, inplace=False, in_prefix="", out_prefix=""):
    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
    candidates = combined_index_data["candidates"]
    # print("candidates", candidates)

    scores_df = pd.read_csv(util_score_csv)
    # print("scores_df", scores_df)
    # assert len(candidates) == len(scores_df)

    for i, candidate in enumerate(candidates):
        # print("i", i)
        name = candidate["properties"]["InstrName"]
        instr_row = scores_df[scores_df["instr"] == name]
        # print("instr_row", instr_row)
        assert len(instr_row) == 1
        util_score = float(instr_row[f"{in_prefix}util_score"].iloc[0])
        metrics = candidate.get("metrics", {})
        # print("metrics", metrics)
        metrics[f"{out_prefix}util_score"] = util_score
        # print("metrics2", metrics)
        candidate["metrics"] = metrics

    if inplace:
        assert output is None
        out_file = index
    else:
        assert output is not None
        out_file = output

    combined_index_data["candidates"] = candidates

    with open(out_file, "w") as f:
        yaml.dump(combined_index_data, f)


def handle(args):
    annotate_util_score(
        args.index,
        args.util_score_csv,
        output=args.output,
        inplace=args.inplace,
        in_prefix=args.in_prefix,
        out_prefix=args.out_prefix,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--util-score-csv", required=True, help="TODO")
    parser.add_argument("--in-prefix", default="", help="TODO")
    parser.add_argument("--out-prefix", default="", help="TODO")
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
