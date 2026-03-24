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


def get_seal5_score_df(
    seal5_status_df: pd.DataFrame,
    seal5_status_compact_df: pd.DataFrame,
):
    seal5_score_data = []
    seal5_pattern_gen_status_df = seal5_status_df[
        seal5_status_df["pass"] == "generate_passes.pattern_gen.behav_to_pat"
    ][["instr", "status"]]
    seal5_passes_status_df = seal5_status_compact_df[["instr", "status"]]
    # print("seal5_passes_status_df", seal5_passes_status_df)
    for instr_name in seal5_status_df["instr"].unique():
        # print("instr_name", instr_name)
        passes_status = seal5_passes_status_df[seal5_passes_status_df["instr"] == instr_name]
        assert len(passes_status) == 1
        passes_status = passes_status["status"].iloc[0]
        passes_score = 1.0 if passes_status == "good" else (0.5 if passes_status == "ok" else 0.0)

        pattern_gen_status = seal5_pattern_gen_status_df[seal5_pattern_gen_status_df["instr"] == instr_name]
        assert len(pattern_gen_status) == 1
        pattern_gen_status = pattern_gen_status["status"].iloc[0]
        # print("pattern_gen_status", pattern_gen_status)
        pattern_gen_score = 1.0 if pattern_gen_status == "success" else -1.0
        # print("pattern_gen_score", pattern_gen_score)

        new = {"instr": instr_name, "pattern_gen_score": pattern_gen_score, "passes_score": passes_score}
        seal5_score_data.append(new)

    def calc_seal5_score(x):
        return x.min()

    seal5_score_df = pd.DataFrame(seal5_score_data)
    seal5_score_df["seal5_score"] = seal5_score_df[["pattern_gen_score", "passes_score"]].apply(
        calc_seal5_score, axis=1
    )
    return seal5_score_df


def calc_seal5_score(seal5_status_csv, seal5_status_compact_csv, output=None):
    seal5_status_df = pd.read_csv(seal5_status_csv)
    seal5_status_compact_df = pd.read_csv(seal5_status_compact_csv)

    seal5_score_df = get_seal5_score_df(seal5_status_df, seal5_status_compact_df)

    if output is None:
        print(seal5_score_df)
    else:
        seal5_score_df.to_csv(output, index=False)


def handle(args):
    calc_seal5_score(args.seal5_status_csv, args.seal5_status_compact_csv, output=args.output)


def get_parser():
    parser = argparse.ArgumentParser(description="Combine scores into single CSV")
    parser.add_argument("--seal5-status-csv", required=True, help="Seal5 status csv file")
    parser.add_argument("--seal5-status-compact-csv", required=True, help="Seal5 status csv file")
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
