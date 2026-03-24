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


def annotate_per_instr_metrics(index, report, output=None, inplace=False, multi=False, multi_agg_func="sum"):
    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
    candidates = combined_index_data["candidates"]
    # print("candidates", candidates)

    report_df = pd.read_csv(report)
    # print("report_df", report_df)
    run_instrs_rel_col = "Run Instructions (rel.)"
    rom_code_rel_col = "ROM code (rel.)"
    has_mem = rom_code_rel_col in report_df.columns
    report_df["runtime_reduction_rel"] = (report_df[run_instrs_rel_col] * -1) + 1
    if has_mem:
        report_df["code_size_reduction_rel"] = (report_df[rom_code_rel_col] * -1) + 1
    if multi:
        progs = list(report_df["Model"].unique())
        # print("progs", progs)
        assert ((len(candidates) + 1) * len(progs)) == len(report_df)
        for i, candidate in enumerate(candidates):
            # print("i", i)
            candidate_rows = report_df.iloc[1 + i :: (len(candidates) + 1)]
            # print("candidate_rows", candidate_rows)
            metrics = candidate.get("metrics", {})
            # print("metrics", metrics)
            # metrics["runtime_reduction_rel"] = 1 - run_instrs_rel
            # metrics["code_size_reduction_rel"] = 1 - rom_code_rel
            metrics["multi_runtime_reduction_rel"] = float(candidate_rows["runtime_reduction_rel"].agg(multi_agg_func))
            if has_mem:
                metrics["multi_code_size_reduction_rel"] = float(
                    candidate_rows["code_size_reduction_rel"].agg(multi_agg_func)
                )
            # print("metrics2", metrics)
            candidate["metrics"] = metrics
            # input("!!!")
    else:
        assert len(candidates) == (len(report_df) - 1)

        for i, candidate in enumerate(candidates):
            # print("i", i)
            # run_instrs_rel = float(report_df[run_instrs_rel_col].iloc[i + 1])
            # rom_code_rel = float(report_df[rom_code_rel_col].iloc[i + 1])
            metrics = candidate.get("metrics", {})
            # print("metrics", metrics)
            # metrics["runtime_reduction_rel"] = 1 - run_instrs_rel
            # metrics["code_size_reduction_rel"] = 1 - rom_code_rel
            metrics["runtime_reduction_rel"] = float(report_df["runtime_reduction_rel"].iloc[i + 1])
            if has_mem:
                metrics["code_size_reduction_rel"] = float(report_df["code_size_reduction_rel"].iloc[i + 1])
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
    annotate_per_instr_metrics(
        args.index,
        args.report,
        output=args.output,
        inplace=args.inplace,
        multi=args.multi,
        multi_agg_func=args.multi_agg_func,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--report", required=True, help="TODO")
    parser.add_argument("--multi", action="store_true", help="Multi-benchmark flag")
    parser.add_argument(
        "--multi-agg-func", default="sum", choices=["sum", "mean", "max", "min"], help="Multi-benchmark agg func"
    )
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
