import argparse
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--hls-schedules-csv", required=True, help="TODO")
    parser.add_argument("--hls-selected-schedules-yaml", required=True, help="TODO")
    args = parser.parse_args()

    with open(args.index, "r") as f:
        combined_index_data = yaml.safe_load(f)
    candidates = combined_index_data["candidates"]
    # print("candidates", candidates)

    schedules_df = pd.read_csv(args.hls_schedules_csv)
    schedules_df
    # print("schedules_df", schedules_df)
    # TODO: each instr needs its own schedule!

    with open(args.hls_selected_schedules_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    # print("yaml_data", yaml_data)
    instr2ii = {}
    instr2lat = {}
    instr2area = {}
    instr2group = {}
    instr2num_shared = {}
    group2instrs = {}
    for row in yaml_data:
        sharing_group = row["sharing_group"]
        idx = row["solution_idx"]
        name = f"SG_{sharing_group}_SOL_IDX_{idx}"
        schedules = schedules_df[schedules_df["config"] == name]
        assert len(schedules) == 1
        # print("schedules", schedules)
        schedule = schedules.iloc[0]
        ii = int(schedule["II"])
        import ast

        lats = ast.literal_eval(schedule["Instruction latencies"])
        area = float(schedule["Area estimate w/o lifetimes"])
        instrs = list(lats.keys())
        group2instrs[idx] = instrs
        for instr, lat in lats.items():
            instr2ii[instr] = ii
            instr2lat[instr] = lat
            instr2area[instr] = area
            instr2group[instr] = sharing_group
            instr2num_shared[instr] = len(lats)
    assert len(candidates) == len(instr2ii)

    for i, candidate in enumerate(candidates):
        name = candidate["properties"]["InstrName"]
        metrics = candidate.get("metrics", {})
        metrics["hls_ii"] = instr2ii[name]
        metrics["hls_latency"] = instr2lat[name]
        metrics["hls_area"] = instr2area[name]
        metrics["hls_num_shared"] = instr2num_shared[name]
        area_scaled = instr2area[name] / instr2num_shared[name]
        metrics["hls_area_scaled"] = area_scaled
        # print("metrics", metrics)
        candidate["metrics"] = metrics

    if args.inplace:
        assert args.output is None
        out_file = args.index
    else:
        assert args.output is not None
        out_file = args.output

    combined_index_data["candidates"] = candidates

    with open(out_file, "w") as f:
        yaml.dump(combined_index_data, f)


if __name__ == "__main__":
    main()
