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
import os
import ast
import sys
import shutil
import subprocess

import yaml
import argparse
from typing import Optional, Union
from pathlib import Path
from collections import defaultdict
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()

STRATEGIES = ["best", "worst", "random"]


def run_fake_hls(
    sess: Session,
    set_name: str = "XIsaac",
    core: Optional[str] = None,
    strategy: str = "best",
    index: Optional[Union[str, Path]] = None,
    workdir: Optional[Union[str, Path]] = None,
    label: Optional[str] = None,
    force: bool = False,
    verbose: bool = False,
    cleanup: bool = False,
):
    assert workdir is not None
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert workdir.is_dir()
    # TODO: handle suffix
    out_dir = workdir / "local" / "fake_hls"
    out_dir.mkdir(exist_ok=True, parents=True)
    print("FAKE HLS")
    assert core is not None
    assert set_name is not None
    assert index is not None
    index_file = Path(index)
    assert index_file.is_file()
    instr_schedules = defaultdict(list)
    with open(index_file, "r") as f:
        index_data = yaml.safe_load(f)
    candidates = index_data["candidates"]
    num_candidates = len(candidates)
    print("candidates", candidates, len(candidates))
    for candidate in candidates:
        print("candidate", candidate)
        properties = candidate["properties"]
        metrics = candidate["metrics"]
        name = properties["InstrName"]
        num_instrs = int(properties["#Instrs"])
        max_path_len = int(properties["ScheduleLength"])

        assert num_instrs >= 2
        min_lat = 1
        max_lat = num_instrs
        lats = list(range(min_lat, max_lat + 1))
        print("name", name)
        print("num_instrs", num_instrs)
        print("lats", lats)
        for lat in lats:
            print("lat", lat)
            min_ii = 1
            max_ii = lat
            iis = list(range(min_ii, max_ii + 1))
            print("iis", iis)
            for ii in iis:
                legal = (lat % ii) == 0
                if not legal:
                    continue
                sched = {"lat": lat, "ii": ii}
                instr_schedules[name].append(sched)

        print("instr_schedules", instr_schedules)
    # strategies = ["fast", "slow"]
    # strategy = "worst"  # TODO: implement others
    selected_schedules = defaultdict(dict)

    def apply_strategy(scheds, strategy):
        assert len(scheds) > 0
        if strategy == "best":
            sorted_scheds = sorted(scheds, key=lambda x: (x["lat"], x["ii"]))
            # print("sorted_scheds", sorted_scheds)
            selected = sorted_scheds[0]
            return selected
        elif strategy == "worst":
            sorted_scheds = sorted(scheds, key=lambda x: (x["lat"], x["ii"]))
            # print("sorted_scheds", sorted_scheds)
            selected = sorted_scheds[-1]
            return selected
        elif strategy == "random":
            import random

            selected = random.choice(scheds)
            return selected
        else:
            raise NotImplementedError(f"Unsupported strategy: {strategy}")

    for instr_name, scheds in instr_schedules.items():
        selected = apply_strategy(scheds, strategy)
        selected_schedules[instr_name] = selected
    print("selected_schedules", selected_schedules)
    # TODO: allow sharing
    # TODO: analyze nodes (find longes paths,...)
    # TODO: use external cost/latency model
    # TODO: use constraints
    # TODO: sample schedule-combinations
    # Output format
    selected_solutions_yaml_data = []
    hls_schedules_csv_data = []
    isax_xisaac_yaml_data = []
    sg = 1
    first_stage = 2
    max_stage = first_stage
    for instr_name, sched in selected_schedules.items():
        sol_idx = 0
        new = {"sharing_group": sg, "solution_idx": sol_idx}
        selected_solutions_yaml_data.append(new)
        config = f"SG_{sg}_SOL_IDX_{sol_idx}"
        ii = sched["ii"]
        lat = sched["lat"]
        full_lat = lat + first_stage
        lats = {instr_name: full_lat}
        allocs = {}
        area_est = 123.0  # DUMMY
        area_est2 = area_est
        new2 = {
            "config": config,
            "idx": sol_idx,
            "II": ii,
            "Fallback": False,
            "Instruction latencies": lats,
            "Allocation": allocs,
            "Overall latency": max_lat,
            "Area estimate w/o lifetimes": area_est,
            "Area estimate w/ lifetimes": area_est2,
            "Total lifetime": 0.0,
            "Total decoupled ops": 0,
        }
        hls_schedules_csv_data.append(new2)
        stage = first_stage + lat - 1
        max_stage = max(max_stage, stage)
        dummy_sched = [{"interface": "foo", "stage": first_stage}, {"interface": "bar", "stage": stage}]
        new3 = {"instruction": instr_name, "schedule": dummy_sched}
        isax_xisaac_yaml_data.append(new3)
        sg += 1
    isax_xisaac_yaml_data.append({"last_stage": max_stage + 1})
    print("selected_solutions_yaml_data", selected_solutions_yaml_data)
    print("hls_schedules_csv_data", hls_schedules_csv_data)
    print("isax_xisaac_yaml_data", isax_xisaac_yaml_data)
    hls_schedules_csv_path = out_dir / "hls_schedules.csv"
    hls_outputs_path = out_dir / "output"
    hls_outputs_path.mkdir(exist_ok=True)
    selected_solutions_yaml_path = hls_outputs_path / "selected_solutions.yaml"
    isax_xisaac_yaml_path = hls_outputs_path / "ISAX_XIsaac.yaml"
    hls_schedules_df = pd.DataFrame(hls_schedules_csv_data)
    with open(selected_solutions_yaml_path, "w") as f:
        yaml.dump(selected_solutions_yaml_data, f)
    with open(isax_xisaac_yaml_path, "w") as f:
        yaml.dump(isax_xisaac_yaml_data, f)
    hls_schedules_df.to_csv(hls_schedules_csv_path)

    total_area_estimate = 0
    total_area_estimate_with_lifetimes = 0
    iis = []
    all_lats = []
    num_groups = 0
    num_instrs = 0
    group2instrs = {}
    for row in selected_solutions_yaml_data:
        num_groups += 1
        sharing_group = row["sharing_group"]
        idx = row["solution_idx"]
        name = f"SG_{sharing_group}_SOL_IDX_{idx}"
        schedules = hls_schedules_df[hls_schedules_df["config"] == name]
        assert len(schedules) == 1
        # print("schedules", schedules)
        schedule = schedules.iloc[0]
        ii = schedule["II"]
        iis.append(ii)
        lats = schedule["Instruction latencies"]
        if isinstance(lats, str):
            lats = ast.literal_eval(lats)
        assert isinstance(lats, dict)
        group2instrs[idx] = list(lats.keys())
        num_instrs += len(lats)
        all_lats += list(lats.values())
        area_estimate = schedule["Area estimate w/o lifetimes"]
        total_area_estimate += area_estimate
        area_estimate_with_lifetimes = schedule["Area estimate w/ lifetimes"]
        total_area_estimate_with_lifetimes += area_estimate_with_lifetimes
    max_instrs = max(map(len, group2instrs.values()))
    min_instrs = min(map(len, group2instrs.values()))
    avg_instrs = num_instrs / num_groups
    min_ii = min(iis)
    max_ii = max(iis)
    avg_ii = sum(iis) / len(iis)
    min_lat = min(all_lats)
    max_lat = max(all_lats)
    avg_lat = sum(all_lats) / len(all_lats)
    hls_selected_schedule_metrics_data = [
        {
            "num_groups": num_groups,
            "num_instrs": num_instrs,
            "max_instrs": max_instrs,
            "min_instrs": min_instrs,
            "avg_instrs": avg_instrs,
            "min_ii": min_ii,
            "max_ii": max_ii,
            "avg_ii": avg_ii,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "avg_lat": avg_lat,
            "total_area_estimate": total_area_estimate,
            "total_area_estimate_with_lifetimes": total_area_estimate_with_lifetimes,
        }
    ]
    hls_selected_schedule_metrics_df = pd.DataFrame([hls_selected_schedule_metrics_data])
    print("hls_selected_schedule_metrics_df", hls_selected_schedule_metrics_df)
    hls_selected_schedule_metrics_csv_path = out_dir / "hls_selected_schedule_metrics.csv"
    hls_selected_schedule_metrics_df.to_csv(hls_selected_schedule_metrics_csv_path)


def handle(args):
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    run_fake_hls(
        sess,
        set_name=args.set_name,
        core=args.core,
        strategy=args.strategy,
        index=args.index,
        force=args.force,
        workdir=args.workdir,
        verbose=args.verbose,
    )
    if sess is not None:
        sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    # parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--session", "--sess", "-s", type=str, required=False)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--set-name", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--core", type=str, choices=["cv32e40p"], default=None)
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, default="best")
    parser.add_argument("--verbose", action="store_true")

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
