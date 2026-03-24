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
import re
import sys
import itertools
import random
from math import ceil

import yaml
import argparse
from typing import Optional, Union, List
from pathlib import Path
from collections import defaultdict
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()

# SUPPORTED_STRATEGIES = ["best", "worst", "random"]
SUPPORTED_VARIANT_STRATEGIES = [
    "min_ii",
    "min_lat",
    "min_area",
    "balanced",
    "random",
    "all",
]
DEFAULT_STRATEGIES = ["all"]
# DEFAULT_STRATEGIES = ["min_ii(topk=1)", "min_lat(topk=1)", "min_area(topk=1)", "balanced", "random(n=10)", "all(limit=100)"]


def parse_strategy_string(strategy_str: str):
    """
    Example:
    min_ii(topk=2)
    balanced(alpha=1,beta=1,gamma=1)
    random(n=5)
    all(limit=100,shuffle=true)
    """
    pattern = r"(\w+)(?:\((.*)\))?"
    match = re.match(pattern, strategy_str.strip())
    if not match:
        raise ValueError(f"Invalid strategy format: {strategy_str}")

    name = match.group(1)
    args_str = match.group(2)

    kwargs = {}
    if args_str:
        for part in args_str.split(","):
            k, v = part.split("=")
            k = k.strip()
            v = v.strip()
            if v.lower() == "true":
                v = True
            elif v.lower() == "false":
                v = False
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            kwargs[k] = v

    return name, kwargs


def generate_variants(sg_schedules, strategy_strings):
    """
    Returns:
        List[Dict[sg_id -> sol_idx]]
    """

    parsed = [parse_strategy_string(s) for s in strategy_strings]

    # deterministic ordering
    sg_ids = sorted(sg_schedules.keys())

    all_variants = []
    seen = set()

    def variant_key(v):
        return tuple((sg, v[sg]) for sg in sg_ids)

    for strategy_name, kwargs in parsed:

        # ---------------------------------
        # ALL
        # ---------------------------------
        if strategy_name == "all":
            limit = kwargs.get("limit", None)
            shuffle = kwargs.get("shuffle", False)

            per_sg_indices = [list(range(len(sg_schedules[sg]))) for sg in sg_ids]

            product = list(itertools.product(*per_sg_indices))

            if shuffle:
                random.shuffle(product)

            if limit is not None:
                product = product[:limit]
            else:
                MAX_NUM_ALL = 200
                num = len(product)
                if num >= MAX_NUM_ALL:
                    raise RuntimeError(
                        f"Strategy all would generate {num} variants! Use all(limit={num}) to allow this explicitly."
                    )

            for local_idx, combo in enumerate(product):
                variant = {sg: sol for sg, sol in zip(sg_ids, combo)}
                key = variant_key(variant)
                if key not in seen:
                    seen.add(key)
                    all_variants.append((variant, f"all:{local_idx}"))

        # ---------------------------------
        # RANDOM
        # ---------------------------------
        elif strategy_name == "random":
            n = kwargs.get("n", 1)

            for i in range(n):
                variant = {}
                for sg in sg_ids:
                    sol = random.randrange(len(sg_schedules[sg]))
                    variant[sg] = sol

                key = variant_key(variant)
                if key not in seen:
                    seen.add(key)
                    all_variants.append((variant, f"random(n={n}):{i}"))

        # ---------------------------------
        # MIN-II / MIN-LAT / MIN-AREA
        # ---------------------------------
        elif strategy_name in ["min_ii", "min_lat", "min_area", "balanced"]:

            topk = kwargs.get("topk", 1)

            per_sg_topk = []

            for sg in sg_ids:
                scheds = sg_schedules[sg]

                if strategy_name == "min_ii":
                    ranked = sorted(enumerate(scheds), key=lambda x: x[1]["ii"])

                elif strategy_name == "min_lat":
                    ranked = sorted(enumerate(scheds), key=lambda x: sum(x[1]["lats"].values()))

                elif strategy_name == "min_area":
                    ranked = sorted(enumerate(scheds), key=lambda x: 123.0)  # TODO real area

                elif strategy_name == "balanced":
                    alpha = kwargs.get("alpha", 1.0)
                    beta = kwargs.get("beta", 1.0)
                    gamma = kwargs.get("gamma", 1.0)

                    def score(item):
                        _, sched = item
                        ii = sched["ii"]
                        lat = sum(sched["lats"].values())
                        area = 123.0  # TODO: replace
                        return alpha * ii + beta * lat + gamma * area

                    ranked = sorted(enumerate(scheds), key=score)

                per_sg_topk.append([idx for idx, _ in ranked[:topk]])

            product = list(itertools.product(*per_sg_topk))

            for local_idx, combo in enumerate(product):
                variant = {sg: sol for sg, sol in zip(sg_ids, combo)}
                key = variant_key(variant)
                if key not in seen:
                    seen.add(key)
                    all_variants.append((variant, f"{strategy_name}:{local_idx}"))

        else:
            raise NotImplementedError(f"Unknown strategy {strategy_name}")

    return all_variants


def run_fake_hls(
    sess: Session,
    set_name: str = "XIsaac",
    core: Optional[str] = None,
    strategies: Optional[List[str]] = None,
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
    if strategies is None:
        # strategies = [DEFAULT_STRATEGY]
        strategies = DEFAULT_STRATEGIES
    # assert len(strategies) == 1
    strategy = strategies[0]
    assert core is not None
    assert set_name is not None
    assert index is not None
    index_file = Path(index)
    assert index_file.is_file()
    instr_schedules = defaultdict(list)
    sg_schedules = defaultdict(list)
    with open(index_file, "r") as f:
        index_data = yaml.safe_load(f)
    candidates = index_data["candidates"]
    num_candidates = len(candidates)
    print("candidates", candidates, len(candidates))
    sharing_groups = defaultdict(list)
    first_stage = 2
    for i, candidate in enumerate(candidates):
        print("candidate", candidate)
        properties = candidate["properties"]
        metrics = candidate["metrics"]
        name = properties["InstrName"]
        # TODO: handle actual sharing
        sg = i + 1
        sg_ = [name]
        sharing_groups[sg] = sg_

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
            # max_ii = 1
            max_ii = lat
            iis = list(range(min_ii, max_ii + 1))
            print("iis", iis)
            for ii in iis:
                # stages should have the same latency? (can be maybe dropped)
                legal = (lat % ii) == 0
                if not legal:
                    continue
                # Filter candidates with unsupported timing model
                max_stages = 3
                required_stages = ceil(lat / ii)
                legal2 = required_stages <= max_stages
                if not legal2:
                    continue
                full_lat = lat + first_stage
                sched = {"lat": lat, "full_lat": full_lat, "ii": ii}
                instr_schedules[name].append(sched)
                sg_sched = {"lats": {name: lat}, "full_lats": {name: full_lat}, "ii": ii}
                sg_schedules[sg].append(sg_sched)

    print("instr_schedules", instr_schedules)
    print("sg_schedules", sg_schedules)
    # strategies = ["fast", "slow"]
    # strategy = "worst"  # TODO: implement others
    # selected_schedules = defaultdict(dict)
    # selected_schedules_idx = defaultdict(int)
    print("sharing_groups", sharing_groups)

    variants = generate_variants(sg_schedules, strategies)
    # print("variants", variants, len(variants))
    # input("!")
    # print("selected_schedules", selected_schedules)
    # print("selected_schedules_idx", selected_schedules_idx)
    # print("selected_sg_schedules", selected_sg_schedules)
    # print("selected_sg_schedules_idx", selected_sg_schedules_idx)
    # TODO: allow sharing
    # TODO: analyze nodes (find longes paths,...)
    # TODO: use external cost/latency model
    # TODO: use constraints
    # TODO: sample schedule-combinations
    # Output format
    hls_schedules_csv_data = []
    all_hls_schedules_csv_data = []
    max_stage = first_stage
    # for instr_name, scheds in hls_schedules.items():
    #     for sol_idx, sched in enumerate(scheds):
    #         config = f"SG_{sg}_SOL_IDX_{sol_idx}"
    #         ii = sched["ii"]
    #         lat = sched["lat"]
    #         full_lat = lat + first_stage
    #         lats = {instr_name: full_lat}
    #         allocs = {}
    #         area_est = 123.0  # DUMMY
    #         area_est2 = area_est
    #         new2 = {
    #             "config": config,
    #             "idx": sol_idx,
    #             "II": ii,
    #             "Fallback": False,
    #             "Instruction latencies": lats,
    #             "Allocation": allocs,
    #             "Overall latency": max_lat,
    #             "Area estimate w/o lifetimes": area_est,
    #             "Area estimate w/ lifetimes": area_est2,
    #             "Total lifetime": 0.0,
    #             "Total decoupled ops": 0,
    #         }
    #         hls_schedules_csv_data.append(new2)
    #         sg += 1
    # TODO: fill
    default_area = 1.0
    default_delay = 1.0  # TODO
    default_sharing_weight = 0.2  # TODO
    instr_cost_dict = {
        "ADD": {
            "*": (default_area, default_delay, default_sharing_weight),
            "i32": (default_area, default_delay, default_sharing_weight),
            "i64": (default_area, default_delay, default_sharing_weight),
        },
        "ADDI": {
            "*": (default_area, default_delay, default_sharing_weight),
            "i32": (default_area, default_delay, default_sharing_weight),
            "i64": (default_area, default_delay, default_sharing_weight),
        },
        "SLL": {
            "*": (default_area, default_delay, default_sharing_weight),
            "i32": (default_area, default_delay, default_sharing_weight),
            "i64": (default_area, default_delay, default_sharing_weight),
        },
        "SLLI": {
            "*": (default_area, default_delay, default_sharing_weight),
            "i32": (default_area, default_delay, default_sharing_weight),
            "i64": (default_area, default_delay, default_sharing_weight),
        },
        "*": {
            "*": (default_area, default_delay, default_sharing_weight),
            "i32": (default_area, default_delay, default_sharing_weight),
            "i64": (default_area, default_delay, default_sharing_weight),
        },
    }

    def get_estimated_area(sg_sched, instr_cost_dict):
        instrs = list(sg_sched["lats"].keys())
        total_area = 0
        for instr in instrs:
            dtype = "unknown"
            lookup_key = instr if instr in instr_cost_dict else "*"
            assert lookup_key in instr_cost_dict
            dtype_cost_dict = instr_cost_dict[lookup_key]
            lookup_key2 = dtype if dtype in dtype_cost_dict else "*"
            assert lookup_key2 in dtype_cost_dict
            cost = dtype_cost_dict[lookup_key2]
            area, _, _ = cost
            total_area += area
        return total_area

    for sg, sg_scheds in sg_schedules.items():
        for sol_idx, sg_sched in enumerate(sg_scheds):
            config = f"SG_{sg}_SOL_IDX_{sol_idx}"
            ii = sg_sched["ii"]
            full_lats = sg_sched["full_lats"]
            allocs = {}
            area_est = get_estimated_area(sg_sched, instr_cost_dict)  # WIP
            area_est2 = area_est
            new2 = {
                "config": config,
                "idx": sol_idx,
                "II": ii,
                "Fallback": False,
                "Instruction latencies": full_lats,
                "Allocation": allocs,
                "Overall latency": max_lat,
                "Area estimate w/o lifetimes": area_est,
                "Area estimate w/ lifetimes": area_est2,
                "Total lifetime": 0.0,
                "Total decoupled ops": 0,
            }
            hls_schedules_csv_data.append(new2)
    # for instr_name, sched in selected_schedules.items():
    #     sol_idx = selected_schedules_idx[instr_name]
    #     new = {"sharing_group": sg, "solution_idx": sol_idx}
    #     selected_solutions_yaml_data.append(new)
    #     config = f"SG_{sg}_SOL_IDX_{sol_idx}"
    #     ii = sched["ii"]
    #     lat = sched["lat"]
    #     full_lat = lat + first_stage
    #     lats = {instr_name: full_lat}
    #     allocs = {}
    #     area_est = 123.0  # DUMMY
    #     area_est2 = area_est
    #     new2 = {
    #         "config": config,
    #         "idx": sol_idx,
    #         "II": ii,
    #         "Fallback": False,
    #         "Instruction latencies": lats,
    #         "Allocation": allocs,
    #         "Overall latency": max_lat,
    #         "Area estimate w/o lifetimes": area_est,
    #         "Area estimate w/ lifetimes": area_est2,
    #         "Total lifetime": 0.0,
    #         "Total decoupled ops": 0,
    #     }
    #     hls_schedules_csv_data.append(new2)
    #     stage = first_stage + lat - 1
    #     max_stage = max(max_stage, stage)
    #     dummy_sched = [{"interface": "foo", "stage": first_stage}, {"interface": "bar", "stage": stage}]
    #     new3 = {"instruction": instr_name, "schedule": dummy_sched}
    #     isax_xisaac_yaml_data.append(new3)
    #     sg += 1
    # print("hls_schedules_csv_data", hls_schedules_csv_data)
    # print("isax_xisaac_yaml_data", isax_xisaac_yaml_data)
    hls_schedules_csv_path = out_dir / "hls_schedules.csv"
    hls_outputs_path = out_dir / "output"
    hls_outputs_path.mkdir(exist_ok=True)
    hls_schedules_df = pd.DataFrame(hls_schedules_csv_data)
    hls_schedules_df.to_csv(hls_schedules_csv_path)

    # total_area_estimate = 0
    # total_area_estimate_with_lifetimes = 0
    # iis = []
    # all_lats = []
    # num_groups = 0
    # num_instrs = 0
    # group2instrs = {}
    # for row in selected_solutions_yaml_data:
    #     num_groups += 1
    #     sharing_group = row["sharing_group"]
    #     idx = row["solution_idx"]
    #     name = f"SG_{sharing_group}_SOL_IDX_{idx}"
    #     schedules = hls_schedules_df[hls_schedules_df["config"] == name]
    #     assert len(schedules) == 1
    #     # print("schedules", schedules)
    #     schedule = schedules.iloc[0]
    #     ii = schedule["II"]
    #     iis.append(ii)
    #     lats = schedule["Instruction latencies"]
    #     if isinstance(lats, str):
    #         lats = ast.literal_eval(lats)
    #     assert isinstance(lats, dict)
    #     group2instrs[idx] = list(lats.keys())
    #     num_instrs += len(lats)
    #     all_lats += list(lats.values())
    #     area_estimate = schedule["Area estimate w/o lifetimes"]
    #     total_area_estimate += area_estimate
    #     area_estimate_with_lifetimes = schedule["Area estimate w/ lifetimes"]
    #     total_area_estimate_with_lifetimes += area_estimate_with_lifetimes
    # max_instrs = max(map(len, group2instrs.values()))
    # min_instrs = min(map(len, group2instrs.values()))
    # avg_instrs = num_instrs / num_groups
    # min_ii = min(iis)
    # max_ii = max(iis)
    # avg_ii = sum(iis) / len(iis)
    # min_lat = min(all_lats)
    # max_lat = max(all_lats)
    # avg_lat = sum(all_lats) / len(all_lats)
    # hls_selected_schedule_metrics_data = [
    #     {
    #         "num_groups": num_groups,
    #         "num_instrs": num_instrs,
    #         "max_instrs": max_instrs,
    #         "min_instrs": min_instrs,
    #         "avg_instrs": avg_instrs,
    #         "min_ii": min_ii,
    #         "max_ii": max_ii,
    #         "avg_ii": avg_ii,
    #         "min_lat": min_lat,
    #         "max_lat": max_lat,
    #         "avg_lat": avg_lat,
    #         "total_area_estimate": total_area_estimate,
    #         "total_area_estimate_with_lifetimes": total_area_estimate_with_lifetimes,
    #     }
    # ]
    # hls_selected_schedule_metrics_df = pd.DataFrame([hls_selected_schedule_metrics_data])
    # print("hls_selected_schedule_metrics_df", hls_selected_schedule_metrics_df)
    hls_selected_schedule_metrics_csv_path = out_dir / "hls_selected_schedule_metrics.csv"
    # hls_selected_schedule_metrics_df.to_csv(hls_selected_schedule_metrics_csv_path)

    def process_variant(variant_ifx, variant_name, variant, description, variant_dir):
        variant_dir.mkdir(exist_ok=True)

        variant_selected_solutions_yaml_data = []
        variant_isax_xisaac_yaml_data = []

        total_area_estimate = 0
        total_area_estimate_with_lifetimes = 0
        iis = []
        all_lats = []
        num_groups = 0
        num_instrs = 0
        group_instr_counts = []
        details = []
        max_stage = 0

        for sg, sol_idx in variant.items():
            num_groups += 1

            sg_sched = sg_schedules[sg][sol_idx]

            ii = sg_sched["ii"]
            iis.append(ii)

            lats = sg_sched["lats"]
            full_lats = sg_sched["full_lats"]
            instr_count = len(full_lats)
            group_instr_counts.append(instr_count)
            num_instrs += instr_count
            all_lats += list(full_lats.values())

            area_estimate = 123.0
            total_area_estimate += area_estimate
            total_area_estimate_with_lifetimes += area_estimate
            new = {"sharing_group": sg, "solution_idx": sol_idx}
            variant_selected_solutions_yaml_data.append(new)
            for instr_name, lat in lats.items():
                stage = first_stage + lat - 1  # TODO!
                max_stage = max(max_stage, stage)
                dummy_sched = [{"interface": "foo", "stage": first_stage}, {"interface": "bar", "stage": stage}]
                new3 = {"instruction": instr_name, "schedule": dummy_sched}
                variant_isax_xisaac_yaml_data.append(new3)
            lats_str = ", ".join(f"{instr}: {lat}" for instr, lat in lats.items())
            full_lats_str = ", ".join(f"{instr}: {lat}" for instr, lat in full_lats.items())
            detail = f"SG{sg}(II={ii}, lats={{{lats_str}}}, full_lats={{{full_lats_str}}})"
            details.append(detail)
        details_str = ", ".join(details)
        # print("details_str", details_str)
        # input("!")
        # variant_isax_xisaac_yaml_data.append({"last_stage": max_stage + 1})
        variant_isax_xisaac_yaml_data.append({"last stage": max_stage + 1})
        variant_selected_solutions_yaml_path = variant_dir / "selected_solutions.yaml"
        # print("variant_selected_solutions_yaml_data", variant_selected_solutions_yaml_data)
        with open(variant_selected_solutions_yaml_path, "w") as f:
            yaml.dump(variant_selected_solutions_yaml_data, f)
        variant_isax_xisaac_yaml_path = variant_dir / "ISAX_XIsaac.yaml"
        with open(variant_isax_xisaac_yaml_path, "w") as f:
            yaml.dump(variant_isax_xisaac_yaml_data, f)

        max_instrs = max(group_instr_counts)
        min_instrs = min(group_instr_counts)
        avg_instrs = num_instrs / num_groups

        min_ii = min(iis)
        max_ii = max(iis)
        avg_ii = sum(iis) / len(iis)

        min_lat = min(all_lats)
        max_lat = max(all_lats)
        avg_lat = sum(all_lats) / len(all_lats)

        variant_metrics_row = {
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
            "Variant idx": variant_idx,
            "Variant name": variant_name,
            "Variant description": description,
            "Variant details": details_str,
        }
        return variant_metrics_row

    variant_metrics_rows = []

    variants_dir = hls_outputs_path / "variants"
    for variant_idx, (variant, description) in enumerate(variants):
        variant_name = f"V{variant_idx}"
        variant_dir = hls_outputs_path / variant_name
        variant_metrics_row = process_variant(variant_idx, variant_name, variant, description, variant_dir)
        variant_metrics_rows.append(variant_metrics_row)
        if variant_idx == 0:
            # export first variant to base out dir for annotation (assign_hls)
            _ = process_variant(variant_idx, variant_name, variant, description, hls_outputs_path)
    hls_selected_schedule_metrics_df = pd.DataFrame(variant_metrics_rows)
    # print("hls_selected_schedule_metrics_df")
    # print(hls_selected_schedule_metrics_df)
    hls_selected_schedule_metrics_df.to_csv(hls_selected_schedule_metrics_csv_path)


def handle(args):
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    # strategy = args.strategy
    strategies = args.strategies
    # assert strategy is not None or strategies is not None
    # assert strategy is not None or strategies is not None
    # if strategy is not None:
    #     assert strategies is None
    #     strategies = [strategy]
    if strategies is not None:
        # assert strategy is None
        assert isinstance(strategies, str)
        strategies = strategies.split(",")
        # assert all(x in SUPPORTED_STRATEGIES for x in strategies)
    run_fake_hls(
        sess,
        set_name=args.set_name,
        core=args.core,
        strategies=strategies,
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
    # parser.add_argument("--strategy", type=str, choices=SUPPORTED_STRATEGIES, default=None)  # TODO: drop
    # parser.add_argument("--strategy", type=str, default=None)  # TODO: drop
    parser.add_argument("--strategies", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
