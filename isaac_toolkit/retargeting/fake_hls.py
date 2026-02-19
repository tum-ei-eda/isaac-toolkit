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
import sys
import shutil
import subprocess

import yaml
import argparse
from typing import Optional, Union
from pathlib import Path
from collections import defaultdict

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()

DEFAULT_DOCKER_IMAGE = "isaac-quickstart-etiss:latest"


def run_fake_hls(
    sess: Session,
    set_name: str = "XIsaac",
    core: Optional[str] = None,
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
    strategy = "best"  # TODO: implement others
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
    input("!")
    # Output format
    """
    outputs/selected_solutions.yaml
    - sharing_group: 1
      solution_idx: 1
    - sharing_group: 2
      solution_idx: 1
    hls_schedules.csv:
    ,config,idx,II,Fallback,Instruction latencies,Allocation,Overall latency,Area estimate w/o lifetimes,Area estimate w/ lifetimes,Total lifetime,Total decoupled ops
    0,SG_1_SOL_IDX_1,1,1,False,{'CUSTOM0': 4},{'mul': 1},,28071.929688,28071.929688,0.0,0.0
    1,SG_1_SOL_IDX_0,0,4,True,{},{},4.0,,,,
    2,SG_2_SOL_IDX_1,1,1,False,{'CUSTOM3': 3},{},,2014.431885,2014.431885,0.0,0.0
    3,SG_2_SOL_IDX_0,0,1,True,{},{},3.0,,,,
    hls_selected_schedule_metrics.csv:
    ,num_groups,num_instrs,max_instrs,min_instrs,avg_instrs,min_ii,max_ii,avg_ii,min_lat,max_lat,avg_lat,total_area_estimate,total_area_estimate_with_lifetimes
    0,2,2,1,1,1.0,1,1,1.0,3,4,3.5,30086.361573000002,30086.361573000002
    output/ISAX_XIsaac.yaml
    - instruction: CUSTOM0
      schedule:
        - interface: RdRS1
          stage: 2
        - interface: RdRS2
          stage: 2
        - interface: RdIValid
          stage: 2
        - interface: RdIValid
          stage: 3
        - interface: RdStall
          stage: 2
        - interface: RdStall
          stage: 3
        - interface: WrRD
          stage: 3
    - instruction: CUSTOM3
      schedule:
        - interface: RdRS1
          stage: 2
        - interface: RdRS2
          stage: 2
        - interface: RdIValid
          stage: 2
        - interface: RdStall
          stage: 2
        - interface: WrRD
          stage: 2
    - last stage: 4
    """


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
    parser.add_argument("--verbose", action="store_true")

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
