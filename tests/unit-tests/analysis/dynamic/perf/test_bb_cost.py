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
import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).parents[5] / "isaac_toolkit" / "analysis" / "dynamic" / "perf" / "bb_cost.py"
SPEC = importlib.util.spec_from_file_location("bb_cost", MODULE_PATH)
bb_cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bb_cost)


def test_collect_bb_costs_supports_cva6_stages_and_cache_misses():
    bb_trace = pd.DataFrame(
        {
            "bb_idx": [0, 1],
            "bb_call": [0, 0],
            "bb_end_trace_idx": [1, 2],
        }
    )
    timing = pd.DataFrame(
        {
            "Enter": [10, 12, 15],
            "PC_stage": [11, 13, 16],
            "IF_stage": [12, 14, 17],
            "IQ_stage": [13, 15, 18],
            "ID_stage": [14, 16, 19],
            "IS_stage": [15, 17, 20],
            "EX_stage": [16, 18, 21],
            "COM_stage": [17, 19, 23],
            "br:mispredict": [0, 1, 0],
            "L1I:miss": [1, 0, 1],
            "L1D:miss": [0, 1, 1],
        }
    )

    costs, distribution = bb_cost.collect_bb_costs(bb_trace, timing)

    assert costs["Cycles"].tolist() == [5, 9]
    assert costs["BranchMispredicts"].tolist() == [1, 0]
    assert costs["L1IMisses"].tolist() == [1, 1]
    assert costs["L1DMisses"].tolist() == [1, 1]
    assert {"L1IMisses", "L1DMisses"} <= set(distribution.columns)


def test_summarize_bb_costs_includes_cache_misses():
    bb_trace = pd.DataFrame(
        {
            "bb_idx": [0],
            "bb_call": [0],
            "bb_end_trace_idx": [0],
        }
    )
    timing = pd.DataFrame(
        {
            "Enter": [10],
            "COM_stage": [12],
            "L1I:miss": [1],
            "L1D:miss": [2],
        }
    )
    costs, distribution = bb_cost.collect_bb_costs(bb_trace, timing)

    stats = bb_cost.summarize_bb_costs(costs, distribution)

    assert stats.loc[0, "L1IMisses_mean"] == 1
    assert stats.loc[0, "L1DMisses_mean"] == 2


def test_optional_cache_metrics_are_absent_when_trace_does_not_provide_them():
    bb_trace = pd.DataFrame(
        {
            "bb_idx": [0],
            "bb_call": [0],
            "bb_end_trace_idx": [0],
        }
    )
    timing = pd.DataFrame(
        {
            "Enter": [10],
            "WB_stage": [12],
            "br:mispredict": [0],
        }
    )

    costs, distribution = bb_cost.collect_bb_costs(bb_trace, timing)
    stats = bb_cost.summarize_bb_costs(costs, distribution)

    for frame in (costs, distribution, stats):
        assert not any(column.startswith(("L1IMisses", "L1DMisses")) for column in frame.columns)
