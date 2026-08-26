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
import pandas as pd

from isaac_toolkit.report.report_perf import generate_perf_summary, generate_top_bb_perf


def test_generate_perf_summary_only_contains_available_metrics():
    costs = pd.DataFrame(
        {
            "Ir": [2, 3],
            "Cycles": [4, 9],
            "StallCycles": [2, 6],
            "Latency": [2.0, 4.0],
            "BranchMispredicts": [0, 1],
        }
    )

    summary = generate_perf_summary(costs).set_index("Metric")["Value"]

    assert summary["CPI"] == 13 / 5
    assert summary["AverageLatency"] == 16 / 5
    assert "L1IMisses" not in summary
    assert "L1DMisses" not in summary


def test_generate_top_bb_perf_ranks_total_cycle_contribution():
    stats = pd.DataFrame(
        {
            "bb_idx": [0, 1],
            "invocations": [10, 2],
            "Cycles_mean": [3.0, 20.0],
            "CPI_mean": [1.5, 4.0],
            "StallCycles_mean": [1.0, 15.0],
            "Latency_mean": [2.0, 8.0],
            "cost_patterns": [1, 2],
        }
    )
    unique_bbs = pd.DataFrame(
        {
            "first_pc": [0x100, 0x200],
            "last_pc": [0x104, 0x208],
            "num_instrs": [2, 3],
            "func": ["foo", "bar"],
        }
    )

    result = generate_top_bb_perf(stats, unique_bbs)

    assert result["bb_idx"].tolist() == [1, 0]
    assert result["TotalCycles"].tolist() == [40.0, 30.0]
    assert result["CycleShare"].sum() == 1.0
