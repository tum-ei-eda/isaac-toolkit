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

from isaac_toolkit.visualize.mem_access import select_mem_trace_window
from isaac_toolkit.visualize.mem_reuse import analyze_mem_reuse
from isaac_toolkit.visualize.window_metrics import select_metrics


def test_select_mem_trace_window_starts_at_pc_and_limits_instruction_indices():
    trace = pd.DataFrame(
        {
            "idx": [0, 1, 2, 3, 4],
            "pc": [0, 0x100, 0x104, 0x108, 0x10C],
            "mode": ["r"] * 5,
            "addr": range(5),
        }
    )

    result = select_mem_trace_window(trace, start_pc=0x104, idx_count=2)

    assert result["idx"].tolist() == [2, 3]


def test_analyze_mem_reuse_finds_nearest_overlapping_event():
    trace = pd.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "pc": [0x100, 0x104, 0x108, 0x10C],
            "mode": ["r", "w", "r", "w"],
            "addr": [0x200, 0x300, 0x200, 0x200],
            "bytes": [4, 4, 4, 4],
        }
    )

    result = analyze_mem_reuse(trace)

    assert result[["previous_idx", "idx"]].values.tolist() == [[1, 3], [3, 4]]
    assert result["mode"].tolist() == ["R->R", "R->W"]


def test_analyze_mem_reuse_applies_transition_and_distance_filters():
    trace = pd.DataFrame(
        {
            "idx": [1, 3, 8],
            "pc": [0x100, 0x104, 0x108],
            "mode": ["w", "r", "r"],
            "addr": [0x200, 0x200, 0x200],
            "bytes": [4, 4, 4],
        }
    )

    result = analyze_mem_reuse(trace, max_idx_distance=3, modes=["W->R"])

    assert result[["previous_idx", "idx"]].values.tolist() == [[1, 3]]


def test_window_metric_selection_ignores_labels_and_validates_requested_metrics():
    metrics = pd.DataFrame({"range_name": ["window0"], "cycles_num": [10], "note": ["x"]})

    assert select_metrics(metrics) == ["cycles_num"]
    assert select_metrics(metrics, ["cycles_num"]) == ["cycles_num"]
