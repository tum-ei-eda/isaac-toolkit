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
import pytest

from isaac_toolkit.backend.profile.callgrind_new import MEMORY_EVENT_NAMES, collect_memory_event_costs, get_parser


def test_memory_events_are_opt_in():
    parser = get_parser()

    assert not parser.parse_args(["--session", "session"]).memory_events
    assert parser.parse_args(["--session", "session", "--memory-events"]).memory_events


def test_collect_memory_event_costs_by_dynamic_bb_and_pc():
    bb_trace = pd.DataFrame({"bb_end_trace_idx": [1, 3]})
    mem_trace = pd.DataFrame(
        {
            "idx": [0, 1, 2, 3, 3],
            "pc": [0, 0x100, 0x104, 0x108, 0x108],
            "mode": ["r", "r", "w", "r", "w"],
            "bytes": [16, 4, 2, 8, 1],
        }
    )

    bb_costs, pc_costs = collect_memory_event_costs(bb_trace, mem_trace)

    assert bb_costs[MEMORY_EVENT_NAMES].to_dict("records") == [
        {"DataReads": 1, "DataWrites": 1, "DataAccesses": 2, "DataReadBytes": 4, "DataWriteBytes": 2},
        {"DataReads": 1, "DataWrites": 1, "DataAccesses": 2, "DataReadBytes": 8, "DataWriteBytes": 1},
    ]
    assert pc_costs.loc[0x108].to_dict() == {
        "DataReads": 1,
        "DataWrites": 1,
        "DataAccesses": 2,
        "DataReadBytes": 8,
        "DataWriteBytes": 1,
    }


def test_collect_memory_event_costs_rejects_out_of_range_index():
    bb_trace = pd.DataFrame({"bb_end_trace_idx": [1]})
    mem_trace = pd.DataFrame({"idx": [3], "pc": [0x100], "mode": ["r"], "bytes": [4]})

    with pytest.raises(ValueError, match="beyond the basic-block trace"):
        collect_memory_event_costs(bb_trace, mem_trace)
