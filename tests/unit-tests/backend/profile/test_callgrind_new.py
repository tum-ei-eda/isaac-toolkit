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
