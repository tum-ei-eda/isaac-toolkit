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
