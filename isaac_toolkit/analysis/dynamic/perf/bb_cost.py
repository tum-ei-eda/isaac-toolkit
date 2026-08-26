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
"""Attribute pipeline costs to dynamic basic-block invocations."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()

IDENTITY_COLUMNS = ["bb_idx", "bb_call", "bb_end_trace_idx"]
COUNTER_METRICS = {
    "br:mispredict": "BranchMispredicts",
    "L1I:miss": "L1IMisses",
    "L1D:miss": "L1DMisses",
}
DEFAULT_METRICS = ["Ir", "Cycles", "StallCycles", "CPI", "Latency"]


def collect_bb_costs(bb_trace_df: pd.DataFrame, timing_df: pd.DataFrame):
    """Return invocation costs and their per-BB empirical distribution.

    ``Cycles`` uses half-open ownership: an instruction owns the interval from
    its Enter cycle to the next instruction's Enter cycle.  The last
    instruction owns the remaining pipeline-drain interval.  Consequently the
    invocation costs are additive, stalls are charged to the instruction/BB
    which precedes them, and ``sum(Cycles)`` equals the complete trace span.
    """
    if not len(timing_df):
        raise ValueError("timing trace is empty")
    if not len(bb_trace_df):
        empty = pd.DataFrame(columns=IDENTITY_COLUMNS + DEFAULT_METRICS)
        return empty, pd.DataFrame(columns=["bb_idx", *DEFAULT_METRICS, "occurrences", "probability"])
    if "Enter" not in timing_df:
        raise ValueError("timing trace has no 'Enter' column")

    ends = bb_trace_df["bb_end_trace_idx"].to_numpy(dtype=np.int64)
    if np.any(ends[1:] <= ends[:-1]) or ends[-1] >= len(timing_df):
        raise ValueError("bb_trace boundaries are not strictly increasing or exceed the timing trace")

    enters = timing_df["Enter"].to_numpy(dtype=np.int64)
    completion_columns = [c for c in timing_df.columns if c.endswith("_stage")]
    if not completion_columns:
        raise ValueError("timing trace contains no '*_stage' pipeline columns")
    completion = timing_df[completion_columns].max(axis=1).to_numpy(dtype=np.int64)

    instruction_cycles = np.empty(len(timing_df), dtype=np.int64)
    instruction_cycles[:-1] = enters[1:] - enters[:-1]
    instruction_cycles[-1] = max(1, completion[-1] - enters[-1] + 1)
    if np.any(instruction_cycles < 0):
        raise ValueError("timing trace Enter cycles are not monotonic")

    starts = np.r_[0, ends[:-1] + 1]
    ir = ends - starts + 1
    cycles = np.add.reduceat(instruction_cycles, starts)
    latency = completion - enters + 1
    latency_sum = np.add.reduceat(latency, starts)

    costs = bb_trace_df.reset_index(drop=True).copy()
    costs["Ir"] = ir
    costs["Cycles"] = cycles
    costs["StallCycles"] = cycles - ir
    costs["CPI"] = cycles / ir
    costs["Latency"] = latency_sum / ir
    for trace_column, metric in COUNTER_METRICS.items():
        if trace_column in timing_df:
            counter = timing_df[trace_column].to_numpy(dtype=np.int64)
            costs[metric] = np.add.reduceat(counter, starts)

    metric_columns = [*DEFAULT_METRICS, *(metric for metric in COUNTER_METRICS.values() if metric in costs)]
    distribution = (
        costs.groupby(["bb_idx", *metric_columns], observed=True, dropna=False)
        .size()
        .rename("occurrences")
        .reset_index()
    )
    totals = distribution.groupby("bb_idx", observed=True)["occurrences"].transform("sum")
    distribution["probability"] = distribution["occurrences"] / totals
    distribution.sort_values(["bb_idx", "occurrences"], ascending=[True, False], inplace=True)
    distribution.reset_index(drop=True, inplace=True)
    return costs, distribution


def summarize_bb_costs(costs: pd.DataFrame, distribution: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics across invocations of each unique BB."""
    metrics = [
        "Cycles",
        "StallCycles",
        "CPI",
        "Latency",
        *(metric for metric in COUNTER_METRICS.values() if metric in costs),
    ]
    if costs.empty:
        return pd.DataFrame(columns=["bb_idx", "invocations", "cost_patterns"])
    grouped = costs.groupby("bb_idx", observed=True)
    parts = [grouped.size().rename("invocations")]
    for metric in metrics:
        values = grouped[metric]
        parts.extend(
            [
                values.mean().rename(f"{metric}_mean"),
                values.std(ddof=0).fillna(0).rename(f"{metric}_std"),
                values.min().rename(f"{metric}_min"),
                values.quantile(0.5).rename(f"{metric}_p50"),
                values.quantile(0.95).rename(f"{metric}_p95"),
                values.max().rename(f"{metric}_max"),
            ]
        )
    result = pd.concat(parts, axis=1).reset_index()
    patterns = distribution.groupby("bb_idx", observed=True).size().rename("cost_patterns")
    result = result.merge(patterns, left_on="bb_idx", right_index=True, how="left")
    return result


def collect_bb_cost_artifacts(sess: Session, force: bool = False):
    print("collect_bb_cost_artifacts")
    bb_artifacts = filter_artifacts(sess.artifacts, lambda x: x.name == "bb_trace")
    timing_artifacts = filter_artifacts(
        sess.artifacts, lambda x: x.flags & ArtifactFlag.TRACE and x.attrs.get("kind") == "timing_trace"
    )
    if len(bb_artifacts) != 1 or len(timing_artifacts) != 1:
        raise ValueError("exactly one bb_trace and one timing_trace artifact are required")
    bb_artifact, timing_artifact = bb_artifacts[0], timing_artifacts[0]
    costs, distribution = collect_bb_costs(bb_artifact.df, timing_artifact.df)
    stats = summarize_bb_costs(costs, distribution)
    print("costs", costs)
    print("distribution", distribution)
    print("stats", stats)
    attrs = {"bb_trace": bb_artifact.name, "timing_trace": timing_artifact.name, "by": __name__}
    sess.add_artifact(TableArtifact("bb_cost", costs, attrs={**attrs, "kind": "bb_cost"}), override=force)
    sess.add_artifact(
        TableArtifact("bb_cost_distribution", distribution, attrs={**attrs, "kind": "bb_cost_distribution"}),
        override=force,
    )
    sess.add_artifact(TableArtifact("bb_cost_stats", stats, attrs={**attrs, "kind": "bb_cost_stats"}), override=force)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args(argv)
    session_dir = Path(args.session)
    if not session_dir.is_dir():
        parser.error(f"session directory does not exist: {session_dir}")
    set_log_level(console_level=args.log, file_level=args.log)
    sess = Session.from_dir(session_dir)
    collect_bb_cost_artifacts(sess, force=args.force)
    sess.save()


if __name__ == "__main__":
    main(sys.argv[1:])
