#
# Copyright (c) 2025 TUM Department of Electrical and Computer Engineering.
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
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)

# Inspired by: https://github.com/tum-ei-eda/etiss/blob/master/src/bare_etiss_processor/get_metrics.py


def collect_perf_metrics_seq(timing_df):
    stages = timing_df.columns
    temp_df = timing_df.copy()
    # print(temp_df)
    # print(temp_df[[stages[0]]])
    temp_df2 = pd.DataFrame(temp_df.values < temp_df[[stages[0]]].values, columns=stages)
    temp_df2.index = temp_df.index
    # pd.set_option("display.max_rows", None)
    # print(temp_df2)
    # print(temp_df2.value_counts())
    # print("stages", stages)
    mask_df = temp_df2

    timing_df[mask_df] = np.nan
    # print(timing_df)
    # num_stages_per_instruction = timing_df.notna().sum(axis=1).value_counts()
    # print("num_stages_per_instruction", num_stages_per_instruction)
    avg_stages_per_instr = timing_df.notna().sum(axis=1).mean(axis=0)
    # print("avg_stages_per_instr", avg_stages_per_instr)
    latencies_df = (timing_df[stages].max(axis=1) - timing_df[stages].min(axis=1)) + 1
    # print(latencies_df)
    # print(latencies_df[latencies_df == 34.0])
    # print(latencies_df.value_counts())
    avg_latency = latencies_df.mean()
    median_latency = latencies_df.median()
    max_latency = latencies_df.max()
    # print("avg_latency", avg_latency)
    # print("median_latency", median_latency)
    # print("max_latency", max_latency)
    num_stages = len(stages)
    total_instructions = len(timing_df)
    total_cycles = timing_df[stages].max().max() - timing_df[stages].min().min()
    stall_cycles = total_cycles - total_instructions
    stall_rate = stall_cycles / total_cycles
    cpi = total_cycles / total_instructions
    # TODO: minus latency of first instruction?
    metrics = {
        "num_stages": num_stages,
        "avg_stages_per_instr": avg_stages_per_instr,
        "total_cycles": total_cycles,
        "total_instructions": total_instructions,
        "stall_cycles": stall_cycles,
        "stall_rate": stall_rate,
        "cpi": cpi,
        "avg_latency": avg_latency,
        "median_latency": median_latency,
        "max_latency": max_latency,
    }

    if cpi > 10:
        print("metrics", metrics)
        print(timing_df)
    # print("metrics", metrics)
    return metrics


def collect_perf_metrics(
    # perf_trace_df,
    timing_df,
    # verbose: bool = False,
    # TODO: add agg=avg!
):
    # stages = timing_df.columns
    # print(timing_df)

    # timing_df.drop(index=[*range(100000, 200000), *range(2000000, 3000000)], inplace=True)
    index_values = np.array(timing_df.index)
    # index_values = np.concatenate((index_values[:1000], index_values[1001:]))
    # print("index_values", index_values, len(index_values))
    index_values_ref = np.arange(index_values[0], index_values[0] + len(index_values))
    # print("index_values_ref", index_values_ref)
    index_values_diff = index_values_ref - index_values
    # print("index_values_diff", index_values_diff)
    counts = np.unique(index_values_diff, return_counts=True)
    print("counts", counts)
    # unique = pd.unique(timing_df[stages].values.ravel())
    # print("timing_df", len(timing_df))
    # print("unique", unique, len(unique))
    all_metrics = []
    if len(counts[0]) > 1:
        seqs = []
        while len(index_values) > 0:
            print("remaining", len(index_values))
            print("index_values_diff", index_values_diff)
            until = np.argmax(index_values_diff < 0)
            print("until", until)
            if until == 0:
                print("until==0")
                index_values_this = index_values
                index_values = []
                seqs.append(index_values_this)
            else:
                index_values_this = index_values[:until]
                print("index_values_this", index_values_this)
                index_values = index_values[until:]
                print("index_values", index_values)
                print("index_values_diff0", index_values_diff)
                index_values_diff = index_values_diff[until:]
                print("index_values_diff1", index_values_diff)
                index_values_diff = index_values_diff - index_values_diff[0]
                print("index_values_diff2", index_values_diff)
                seqs.append(index_values_this)
            # input(">")
        print("seqs", seqs, len(seqs))
        for seq in seqs:
            print("seq", seq, len(seq))
            # print("timing_df")
            # print(timing_df)
            seq_timing_df = timing_df.loc[seq]
            start_idx = seq_timing_df.index[0]
            end_idx = seq_timing_df.index[-1]
            print("start,end", start_idx, end_idx)
            # print("seq_timing_df")
            print(seq_timing_df)
            metrics = collect_perf_metrics_seq(seq_timing_df)
            metrics["start_idx"] = start_idx
            metrics["end_idx"] = end_idx
            all_metrics.append(metrics)
        # input(">2")

    else:
        assert len(counts[0]) == 1
        assert counts[0][0] == 0
        # assert
        metrics = collect_perf_metrics_seq(timing_df)
        all_metrics.append(metrics)
    metrics_df = pd.DataFrame(all_metrics)
    # print("metrics_df")
    print(metrics_df)
    print(metrics_df[["total_instructions", "cpi"]])
    # assert False

    return metrics_df


def get_perf_metrics(
    sess: Session,
    force: bool = False,
    # verbose: bool = False,
):
    artifacts = sess.artifacts

    timing_trace_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TRACE and x.attrs.get("kind") == "timing_trace"
    )
    assert len(timing_trace_artifacts) == 1
    timing_trace_artifact = timing_trace_artifacts[0]
    assert timing_trace_artifact.attrs.get("simulator") in ["etiss_perf", "etiss"]

    metrics_df = collect_perf_metrics(
        timing_trace_artifact.df,
    )

    attrs = {
        "timing_trace": timing_trace_artifact.name,
        "kind": "metrics",
        "by": __name__,
    }

    perf_metrics_artifact = TableArtifact("perf_metrics", metrics_df, attrs=attrs)
    sess.add_artifact(perf_metrics_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    get_perf_metrics(sess, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
