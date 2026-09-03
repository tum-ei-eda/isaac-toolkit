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
"""Dynamic source-location histograms."""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from isaac_toolkit.analysis.dynamic.histogram.opcode import decode_opcode
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts

logger = get_logger()
DEFAULT_METRICS = ("count",)
SUPPORTED_METRICS = ("count", "opcode", "instr")
METRIC_ALIASES = {"per_opcode": "opcode", "per_instr": "instr"}


def _normalise_metrics(metrics: Iterable[str] | str | None) -> tuple[str, ...]:
    if metrics is None:
        return DEFAULT_METRICS
    if isinstance(metrics, str):
        metrics = (metrics,)
    result = tuple(dict.fromkeys(METRIC_ALIASES.get(metric, metric) for metric in metrics))
    invalid = set(result).difference(SUPPORTED_METRICS)
    if invalid:
        raise ValueError(f"Unsupported LOC histogram metrics: {sorted(invalid)}")
    return result


def _location_map(pc2locs_df: pd.DataFrame) -> pd.DataFrame:
    missing = {"pc", "locs"}.difference(pc2locs_df.columns)
    if missing:
        raise ValueError(f"pc2locs is missing columns: {sorted(missing)}")
    result = pc2locs_df[["pc", "locs"]].explode("locs").rename(columns={"locs": "loc"})
    return result.dropna(subset=["loc"]).drop_duplicates(["pc", "loc"])


def _collect_histogram(locations, values, dimensions):
    keys = ["loc", *dimensions]
    counts = values.groupby(["pc", *dimensions], observed=True, as_index=False).size()
    result = locations.merge(counts, on="pc", how="inner", copy=False)
    result = result.groupby(keys, observed=True, dropna=False, as_index=False)["size"].sum()
    result = result.rename(columns={"size": "count"})
    total = result["count"].sum()
    result["rel_count"] = result["count"] / total if total else 0.0
    return result.sort_values("count", ascending=False, kind="stable").reset_index(drop=True)


def collect_loc_histograms(pc2locs_df, trace_df, metrics: Iterable[str] | str | None = None):
    """Build selected source-location histograms from an instruction trace."""
    metrics = _normalise_metrics(metrics)
    locations = _location_map(pc2locs_df)
    if "pc" not in trace_df:
        raise ValueError("trace is missing column: pc")
    results = {}
    if "count" in metrics:
        results["count"] = _collect_histogram(locations, trace_df[["pc"]], [])
    if "instr" in metrics:
        if "instr" not in trace_df:
            raise ValueError("trace is missing column: instr")
        results["instr"] = _collect_histogram(locations, trace_df[["pc", "instr"]], ["instr"])
    if "opcode" in metrics:
        if "bytecode" not in trace_df:
            raise ValueError("trace is missing column: bytecode")
        words = trace_df["bytecode"].drop_duplicates()
        opcode_map = dict(zip(words, words.map(decode_opcode)))
        values = trace_df[["pc"]].assign(opcode=trace_df["bytecode"].map(opcode_map))
        results["opcode"] = _collect_histogram(locations, values, ["opcode"])
    return results


def create_loc_hists(sess: Session, metrics=None, force: bool = False):
    metrics = _normalise_metrics(metrics)
    mappings = filter_artifacts(sess.artifacts, lambda artifact: artifact.name == "pc2locs")
    traces = filter_artifacts(sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.INSTR_TRACE)
    if len(mappings) != 1:
        raise ValueError(f"Expected exactly one pc2locs artifact, found {len(mappings)}")
    if len(traces) != 1:
        raise ValueError(f"Expected exactly one instruction trace artifact, found {len(traces)}")
    trace = traces[0]
    logger.info("Creating dynamic LOC histograms for metrics: %s", ", ".join(metrics))
    histograms = collect_loc_histograms(mappings[0].df, trace.df, metrics)
    names = {"count": "locs_hist", "opcode": "locs_opcode_hist", "instr": "locs_instr_hist"}
    created = []
    for metric, dataframe in histograms.items():
        artifact = TableArtifact(
            names[metric],
            dataframe,
            attrs={
                "trace": trace.name,
                "kind": "histogram",
                "metric": metric,
                "by": __name__,
            },
        )
        sess.add_artifact(artifact, override=force)
        created.append(artifact)
    return created


create_loc_hist = create_loc_hists


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument(
        "--metrics", nargs="+", default=list(DEFAULT_METRICS), choices=(*SUPPORTED_METRICS, *METRIC_ALIASES)
    )
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"])
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    session_dir = Path(args.session)
    if not session_dir.is_dir():
        raise ValueError(f"Session dir does not exist: {session_dir}")
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    create_loc_hists(sess, metrics=args.metrics, force=args.force)
    sess.save()


if __name__ == "__main__":
    main(sys.argv[1:])
