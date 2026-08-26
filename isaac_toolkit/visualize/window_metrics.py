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
"""Visualize trace-analyzer metrics across instruction windows."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


def load_window_metrics(sess: Session, groups=None) -> pd.DataFrame:
    artifacts = filter_artifacts(
        sess.artifacts,
        lambda artifact: artifact.flags & ArtifactFlag.TABLE
        and artifact.attrs.get("kind") == "analysis"
        and artifact.name.startswith("analysis_"),
    )
    requested = set(groups or ())
    if requested:
        artifacts = [
            artifact
            for artifact in artifacts
            if artifact.attrs.get("group") in requested or artifact.name.removeprefix("analysis_") in requested
        ]
    if not artifacts:
        raise ValueError("no matching analysis_* artifacts found")
    frames = [artifact.df.copy() for artifact in artifacts]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="range_name", how="outer", validate="one_to_one")
    return result


def select_metrics(df: pd.DataFrame, metrics=None):
    available = [column for column in df if column != "range_name" and pd.api.types.is_numeric_dtype(df[column])]
    if metrics:
        missing = sorted(set(metrics) - set(available))
        if missing:
            raise ValueError(f"metrics not found: {', '.join(missing)}")
        return list(metrics)
    return available


def create_window_metrics_figure(df, metrics=None, plot_type="area", normalize=False, group_prefixes=False):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics = select_metrics(df, metrics)
    if not metrics:
        raise ValueError("no numeric metrics selected")
    x = np.arange(len(df))
    if plot_type in ("line", "area", "scatter"):
        fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, subplot_titles=metrics)
        for row, metric in enumerate(metrics, start=1):
            mode = "markers" if plot_type == "scatter" else "lines"
            fig.add_trace(
                go.Scatter(x=x, y=df[metric], mode=mode, fill="tozeroy" if plot_type == "area" else None, name=metric),
                row=row,
                col=1,
            )
        fig.update_layout(height=max(350, 220 * len(metrics)), showlegend=False, title="Window Metrics")
        fig.update_xaxes(title_text="Window index", row=len(metrics), col=1)
        return fig

    if plot_type == "heatmap":
        values = df[metrics].T
        if normalize:
            span = values.max(axis=1) - values.min(axis=1)
            values = values.sub(values.min(axis=1), axis=0).div(span.replace(0, 1), axis=0)
        return go.Figure(go.Heatmap(z=values.values, x=x, y=metrics, colorscale="Viridis")).update_layout(
            title="Window Metric Heatmap", xaxis_title="Window index"
        )

    if plot_type != "heatmap-multi":
        raise ValueError(f"unsupported plot type: {plot_type}")
    groups = defaultdict(list)
    for metric in metrics:
        groups[metric.split("_", 1)[0] if group_prefixes else metric].append(metric)
    fig = make_subplots(rows=len(groups), cols=1, shared_xaxes=True, subplot_titles=list(groups))
    for row, group_metrics in enumerate(groups.values(), start=1):
        values = df[group_metrics].T
        if normalize:
            span = values.max(axis=1) - values.min(axis=1)
            values = values.sub(values.min(axis=1), axis=0).div(span.replace(0, 1), axis=0)
        fig.add_trace(
            go.Heatmap(z=values.values, x=x, y=group_metrics, colorscale="Viridis", showscale=True), row=row, col=1
        )
    fig.update_layout(height=max(350, 220 * len(groups)), title="Window Metric Heatmaps")
    return fig


def visualize_window_metrics(
    sess, output=None, groups=None, metrics=None, plot_type="area", normalize=False, group_prefixes=False, force=False
):
    df = load_window_metrics(sess, groups=groups)
    output = Path(output) if output else sess.directory / "plots" / "window_metrics.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"file already exists: {output}")
    create_window_metrics_figure(df, metrics, plot_type, normalize, group_prefixes).write_html(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot trace-analyzer metrics across windows in an ISAAC session.")
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--out")
    parser.add_argument("--groups", type=lambda value: value.split(","))
    parser.add_argument("--metrics", type=lambda value: value.split(","))
    parser.add_argument("--plot-type", choices=["line", "area", "scatter", "heatmap", "heatmap-multi"], default="area")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--group-prefixes", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args(argv)
    sess = Session.from_dir(Path(args.session))
    visualize_window_metrics(
        sess, args.out, args.groups, args.metrics, args.plot_type, args.normalize, args.group_prefixes, args.force
    )


if __name__ == "__main__":
    main(sys.argv[1:])
