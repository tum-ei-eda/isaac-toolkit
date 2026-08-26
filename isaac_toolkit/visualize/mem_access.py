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
"""Visualize memory accesses from a session ``mem_trace`` artifact."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


def select_mem_trace_window(df: pd.DataFrame, start_pc=None, idx_count=None) -> pd.DataFrame:
    result = df[df["idx"] != 0].copy()
    if start_pc is not None:
        matches = result.index[result["pc"] == start_pc]
        if matches.empty:
            raise ValueError(f"start PC {hex(start_pc)} is not present in the memory trace")
        start_idx = result.loc[matches[0], "idx"]
        result = result[result["idx"] >= start_idx]
    if idx_count is not None:
        if idx_count <= 0:
            raise ValueError("idx_count must be positive")
        first_idx = result["idx"].iloc[0]
        result = result[result["idx"] < first_idx + idx_count]
    return result


def compute_access_heatmap(data: pd.DataFrame, time_bins: int, addr_bins: int):
    if data.empty:
        return np.zeros((addr_bins, time_bins)), np.arange(time_bins), np.arange(addr_bins)
    heatmap, xedges, yedges = np.histogram2d(data["idx"], data["addr"], bins=[time_bins, addr_bins])
    return np.log1p(heatmap.T), (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2


def create_mem_access_figure(df: pd.DataFrame, time_bins=200, addr_bins=200):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    reads = compute_access_heatmap(df[df["mode"].str.lower() == "r"], time_bins, addr_bins)
    writes = compute_access_heatmap(df[df["mode"].str.lower() == "w"], time_bins, addr_bins)
    zmax = max(float(reads[0].max()), float(writes[0].max()), 1.0)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reads", "Writes"), shared_yaxes=True)
    for col, (title, (z, x, y)) in enumerate(zip(("Reads", "Writes"), (reads, writes)), start=1):
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale="Viridis",
                zmin=0,
                zmax=zmax,
                showscale=col == 2,
                colorbar={"title": "log(1 + accesses)"} if col == 2 else None,
                name=title,
            ),
            row=1,
            col=col,
        )
    fig.update_layout(title="Memory Access Heatmaps", height=600)
    fig.update_xaxes(title_text="Instruction trace index")
    fig.update_yaxes(title_text="Address", row=1, col=1)
    return fig


def visualize_mem_access(sess, output=None, time_bins=200, addr_bins=200, start_pc=None, idx_count=None, force=False):
    artifacts = filter_artifacts(
        sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.TABLE and artifact.name == "mem_trace"
    )
    if len(artifacts) != 1:
        raise ValueError("exactly one mem_trace artifact is required")
    df = select_mem_trace_window(artifacts[0].df, start_pc=start_pc, idx_count=idx_count)
    if df.empty:
        raise ValueError("selected memory trace window is empty")
    output = Path(output) if output else sess.directory / "plots" / "mem_access.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"file already exists: {output}")
    create_mem_access_figure(df, time_bins=time_bins, addr_bins=addr_bins).write_html(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot read/write memory-access heatmaps from an ISAAC session.")
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--out")
    parser.add_argument("--time-bins", type=int, default=200)
    parser.add_argument("--addr-bins", type=int, default=200)
    parser.add_argument("--start-pc", type=lambda value: int(value, 0))
    parser.add_argument("--idx-count", type=int)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args(argv)
    sess = Session.from_dir(Path(args.session))
    visualize_mem_access(sess, args.out, args.time_bins, args.addr_bins, args.start_pc, args.idx_count, args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
