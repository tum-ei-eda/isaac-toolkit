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
"""Analyze and visualize nearest memory-access reuse events."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


def create_mem_reuse_figure(reuse):
    import plotly.express as px

    if reuse.empty:
        raise ValueError("no reuse events match the selected filters")
    display = reuse.copy()
    display["PC"] = display["pc"].map(hex)
    display["Previous PC"] = display["previous_pc"].map(hex)
    return px.scatter(
        display,
        x="idx",
        y="idx_distance",
        color="mode",
        size="overlap_accesses",
        hover_data=["PC", "Previous PC", "overlap_fraction"],
        title="Nearest Memory Reuse Events",
        labels={"idx": "Instruction trace index", "idx_distance": "Reuse distance"},
    )


def visualize_mem_reuse(sess, output=None, pcs=None, max_idx_distance=1000, modes=None, max_points=50000, force=False):
    artifacts = filter_artifacts(
        sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.TABLE and artifact.name == "mem_reuse"
    )
    if len(artifacts) != 1:
        raise ValueError("exactly one mem_reuse artifact is required; run isaac_toolkit.analysis.dynamic.mem_reuse")
    reuse = artifacts[0].df.copy()
    if pcs:
        reuse = reuse[reuse["pc"].isin(pcs) | reuse["previous_pc"].isin(pcs)]
    if max_idx_distance is not None:
        reuse = reuse[reuse["idx_distance"] <= max_idx_distance]
    if modes:
        reuse = reuse[reuse["mode"].isin(modes)]
    if max_points is not None and len(reuse) > max_points:
        # Deterministic, evenly spaced sampling preserves the full time span.
        positions = pd.Series(np.linspace(0, len(reuse) - 1, max_points, dtype=int)).unique()
        reuse = reuse.iloc[positions]
    output = Path(output) if output else sess.directory / "plots" / "mem_reuse.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"file already exists: {output}")
    create_mem_reuse_figure(reuse).write_html(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot nearest memory-reuse events from an ISAAC session.")
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--out")
    parser.add_argument("--pcs", type=lambda value: [int(pc, 0) for pc in value.split(",")])
    parser.add_argument("--max-idx-distance", type=int, default=1000)
    parser.add_argument("--modes", type=lambda value: value.upper().split(","))
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args(argv)
    sess = Session.from_dir(Path(args.session))
    visualize_mem_reuse(sess, args.out, args.pcs, args.max_idx_distance, args.modes, args.max_points, args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
