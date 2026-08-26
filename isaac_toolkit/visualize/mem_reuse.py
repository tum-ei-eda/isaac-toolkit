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

import pandas as pd
from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


REUSE_COLUMNS = [
    "previous_idx",
    "idx",
    "idx_distance",
    "previous_pc",
    "pc",
    "mode",
    "overlap_accesses",
    "overlap_fraction",
]


def analyze_mem_reuse(df: pd.DataFrame, pcs=None, max_idx_distance=1000, modes=None) -> pd.DataFrame:
    """Find the nearest prior event sharing at least one ``(address, bytes)`` access.

    Indexing the most recent event by access and mode keeps this linear in the
    number of memory accesses instead of scanning every earlier event.
    """
    trace = df[df["idx"] != 0]
    if pcs:
        trace = trace[trace["pc"].isin(pcs)]
    allowed_modes = set(modes or ())
    events = {}
    latest = {}
    rows = []

    for idx, group in trace.groupby("idx", sort=True):
        unique_pcs = group["pc"].unique()
        unique_modes = group["mode"].str.lower().unique()
        if len(unique_pcs) != 1 or len(unique_modes) != 1:
            raise ValueError(f"memory event {idx} contains multiple PCs or modes")
        pc = int(unique_pcs[0])
        mode = str(unique_modes[0])
        accesses = set(group[["addr", "bytes"]].itertuples(index=False, name=None))

        candidates = set()
        for access in accesses:
            for previous_mode in ("r", "w"):
                candidate = latest.get((access, previous_mode))
                if candidate is not None:
                    transition = f"{previous_mode.upper()}->{mode.upper()}"
                    if not allowed_modes or transition in allowed_modes:
                        candidates.add(candidate)
        if candidates:
            previous_idx = max(candidates)
            previous = events[previous_idx]
            distance = int(idx) - previous_idx
            if max_idx_distance is None or distance <= max_idx_distance:
                overlap = accesses & previous["accesses"]
                rows.append(
                    {
                        "previous_idx": previous_idx,
                        "idx": int(idx),
                        "idx_distance": distance,
                        "previous_pc": previous["pc"],
                        "pc": pc,
                        "mode": f"{previous['mode'].upper()}->{mode.upper()}",
                        "overlap_accesses": len(overlap),
                        "overlap_fraction": len(overlap) / max(len(accesses), len(previous["accesses"])),
                    }
                )

        events[int(idx)] = {"pc": pc, "mode": mode, "accesses": accesses}
        for access in accesses:
            latest[(access, mode)] = int(idx)

    return pd.DataFrame(rows, columns=REUSE_COLUMNS)


def create_mem_reuse_figure(reuse: pd.DataFrame):
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


def visualize_mem_reuse(sess, output=None, pcs=None, max_idx_distance=1000, modes=None, force=False):
    artifacts = filter_artifacts(
        sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.TABLE and artifact.name == "mem_trace"
    )
    if len(artifacts) != 1:
        raise ValueError("exactly one mem_trace artifact is required")
    reuse = analyze_mem_reuse(artifacts[0].df, pcs=pcs, max_idx_distance=max_idx_distance, modes=modes)
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
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args(argv)
    sess = Session.from_dir(Path(args.session))
    visualize_mem_reuse(sess, args.out, args.pcs, args.max_idx_distance, args.modes, args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
