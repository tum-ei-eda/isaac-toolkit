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
"""Create persistent nearest memory-reuse data from a memory trace."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts

REUSE_COLUMNS = [
    "previous_idx", "idx", "idx_distance", "previous_pc", "pc", "mode", "overlap_accesses", "overlap_fraction"
]


def select_critical_bb_pcs(unique_bbs: pd.DataFrame, topk: int):
    """Return PCs and BB indices for the hottest static BBs by dynamic instruction weight."""
    if topk <= 0:
        raise ValueError("critical_bbs must be positive")
    required = {"first_pc", "last_pc", "num_instrs", "freq"}
    if not required <= set(unique_bbs.columns):
        raise ValueError(f"unique_bbs is missing columns: {', '.join(sorted(required - set(unique_bbs.columns)))}")
    ranked = unique_bbs.assign(_weight=unique_bbs["freq"] * unique_bbs["num_instrs"]).nlargest(topk, "_weight")
    pcs = set()
    for bb in ranked.itertuples():
        pcs.update(range(int(bb.first_pc), int(bb.last_pc) + 1, 2))
    return pcs, ranked.index.astype(int).tolist()


def analyze_mem_reuse(df: pd.DataFrame, pcs=None, max_idx_distance=None, modes=None) -> pd.DataFrame:
    """Find the nearest prior event sharing an ``(address, bytes)`` access."""
    trace = df[df["idx"] != 0]
    if pcs:
        trace = trace[trace["pc"].isin(pcs)]
    allowed_modes = set(modes or ())
    events, latest, rows = {}, {}, []
    for idx, group in trace.groupby("idx", sort=True):
        unique_pcs = group["pc"].unique()
        unique_modes = group["mode"].str.lower().unique()
        if len(unique_pcs) != 1 or len(unique_modes) != 1:
            raise ValueError(f"memory event {idx} contains multiple PCs or modes")
        pc, mode = int(unique_pcs[0]), str(unique_modes[0])
        accesses = set(group[["addr", "bytes"]].itertuples(index=False, name=None))
        candidates = set()
        for access in accesses:
            for previous_mode in ("r", "w"):
                candidate = latest.get((access, previous_mode))
                transition = f"{previous_mode.upper()}->{mode.upper()}"
                if candidate is not None and (not allowed_modes or transition in allowed_modes):
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


def collect_mem_reuse_artifact(
    sess: Session, force=False, pcs=None, max_idx_distance=None, modes=None, critical_bbs=None
):
    artifacts = filter_artifacts(
        sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.TABLE and artifact.name == "mem_trace"
    )
    if len(artifacts) != 1:
        raise ValueError("exactly one mem_trace artifact is required")
    source = artifacts[0]
    critical_bb_idxs = None
    if critical_bbs is not None:
        bb_artifacts = filter_artifacts(
            sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.TABLE and artifact.name == "unique_bbs"
        )
        if len(bb_artifacts) != 1:
            raise ValueError("exactly one unique_bbs artifact is required for --critical-bbs")
        critical_pcs, critical_bb_idxs = select_critical_bb_pcs(bb_artifacts[0].df, critical_bbs)
        pcs = critical_pcs if not pcs else set(pcs) & critical_pcs
    reuse = analyze_mem_reuse(source.df, pcs=pcs, max_idx_distance=max_idx_distance, modes=modes)
    attrs = {
        "mem_trace": source.name,
        "kind": "mem_reuse",
        "by": __name__,
        "pcs": pcs,
        "max_idx_distance": max_idx_distance,
        "modes": modes,
        "critical_bbs": critical_bbs,
        "critical_bb_idxs": critical_bb_idxs,
    }
    sess.add_artifact(TableArtifact("mem_reuse", reuse, attrs=attrs), override=force)
    return reuse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze nearest memory reuse in an ISAAC session.")
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--pcs", type=lambda value: [int(pc, 0) for pc in value.split(",")])
    parser.add_argument("--max-idx-distance", type=int)
    parser.add_argument("--modes", type=lambda value: value.upper().split(","))
    parser.add_argument("--critical-bbs", type=int, help="Only analyze the N hottest static basic blocks")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args(argv)
    sess = Session.from_dir(Path(args.session))
    collect_mem_reuse_artifact(sess, args.force, args.pcs, args.max_idx_distance, args.modes, args.critical_bbs)
    sess.save()


if __name__ == "__main__":
    main(sys.argv[1:])
