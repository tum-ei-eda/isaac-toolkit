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
"""Static source-location histograms derived from disassembly."""

import sys
from pathlib import Path

from isaac_toolkit.analysis.dynamic.histogram.loc import (
    _collect_histogram,
    _location_map,
    _normalise_metrics,
    get_parser,
)
from isaac_toolkit.analysis.dynamic.histogram.opcode import decode_opcode
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import TableArtifact, filter_artifacts

logger = get_logger()


def collect_loc_histograms(pc2locs_df, disass_df, metrics=None):
    """Build selected source-location histograms from disassembly."""
    metrics = _normalise_metrics(metrics)
    locations = _location_map(pc2locs_df)
    if "pc" not in disass_df:
        raise ValueError("disass is missing column: pc")
    results = {}
    if "count" in metrics:
        results["count"] = _collect_histogram(locations, disass_df[["pc"]], [])
    if "instr" in metrics:
        if "instr" not in disass_df:
            raise ValueError("disass is missing column: instr")
        results["instr"] = _collect_histogram(locations, disass_df[["pc", "instr"]], ["instr"])
    if "opcode" in metrics:
        if "bytecode" not in disass_df:
            raise ValueError("disass is missing column: bytecode")
        words = disass_df["bytecode"].drop_duplicates()
        opcode_map = dict(zip(words, words.map(decode_opcode)))
        values = disass_df[["pc"]].assign(opcode=disass_df["bytecode"].map(opcode_map))
        results["opcode"] = _collect_histogram(locations, values, ["opcode"])
    return results


def create_loc_hists(sess: Session, metrics=None, force: bool = False):
    metrics = _normalise_metrics(metrics)
    mappings = filter_artifacts(sess.artifacts, lambda artifact: artifact.name == "pc2locs")
    disassemblies = filter_artifacts(sess.artifacts, lambda artifact: artifact.name == "disass")
    if len(mappings) != 1:
        raise ValueError(f"Expected exactly one pc2locs artifact, found {len(mappings)}")
    if len(disassemblies) != 1:
        raise ValueError(f"Expected exactly one disass artifact, found {len(disassemblies)}")
    logger.info("Creating static LOC histograms for metrics: %s", ", ".join(metrics))
    histograms = collect_loc_histograms(mappings[0].df, disassemblies[0].df, metrics)
    names = {"count": "disass_locs_hist", "opcode": "disass_locs_opcode_hist", "instr": "disass_locs_instr_hist"}
    created = []
    for metric, dataframe in histograms.items():
        artifact = TableArtifact(
            names[metric], dataframe, attrs={"kind": "histogram", "metric": metric, "by": __name__}
        )
        sess.add_artifact(artifact, override=force)
        created.append(artifact)
    return created


create_loc_hist = create_loc_hists


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
