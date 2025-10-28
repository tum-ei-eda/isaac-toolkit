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
from typing import List, Optional, Union


import sys
import logging
import argparse
from pathlib import Path

import yaml
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def generate_ranges_yaml(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    force: bool = False,
    filter_funcs: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = False,
    topk: Optional[int] = None,
):
    artifacts = sess.artifacts

    bb_trace_artifacts = filter_artifacts(artifacts, lambda x: x.name == "bb_trace")
    assert len(bb_trace_artifacts) == 1
    bb_trace_artifact = bb_trace_artifacts[0]
    bb_trace_df = bb_trace_artifact.df

    unique_bbs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "unique_bbs")
    assert len(unique_bbs_artifacts) == 1
    unique_bbs_artifact = unique_bbs_artifacts[0]
    unique_bbs_df = unique_bbs_artifact.df.copy()
    print("unique_bbs_df", unique_bbs_df)
    # unique_bbs_df

    if filter_funcs is not None and len(filter_funcs) > 0:
        unique_bbs_df = unique_bbs_df[unique_bbs_df["func"].isin(filter_funcs)]

    unique_bbs_df["func_bb_idx"] = unique_bbs_df.groupby("func").cumcount()
    unique_bbs_df["runtime_weight"] = unique_bbs_df["freq"] * unique_bbs_df["num_instrs"]

    if sort_by is not None:
        unique_bbs_df.sort_values(sort_by, inplace=True, ascending=sort_ascending)
    if topk is not None:
        unique_bbs_df = unique_bbs_df.iloc[:topk]
    print("unique_bbs_df", unique_bbs_df)

    print("bb_trace_df", bb_trace_df)
    # TODO: expose strategy for selecting which call (first, last, first+last, random, single, multiple,...)
    # per_bb_trace_df = bb_trace_df.groupby("bb_idx").first()
    per_bb_trace_df = bb_trace_df.groupby("bb_idx").last()
    print("per_bb_trace_df", per_bb_trace_df)

    merged_df = pd.merge(per_bb_trace_df, unique_bbs_df, how="inner", left_index=True, right_index=True)
    KEEP_COLS = ["bb_call", "trace_idx", "num_instrs", "size", "func", "func_bb_idx"]
    if sort_by is not None:
        KEEP_COLS.append(sort_by)
        merged_df.sort_values(sort_by, inplace=True, ascending=sort_ascending)
    merged_df = merged_df[KEEP_COLS]
    print("merged_df", merged_df)

    ranges_data = []

    for bb_idx, bb_row in merged_df.iterrows():
        func = bb_row.func
        start = bb_row.trace_idx
        num_instrs = bb_row.num_instrs
        func_bb_idx = bb_row.func_bb_idx
        call = bb_row.bb_call
        end = start + num_instrs - 1
        # name = f"{func}@bb{func_bb_idx}-BB{bb_idx}-I{call}"
        # TODO: add func_bb_idx to artifact!
        name = f"bb{func_bb_idx}@{func}-BB{bb_idx}-I{call}"
        new = [name, start, end]
        ranges_data.append(new)

    yaml_data = {"ranges": ranges_data}

    if output is None:
        profile_dir = sess.directory / "output"
        profile_dir.mkdir(exist_ok=True)
        out_name = "ranges.yml"
        output = profile_dir / out_name
    if Path(output).is_file():
        assert force, f"Output file '{output}' already exists. Use --force to override!"
    with open(output, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_ranges_yaml(
        sess,
        output=args.output,
        force=args.force,
        filter_funcs=args.func_filter.split(",") if args.func_filter is not None else None,
        sort_by=args.sort_by,
        sort_ascending=args.ascending,
        topk=args.topk,
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--func-filter", type=str, default=None)  # comma-separated
    parser.add_argument("--sort-by", type=str, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--ascending", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
