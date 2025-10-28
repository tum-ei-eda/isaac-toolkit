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
# Parts of the following code have been provided by MINRES Technologies.
#

import argparse
import logging
import os.path as op
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union
from collections import defaultdict

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import filter_artifacts


def create_output(histogram, outputfile):
    """
    Write the histogram to the specified outputfile in lcov format
    """
    with open(outputfile, "w") as f:
        for file, lst in histogram.items():
            f.write(f"SF:{file}\n")
            executed = lst[0]
            functions = lst[1]
            # print("file", file)
            # print("executed", executed)
            # print("functions", functions)
            # input("*")
            for line, count in executed.items():
                if line in functions:
                    fcount = functions[line][0]
                    name = functions[line][1]
                    f.write(f"FN:{line},{name}\n")
                    f.write(f"FNDA:{fcount},{name}\n")
                f.write(f"DA:{line},{count}\n")
            f.write("end_of_record\n")


def generate_lcov_output(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    binary: bool = False,
    genhtml: Optional[Union[str, Path]] = None,
    force: bool = False,
):
    del force  # TODO: use
    artifacts = sess.artifacts

    func2pc_artifacts = filter_artifacts(artifacts, lambda x: x.name == "func2pc")
    assert len(func2pc_artifacts) == 1
    func2pc_artifact = func2pc_artifacts[0]
    func2pc_df = func2pc_artifact.df

    pcs_hist_artifacts = filter_artifacts(artifacts, lambda x: x.name == "pcs_hist")
    assert len(pcs_hist_artifacts) == 1
    pcs_hist_artifact = pcs_hist_artifacts[0]
    pcs_hist_df = pcs_hist_artifact.df

    pc2locs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "pc2locs")
    assert len(pc2locs_artifacts) == 1
    pc2locs_artifact = pc2locs_artifacts[0]
    pc2locs_df = pc2locs_artifact.df

    # Build IntervalIndex
    if "start" not in func2pc_df.columns:
        func2pc_df[["start", "end"]] = func2pc_df["pc_range"].apply(pd.Series)
    intervals = pd.IntervalIndex.from_arrays(func2pc_df["start"], func2pc_df["end"], closed="both")
    func_names = func2pc_df["func"].values

    # Map function to a PC
    def get_func_name(pc):
        matches = intervals.contains(pc)
        if matches.any():
            return func_names[matches.argmax()]  # first matching interval
        return None

    # Annotate pcs_hist_df
    func_agg = "all"
    if func_agg == "all":
        pcs_hist_df["func_name"] = pcs_hist_df["pc"].apply(get_func_name)
    elif func_agg == "first":
        raise NotImplementedError(f"func_agg: {func_agg}")
    else:
        raise ValueError(f"Invalid func_agg: {func_agg}")

    pc2locs_df["locs"] = pc2locs_df["locs"].apply(list)

    # Extract the loc with the smallest line number
    def pick_min_loc(locs):
        # locs is a list like ["src/foo.c:260", "src/foo.c:389"]
        return min(locs, key=lambda x: int(x.split(":")[1]))

    pc2locs_df["locs"] = pc2locs_df["locs"].apply(pick_min_loc)

    merged_df = pcs_hist_df.merge(pc2locs_df, on="pc", how="left")
    # Sort by function and pc
    merged_df = merged_df.sort_values(["func_name", "pc"]).reset_index(drop=True)
    # Forward-fill locs *within each function*
    merged_df["locs"] = merged_df.groupby("func_name")["locs"].ffill()

    # Now group by locs and sum counts
    locs_hist_df = (
        merged_df.groupby(["locs", "func_name"], dropna=False)
        .agg(count=("count", "sum"), rel_count=("rel_count", "sum"))
        .reset_index()
    )
    # print("locs_hist_df", locs_hist_df)

    # Sort by count if desired
    locs_hist_df = locs_hist_df.sort_values("count", ascending=False)
    # If locs are strings like "src/core_list_join.c:260"
    # Split into file and line
    locs_hist_df[["file", "line"]] = locs_hist_df["locs"].str.split(":", n=1, expand=True)
    locs_hist_df = locs_hist_df.drop(columns=["locs", "rel_count"])
    locs_hist_df["line"] = locs_hist_df["line"].fillna(0)
    locs_hist_df["line"] = locs_hist_df["line"].astype(int)

    # Structure: {filename: [executed_dict, functions_dict]}
    # executed_dict: line → count
    # functions_dict: line → [count, name]
    histogram = defaultdict(lambda: [defaultdict(int), defaultdict(lambda: [0, ""])])
    for _, row in locs_hist_df.iterrows():
        file = row["file"]
        line = row["line"]
        count = row["count"]
        func_name = row["func_name"]

        # Increment executed count
        if binary:
            histogram[file][0][line] = 1
        else:
            histogram[file][0][line] += count

        # If function info is available
        if func_name is not None:
            if binary:
                histogram[file][1][line][0] = 1
            else:
                histogram[file][1][line][0] += count
            histogram[file][1][line][1] = func_name  # store function name

    if output is None:
        profile_dir = sess.directory / "profile"
        profile_dir.mkdir(exist_ok=True)
        out_name = "coverage.info"
        output = profile_dir / out_name
        print("genhtml", genhtml)
        if genhtml is None:  # TODO: make optional?
            genhtml = profile_dir / "html"

    create_output(
        histogram,
        output,
    )

    if binary:
        logging.info("Displaying binary coverage")

    if op.getsize("coverage.info") == 0:
        logging.critical("Something went wrong, couldn't generate output")
        sys.exit(1)

    if genhtml:
        logging.info(f"Output directory set to {genhtml}")
        try:
            subprocess.run(
                [
                    "genhtml",
                    "-o",
                    genhtml,
                    "-q",
                    output,
                    "--ignore-errors",
                    "source",
                ],
                check=True,
            )
        except FileNotFoundError:
            logging.error('Could not find "genhtml", is the lcov package installed?')
            sys.exit(1)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_lcov_output(
        sess,
        output=args.output,
        genhtml=args.genhtml,
        binary=args.binary,
        force=args.force,
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
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument(
        "--genhtml",
        metavar="<directory>",
        nargs="?",
        default=None,  # TODO: ?
        help=(
            "calls the genhtml command from lcov, will generate a directory for the "
            "generated output if not specified otherwise"
        ),
    )
    parser.add_argument(
        "--binary",
        "-b",
        action="store_true",
        default=False,
        help="toggle boolean coverage",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="<outputfile>",
        nargs="?",
        default=None,
        help="specify the putput file",
    )
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
