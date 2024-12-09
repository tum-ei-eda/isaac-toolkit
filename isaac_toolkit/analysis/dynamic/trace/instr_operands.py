#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
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
import re
import sys
import logging
import argparse
from typing import Optional
from math import ceil, log2
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def collect_operands(trace_df):
    instrs_operands = defaultdict(list)
    for row in trace_df.itertuples(index=False):
        # pc = row.pc
        instr = row.instr
        instr = instr.strip()  # TODO: fix in frontend
        operands = row.operands
        instr_operands = instrs_operands[instr].append(operands)
    operands_data = []
    operand_names = set()
    for instr, instr_operands in instrs_operands.items():
        for operands in instr_operands:
            operand_names |= set(operands.keys())
            operands_data.append({"instr": instr, **operands})
    operands_df = pd.DataFrame(operands_data)
    operands_df["instr"] = operands_df["instr"].astype("category")
    for op in operand_names:
        operands_df[op] = operands_df[op].astype("UInt32")
        # TODO: for smaller immediates, use smaller types?

    return operands_df


def analyze_instr_operands(
    sess: Session,
    force: bool = False,
    imm_only: bool = True,
    signed: bool = False,
    subplots: bool = False,
    plot_fmt: str = "pdf",
    topk: int = 20,
    filter_instrs: Optional[str] = None,
    filter_operands: Optional[str] = None,
):
    artifacts = sess.artifacts
    trace_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE
    )
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    # filter_instrs = "addi"
    # filter_operands = "imm"

    operands_df = collect_operands(trace_artifact.df)

    operand_names = sorted([x for x in operands_df.columns if x != "instr"])

    to_drop = {"rd", "rs1", "rs2", "rs3"} if imm_only else set()
    if filter_operands is not None:
        keep = set(
            filter(lambda x: re.compile(filter_operands).match(x), operand_names)
        )
        drop = set(operand_names) - keep
        to_drop |= drop
        operand_names = sorted(list(keep))
    if to_drop:
        for col in to_drop:
            if col in operands_df.columns:
                operands_df.drop(columns=[col], inplace=True)
    instrs = sorted(operands_df["instr"].unique())

    to_keep = instrs
    if filter_instrs is not None:
        to_keep = list(filter(lambda x: re.compile(filter_instrs).match(x), instrs))

    operand_names = sorted([x for x in operands_df.columns if x != "instr"])
    to_keep = [
        instr_name
        for instr_name in to_keep
        if not pd.isna(operands_df[operands_df["instr"] == instr_name][operand_names])
        .all()
        .all()
    ]

    if len(to_keep) < len(instrs):
        res = operands_df[operands_df["instr"].astype(str).isin(to_keep)]
        operands_df = res

    instrs = sorted(operands_df["instr"].unique())

    attrs = {
        "trace": trace_artifact.name,
        "kind": "table",
        "by": __name__,
    }
    operands_artifact = TableArtifact("instr_operands", operands_df, attrs=attrs)
    sess.add_artifact(operands_artifact, override=force)

    plot = True
    if plot:
        plots_dir = sess.directory / "plots"
        plots_dir.mkdir(exist_ok=True)

    operands_hist_data = []
    # TODO: share code with below
    # TODO: split artifact creation and plotting
    plot_data = defaultdict(dict)
    for i, instr_name in enumerate(instrs):
        instr_df = operands_df[operands_df["instr"] == instr_name]
        for j, operand_name in enumerate(operand_names):
            if (
                operand_name not in instr_df.columns
                or pd.isna(instr_df[operand_name]).all()
            ):
                continue
            counts = instr_df[operand_name].value_counts()
            bit_counts = (
                instr_df[operand_name]
                .apply(lambda x: max(1, ceil(log2(1 + x))))
                .value_counts()
            )
            plot_data[operand_name][instr_name] = (counts, bit_counts)
            operands_hist_data.append(
                {
                    "instr": instr_name,
                    "op": operand_name,
                    "counts": counts.to_dict(),
                    "bit_counts": bit_counts.to_dict(),
                }
            )

    if plot:
        num_operands = len(operand_names)
        assert num_operands > 0
        if subplots:
            num_instrs = len(instrs)
            assert num_instrs > 0
            # VALUES
            fig, axes = plt.subplots(
                nrows=num_operands,
                ncols=num_instrs,
                figsize=(3 * num_instrs, 2 * num_operands),
            )
            plt.tight_layout()
            if num_instrs == 1 and num_operands == 1:
                axes = [[axes]]
            elif num_instrs == 1 or num_operands == 1:
                axes = [axes]
            for i, operand_name in enumerate(operand_names):
                for j, instr_name in enumerate(instrs):
                    axes[i][j].axes.xaxis.set_ticklabels([])
                    axes[i][j].axes.yaxis.set_ticklabels([])
                    axes[i][j].tick_params(axis="x", labelsize=8)
                    axes[i][j].tick_params(axis="y", labelsize=8)
            for operand_name, plot_data_ in plot_data.items():
                i = operand_names.index(operand_name)
                for instr_name, plot_data__ in plot_data_.items():
                    j = instrs.index(instr_name)
                    val_counts = plot_data__[0]
                    val_counts.head(n=topk).plot(ax=axes[i][j], kind="bar")
                    axes[i][j].set_xlabel("", fontsize=8)
            for i, operand_name in enumerate(operand_names):
                axes[i][0].set_ylabel(operand_name, fontsize=8)
            for j, instr_name in enumerate(instrs):
                axes[0][j].set_title(instr_name)
                axes[-1][j].set_xlabel("Value", fontsize=8)
            plot_file = plots_dir / f"operands-hist.{plot_fmt}"
            fig.savefig(plot_file, bbox_inches="tight")
            plt.close()
            # BITS
            fig, axes = plt.subplots(
                nrows=num_operands,
                ncols=num_instrs,
                figsize=(3 * num_instrs, 2 * num_operands),
            )
            plt.tight_layout()
            if num_instrs == 1 and num_operands == 1:
                axes = [[axes]]
            elif num_instrs == 1 or num_operands == 1:
                axes = [axes]
            for i, operand_name in enumerate(operand_names):
                for j, instr_name in enumerate(instrs):
                    axes[i][j].axes.xaxis.set_ticklabels([])
                    axes[i][j].axes.yaxis.set_ticklabels([])
                    axes[i][j].tick_params(axis="x", labelsize=8)
                    axes[i][j].tick_params(axis="y", labelsize=8)
            for operand_name, plot_data_ in plot_data.items():
                i = operand_names.index(operand_name)
                for instr_name, plot_data__ in plot_data_.items():
                    j = instrs.index(instr_name)
                    bit_counts = plot_data__[1]
                    bit_counts.head(n=topk).plot(ax=axes[i][j], kind="bar")
                    axes[i][j].set_xlabel("", fontsize=8)
            for i, operand_name in enumerate(operand_names):
                axes[i][0].set_ylabel(operand_name, fontsize=8)
            for j, instr_name in enumerate(instrs):
                axes[0][j].set_title(instr_name)
                axes[-1][j].set_xlabel("#Bits", fontsize=8)
            plot_file = plots_dir / f"operands-hist-bits.{plot_fmt}"
            fig.savefig(plot_file, bbox_inches="tight")
            plt.close()
        else:
            plot_data2 = {
                instr_name: {
                    operand_name: plot_data[operand_name][instr_name]
                    for operand_name in operand_names
                    if operand_name in plot_data
                    and instr_name in plot_data[operand_name].keys()
                }
                for instr_name in instrs
            }
            for instr_name, plot_data_ in plot_data2.items():
                operand_names_ = sorted(list(plot_data_.keys()))
                num_operands_ = len(operand_names_)
                if num_operands_ == 0:
                    continue
                # VALUES
                fig, axes = plt.subplots(
                    nrows=num_operands_, ncols=1, figsize=(10, 2 * num_operands)
                )
                plt.tight_layout()
                plot_data = {}
                if num_operands_ == 1:
                    axes = [axes]
                for operand_name, plot_data__ in plot_data_.items():
                    i = operand_names_.index(operand_name)
                    val_counts = plot_data__[0]
                    val_counts.head(n=topk).plot(ax=axes[i], kind="bar")
                    axes[i].tick_params(axis="x", labelsize=8)
                    axes[i].tick_params(axis="y", labelsize=8)
                    axes[i].set_ylabel(operand_name, fontsize=8)
                    axes[i].set_xlabel("", fontsize=8)
                axes[0].set_title(instr_name)
                axes[-1].set_xlabel("Value", fontsize=8)
                plot_file = plots_dir / f"operands-hist-{instr_name}.{plot_fmt}"
                fig.savefig(plot_file, bbox_inches="tight")
                plt.close()
                # BITS
                fig, axes = plt.subplots(
                    nrows=num_operands_, ncols=1, figsize=(10, 2 * num_operands)
                )
                plt.tight_layout()
                plot_data = {}
                if num_operands_ == 1:
                    axes = [axes]
                for operand_name, plot_data__ in plot_data_.items():
                    i = operand_names_.index(operand_name)
                    bit_counts = plot_data__[1]
                    bit_counts.head(n=topk).plot(ax=axes[i], kind="bar")
                    axes[i].tick_params(axis="x", labelsize=8)
                    axes[i].tick_params(axis="y", labelsize=8)
                    axes[i].set_ylabel(operand_name, fontsize=8)
                    axes[i].set_xlabel("", fontsize=8)
                axes[0].set_title(instr_name)
                axes[-1].set_xlabel("#Bits", fontsize=8)
                plot_file = plots_dir / f"operands-hist-{instr_name}-bits.{plot_fmt}"
                fig.savefig(plot_file, bbox_inches="tight")
                plt.close()
    operands_hist_df = pd.DataFrame(operands_hist_data)

    attrs2 = {
        "trace": trace_artifact.name,
        "kind": "histogram",
        "by": __name__,
    }

    operands_hist_artifact = TableArtifact(
        "instr_operands_hist", operands_hist_df, attrs=attrs2
    )
    sess.add_artifact(operands_hist_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_instr_operands(
        sess,
        force=args.force,
        signed=args.signed,
        imm_only=args.imm_only,
        plot_fmt=args.fmt,
        topk=args.topk,
        filter_instrs=args.filter_instrs,
        filter_operands=args.filter_operands,
        subplots=args.subplots,
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
    parser.add_argument("--signed", action="store_true")
    parser.add_argument("--imm-only", action="store_true")
    parser.add_argument("--subplots", action="store_true")
    parser.add_argument("--fmt", type=str, default="pdf")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--filter-instrs", type=str, default=None)
    parser.add_argument("--filter-operands", type=str, default=None)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
