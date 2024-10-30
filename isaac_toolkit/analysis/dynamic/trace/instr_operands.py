import sys
import logging
import argparse
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
    # print("trace_df", trace_df)
    instrs_operands = defaultdict(list)
    for row in trace_df.itertuples(index=False):
        # print("row", row)
        # pc = row.pc
        instr = row.instr
        instr = instr.strip()  # TODO: fix in frontend
        operands = row.operands
        instr_operands = instrs_operands[instr].append(operands)
    # print("instrs_operands", instrs_operands)
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
    plot_fmt: str = "pdf",
    topk: int = 20,
):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    # print("elf_artifacts", elf_artifacts)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]

    operands_df = collect_operands(trace_artifact.df)
    # print("operands_df", operands_df)
    operand_names = [x for x in operands_df.columns if x != "instr"]
    for operand_name in operand_names:
        # print(f"ALL & {operand_name}")
        counts = operands_df[operand_name].value_counts()
        # print("counts", counts)
    if imm_only:
        to_drop = ["rd", "rs1", "rs2", "rs3"]  # TODO: use regex instead
        for col in to_drop:
            if col in operands_df.columns:
                operands_df.drop(columns=[col], inplace=True)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "table",
        "by": __name__,
    }
    operands_artifact = TableArtifact("instr_operands", operands_df, attrs=attrs)
    sess.add_artifact(operands_artifact, override=force)

    plot = True
    print_ = False
    if plot:
        plots_dir = sess.directory / "plots"
        plots_dir.mkdir(exist_ok=True)

    operands_hist_data = []
    filter_instrs = None
    filter_ops = None
    for instr_name, instr_df in operands_df.groupby("instr"):
        if filter_instrs is not None and instr_name not in filter_instrs:
            continue
        plot_data = {}
        for operand_name in operand_names:
            if operand_name not in instr_df.columns or pd.isna(instr_df[operand_name]).all():
                continue
            if filter_ops is not None and operand_name not in filter_ops:
                continue
            print("instr_df", instr_df)
            counts = instr_df[operand_name].value_counts()
            print("counts", counts)
            bit_counts = instr_df[operand_name].apply(lambda x: max(1, ceil(log2(1 + x)))).value_counts()
            print("bit_counts", bit_counts)
            # input(">")
            plot_data[operand_name] = (counts, bit_counts)
            if print_:
                print(f"{instr_name} & {operand_name}")
                print("counts", counts)
                input(">!")
            operands_hist_data.append(
                {
                    "instr": instr_name,
                    "op": operand_name,
                    "counts": counts.to_dict(),
                    "bit_counts": bit_counts.to_dict(),
                }
            )
        if plot and len(plot_data) > 0:
            fig, axes = plt.subplots(nrows=len(plot_data), ncols=1)
            plt.tight_layout()
            if len(plot_data) == 1:
                axes = [axes]
            i = 0
            for op, op_counts in plot_data.items():
                val_counts = op_counts[0]
                val_counts.head(n=topk).plot(ax=axes[i], kind="bar")
                axes[i].tick_params(axis="x", labelsize=8)
                axes[i].tick_params(axis="y", labelsize=8)
                axes[i].set_ylabel(op, fontsize=8)
                if i == len(plot_data) - 1:
                    axes[i].set_xlabel("Value", fontsize=8)
                i += 1
            axes[0].set_title(instr_name)
            plot_file = plots_dir / f"operands-hist-{instr_name}.{plot_fmt}"
            fig.savefig(plot_file, bbox_inches="tight")
            plt.close()
            fig, axes = plt.subplots(nrows=len(plot_data), ncols=1)
            plt.tight_layout()
            if len(plot_data) == 1:
                axes = [axes]
            i = 0
            for op, op_counts in plot_data.items():
                bit_counts = op_counts[1]
                bit_counts.head(n=topk).plot(ax=axes[i], kind="bar")
                axes[i].tick_params(axis="x", labelsize=8)
                axes[i].tick_params(axis="y", labelsize=8)
                axes[i].set_ylabel(op, fontsize=8)
                if i == len(plot_data) - 1:
                    axes[i].set_xlabel("#Bits", fontsize=8)
                i += 1
            axes[0].set_title(instr_name)
            plot_file = plots_dir / f"operands-hist-{instr_name}-bits.{plot_fmt}"
            fig.savefig(plot_file, bbox_inches="tight")
            plt.close()
    operands_hist_df = pd.DataFrame(operands_hist_data)

    attrs2 = {
        "trace": trace_artifact.name,
        "kind": "histogram",
        "by": __name__,
    }

    operands_hist_artifact = TableArtifact("instr_operands_hist", operands_hist_df, attrs=attrs2)
    sess.add_artifact(operands_hist_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_instr_operands(
        sess, force=args.force, signed=args.signed, imm_only=args.imm_only, plot_fmt=args.fmt, topk=args.topk
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--signed", action="store_true")
    parser.add_argument("--imm-only", action="store_true")
    parser.add_argument("--fmt", type=str, default="pdf")
    parser.add_argument("--topk", type=int, default=20)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
