import sys
import logging
import argparse
import posixpath
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def collect_operands(trace_df):
    # print("trace_df", trace_df)
    instrs_operands = defaultdict(list)
    for row in trace_df.itertuples(index=False):
        # print("row", row)
        pc = row.pc
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

    return operands_df


def analyze_instr_operands(sess: Session, force: bool = False):
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
    for instr_name, instr_df in operands_df.groupby("instr"):
        for operand_name in operand_names:
            if operand_name not in instr_df.columns or pd.isna(instr_df[operand_name]).all():
                continue
            # print(f"{instr_name} & {operand_name}")
            counts = instr_df[operand_name].value_counts()
            # print("counts", counts)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "table",
        "by": __name__,
    }
    attrs2 = {
        "trace": trace_artifact.name,
        "kind": "histogram",
        "by": __name__,
    }

    operands_artifact = TableArtifact(f"instr_operands", operands_df, attrs=attrs)
    # TODO:
    # operands_hist_artifact = TableArtifact(f"instr_operands_hist", operands_hist_df, attrs=attrs2)
    sess.add_artifact(operands_artifact, override=force)
    # sess.add_artifact(operands_hist_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_instr_operands(sess, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
