import io
import sys
import leb128
import logging
import argparse
import posixpath
from typing import Optional
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("llvm_bbs")


def choose_bbs(
    sess: Session,
    threshold: float = 0.9,
    min_weight: float = 0.01,
    max_num: Optional[int] = None,
    force: bool = False,
):
    artifacts = sess.artifacts
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]
    llvm_bbs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs_new"
    )
    assert len(llvm_bbs_artifacts) == 1
    llvm_bbs_artifact = llvm_bbs_artifacts[0]
    llvm_bbs_df = llvm_bbs_artifact.df.copy()
    # print("llvm_bbs_df", llvm_bbs_df)
    sum_weights = 0.0
    choices = []
    for index, row in llvm_bbs_df.sort_values("rel_weight", ascending=False).iterrows():
        rel_weight = row["rel_weight"]
        if pd.isna(rel_weight):
            continue
        if rel_weight < min_weight:
            continue
        func_name = row["func_name"]
        bb_name = row["bb_name"]
        num_instrs = row["num_instrs"]
        freq = row["freq"]
        choice = {
            "func_name": func_name,
            "bb_name": bb_name,
            "rel_weight": rel_weight,
            "num_instrs": num_instrs,
            "freq": freq,
        }
        choices.append(choice)
        sum_weights += rel_weight
        if sum_weights >= threshold:
            break
        if max_num is not None:
            if len(choices) >= max_num:
                break
    # print("sum_weights", sum_weights)
    # print("choices", choices, len(choices))
    choices_df = pd.DataFrame(choices)
    # print("choices_df", choices_df)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "list",
        "by": __name__,
    }

    artifact = TableArtifact("choices", choices_df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    choose_bbs(
        sess,
        threshold=args.threshold,
        min_weight=args.min_weight,
        max_num=args.max_num,
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
    parser.add_argument("--min-weight", type=float, default=0.01)
    parser.add_argument("--max-num", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
