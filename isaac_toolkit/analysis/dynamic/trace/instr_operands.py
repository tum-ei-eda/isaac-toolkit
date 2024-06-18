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


def collect_bbs(trace_df):
    print("trace_df", trace_df)
    first_pc = None
    # TODO: make this generic!
    branch_instrs = ["jalr", "jal", "beq", "bne", "blt", "bge", "bltu", "bgeu", "ecall"]
    bbs = []
    prev_pc = None
    prev_instr = None
    for row in trace_df.iterrows():
        pc = row["pc"]
        instr = row["instr"]
        sz = 4  # TODO: generalize
        if prev_pc:
            step = pc - prev_pc
            if step in [2, 4]:
                assert step == sz
            else:
                # is_jump = True
                if first_pc is None:
                    pass
                else:
                    assert False, f"Sub basic block not found at: pc = {prev_pc:x} -> {pc:x}"
                    func = None
                    bb = BasicBlock(first_pc=first_pc, last_pc=prev_pc, end_instr=instr, func=func)
                    first_pc = pc
                    bbs.append(bb)

        # At the first pc of a basic block
        if first_pc is None:
            first_pc = pc

        if instr in branch_instrs:
            # func = self.find_func_name(pc)
            func = None
            bb = BasicBlock(first_pc=first_pc, last_pc=pc, end_instr=instr, func=func)
            # self.func_set.add(func)
            bbs.append(bb)
            # if bb.get_freq() == 1:
            #     self.func_BB_dict[bb.func].append(bb)

            # Hard coded where mlonmcu_exit() calls exit() explicitly
            # Note that in mlonmcu main() does never return
            # if pc == int("0xe0", 0):
            #     break
            first_pc = None
        prev_pc = pc
        prev_instr = instr
        prev_size = sz
    if first_pc is not None:
        func = None
        bb = BasicBlock(first_pc=first_pc, last_pc=prev_pc, end_instr=instr, func=func)
        bbs.append(bb)
    print("bbs", bbs)
    input("zzz")

    return None


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    override = args.force
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    # print("elf_artifacts", elf_artifacts)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]

    pc2bb = collect_bbs(trace_artifact.df)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "mapping",
        "by": __name__,
    }

    pc2bb_artifact = TableArtifact(f"pc2bb", footprint_df, attrs=attrs)
    sess.add_artifact(pc2bb_artifact, override=override)
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
