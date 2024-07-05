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


class BasicBlock(object):
    _instances = {}

    def __new__(cls, first_pc: int, last_pc: int, end_instr: str, func: str):
        # print("__new__", first_pc, last_pc, end_instr, func)
        # print("_instances", BasicBlock._instances)
        key = (first_pc, last_pc, end_instr, func)
        # print("key", key)
        instance = cls._instances.get(key)
        # print("instance", instance)
        if instance is None:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            instance.__initialized = False
            instance._freq = 0
        instance._freq += 1
        return instance

    def __init__(self, first_pc: int, last_pc: int, end_instr: str, func: str) -> None:
        if not self.__initialized:
            self.first_pc = first_pc
            self.last_pc = last_pc
            self.func = func
            self.end_instr = end_instr
            self.__initialized = True

    def __str__(self) -> str:
        return f"start:{hex(self.first_pc)}, end:{hex(self.last_pc)}, end_instr:{self.end_instr}, func:{self.func}\n"

    def __eq__(self, other) -> bool:
        # print("__eq__", self, other, end="")
        if not isinstance(other, BasicBlock):
            # print(" -> False1")
            return False
        ret = self.first_pc == other.first_pc and self.last_pc == other.last_pc and self.func == other.func
        # print(f" -> {ret}2")
        return ret

    def __hash__(self) -> int:
        return hash((self.first_pc, self.last_pc, self.end_instr, self.func))

    def get_freq(self) -> int:
        return self._freq


def collect_bbs(trace_df):
    print("trace_df", trace_df)
    first_pc = None
    # TODO: make this generic!
    branch_instrs = ["jalr", "jal", "beq", "bne", "blt", "bge", "bltu", "bgeu", "ecall"]
    bbs = []
    prev_pc = None
    prev_instr = None
    for row in trace_df.itertuples(index=False):
        # print("row", row)
        pc = row.pc
        instr = row.instr
        instr = instr.strip()  # TODO: fix in frontend
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
                    # assert False, f"Sub basic block not found at: pc = {prev_pc:x} -> {pc:x}"
                    logger.warning("Sub basic block not found at: pc = %x -> %x", prev_pc, pc)
                    if False:
                        func = None
                        bb = BasicBlock(first_pc=first_pc, last_pc=prev_pc, end_instr=instr, func=func)
                        first_pc = pc
                        if bb.get_freq() == 1:
                            bbs.append(bb)

        # At the first pc of a basic block
        if first_pc is None:
            first_pc = pc

        if instr in branch_instrs:
            # func = self.find_func_name(pc)
            func = None
            bb = BasicBlock(first_pc=first_pc, last_pc=pc, end_instr=instr, func=func)
            # self.func_set.add(func)
            if bb.get_freq() == 1:
                bbs.append(bb)
            first_pc = None
        prev_pc = pc
        prev_instr = instr
        prev_size = sz
    if first_pc is not None:
        func = None
        bb = BasicBlock(first_pc=first_pc, last_pc=prev_pc, end_instr=instr, func=func)
        if bb.get_freq() == 1:
            bbs.append(bb)
    # print("bbs", bbs)
    bbs_data = []
    for bb in bbs:
        # print("bb", bb, dir(bb))
        start = bb.first_pc
        end = bb.last_pc
        freq = bb.get_freq()
        last_size = 4  # TODO: do not hardcode
        size = end - start + 4
        num = size // 4  # TODO
        bb_data = {"bb": (start, end), "freq": freq, "size": size, "num_instrs": num}
        bbs_data.append(bb_data)

    bbs_df = pd.DataFrame(bbs_data)
    bbs_df.sort_values("freq", inplace=True, ascending=False)
    bbs_df["weight"] = bbs_df["freq"] * bbs_df["num_instrs"]
    bbs_df["rel_weight"] = bbs_df["weight"] / sum(bbs_df["weight"])
    print("bbs_df", bbs_df)

    return bbs_df


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

    pc2bb_artifact = TableArtifact(f"pc2bb", pc2bb, attrs=attrs)
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
