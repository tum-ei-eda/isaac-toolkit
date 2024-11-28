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


# def find_sub_bbs(bbs, first_pc, last_pc):
#     ret = set()
#     for bb_ in bbs:
#         if bb_.first_pc >= first_pc and bb_.last_pc <= last_pc:
#             ret.add(bb_)
#     return ret
#
# def find_super_bbs(bbs, first_pc, last_pc):
#     ret = set()
#     for bb_ in bbs:
#         if bb_.first_pc <= first_pc and bb_.last_pc >= last_pc:
#             ret.add(bb_)
#     return ret
#
# def get_diff_range(bb, first_pc, last_pc):
#     if bb.last_pc == last_pc:
#         first_pc = first_pc
#         last_pc = bb.first_pc - 4  # TODO: this breaks for sz=2
#     else:
#         raise NotImplementedError
#     return (first_pc, last_pc)


class BasicBlock(object):
    _instances = {}
    # _children = defaultdict(list)
    # _parents = defauktdict(set)

    @staticmethod
    def get_instances():
        return list(BasicBlock._instances.values())

    def __new__(cls, first_pc: int, last_pc: int, end_instr: str, func: str):
        # print("__new__", first_pc, last_pc, end_instr, func)
        # print("_instances", BasicBlock._instances)
        key = (first_pc, last_pc, end_instr, func)
        # print("key", key)
        instance = cls._instances.get(key)
        # children = cls._children.get(key)
        # print("instance", instance)
        # print("children", children)
        if instance is None:
            # if children is not None:
            #     assert len(children) > 1
            #     for x in children:
            #         x._freq += 1
            #     return children[0]
            # sub_bbs = find_sub_bbs(cls.get_instances(), first_pc, last_pc)
            # new_children = []
            # if len(sub_bbs) > 0:
            #     print("sub_bbs", sub_bbs)
            #     assert len(sub_bbs) == 1
            #     sub_bb = list(sub_bbs)[0]
            #     new_children.append(sub_bb)
            #     sub_bb._freq += 1
            #     diff_range = get_diff_range(sub_bb, first_pc, last_pc)
            #     print("diff_range", diff_range)
            #     # new_bb = BasicBlock(first_pc=diff_range[0], last_pc=diff_range[1], end_instr="?", func=None)
            #     # new_bb._freq += sub_bb._freq
            #     first_pc = diff_range[0]
            #     last_pc = diff_range[1]
            #     end_instr = "?"
            #     func = None
            #     # print("new_bb", new_bb)
            #     input("SUB")
            #     pass
            # super_bbs = find_sub_bbs(cls.get_instances(), first_pc, last_pc)
            # if len(super_bbs) > 0:
            #     print("super_bbs", super_bbs)
            #     assert len(super_bbs) == 1
            #     input("SUPER")
            #     pass
            instance = super().__new__(cls)
            # new_children.append(instance)
            # if len(new_children) > 1:
            #     for x in new_children:
            #         cls._children[key].append(x)
            #     for x in new_children[:-1]:
            #         cls._parents[key].append(instance)
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

    def __repr__(self) -> str:
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
    # print("trace_df", len(trace_df))
    # input("{}{}{}{}")
    first_pc = None
    # TODO: make this generic!
    branch_instrs = [
        "j",  # pseudo
        "jr",  # pseudo
        "ret",  # pseudo
        "mret",  # pseudo
        "sret",  # pseudo
        "uret",  # pseudo
        "call",  # pseudo
        "tail",  # pseudo
        "jalr",
        "jal",
        "beq",
        "beqz",  # pseudo
        "bne",
        "blt",
        "bltz",  # pseudo
        "bgt",  # pseudo
        "bgtz",  # pseudo
        "bge",  # pseudo
        "bgez",  # pseudo
        "ble",
        "bltu",
        "bgtu",  # pseudo
        "bgeu",  # pseudo
        "bleu",
        "ecall",
        "bnez",  # bseudo
        "cbnez",
        "c.bnez",
        "cjr",
        "cj",  # pseudo
        "cbeqz",
        "cjalr",
        "cjal",
        "c.j",
        "c.jr",
        "c.j",
        "c.beqz",
        "c.jalr",
        "c.jal",
    ]
    # bbs = []
    prev_pc = None
    prev_size = None
    prev_instr = None
    for row in trace_df.itertuples(index=False):
        # print("row", row)
        pc = row.pc
        instr = row.instr
        instr = instr.strip()  # TODO: fix in frontend
        # def detect_compressed(name):
        #     print("name", name)
        #     if name[0] == "c":
        #         return True
        #     return False
        # is_compressed = detect_compressed(instr)
        is_compressed = row.size == 2
        sz = 2 if is_compressed else 4
        # print("sz", sz)

        if prev_pc:
            # print("pc", pc)
            # print("prev_pc", prev_pc)
            # print("prev_size", prev_size)
            step = pc - prev_pc
            # print("step", step)
            if step in [2, 4] and step == prev_size:
                pass
            else:
                # is_jump = True
                if first_pc is None:
                    pass
                else:
                    # assert False, f"Sub basic block not found at: pc = {prev_pc:x} -> {pc:x}"
                    logger.warning("Detected potential trap @ pc = 0x%x -> 0x%x", prev_pc, pc)
                    # input("OOPS")
                    if True:
                        func = None
                        bb = BasicBlock(first_pc=first_pc, last_pc=prev_pc, end_instr=instr, func=func)
                        first_pc = pc
                        # if bb.get_freq() == 1:
                        #     bbs.append(bb)

        # At the first pc of a basic block
        if first_pc is None:
            first_pc = pc

        if instr in branch_instrs:
            # func = self.find_func_name(pc)
            func = None
            bb = BasicBlock(first_pc=first_pc, last_pc=pc, end_instr=instr, func=func)
            # self.func_set.add(func)
            # if bb.get_freq() == 1:
            #     # sub_bbs = find_overlapping_bbs(bb, bbs)
            #     # print("sub_bbs", sub_bbs)
            #     # if len(sub_bbs) > 0:
            #     #     assert len(sub_bbs) == 1
            #     #     for sub_bb in sub_bbs:
            #     #         sub_bb._freq += bb._freq
            #     #         diff_range = get_diff_range(bb, sub_bb)
            #     #         print("diff_range", diff_range)
            #     #         new_bb = BasicBlock(first_pc=diff_range[0], last_pc=diff_range[1], end_instr="?", func=None)
            #     #         new_bb._freq = bb._freq
            #     #         BasicBlock.instances
            #     #         print("new_bb", new_bb)
            #     #     input("OVERLAP")
            #     bbs.append(bb)
            first_pc = None
        prev_pc = pc
        prev_instr = instr
        prev_size = sz
    if first_pc is not None:
        func = None
        # bb = BasicBlock(first_pc=first_pc, last_pc=prev_pc, end_instr=instr, func=func)
        # if bb.get_freq() == 1:
        #     bbs.append(bb)
    bbs = BasicBlock.get_instances()
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
    # print("bbs_df", bbs_df)

    return bbs_df


def analyze_basic_blocks(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    # print("elf_artifacts", elf_artifacts)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]

    pc2bb = collect_bbs(trace_artifact.df)
    # print("pc2bb", pc2bb)
    func2pc_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "func2pc")
    if len(func2pc_artifacts) > 0:
        assert len(func2pc_artifacts) == 1
        func2pc_artifact = func2pc_artifacts[0]
        func2pc_df = func2pc_artifact.df.copy()
        func2pc_df[["start", "end"]] = func2pc_df["pc_range"].apply(pd.Series)

        # print("func2pc_df", func2pc_df)
        def helper(x):
            # print("x", x)
            # print("!", func2pc_df["start"] >= x[0] & func2pc_df["end"] <= x[1])
            matches = func2pc_df[func2pc_df["start"] <= x[0]]
            matches = matches[matches["end"] >= x[1]]
            # print("matches", matches)
            if len(matches) == 0:
                return None
            # print("len(matches)", len(matches))
            # assert len(matches) == 1
            # match_ = matches.iloc[0]
            # func = match_["func"]
            # TODO: assert that alias!
            funcs = set(matches["func"].values)
            # return func
            return funcs

        ANNOTATE_FUNC = True
        if ANNOTATE_FUNC:
            pc2bb["func_name"] = pc2bb["bb"].apply(helper)  # .astype("category")
        pc2bb[["start", "end"]] = pc2bb["bb"].apply(pd.Series)
        del pc2bb["bb"]
        # print("pc2bb", pc2bb)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "mapping",
        "by": __name__,
    }

    pc2bb_artifact = TableArtifact(f"pc2bb", pc2bb, attrs=attrs)
    sess.add_artifact(pc2bb_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_basic_blocks(sess, force=args.force)
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
