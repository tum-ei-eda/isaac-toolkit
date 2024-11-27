from typing import Dict, List, Set, Tuple, Optional

import bisect

import sys
import logging
import argparse
import posixpath
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile

from isaac_toolkit.session import Session
from isaac_toolkit.analysis.dynamic.trace.basic_blocks import BasicBlock  # TODO: move
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def find_func_name(mapping: Dict[str, Tuple[int, int]], pc: int) -> str:
    # TODO: refactor to use existing tables
    """
    Given a program counter, find the function it belongs to
    """
    for func, range in mapping.items():
        if pc >= range[0] and pc <= range[1]:
            return func


def collect_bbs(trace_df, mapping):
    first_pc = None
    # TODO: make this generic!
    branch_instrs = [
        "jalr",
        "jal",
        "beq",
        "bne",
        "blt",
        "bge",
        "bltu",
        "bgeu",
        "ecall",
        "cbnez",
        "cjr",
        "cj",
        "cbeqz",
        "cjalr",
        "cjal",
    ]
    func2bbs = defaultdict(list)  # TODO: only track func_set?
    bb_freq = defaultdict(int)
    prev_pc = None
    prev_size = None
    prev_instr = None
    for row in trace_df.itertuples(index=False):
        pc = row.pc
        instr = row.instr
        instr = instr.strip()  # TODO: fix in frontend
        is_compressed = row.size == 2
        sz = 2 if is_compressed else 4

        if prev_pc:
            step = pc - prev_pc
            if step in [2, 4] and step == prev_size:
                pass
            else:
                if first_pc is None:
                    pass
                else:
                    logger.warning("Sub basic block not found at: pc = %x -> %x", prev_pc, pc)
                    input("OOPS")
                    if False:
                        func = find_func_name(mapping, prev_pc)
                        bb = BasicBlock(first_pc=first_pc, last_pc=prev_pc, end_instr=instr, func=func)
                        if bb.get_freq() == 1:
                            func2bbs[bb.func].append(bb)
                        bb_freq[bb] += 1

                        first_pc = pc

        # At the first pc of a basic block
        if first_pc is None:
            first_pc = pc

        if instr in branch_instrs:
            func = find_func_name(mapping, pc)
            bb = BasicBlock(first_pc=first_pc, last_pc=pc, end_instr=instr, func=func)
            if bb.get_freq() == 1:
                func2bbs[bb.func].append(bb)
                bb_freq[bb] += 1
            first_pc = None
        prev_pc = pc
        prev_instr = instr
        prev_size = sz
    if first_pc is not None:
        func = None
    bbs = BasicBlock.get_instances()

    return bbs, func2bbs, bb_freq


# TODO: consistent arg naming
# TODO: re-format quotes


def callgrind_format_get_inclusive_cost(bbs: List[BasicBlock]):
    print("callgrind_format_get_inclusive_cost")
    print("bbs", bbs)
    call_stack = []
    bb_stack = []
    inclusive_cost_dict = defaultdict(lambda: defaultdict(list))
    prev_bb = None

    # call_stack: [A, B]
    # [Given] bb_stack: [[bb1, bb2, bb3], [bb4, bb5, bb6]]
    # [B returns]: [[bb1, bb2, [bb3, # cost from B]]]
    # where bb1 - bb3 belong to func A and bb4 - bb6 belong to func B
    for i, bb in enumerate(bbs):
        print("i", i)
        print("bb", bb)
        print("prev_bb", prev_bb)
        if prev_bb is None or (
            prev_bb.end_instr in ["jal", "beq", "bne", "blt", "bltu", "bge", "bgeu", "ecall"]
            and prev_bb.func != bb.func
        ):  # TODO: do not hardcode
            # first bb in the trace
            call_stack.append(bb.func)
            bb_stack.append([bb])
        elif prev_bb.func == bb.func:
            # jalr doesn't necessarily mean return
            # 0x2ac8, jalr, memset -> 0x2ae8, memset
            bb_stack[-1].append(bb)
        elif prev_bb.end_instr == "jalr":
            # Check whether jalr refer to return
            # sometimes jalr simply means indirect jump
            # TODO: Redundant? Is it already handled in the above condition?
            if bb.func not in call_stack:
                call_stack.append(bb.func)
                bb_stack.append([bb])
                continue

            # ret case for the rest code snippet
            diff = len(call_stack) - call_stack.index(bb.func) - 1

            for i in range(diff):
                cost = 0
                callee_first_bb = bb_stack[-1][0][0] if isinstance(bb_stack[-1][0], list) else bb_stack[-1][0]
                for bb_stack_elem in bb_stack[-1]:
                    if not isinstance(bb_stack_elem, list):
                        cost += 1 + (bb_stack_elem.last_pc - bb_stack_elem.first_pc) // 4
                    else:
                        subroutine_cost = bb_stack_elem[1]
                        cost += (
                            subroutine_cost + 1 + (bb_stack_elem[0].last_pc - bb_stack_elem[0].first_pc) // 4
                        )  # TODO: do not hardcode

                bb_stack.pop()
                call_stack.pop()

                caller = bb_stack[-1][-1]
                if isinstance(caller, list):
                    caller_bb = caller[0]
                    inclusive_cost_dict[caller_bb.last_pc][callee_first_bb.first_pc].append(cost)
                    bb_stack[-1][-1][1] += cost
                else:
                    caller_bb = caller
                    inclusive_cost_dict[caller_bb.last_pc][callee_first_bb.first_pc].append(cost)
                    bb_stack[-1][-1] = [caller_bb, cost]

                # When callee function returns, it jumps to the start of the "next" basic block
                # in caller function. For example:
                # 0x6b0  jal    rd, imm <--- in function A
                # ...
                # 0x5ae0 jalr   rd, rs, imm <--- in function B
                # 0x6b4  ... <--- return to function A
                if i == diff - 1:
                    bb_stack[-1].append(bb)

        prev_bb = bb

    return inclusive_cost_dict


def callgrind_format_converter(
    bbs: List[BasicBlock],
    mapping: Dict[str, Tuple[int, int]],
    func2bbs: Dict[str, List[BasicBlock]],
    bb_freq: Dict[BasicBlock, int],
    srcFile_func_dict: Dict[str, List[str]],
    func_set: Set[str],
    file2pc_loc: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    dump_pc: bool = True,
    dump_pos: bool = False,
    elf_file_path: str = "/path/to/elf",
):
    """
    Collect necessary information for callgrind output format.
    Then dump it to output_file

    Necessary information is:
    1. function name and its source file name
    Output:
        ...
        fl={source_file_name}
        fn={func_name}

    2. If --dump-instr=yes
    Output:
        ...
        {pc} {frequency}
        ...

    3. If --dump-line=yes
    Output:
        ...
        {line_number_in_source_code} {frequency}
        ...

    4. Caller-callee relationship
    Output:
        ... in caller's context
        {pc} {frequency}
        cfi={callee's source file}
        calls={number of times caller calls callee} {callee pc}
        {pc} {inclusive_cost}
    """
    assert dump_pc or dump_pos
    inclusive_cost_dict = callgrind_format_get_inclusive_cost(bbs)
    print("inclusive_cost_dict", inclusive_cost_dict)

    # Find the basic block, of which the first pc is equal to pc
    # TODO: time complexity is suboptimal
    def find_bb(pc):
        for bb in bbs:
            if pc == bb.first_pc:
                return bb
        return None

    # Find the source file where the function is defined
    # Some functions come from libc. In this case ??? is returned.
    def find_srcFile(target_func: str) -> str:
        for srcFile, funcs in srcFile_func_dict.items():
            for func in funcs:
                if func == target_func:
                    return srcFile
        return "???"

    #
    def aggregate_pos_cost_of_func(sorted_bb_lists: list):
        position_cost_dict = defaultdict(int)
        for bb in sorted_bb_lists:
            pc = bb.first_pc
            while pc <= bb.last_pc:
                position_cost_dict[pc] += bb.get_freq()
                pc += 4
        return position_cost_dict

    positions = "instr" if dump_pc else "line"
    print("positions", positions)

    prologue = f"""\
# callgrind format
version: 1
creator: callgrind-3.15.0
pid: 2577934
cmd:  ./dhrystone
part: 1

desc: I1 cache:
desc: D1 cache:
desc: LL cache:

desc: Timerange:
desc: Trigger: Program termination

positions: {positions}
events: Ir
summary:

"""
    callgrind_output = ""

    def callgrind_format_dump_instr(pc: int, source_file: str) -> str:
        return hex(pc)

    def callgrind_format_dump_line(pc: int, source_file: str) -> str:
        if source_file == "???":
            return "0"
        assert file2pc_loc is not None
        mapping_ = file2pc_loc[source_file]
        i = bisect.bisect_right(mapping_, pc, key=lambda x: x[0])
        if i:
            start_pc, line = mapping_[i - 1]
            if pc >= start_pc:
                return f"{line}"
        return "0"

    dump_positions = callgrind_format_dump_instr if dump_pc else callgrind_format_dump_line

    # source file to functions mapping
    srcFile_to_func = defaultdict(list)
    for func in func_set:
        print("func", func)
        srcFile_to_func[find_srcFile(func)].append(func)

    print("srcFile_to_func", srcFile_to_func)

    for srcFile, funcs in srcFile_to_func.items():
        callgrind_output += f"ob={posixpath.abspath(elf_file_path)}\n"
        callgrind_output += f"fl={srcFile}\n"

        for func in funcs:
            print(func)
            callgrind_output += f"fn={func}\n"

            bb_list = func2bbs[func]
            bb_list.sort(key=lambda bb: bb.first_pc)

            position_cost_dict = aggregate_pos_cost_of_func(bb_list)

            branch_pc_list = [bb.last_pc for bb in bb_list]

            for pc in sorted(position_cost_dict.keys()):
                callgrind_output += f"{hex(pc)} {position_cost_dict[pc]}\n"
                if pc in branch_pc_list and pc in inclusive_cost_dict:
                    for callee_pc, inclusive_cost in inclusive_cost_dict[pc].items():
                        # TODO Share object files case not implemented here
                        callgrind_output += f"cob={posixpath.abspath(elf_file_path)}\n"
                        callee_func = find_func_name(mapping, callee_pc)
                        callgrind_output += f"cfi={find_srcFile(callee_func)}\n"
                        callgrind_output += f"cfn={callee_func}\n"
                        callgrind_output += f"calls={len(inclusive_cost)} {hex(callee_pc)}\n"
                        callgrind_output += f"{hex(pc)} {sum(inclusive_cost)}\n"

            callgrind_output += "\n"

    content = prologue + callgrind_output
    return content


def generate_callgrind_output(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]

    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]

    func2pc_artifacts = filter_artifacts(artifacts, lambda x: x.name == "func2pc")
    assert len(func2pc_artifacts) == 1
    func2pc_artifact = func2pc_artifacts[0]
    func2pc_df = func2pc_artifact.df

    file2funcs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "file2funcs")
    assert len(file2funcs_artifacts) == 1
    file2funcs_artifact = file2funcs_artifacts[0]
    file2funcs_df = file2funcs_artifact.df

    mapping = dict(list(func2pc_df[["func", "pc_range"]].to_records(index=False)))
    bbs, func2bbs, bb_freq = collect_bbs(trace_artifact.df, mapping)
    file2funcs = dict(list((file2funcs_df.to_records(index=False))))
    func_set = set(func2bbs.keys())
    print("func_set", func_set)
    # dump_pc = False
    dump_pc = True
    dump_pos = False
    # dump_pos = True
    elf_file_path = elf_artifact.path
    file2pc_loc = None  # TODO

    content = callgrind_format_converter(
        bbs=bbs,
        mapping=mapping,
        func2bbs=func2bbs,
        bb_freq=bb_freq,
        srcFile_func_dict=file2funcs,
        func_set=func_set,
        file2pc_loc=file2pc_loc,
        dump_pc=dump_pc,
        dump_pos=dump_pos,
        elf_file_path=elf_file_path,
    )
    print("content:")
    print(content)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_callgrind_output(sess, force=args.force)
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
