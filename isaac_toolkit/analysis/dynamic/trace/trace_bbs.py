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
from typing import Dict, List, Set, Tuple, Optional, Union

import bisect
import time

import sys
import logging
import argparse
import posixpath
from pathlib import Path
from collections import defaultdict
from cpp_demangle import demangle

import pandas as pd
import numpy as np

from isaac_toolkit.session import Session
from isaac_toolkit.analysis.dynamic.trace.basic_blocks import BasicBlock  # TODO: move
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts, TableArtifact
from isaac_toolkit.arch.riscv import riscv_branch_instrs, riscv_return_instrs


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def unmangle_helper(func_name: Optional[str]):
    if func_name is None:
        return None
    if not func_name.startswith("_Z"):
        return func_name
    return demangle(func_name)


PC_FUNC_NAME_CACHE = {}


# TODO: reset?


def find_func_name(mapping: Dict[str, Tuple[int, int]], pc: int) -> str:
    # TODO: refactor to use existing tables
    """
    Given a program counter, find the function it belongs to
    """
    # t0 = time.time()
    found = PC_FUNC_NAME_CACHE.get(pc)
    if found is not None:
        return found
    for func, ranges in mapping.items():
        for range_ in ranges:
            if pc >= range_[0] and pc <= range_[1]:
                PC_FUNC_NAME_CACHE[pc] = func
                return func
    ret = hex(pc)
    PC_FUNC_NAME_CACHE[pc] = ret
    return ret


# def collect_bbs(trace_df, mapping):
#     first_pc = None
#     # TODO: make this generic!
#     func2bbs = defaultdict(list)  # TODO: only track func_set?
#     # bb_freq = defaultdict(int)
#     prev_pc = None
#     prev_size = None
#     # prev_instr = None
#     bbs = []
#     # bb_instrs = []
#     num_bb_instrs = 0
#     bb_size = 0
#     # t0 = time.time()
#     trace_pcs = set(trace_df["pc"].unique())
#     # td = time.time() - t0
#     # print("td", td)
#     unique_bbs = []
#     # unique_bb_hashes = []
#     unique_bb_map = {}
#     unique_bb_freq = defaultdict(int)
#     bb_trace_data = []
#     # t0 = time.time()
#     trace_df["instr"] = trace_df["instr"].str.strip()  # TODO: fix in frontend
#     # td = time.time() - t0
#     # print("td_", td)
#
#     # t0 = time.time()
#     branch_ret_instrs = set(riscv_branch_instrs + riscv_return_instrs)
#     trace_df["is_branch_return"] = trace_df["instr"].isin(branch_ret_instrs)
#     # td = time.time() - t0
#     # print("td__", td)
#
#     # t0 = time.time()
#     pcs = trace_df["pc"].to_numpy()
#     instrs = trace_df["instr"].to_numpy()
#     sizes = trace_df["size"].to_numpy()
#     is_branch_return_list = trace_df["is_branch_return"].to_numpy()
#     # td = time.time() - t0
#     # print("td___", td)
#     # for row in trace_df.itertuples():
#     # Numpy based iter for speed
#
#     # t0 = time.time()
#     steps = pcs[1:] - pcs[:-1]
#     expected_steps = sizes[:-1]
#     irregular_step = steps != expected_steps
#     mask = is_branch_return_list.copy()
#     mask[:-1] |= irregular_step  # propagate irregular step to the earlier instr
#     bb_end_indices = np.where(mask)[0]
#     bb_start_indices = np.empty_like(bb_end_indices)
#     bb_start_indices[0] = 0
#     bb_start_indices[1:] = bb_end_indices[:-1] + 1
#     bbs = pd.DataFrame(
#         {
#             "first_pc": pcs[bb_start_indices],
#             "last_pc": pcs[bb_end_indices],
#             "num_instrs": bb_end_indices - bb_start_indices + 1,
#             "size": np.add.reduceat(sizes, bb_start_indices),  # grouped sum
#             "end_instr": instrs[bb_end_indices],
#         }
#     )
#     # td = time.time() - t0
#     # print("td___exp", td)
#     # print("irregular_step", irregular_step, sum(irregular_step))
#     # print("mask", mask, sum(mask))
#     # print("bbs", bbs)
#
#     # t0 = time.time()
#     tda = 0
#     tdb = 0
#     tdc = 0
#     tdd = 0
#     tde = 0
#     for i in range(len(pcs)):
#         # t0a = time.time()
#         pc = pcs[i]
#         instr = instrs[i]
#         sz = 2 if sizes[i] == 2 else 4
#         is_compressed = sz == 2
#         is_branch_return = is_branch_return_list[i]
#         trace_idx = i
#
#         # trace_idx = row.Index
#         # pc = row.pc
#         # instr = row.instr
#         # is_compressed = row.size == 2
#         # sz = 2 if is_compressed else 4
#         # tda += (time.time() - t0a)
#
#         # t0b = time.time()
#         if prev_pc:
#             step = pc - prev_pc
#             if step in [2, 4] and step == prev_size:
#                 pass
#             else:
#                 if first_pc is None:
#                     pass
#                 else:
#                     logger.warning("Detected potential trap @ pc = 0x%x -> 0x%x", prev_pc, pc)
#                     if True:
#                         func = find_func_name(mapping, prev_pc)
#                         # bb = BasicBlock(
#                         #     first_pc=first_pc,
#                         #     last_pc=prev_pc,
#                         #     num_instrs=len(bb_instrs),
#                         #     size=bb_size,
#                         #     end_instr=instr,
#                         #     func=func,
#                         # )
#                         # first_pc: int,
#                         # last_pc: int,
#                         # num_instrs: int,
#                         # size: int,
#                         # end_instr: str,
#                         # func: str,
#                         # bb_info = (first_pc, prev_pc, len(bb_instrs), bb_size, instr, func)
#                         bb_info = (first_pc, prev_pc, num_bb_instrs, bb_size, instr, func)
#                         # bbs.append(bb_info)
#                         # bb_hash = hash((first_pc, prev_pc))
#                         bb_hash = (first_pc << 32) | prev_pc
#                         ### bb_hash = bb.__hash__()
#                         # unique_bb_idx = None
#                         # if bb_hash in unique_bb_hashes:
#                         #     unique_bb_idx = unique_bb_hashes.index(bb_hash)
#                         unique_bb_idx = unique_bb_map.get(bb_hash)
#                         if unique_bb_idx is None:
#                             unique_bb_idx = len(unique_bbs)
#                             unique_bbs.append(bb_info)
#                             unique_bb_map[bb_hash] = unique_bb_idx
#                             func2bbs[func].append(unique_bb_idx)
#                         unique_bb_call = unique_bb_freq[unique_bb_idx]
#                         unique_bb_freq[unique_bb_idx] += 1
#
#                         bb_trace_data_new = (unique_bb_idx, unique_bb_call, trace_idx)
#                         bb_trace_data.append(bb_trace_data_new)
#
#                         # bb_instrs = []
#                         num_bb_instrs = 0
#                         bb_size = 0
#                         # bbs.append(bb)
#                         # if bb.get_freq() == 1:
#                         #     func2bbs[bb.func].append(bb)
#                         # bb_freq[bb] += 1
#
#                         first_pc = pc
#
#         # tdb += (time.time() - t0b)
#
#         # t0c = time.time()
#         # At the first pc of a basic block
#         if first_pc is None:
#             first_pc = pc
#
#         # bb_instrs.append(instr)
#         num_bb_instrs += 1
#         bb_size += sz
#         # tdc += (time.time() - t0c)
#
#         # t0d = time.time()
#         # if instr in riscv_branch_instrs + riscv_return_instrs:
#         if is_branch_return:
#             func = find_func_name(mapping, pc)
#             # bb = BasicBlock(
#             #     first_pc=first_pc,
#             #     last_pc=pc,
#             #     num_instrs=len(bb_instrs),
#             #     size=bb_size,
#             #     end_instr=instr,
#             #     func=func,
#             # )
#             # bb_info = (first_pc, pc, len(bb_instrs), bb_size, instr, func)
#             bb_info = (first_pc, pc, num_bb_instrs, bb_size, instr, func)
#             # bbs.append(bb_info)
#             # bb_hash = hash((first_pc, pc))
#             bb_hash = (first_pc << 32) | pc
#             # bb_hash = bb.__hash__()
#             # unique_bb_idx = None
#             # if bb_hash in unique_bb_hashes:
#             #     unique_bb_idx = unique_bb_hashes.index(bb_hash)
#             unique_bb_idx = unique_bb_map.get(bb_hash)
#             if unique_bb_idx is None:
#                 unique_bb_idx = len(unique_bbs)
#                 unique_bbs.append(bb_info)
#                 unique_bb_map[bb_hash] = unique_bb_idx
#                 func2bbs[func].append(unique_bb_idx)
#             unique_bb_call = unique_bb_freq[unique_bb_idx]
#             unique_bb_freq[unique_bb_idx] += 1
#
#             bb_trace_data_new = (unique_bb_idx, unique_bb_call, trace_idx)
#             bb_trace_data.append(bb_trace_data_new)
#             # bb_instrs = []
#             num_bb_instrs = 0
#             bb_size = 0
#             # bbs.append(bb)
#             # if bb.get_freq() == 1:
#             #     func2bbs[bb.func].append(bb)
#             #     bb_freq[bb] += 1
#             first_pc = None
#         # tdd += (time.time() - t0d)
#
#         # t0e = time.time()
#         prev_pc = pc
#         # prev_instr = instr
#         prev_size = sz
#         # tde += (time.time() - t0e)
#     # td = time.time() - t0
#     # print("tdloop", td, td/len(pcs))
#     # print("tda", tda, tda/td)
#     # print("tdb", tdb, tdb/td)
#     # print("tdc", tda, tdc/td)
#     # print("tdd", tdd, tdd/td)
#     # print("tde", tde, tde/td)
#     # t0 = time.time()
#     bb_trace_df = pd.DataFrame(bb_trace_data, columns=["bb_idx", "bb_call", "trace_idx"])
#     bb_trace_df["bb_idx"] = pd.to_numeric(bb_trace_df["bb_idx"], downcast="unsigned").astype("category")
#     bb_trace_df["bb_call"] = pd.to_numeric(bb_trace_df["bb_call"], downcast="unsigned")
#     bb_trace_df["trace_idx"] = pd.to_numeric(bb_trace_df["trace_idx"], downcast="unsigned")
#     # print("bb_trace_df", bb_trace_df)
#     # input("1")
#     unique_bbs_df = pd.DataFrame(unique_bbs, columns=["first_pc", "last_pc", "num_instrs", "size", "end_instr", "func"])
#     unique_bbs_df["freq"] = list(unique_bb_freq.values())  # assume ordered dict
#     # td = time.time() - t0
#     # print("tdfin", td, td/len(pcs))
#     # input("!!!")
#     if first_pc is not None:
#         func = None
#
#     return trace_pcs, func2bbs, unique_bbs_df, bb_trace_df


import numpy as np
import pandas as pd
from collections import defaultdict


def collect_bbs_new(trace_df, mapping):
    # --- Extract columns as NumPy ---
    pcs = trace_df["pc"].to_numpy()
    instrs = trace_df["instr"].str.strip().to_numpy()
    sizes = np.where(trace_df["size"].to_numpy() == 2, 2, 4)

    # --- Fast branch/return detection ---
    branch_ret_set = set(riscv_branch_instrs + riscv_return_instrs)
    is_branch_return = trace_df["instr"].isin(branch_ret_set).to_numpy()

    # --- Detect irregular steps ---
    steps = pcs[1:] - pcs[:-1]
    expected_steps = sizes[:-1]
    irregular_step = steps != expected_steps

    # --- Combine into a mask marking BB ends ---
    mask = is_branch_return.copy()
    mask[:-1] |= irregular_step

    # --- Indices of BB ends ---
    bb_end_indices = np.where(mask)[0]
    if len(bb_end_indices) == 0:
        return set(pcs), {}, pd.DataFrame(), pd.DataFrame()

    # --- Indices of BB starts ---
    bb_start_indices = np.empty_like(bb_end_indices)
    bb_start_indices[0] = 0
    bb_start_indices[1:] = bb_end_indices[:-1] + 1

    # --- Build unique BBs ---
    first_pcs = pcs[bb_start_indices]
    last_pcs = pcs[bb_end_indices]
    num_instrs = bb_end_indices - bb_start_indices + 1
    bb_sizes = np.add.reduceat(sizes, bb_start_indices)
    end_instrs = instrs[bb_end_indices]

    # Map PCs to funcs once
    unique_pcs = np.unique(np.concatenate([first_pcs, last_pcs]))
    func_map = {pc: find_func_name(mapping, pc) for pc in unique_pcs}
    funcs = [func_map[pc] for pc in last_pcs]

    # Deduplicate BBs using a deterministic key
    unique_bbs = []
    unique_bb_map = {}
    unique_bb_freq = defaultdict(int)
    func2bbs = defaultdict(list)

    bb_trace_records = []

    for idx, (fp, lp, n, sz, ei, func) in enumerate(zip(first_pcs, last_pcs, num_instrs, bb_sizes, end_instrs, funcs)):
        bb_hash = (fp << 32) | lp
        # unique_bb_idx = bb_hash
        unique_bb_idx = unique_bb_map.get(bb_hash)
        if unique_bb_idx is None:
            unique_bb_idx = len(unique_bbs)
            unique_bbs.append((fp, lp, n, sz, ei, func))
            unique_bb_map[bb_hash] = unique_bb_idx
            func2bbs[func].append(unique_bb_idx)

        bb_call = unique_bb_freq[unique_bb_idx]
        unique_bb_freq[unique_bb_idx] += 1
        bb_trace_records.append((unique_bb_idx, bb_call, bb_end_indices[idx]))

    # --- Convert to DataFrames ---
    bb_trace_df = pd.DataFrame(bb_trace_records, columns=["bb_idx", "bb_call", "trace_idx"])
    bb_trace_df["bb_idx"] = bb_trace_df["bb_idx"].astype("category")
    bb_trace_df["bb_call"] = pd.to_numeric(bb_trace_df["bb_call"], downcast="unsigned")
    bb_trace_df["trace_idx"] = pd.to_numeric(bb_trace_df["trace_idx"], downcast="unsigned")

    unique_bbs_df = pd.DataFrame(unique_bbs, columns=["first_pc", "last_pc", "num_instrs", "size", "end_instr", "func"])
    unique_bbs_df["freq"] = [unique_bb_freq[i] for i in range(len(unique_bbs))]

    return set(pcs), func2bbs, unique_bbs_df, bb_trace_df


# TODO: consistent arg naming
# TODO: re-format quotes


# def callgrind_format_get_inclusive_cost(bbs: List[BasicBlock]):
def callgrind_format_get_inclusive_cost(bb_trace_df: pd.DataFrame, unique_bbs_df: pd.DataFrame):
    call_stack = []
    bb_stack = []
    total_cost = 0
    inclusive_cost_dict = defaultdict(lambda: defaultdict(list))
    prev_bb_idx = None
    unique_bbs_records = unique_bbs_df.to_records(index=False)

    # call_stack: [A, B]
    # [Given] bb_stack: [[bb1, bb2, bb3], [bb4, bb5, bb6]]
    # [B returns]: [[bb1, bb2, [bb3, # cost from B]]]
    # where bb1 - bb3 belong to func A and bb4 - bb6 belong to func B
    # for i, bb in enumerate(bbs):
    # for i, bb_trace_row in bb_trace_df.iterrows():
    bb_funcs = unique_bbs_df["func"].values  # dtype=object or better: categorical/int
    bb_num_instrs = unique_bbs_df["num_instrs"].values  # dtype=int32/int64
    bb_end_instrs = unique_bbs_df["end_instr"].values  # dtype=object or categorical
    bb_first_pc = unique_bbs_df["first_pc"].values
    bb_last_pc = unique_bbs_df["last_pc"].values
    # t0 = time.time()
    # ts = [t0]
    # for i, bb_idx in enumerate(bb_trace_df["bb_idx"].values):
    for i, bb_idx in enumerate(bb_trace_df["bb_idx"].values):
        # ts.append(time.time())
        # if i == 100:
        #     break
        func = bb_funcs[bb_idx]
        n_instr = bb_num_instrs[bb_idx]
        end_instr = bb_end_instrs[bb_idx]
        first_pc = bb_first_pc[bb_idx]
        last_pc = bb_last_pc[bb_idx]
        # print("progress", float(i)/len(bb_trace_df))
        # bb_idx = bb_trace_row.bb_idx
        # bb = unique_bbs_df.iloc[bb_idx]
        bb = unique_bbs_records[bb_idx]
        # print("i,bb_idx,bb", i, bb_idx, bb)
        total_cost += n_instr
        # print("bb", bb)
        # print("prev_bb", prev_bb)
        if prev_bb_idx is None or (bb_end_instrs[prev_bb_idx] in riscv_branch_instrs and bb_funcs[prev_bb_idx] != func):
            # first bb in the trace
            # print("append call_stack + bb_stack")
            call_stack.append(func)
            bb_stack.append([bb_idx])
        elif bb_funcs[prev_bb_idx] == func:
            # jalr doesn't necessarily mean return
            # 0x2ac8, jalr, memset -> 0x2ae8, memset
            # print("append bb_stack")
            bb_stack[-1].append(bb_idx)
        # elif prev_bb.end_instr in return_instrs:
        # elif prev_bb.end_instr in riscv_return_instrs:
        elif bb_end_instrs[prev_bb_idx] in riscv_return_instrs or bb_num_instrs[prev_bb_idx].num_instrs == 1:
            # Check whether jalr refer to return
            # sometimes jalr simply means indirect jump
            # TODO: Redundant? Is it already handled in the above condition?
            if func not in call_stack:
                call_stack.append(func)
                bb_stack.append([bb_idx])
                # print("append call_stack + bb_stack 2")
                continue

            # ret case for the rest code snippet
            diff = len(call_stack) - call_stack.index(func) - 1

            for j in range(diff):
                cost = 0
                callee_first_bb_idx = bb_stack[-1][0][0] if isinstance(bb_stack[-1][0], list) else bb_stack[-1][0]
                for bb_stack_elem in bb_stack[-1]:
                    if not isinstance(bb_stack_elem, list):
                        # cost += 1 + (bb_stack_elem.last_pc - bb_stack_elem.first_pc) // 4
                        cost += bb_num_instrs[bb_stack_elem]
                    else:
                        subroutine_cost = bb_stack_elem[1]
                        cost += (
                            # subroutine_cost + 1 + (bb_stack_elem[0].last_pc - bb_stack_elem[0].first_pc) // 4
                            subroutine_cost
                            + bb_num_instrs[bb_stack_elem[0]]
                        )

                # print("pop call_stack + bb_stack")
                bb_stack.pop()
                call_stack.pop()

                caller = bb_stack[-1][-1]
                if isinstance(caller, list):
                    caller_bb_idx = caller[0]
                    inclusive_cost_dict[bb_last_pc[caller_bb_idx]][bb_first_pc[callee_first_bb_idx]].append(cost)
                    bb_stack[-1][-1][1] += cost
                else:
                    caller_bb_idx = caller
                    inclusive_cost_dict[bb_last_pc[caller_bb_idx]][bb_first_pc[callee_first_bb_idx]].append(cost)
                    bb_stack[-1][-1] = [caller_bb_idx, cost]

                # When callee function returns, it jumps to the start of the "next" basic block
                # in caller function. For example:
                # 0x6b0  jal    rd, imm <--- in function A
                # ...
                # 0x5ae0 jalr   rd, rs, imm <--- in function B
                # 0x6b4  ... <--- return to function A
                if j == diff - 1:
                    # print("append bb stack")
                    bb_stack[-1].append(bb_idx)

        prev_bb_idx = bb_idx
    # print("ts", ts)
    # import numpy as np
    # tds = np.array(ts)[1:] - np.array(ts)[:-1]
    # print("tds", tds)
    # input()

    # print("inclusive_cost_dict", inclusive_cost_dict)
    # inclusive_cost_dict_sum = {pc: sum([sum(x) for x in costs.values()]) for pc, costs in inclusive_cost_dict.items()}
    # print("inclusive_cost_dict_sum", inclusive_cost_dict_sum)
    # inclusive_cost_dict_total = sum(inclusive_cost_dict_sum.values())
    # print("inclusive_cost_dict_total", inclusive_cost_dict_total)
    # inclusive_cost_dict_max = max(inclusive_cost_dict_sum.values())
    # print("inclusive_cost_dict_sum_max", inclusive_cost_dict_max)
    # print("total_cost", total_cost)
    # TODO: check that all costs are contained

    return inclusive_cost_dict


def callgrind_format_converter(
    # bbs: List[BasicBlock],
    bb_trace_df: pd.DataFrame,
    unique_bbs_df: pd.DataFrame,
    trace_pcs: Set[int],
    mapping: Dict[str, Tuple[int, int]],
    # func2bbs: Dict[str, List[BasicBlock]],
    func2bbs: Dict[str, List[int]],
    # bb_freq: Dict[BasicBlock, int],
    srcFile_func_dict: Dict[str, List[str]],
    srcFile_linkage_name_dict: Dict[str, List[str]],
    srcFile_unmangled_linkage_name_dict: Dict[str, List[str]],
    func_set: Set[str],
    file2pc_loc: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    dump_pc: bool = True,
    dump_pos: bool = False,
    elf_file_path: str = "/path/to/elf",
    unmangle_names: bool = False,
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
    # inclusive_cost_dict = callgrind_format_get_inclusive_cost(bbs)
    inclusive_cost_dict = callgrind_format_get_inclusive_cost(bb_trace_df, unique_bbs_df)

    # Find the source file where the function is defined
    # Some functions come from libc. In this case ??? is returned.
    def find_srcFile(target_func: str) -> str:
        for srcFile, funcs in srcFile_func_dict.items():
            for func in funcs:
                if func == target_func:
                    return srcFile
        for srcFile, funcs in srcFile_linkage_name_dict.items():
            for func in funcs:
                if func == target_func:
                    return srcFile
        return "???"

    #
    def aggregate_pos_cost_of_func(sorted_bb_lists: list):
        position_cost_dict = defaultdict(int)
        for bb_idx in sorted_bb_lists:
            bb = unique_bbs_df.iloc[bb_idx]
            pc = bb.first_pc
            # while pc <= bb.last_pc:
            for pc_ in range(pc, bb.last_pc + 2, 2):
                if pc_ not in trace_pcs:
                    # pc += 2
                    continue
                # position_cost_dict[pc_] += bb.get_freq()
                position_cost_dict[pc_] += bb.freq
                # pc += 4
                # pc += 2
        return position_cost_dict

    positions = "instr" if dump_pc else "line"

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
        mapping_ = file2pc_loc.get(source_file)
        if mapping_ is None:
            return "0"
        # PYTHON 3.10:
        # i = bisect.bisect_right(mapping_, pc, key=lambda x: x[0])
        # if i:
        #     start_pc, line = mapping_[i - 1]
        #     if pc >= start_pc:
        #         return f"{line}"
        # ---
        # PYTHON 3.8
        mapping_first = [x[0] for x in mapping_]
        i = bisect.bisect_right(mapping_first, pc)
        if i:
            start_pc, line = mapping_[i - 1]
            if pc >= start_pc:
                return f"{line}"
        # ---
        return "0"

    dump_positions = callgrind_format_dump_instr if dump_pc else callgrind_format_dump_line

    # source file to functions mapping
    srcFile_to_func = defaultdict(list)
    for func in func_set:
        srcFile_to_func[find_srcFile(func)].append(func)

    for srcFile, funcs in srcFile_to_func.items():
        callgrind_output += f"ob={posixpath.abspath(elf_file_path)}\n"
        callgrind_output += f"fl={srcFile}\n"

        for func in funcs:

            bb_list = func2bbs[func]
            bb_list.sort(key=lambda bb_idx: unique_bbs_df.iloc[bb_idx].first_pc)
            if unmangle_names:
                func = unmangle_helper(func)
            callgrind_output += f"fn={func}\n"

            position_cost_dict = aggregate_pos_cost_of_func(bb_list)

            # branch_pc_list = [bb.last_pc for bb in bb_list]
            branch_pc_list = unique_bbs_df.iloc[bb_list].last_pc.values

            for pc in sorted(position_cost_dict.keys()):
                position_info = dump_positions(pc, srcFile)
                callgrind_output += f"{position_info} {position_cost_dict[pc]}\n"
                if pc in branch_pc_list and pc in inclusive_cost_dict:
                    for callee_pc, inclusive_cost in inclusive_cost_dict[pc].items():
                        # TODO Share object files case not implemented here
                        callgrind_output += f"cob={posixpath.abspath(elf_file_path)}\n"
                        callee_func = find_func_name(mapping, callee_pc)
                        callgrind_output += f"cfi={find_srcFile(callee_func)}\n"
                        if unmangle_names:
                            callee_func = unmangle_helper(callee_func)
                        callgrind_output += f"cfn={callee_func}\n"
                        callgrind_output += f"calls={len(inclusive_cost)} {hex(callee_pc)}\n"
                        callgrind_output += f"{position_info} {sum(inclusive_cost)}\n"

            callgrind_output += "\n"

    content = prologue + callgrind_output
    return content


def collect_trace_bbs(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    force: bool = False,
):
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

    mapping = func2pc_df.groupby("func")["pc_range"].apply(list).to_dict()
    # bbs, trace_pcs, func2bbs, bb_freq = collect_bbs(trace_artifact.df, mapping)
    # t0 = time.time()
    # trace_pcs, func2bbs, unique_bbs_df, bb_trace_df = collect_bbs(trace_artifact.df, mapping)
    # t1 = time.time()
    trace_pcs, func2bbs, unique_bbs_df, bb_trace_df = collect_bbs_new(trace_artifact.df, mapping)
    # t2 = time.time()
    # tdx = t1 - t0
    # tdy = t2 - t1
    # print("tdx", tdx)
    # print("tdy", tdy)
    # print("bb_trace_df", bb_trace_df)
    # print("bb_trace_df2", bb_trace_df2)
    # print("unique_bbs_df", unique_bbs_df)
    # print("unique_bbs_df2", unique_bbs_df2)

    bb_trace_attrs = {
        "trace": trace_artifact.name,
        "kind": "bb_trace",
        "by": __name__,
    }

    unique_bbs_attrs = {
        "trace": trace_artifact.name,
        "kind": "mapping",
        "by": __name__,
    }

    func2bbs_attrs = {
        "trace": trace_artifact.name,
        "kind": "mapping",
        "by": __name__,
    }

    trace_pcs_attrs = {
        "trace": trace_artifact.name,
        "kind": "list",
        "by": __name__,
    }

    trace_pcs_df = pd.Series(list(trace_pcs))  # TODO: to_frame()?
    # print("trace_pcs_df", trace_pcs_df)

    # print("func2bbs", func2bbs)
    func2bbs_df = pd.DataFrame({"func": list(func2bbs.keys()), "bb_idxs": list(func2bbs.values())})
    # print("func2bbs_df", func2bbs_df)

    bb_trace_artifact = TableArtifact("bb_trace", bb_trace_df, attrs=bb_trace_attrs)
    unique_bbs_artifact = TableArtifact("unique_bbs", unique_bbs_df, attrs=unique_bbs_attrs)
    func2bbs_artifact = TableArtifact("func2bbs", func2bbs_df, attrs=func2bbs_attrs)
    trace_pcs_artifact = TableArtifact("trace_pcs", trace_pcs_df, attrs=trace_pcs_attrs)

    sess.add_artifact(bb_trace_artifact, override=force)
    sess.add_artifact(unique_bbs_artifact, override=force)
    sess.add_artifact(func2bbs_artifact, override=force)
    sess.add_artifact(trace_pcs_artifact, override=force)

    # from basic_blocks (for backward compat)
    pc2bb_df = unique_bbs_df.copy()
    # TODO: check if inclusive or exclusive range
    # pc2bb_df.rename(columns={"func": "func_name", "first_pc": "start", "last_pc": "end"}, inplace=True)
    pc2bb_df.rename(columns={"first_pc": "start", "last_pc": "end"}, inplace=True)
    pc2bb_df["func_name"] = pc2bb_df["func"].map(lambda x: {x})
    pc2bb_df.sort_values("freq", inplace=True, ascending=False)
    pc2bb_df["weight"] = pc2bb_df["freq"] * pc2bb_df["num_instrs"]
    pc2bb_df["rel_weight"] = pc2bb_df["weight"] / sum(pc2bb_df["weight"])
    # TODO: size is wrong!!!
    pc2bb_df.drop(columns=["end_instr", "func"], inplace=True)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "mapping",
        "by": __name__,
    }

    # pc2bb_artifact = TableArtifact("pc2bb_compat", pc2bb_df, attrs=attrs)
    pc2bb_artifact = TableArtifact("pc2bb", pc2bb_df, attrs=attrs)
    sess.add_artifact(pc2bb_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    collect_trace_bbs(
        sess,
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
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
