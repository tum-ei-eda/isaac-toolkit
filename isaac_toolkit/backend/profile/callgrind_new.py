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
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts
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
    found = PC_FUNC_NAME_CACHE.get(pc)
    if found is not None:
        return found
    # print("mapping", mapping)
    for func, ranges in mapping.items():
        for range_ in ranges:
            if pc >= range_[0] and pc <= range_[1]:
                PC_FUNC_NAME_CACHE[pc] = func
                return func
    ret = hex(pc)
    # print("hex")
    input("???")
    PC_FUNC_NAME_CACHE[pc] = ret
    return ret


# TODO: consistent arg naming
# TODO: re-format quotes


def build_call_trace(function_trace_df, event_names):
    call_records = []
    call_stack = []
    funcs = function_trace_df["func"].to_numpy()
    depths = function_trace_df["depth"].to_numpy()
    costs = function_trace_df["cost"].to_numpy()
    start_pcs = function_trace_df["start_pc"].to_numpy()

    # for _, row in function_trace_df.iterrows():
    for i in range(len(function_trace_df)):
        # func, depth, metrics = row["func"], row["depth"], row["cost"]
        func, depth, cost = funcs[i], depths[i], costs[i]

        if not call_stack or depth > call_stack[-1][1]:
            # print("if")
            # function entry
            if call_stack:
                # print("if2")
                caller, _, call_pc = call_stack[-1]
                call_records.append(
                    {
                        "caller": caller,
                        "callee": func,
                        "call_pc": call_pc,  # optional: store PC if tracked
                        "calls": 1,
                        "cost": cost,
                    }
                )
            call_stack.append((func, depth, start_pcs[i]))
            # print("end")
        else:
            # print("else")
            # function exit or same-level
            while call_stack and call_stack[-1][1] >= depth:
                # print("while")
                call_stack.pop()
            call_stack.append((func, depth, start_pcs[i]))
            # print("end2")

    call_trace_df = pd.DataFrame(call_records)
    print("ctdf", call_trace_df)

    # Aggregate duplicates (same caller/callee/pc)
    agg_dict = {"calls": "sum"}
    # for ev in event_names:
    #     agg_dict[ev] = "sum"

    call_trace_df = call_trace_df.groupby(["caller", "callee", "call_pc"], as_index=False).agg(agg_dict)
    print("ctdf", call_trace_df)

    return call_trace_df


# def callgrind_format_get_inclusive_cost(bbs: List[BasicBlock]):
def callgrind_format_get_inclusive_cost(
    bb_trace_df: pd.DataFrame, unique_bbs_df: pd.DataFrame, event_names: List[str] = [], event_arrays=None
):
    call_stack = []
    bb_stack = []
    # total_cost = 0
    inclusive_cost_dict = defaultdict(lambda: defaultdict(list))
    prev_bb_idx = None
    print("P", time.time())
    print("bb_trace_df.head()", bb_trace_df.head())
    print("unique_bbs_df.head()", unique_bbs_df.head())
    unique_bbs_records = unique_bbs_df.to_records(index=False)
    print("len(unique_bbs_records)", len(unique_bbs_records))
    print("len(unique_bbs_df)", len(unique_bbs_df))

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
    function_trace = []
    # t0 = time.time()
    # ts = [t0]
    # for i, bb_idx in enumerate(bb_trace_df["bb_idx"].values):
    for i, bb_idx in enumerate(bb_trace_df["bb_idx"].values):
        # print("progress", i/len(bb_trace_df)*100, "%")
        # ts.append(time.time())
        # if i == 100:
        #     break
        func = bb_funcs[bb_idx]
        # n_instr = bb_num_instrs[bb_idx]
        # end_instr = bb_end_instrs[bb_idx]
        first_pc = bb_first_pc[bb_idx]
        last_pc = bb_last_pc[bb_idx]
        # print("progress", float(i)/len(bb_trace_df))
        # bb_idx = bb_trace_row.bb_idx
        # bb = unique_bbs_df.iloc[bb_idx]
        # bb = unique_bbs_records[bb_idx]
        # print("bb_ifx", bb_idx)
        # print("func", func)
        # row = bb_cost_trace_df.iloc[i]
        # print("row", row)
        # metrics = [int(row[ev]) for ev in event_names]
        # print("metrics", metrics)
        # print("i,bb_idx,bb", i, bb_idx, bb)
        # total_cost += n_instr
        # print("bb", bb)
        if prev_bb_idx is None or (bb_end_instrs[prev_bb_idx] in riscv_branch_instrs and bb_funcs[prev_bb_idx] != func):
            # first bb in the trace
            # print("append call_stack + bb_stack")
            call_stack.append(func)
            depth = len(call_stack)
            new_ = [func, depth, first_pc, last_pc, np.zeros(len(event_names), dtype=np.int64)]
            function_trace.append(new_)
            bb_stack.append([bb_idx])
        elif bb_funcs[prev_bb_idx] == func:
            # jalr doesn't necessarily mean return
            # 0x2ac8, jalr, memset -> 0x2ae8, memset
            # print("append bb_stack")
            bb_stack[-1].append(bb_idx)
        # elif prev_bb.end_instr in return_instrs:
        # elif prev_bb.end_instr in riscv_return_instrs:
        elif bb_end_instrs[prev_bb_idx] in riscv_return_instrs or bb_num_instrs[prev_bb_idx] == 1:
            # Check whether jalr refer to return
            # sometimes jalr simply means indirect jump
            # TODO: Redundant? Is it already handled in the above condition?
            if func not in call_stack:
                call_stack.append(func)
                depth = len(call_stack)
                # metrics_vec = [event_arrays[ev][bb_idx] for ev in event_names]
                metrics_vec = event_arrays[:, bb_idx]
                new_ = [func, depth, first_pc, last_pc, metrics_vec]
                function_trace.append(new_)
                bb_stack.append([bb_idx])
                # print("append call_stack + bb_stack 2")
                continue

            # ret case for the rest code snippet
            diff = len(call_stack) - call_stack.index(func) - 1

            for j in range(diff):
                # cost_vec = [0] * len(event_names)
                cost_vec = np.zeros(len(event_names), dtype=np.int64)
                callee_first_bb_idx = bb_stack[-1][0][0] if isinstance(bb_stack[-1][0], list) else bb_stack[-1][0]
                for bb_stack_elem in bb_stack[-1]:
                    # print("bb_stack_elem", bb_stack_elem)
                    if not isinstance(bb_stack_elem, list):
                        # base case: plain BB
                        # bb_metrics = bb_cost_trace_df.iloc[bb_stack_elem]
                        # metrics_vec = [int(bb_metrics[ev]) for ev in event_names]
                        # metrics_vec = [event_arrays[ev][bb_stack_elem] for ev in event_names]
                        metrics_vec = event_arrays[:, bb_stack_elem]
                        # cost_vec = [c + m for c, m in zip(cost_vec, metrics_vec)]
                        cost_vec = cost_vec + metrics_vec
                    else:
                        # recursive case: [bb_idx, subroutine_cost_vec]
                        callee_bb_idx, sub_cost_vec = bb_stack_elem
                        # bb_metrics = bb_cost_trace_df.iloc[callee_bb_idx]
                        # metrics_vec = [int(bb_metrics[ev]) for ev in event_names]
                        # metrics_vec = [event_arrays[ev][callee_bb_idx] for ev in event_names]
                        metrics_vec = event_arrays[:, callee_bb_idx]
                        # cost_vec = [c + m + s for c, m, s in zip(cost_vec, metrics_vec, sub_cost_vec)]
                        cost_vec = cost_vec + metrics_vec + sub_cost_vec

                # print("pop call_stack + bb_stack")
                bb_stack.pop()
                call_stack.pop()
                if (diff - 1) == j:
                    depth = len(call_stack)
                    new_ = [func, depth, first_pc, last_pc, np.zeros(len(event_names), dtype=np.int64)]
                    function_trace.append(new_)

                caller = bb_stack[-1][-1]
                if isinstance(caller, list):
                    caller_bb_idx, caller_cost_vec = caller
                    # record the inclusive cost of the callee relative to caller
                    inclusive_cost_dict[bb_last_pc[caller_bb_idx]][bb_first_pc[callee_first_bb_idx]].append(cost_vec)
                    # update caller’s accumulated cost vector
                    # bb_stack[-1][-1][1] = [c + v for c, v in zip(caller_cost_vec, cost_vec)]
                    bb_stack[-1][-1][1] = caller_cost_vec + cost_vec
                else:
                    caller_bb_idx = caller
                    inclusive_cost_dict[bb_last_pc[caller_bb_idx]][bb_first_pc[callee_first_bb_idx]].append(cost_vec)
                    # replace scalar with [bb_idx, cost_vec]
                    bb_stack[-1][-1] = [caller_bb_idx, cost_vec]

                # When callee function returns, it jumps to the start of the "next" basic block
                # in caller function. For example:
                # 0x6b0  jal    rd, imm <--- in function A
                # ...
                # 0x5ae0 jalr   rd, rs, imm <--- in function B
                # 0x6b4  ... <--- return to function A
                if j == diff - 1:
                    # print("append bb stack")
                    bb_stack[-1].append(bb_idx)

        cost_vec_old = function_trace[-1][-1]
        # metrics_vec = [event_arrays[ev][bb_idx] for ev in event_names]
        metrics_vec = event_arrays[:, bb_idx]
        # cost_vec_new = [c + m for c, m in zip(cost_vec_old, metrics_vec)]
        cost_vec_new = cost_vec_old + metrics_vec
        function_trace[-1][-1] = cost_vec_new
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
    # print("function_trace", "\n".join(map(str, function_trace[:100])))
    # TODO: export call & func trace (&USE!)
    # function_trace_df = pd.DataFrame(function_trace, columns=["func", "depth", "start_pc", "end_pc", "cost"])
    # print("function_trace_df", function_trace_df)
    # call_trace_df = build_call_trace(function_trace_df, event_names)
    # print("call_trace_df", call_trace_df)
    # input("@")
    # print("inclusive_cost_dict", inclusive_cost_dict)
    # input("%")

    return inclusive_cost_dict


def callgrind_format_converter(
    bb_trace_df: pd.DataFrame,
    unique_bbs_df: pd.DataFrame,
    trace_pcs: Set[int],
    mapping: Dict[str, Tuple[int, int]],
    func2bbs: Dict[str, List[int]],
    srcFile_func_dict: Dict[str, List[str]],
    srcFile_linkage_name_dict: Dict[str, List[str]],
    srcFile_unmangled_linkage_name_dict: Dict[str, List[str]],
    func_set: Set[str],
    file2pc_loc: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    dump_pc: bool = True,
    dump_pos: bool = False,
    elf_file_path: str = "/path/to/elf",
    unmangle_names: bool = False,
    event_names: List[str] = ["Ir", "Cycles"],
    # event_names: List[str] = ["Ir"],
    pcs_hist_df=None,
):
    """
    Generate a callgrind-compatible profile.
    Supports pc-only, line-only, or combined instr+line mode.
    """
    bb_cost_trace_df = pd.DataFrame()
    bb_cost_trace_df["Ir"] = bb_trace_df.merge(unique_bbs_df, how="left", left_on="bb_idx", right_index=True)[
        "num_instrs"
    ]
    bb_cost_trace_df["Cycles"] = bb_cost_trace_df["Ir"] * 2
    # print("bb_cost_trace_df", bb_cost_trace_df)
    # event_arrays = {ev: bb_cost_trace_df[ev].to_numpy() for ev in event_names}
    event_arrays = np.vstack([bb_cost_trace_df[ev].to_numpy() for ev in event_names])

    inclusive_cost_dict = callgrind_format_get_inclusive_cost(
        bb_trace_df, unique_bbs_df, event_names=event_names, event_arrays=event_arrays
    )

    # --- helpers -------------------------------------------------------------
    def find_srcFile(target_func: str) -> str:
        for d in (srcFile_func_dict, srcFile_linkage_name_dict):
            for srcFile, funcs in d.items():
                if target_func in funcs:
                    return srcFile
        return "???"

    def aggregate_pos_cost_of_func(
        sorted_bb_lists: list,
        trace_pcs,
        unique_bbs_df,
        bb_trace_df,
        event_arrays,
        event_names,
        pcs_hist_df: pd.DataFrame,
    ):
        """
        Aggregate per-PC cost vectors for a function.
        Returns: dict pc -> np.ndarray (len(event_names))
        """
        # print("sorted_bb_lists", sorted_bb_lists)
        n_events = len(event_names)
        position_cost_dict = defaultdict(lambda: np.zeros(n_events, dtype=np.int64))
        # bb_trace_arr = bb_trace_df["bb_idx"].to_numpy()

        for bb_idx in sorted_bb_lists:
            bb = unique_bbs_df.iloc[bb_idx]
            pc_start, pc_end = bb.first_pc, bb.last_pc

            # Find all executions of this BB in the trace
            # exec_rows = np.where(bb_trace_arr == bb_idx)[0]

            # for row_idx in exec_rows:
            matching = pcs_hist_df[pcs_hist_df["pc"].isin(range(pc_start, pc_end + 2, 2))]
            # print("matching", matching)
            for row in matching.itertuples():
                pc = row.pc
                count = row.count
                metrics_vec = np.array([count, count * 2])
                position_cost_dict[pc] += metrics_vec
            # if True:
            #     # metrics_vec = np.array([event_arrays[ev][row_idx] for ev in event_names], dtype=np.int64)
            #     # metrics_vec = event_arrays[:, row_idx]

            #     for pc in range(pc_start, pc_end + 2, 2):
            #         if pc not in trace_pcs:
            #             continue
            #         # metrics_vec = np.array([event_arrays[ev][row_idx] for ev in event_names], dtype=np.int64)
            #         # position_cost_dict[pc] += metrics_vec
            #         position_cost_dict[pc] += metrics_vec
        # print("position_cost_dict", position_cost_dict)
        # input("k")
        return position_cost_dict

    def get_line_for_pc(pc: int, source_file: str) -> str:
        if source_file == "???":
            return "0"
        if file2pc_loc is None:
            return "0"
        mapping_ = file2pc_loc.get(source_file)
        if not mapping_:
            return "0"
        mapping_first = [x[0] for x in mapping_]
        i = bisect.bisect_right(mapping_first, pc)
        if i:
            start_pc, line = mapping_[i - 1]
            if pc >= start_pc:
                return str(line)
        return "0"

    # --- positions mode ------------------------------------------------------
    if dump_pc and dump_pos:
        positions = "instr line"
    elif dump_pc:
        positions = "instr"
    else:
        positions = "line"

    events_str = " ".join(event_names)
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
events: {events_str}
summary:

"""

    # --- main output ---------------------------------------------------------
    callgrind_output = ""
    srcFile_to_func = defaultdict(list)
    for func in func_set:
        srcFile_to_func[find_srcFile(func)].append(func)

    for srcFile, funcs in srcFile_to_func.items():
        callgrind_output += f"ob={posixpath.abspath(elf_file_path)}\n"
        callgrind_output += f"fl={srcFile}\n"

        for func in funcs:
            # print("func", func)
            bb_list = func2bbs[func]
            bb_list.sort(key=lambda bb_idx: unique_bbs_df.iloc[bb_idx].first_pc)
            if unmangle_names:
                func = unmangle_helper(func)
            callgrind_output += f"fn={func}\n"

            # position_cost_dict = aggregate_pos_cost_of_func(bb_list)
            position_cost_dict = aggregate_pos_cost_of_func(
                bb_list,
                trace_pcs,
                unique_bbs_df,
                bb_trace_df,
                event_arrays,
                event_names,
                pcs_hist_df,
            )
            branch_pc_list = unique_bbs_df.iloc[bb_list].last_pc.values

            for pc in sorted(position_cost_dict.keys()):
                # cost = position_cost_dict[pc]
                costs = position_cost_dict[pc]
                # print("costs", costs)
                costs_str = " ".join(map(str, costs))
                if dump_pc and dump_pos:
                    line = get_line_for_pc(pc, srcFile)
                    callgrind_output += f"{hex(pc)} {line} {costs_str}\n"
                elif dump_pc:
                    callgrind_output += f"{hex(pc)} {costs_str}\n"
                else:  # dump_pos only
                    line = get_line_for_pc(pc, srcFile)
                    callgrind_output += f"{line} {costs_str}\n"

                if pc in branch_pc_list and pc in inclusive_cost_dict:
                    for callee_pc, inclusive_cost in inclusive_cost_dict[pc].items():
                        # print("inclusive_cost", inclusive_cost)
                        # input("123")
                        callgrind_output += f"cob={posixpath.abspath(elf_file_path)}\n"
                        callee_func = find_func_name(mapping, callee_pc)
                        callgrind_output += f"cfi={find_srcFile(callee_func)}\n"
                        if unmangle_names:
                            callee_func = unmangle_helper(callee_func)
                        callgrind_output += f"cfn={callee_func}\n"
                        callgrind_output += f"calls={len(inclusive_cost)} {hex(callee_pc)}\n"
                        total_vec = np.sum(np.stack(inclusive_cost), axis=0)  # shape: (n_events,)
                        sum_str = " ".join(map(str, total_vec))
                        if dump_pc and dump_pos:
                            line = get_line_for_pc(pc, srcFile)
                            # sum all vectors in the list
                            callgrind_output += f"{hex(pc)} {line} {sum_str}\n"
                        elif dump_pc:
                            callgrind_output += f"{hex(pc)} {sum_str}\n"
                        else:
                            line = get_line_for_pc(pc, srcFile)
                            callgrind_output += f"{line} {sum_str}\n"

            callgrind_output += "\n"

    return prologue + callgrind_output


def generate_callgrind_output(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    force: bool = False,
    dump_pc: bool = False,
    dump_pos: bool = False,
    unmangle_names: bool = False,
):
    artifacts = sess.artifacts

    bb_trace_artifacts = filter_artifacts(artifacts, lambda x: x.name == "bb_trace")
    assert len(bb_trace_artifacts) == 1
    bb_trace_artifact = bb_trace_artifacts[0]
    bb_trace_df = bb_trace_artifact.df

    unique_bbs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "unique_bbs")
    assert len(unique_bbs_artifacts) == 1
    unique_bbs_artifact = unique_bbs_artifacts[0]
    unique_bbs_df = unique_bbs_artifact.df

    func2bbs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "func2bbs")
    assert len(func2bbs_artifacts) == 1
    func2bbs_artifact = func2bbs_artifacts[0]
    func2bbs_df = func2bbs_artifact.df
    func2bbs = dict(zip(func2bbs_df["func"], func2bbs_df["bb_idxs"]))

    trace_pcs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "trace_pcs")
    assert len(trace_pcs_artifacts) == 1
    trace_pcs_artifact = trace_pcs_artifacts[0]
    trace_pcs_df = trace_pcs_artifact.df
    trace_pcs = set(trace_pcs_df.values)  # series?

    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]

    func2pc_artifacts = filter_artifacts(artifacts, lambda x: x.name == "func2pc")
    assert len(func2pc_artifacts) == 1
    func2pc_artifact = func2pc_artifacts[0]
    func2pc_df = func2pc_artifact.df

    pcs_hist_artifacts = filter_artifacts(artifacts, lambda x: x.name == "pcs_hist")
    assert len(pcs_hist_artifacts) == 1
    pcs_hist_artifact = pcs_hist_artifacts[0]
    pcs_hist_df = pcs_hist_artifact.df
    # print("pcs_hist_df", pcs_hist_df)

    file2funcs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "file2funcs")
    assert len(file2funcs_artifacts) == 1
    file2funcs_artifact = file2funcs_artifacts[0]
    file2funcs_df = file2funcs_artifact.df

    pc2locs_artifacts = filter_artifacts(artifacts, lambda x: x.name == "pc2locs")
    assert len(pc2locs_artifacts) == 1
    pc2locs_artifact = pc2locs_artifacts[0]
    pc2locs_df = pc2locs_artifact.df

    def helper(pc2locs_df):
        df_ = pc2locs_df.explode("locs")
        df_.reset_index(inplace=True)
        df_[["file", "line"]] = df_["locs"].str.split(":", n=1, expand=True)
        ret = {}
        for file, file_df in df_.groupby("file"):
            temp = list(map(tuple, file_df[["pc", "line"]].values))
            temp = sorted(temp, key=lambda x: x[0])
            ret[file] = temp
        return ret

    file2pc_loc = helper(pc2locs_df)

    mapping = func2pc_df.groupby("func")["pc_range"].apply(list).to_dict()
    # bbs, trace_pcs, func2bbs, bb_freq = collect_bbs(trace_artifact.df, mapping)
    # trace_pcs, func2bbs, unique_bbs_df, bb_trace_df = collect_bbs(trace_artifact.df, mapping)

    # bb_trace_attrs = {
    #     "trace": trace_artifact.name,
    #     "kind": "bb_trace",
    #     "by": __name__,
    # }

    # unique_bbs_attrs = {
    #     "trace": trace_artifact.name,
    #     "kind": "mapping",
    #     "by": __name__,
    # }

    # bb_trace_artifact = TableArtifact("bb_trace", bb_trace_df, attrs=bb_trace_attrs)
    # unique_bbs_artifact = TableArtifact("unique_bbs", unique_bbs_df, attrs=unique_bbs_attrs)

    # sess.add_artifact(bb_trace_artifact, override=force)
    # sess.add_artifact(unique_bbs_artifact, override=force)

    file2funcs = dict(list((file2funcs_df[["file", "func_names"]].to_records(index=False))))
    file2linkage_names = dict(list((file2funcs_df[["file", "linkage_names"]].to_records(index=False))))
    file2unmangled_linkage_names = dict(
        list((file2funcs_df[["file", "unmangled_linkage_names"]].to_records(index=False)))
    )
    func_set = set(func2bbs.keys())
    elf_file_path = elf_artifact.path
    # print("M", time.time())

    content = callgrind_format_converter(
        # bbs=bbs,
        bb_trace_df=bb_trace_df,
        unique_bbs_df=unique_bbs_df,
        trace_pcs=trace_pcs,
        mapping=mapping,
        func2bbs=func2bbs,
        # bb_freq=bb_freq,
        srcFile_func_dict=file2funcs,
        srcFile_linkage_name_dict=file2linkage_names,
        srcFile_unmangled_linkage_name_dict=file2unmangled_linkage_names,
        func_set=func_set,
        file2pc_loc=file2pc_loc,
        dump_pc=dump_pc,
        dump_pos=dump_pos,
        elf_file_path=elf_file_path,
        unmangle_names=unmangle_names,
        pcs_hist_df=pcs_hist_df,
    )
    # print("Z", time.time())

    if output is None:
        profile_dir = sess.directory / "profile"
        profile_dir.mkdir(exist_ok=True)
        out_name = "callgrind"
        if dump_pc:
            out_name += "_pc"
        if dump_pos:
            out_name += "_pos"
        out_name += ".out"
        output = profile_dir / out_name
    with open(output, "w") as f:
        f.write(content)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_callgrind_output(
        sess,
        output=args.output,
        force=args.force,
        dump_pc=args.dump_pc,
        dump_pos=args.dump_pos,
        unmangle_names=args.unmangle,
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
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--dump-pc", action="store_true")
    parser.add_argument("--dump-pos", action="store_true")
    parser.add_argument("--unmangle", action="store_true")
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
