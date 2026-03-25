#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict, deque


def process_trace(input_file, instr_out, perf_out=None):
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)

        last_wb_pc = None
        last_ex_pc = None
        last_instr = None
        # instr_counter = 0
        # Queue of IF cycles per PC
        pc_to_if_cycles = defaultdict(deque)
        pc_to_ex_cycles = defaultdict(deque)

        with open(instr_out, "w") as out_instr:
            out_instr.write("cycle,pc,bytecode\n")

            f_perf = None
            if perf_out:
                f_perf = open(perf_out, "w")
                # f_perf.write("instr_id,pc,instr,if_cycle,wb_cycle\n")
                f_perf.write("if_cycle,ex_cycle,wb_cycle,lat\n")

            for row in reader:
                cycle = int(row["mcycle_o"], 10)
                pc_if = int(row["current_IF_PC"], 16)
                pc_ex = int(row["current_EX_PC"], 16)
                pc_wb = int(row["current_WB_PC"], 16)
                instr_wb = int(row["instruction_wb"], 16)

                # Track IF stage entry
                pc_to_if_cycles[pc_if].append(cycle)

                # Skip empty WB stage
                if instr_wb == 0:
                    continue

                if pc_ex != last_ex_pc:
                    pc_to_ex_cycles[pc_ex].append(cycle)
                last_ex_pc = pc_ex

                # Skip duplicates (pipeline stalls)
                if pc_wb == last_wb_pc and instr_wb == last_instr:
                    continue

                # Get IF cycle for this instruction
                # print("pc_wb", pc_wb)
                # print("len(pc_to_ex_cycles[pc_wb])", pc_to_ex_cycles[pc_wb], len(pc_to_ex_cycles[pc_wb]))
                # print("len(pc_to_if_cycles[pc_wb])", pc_to_if_cycles[pc_wb], len(pc_to_if_cycles[pc_wb]))
                if_cycle = -1
                ex_cycle = -1
                if_dropped = 0
                ex_dropped = 0
                while len(pc_to_if_cycles[pc_wb]) > 0:
                    if if_cycle != -1:
                        if_dropped += 1
                    if_cycle = pc_to_if_cycles[pc_wb].popleft()
                while len(pc_to_ex_cycles[pc_wb]) > 0:
                    if ex_cycle != -1:
                        ex_dropped += 1
                    ex_cycle = pc_to_ex_cycles[pc_wb].popleft()
                # if if_dropped or ex_dropped:
                #     print("IF dropped:", if_dropped)
                #     print("EX dropped:", ex_dropped)
                #     # input(">")

                # if_cycle = pc_to_if_cycles[pc_wb].pop() if pc_to_if_cycles[pc_wb] else -1
                # ex_cycle = pc_to_ex_cycles[pc_wb].pop() if pc_to_ex_cycles[pc_wb] else -1
                # if_cycle = pc_to_if_cycles[pc_wb].popleft() if pc_to_if_cycles[pc_wb] else -1
                # ex_cycle = -1
                # while ex_cycle <= (if_cycle + 1):
                #     print("while", ex_cycle)
                #     ex_cycle = pc_to_ex_cycles[pc_wb].popleft() if pc_to_ex_cycles[pc_wb] else -1
                # print("wb_cycle", cycle)
                # print("ex_cycle", ex_cycle)
                # print("if_cycle", if_cycle)
                # print("pc_to_if_cycles", pc_to_if_cycles)
                # input(">")

                out_instr.write(f"{cycle},{hex(pc_wb)},{hex(instr_wb)}\n")
                if f_perf:
                    # f_perf.write(f"{instr_counter},{hex(pc_wb)},{hex(instr_wb)},{if_cycle},{cycle}\n")
                    lat = cycle - if_cycle + 1
                    # if True:
                    # print("lat", lat)
                    # if lat != 4:
                    #     print("wb_pc", pc_wb)
                    #     input(">")
                    f_perf.write(f"{if_cycle},{ex_cycle},{cycle},{lat}\n")
                    # instr_counter += 1

                last_wb_pc = pc_wb
                last_instr = instr_wb


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: process_vicuna_trace.py <input> <instr_out> [perf_out]")
        sys.exit(1)

    input_file = sys.argv[1]
    instr_out = sys.argv[2]
    perf_out = sys.argv[3] if len(sys.argv) > 3 else None

    process_trace(input_file, instr_out, perf_out)
