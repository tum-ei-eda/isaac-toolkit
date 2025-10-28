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

# import time
import re
import sys
import pandas as pd
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact, TableArtifact


# TODO: logger
logger = logging.getLogger(__name__)


def load_riscvovpsimplus_trace(
    sess,
    input_file: Path,
    force: bool = False,
    progress: bool = False,
    chunksize: int = 2**20,
):
    logger.info("Loading riscvOVPSimPlus instruction trace...")
    assert input_file.is_file()
    name = input_file.name

    instr_chunks = []
    reg_chunks = []
    mem_chunks = []

    instr_idx = 0  # running instruction index

    def flush_chunks():
        """Convert buffers into DataFrames for concatenation."""
        df_instr = pd.DataFrame(instr_chunks)
        df_reg = pd.DataFrame(reg_chunks)
        df_mem = pd.DataFrame(mem_chunks)

        return df_instr, df_reg, df_mem

    with open(input_file, "r") as f:
        buffer_instr = []
        buffer_reg = []
        buffer_mem = []

        for line in tqdm(f, disable=not progress):
            line = line.strip()
            if not line.startswith("Info"):
                continue
            # print("line0", line)
            # Instruction line
            if re.match(r"Info '.*cpu.*',", line):
                m = re.match(
                    r"Info '.*',\s*(0x[0-9a-fA-F]+)\((.*?)\):\s+([0-9a-fA-F]+)\s+(\S+)(?:\s+(.*))?",
                    line,
                )
                if m:
                    pc, symbol, bytecode, instr, args = m.groups()
                    buffer_instr.append(
                        {
                            # "idx": instr_idx,
                            "pc": int(pc, 16),
                            "symbol": symbol,
                            "bytecode": int(bytecode, 16),
                            "instr": instr,
                            "args": args if args else "",
                            "size": 2 if instr.startswith("c.") or len(bytecode) <= 4 else 4,
                        }
                    )
                    instr_idx += 1

            # Register update line
            elif re.match(r"Info\s+[a-zA-Z0-9]+ ", line):
                print("line1", line)
                m = re.match(r"Info\s+(\w+)\s+([0-9a-fA-F]+)\s*->\s*([0-9a-fA-F]+)", line)
                if m:
                    reg, old_val, new_val = m.groups()
                    buffer_reg.append(
                        {
                            "idx": instr_idx - 1,  # belongs to last instr
                            "reg": reg,
                            "old_val": int(old_val, 16),
                            "new_val": int(new_val, 16),
                        }
                    )

                m = re.match(
                    r"Info\s+(MEM\w)\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+(\d+)\s+([0-9a-fA-F]+)",
                    line,
                )

                # if "MEM" in line:
                #    # print("line", line)
                #    print("m", m)
                if m:
                    acc_type, addr_lo, addr_hi, size, data = m.groups()
                    buffer_mem.append(
                        {
                            "idx": instr_idx - 1,  # belongs to last instr
                            "type": acc_type,
                            "addr_lo": int(addr_lo, 16),
                            "addr_hi": int(addr_hi, 16),
                            "size": int(size),
                            "data": int(data, 16),
                        }
                    )

            # Flush to disk in batches
            if len(buffer_instr) >= chunksize:
                instr_chunks.extend(buffer_instr)
                reg_chunks.extend(buffer_reg)
                mem_chunks.extend(buffer_mem)
                buffer_instr, buffer_reg, buffer_mem = [], [], []

        # final flush
        instr_chunks.extend(buffer_instr)
        reg_chunks.extend(buffer_reg)
        mem_chunks.extend(buffer_mem)

    # Build DataFrames
    df_instr = pd.DataFrame(instr_chunks)
    if not df_instr.empty:
        df_instr["instr"] = df_instr["instr"].astype("category")
        df_instr["size"] = df_instr["size"].astype("category")

    df_reg = pd.DataFrame(reg_chunks)
    df_mem = pd.DataFrame(mem_chunks)

    # Wrap artifacts
    attrs = {
        "simulator": "riscvOVPSimPlus",
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.instr_trace.riscvovpsimplus",
    }
    instr_artifact = InstrTraceArtifact(name, df_instr, attrs=attrs)
    sess.add_artifact(instr_artifact, override=force)

    if not df_reg.empty:
        sess.add_artifact(TableArtifact(name + "_reg_updates", df_reg), override=force)
    if not df_mem.empty:
        sess.add_artifact(TableArtifact(name + "_mem_updates", df_mem), override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_file = Path(args.file)
    load_riscvovpsimplus_trace(sess, input_file, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--operands", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
