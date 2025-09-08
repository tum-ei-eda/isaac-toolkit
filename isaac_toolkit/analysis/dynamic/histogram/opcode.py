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
import sys
import logging
import argparse
from typing import Union
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)

RISCV_OPCODE_MAPPING = {
    0b0010011: "OP-IMM",
    0b0110111: "LUI",
    0b0010111: "AUIPC",
    0b0110011: "OP",
    0b1101111: "JAL",
    0b1100111: "JALR",
    0b1100011: "BRANCH",
    0b0000011: "LOAD",
    0b0100011: "STORE",
    0b0001111: "MISC-MEM",
    0b1110011: "SYSTEM",
    0b1000011: "MADD",
    0b1000111: "MSUB",
    0b1001011: "MNSUB",
    0b1001111: "MNADD",
    0b0000111: "LOAD-FP",
    0b0100111: "STORE-FP",
    0b0001011: "custom-0",
    0b0101011: "custom-1",
    0b1011011: "custom-2/rv128",
    0b1111011: "custom-3/rv128",
    0b1101011: "reserved",
    0b0101111: "AMO",
    0b1010011: "OP-FP",
    0b1010111: "OP-V",
    0b1110111: "OP-P",
    0b0011011: "OP-IMM-32",
    0b0111011: "OP-32",
}

RISCV_COMPRESSED_OPCODE_MAPPING = {
    0b00000: "OP-IMM",
    0b00001: "OP-IMM",
    0b00010: "OP-IMM",
    0b00100: "LOAD",
    0b00101: "JAL",
    0b00110: "LOAD-FP",
    0b01000: "LOAD",
    0b01001: "OP-IMM",
    0b01010: "LOAD",
    0b01100: "LOAD-FP",
    0b01101: "OP-IMM",
    0b01110: "LOAD-FP",
    0b10000: "reserved",
    0b10001: "MISC-ALU",
    0b10010: "JALR",
    0b10100: "STORE-FP",
    0b10101: "JAL",
    0b10110: "STORE-FP",
    0b11000: "STORE",
    0b11001: "BRANCH",
    0b11010: "STORE",
    0b11100: "STORE-FP",
    0b11101: "BRANCH",
    0b11110: "STORE-FP",
}


def decode_opcode(instr_word: Union[str, int]):
    if isinstance(instr_word, str):
        instr_word = int(instr_word, 0)
    assert isinstance(instr_word, int)
    opcode = instr_word & 0b1111111
    lsbs = opcode & 0b11
    if lsbs == 0b11:
        major = RISCV_OPCODE_MAPPING.get(opcode, "UNKNOWN")
        return major
    else:
        # 16-bit instruction
        msbs = (instr_word & 0b1110000000000000) >> 13
        combined = msbs << 2 | lsbs
        assert combined in RISCV_COMPRESSED_OPCODE_MAPPING.keys()
        return f"{RISCV_COMPRESSED_OPCODE_MAPPING[combined]} (Compressed)"


def collect_opcodes(trace_df):
    # Step 1: Count bytecode frequencies directly
    bytecode_counts = trace_df["bytecode"].value_counts()

    # Step 2: Decode each unique bytecode only once
    decoded_map = {bc: decode_opcode(bc) for bc in bytecode_counts.index}

    # Step 3: Replace bytecodes with opcodes and sum counts
    opcode_counts = {}
    for bc, count in bytecode_counts.items():
        op = decoded_map[bc]
        opcode_counts[op] = opcode_counts.get(op, 0) + count

    # Step 4: Build DataFrame
    opcodes_df = pd.DataFrame([{"opcode": op, "count": cnt} for op, cnt in opcode_counts.items()])
    total_count = opcodes_df["count"].sum()
    opcodes_df["rel_count"] = opcodes_df["count"] / total_count
    return opcodes_df.sort_values("count", ascending=False)


def create_opcode_hist(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]

    opcodes_df = collect_opcodes(trace_artifact.df)

    attrs = {
        "trace": trace_artifact.name,
        "kind": "histogram",
        "by": __name__,
    }

    opcodes_artifact = TableArtifact("opcodes_hist", opcodes_df, attrs=attrs)
    sess.add_artifact(opcodes_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    create_opcode_hist(sess, force=args.force)
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
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
