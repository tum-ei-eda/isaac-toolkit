#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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
riscv_branch_instrs = [
    "j",  # pseudo
    "jr",  # pseudo
    "call",  # pseudo
    "tail",  # pseudo
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
    "blez",  # new?
    "bltu",
    "bgtu",  # pseudo
    "bgtu",  # pseudo
    "bgeu",  # pseudo
    "bleu",
    "ecall",
    "bnez",  # bseudo
    "cbnez",
    "c.bnez",
    "cj",  # pseudo
    "cbeqz",
    "cjal",
    "c.j",
    "c.j",
    "c.beqz",
    "c.jal",
    # "ret",  # pseudo
    # "mret",  # pseudo
    # "sret",  # pseudo
    # "uret",  # pseudo
    # "jalr"  # return if rd=x0, rs1=x1
    # "c.jr",  # return if rs1=x1
    # "c.jalr",  # return if  rs1=x1
]
riscv_return_instrs = ["jalr", "cjalr", "cjr", "c.jr", "c.jalr", "ret", "mret", "sret", "uret"]  # TODO


def detect_riscv_instr_size(bytecode):  # TODO: move to riscv utils
    major_opcode = bytecode & 0b1111111
    bits10 = major_opcode & 0b11
    bits432 = (major_opcode >> 2) & 0b111
    bits65 = (major_opcode >> 5) & 0b11
    if bits10 != 0b11:
        return 16
    if bits432 != 0b111:
        return 32
    if bits65 == 0b00:
        return 48
    elif bits65 == 0b01:
        return 64
    elif bits65 == 0b10:
        return 48
    elif bits65 == 0b11:
        raise NotImplementedError("Encoding size >=80b not supported")
        return ">=80"
    assert False, "Should not be reached"
    return 0
