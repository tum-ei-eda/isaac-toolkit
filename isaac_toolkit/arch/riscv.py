riscv_branch_instrs = [
    "j",  # pseudo
    "jr",  # pseudo
    "ret",  # pseudo
    "mret",  # pseudo
    "sret",  # pseudo
    "uret",  # pseudo
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
    "c.jr",
    "c.j",
    "c.beqz",
    "c.jalr",
    "c.jal",
]
riscv_return_instrs = ["jalr", "cjalr", "cjr"]  # TODO
