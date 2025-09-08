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
import sys
import pandas as pd
import argparse
from typing import List, Union
from pathlib import Path
from contextlib import contextmanager

from tqdm import tqdm
from capstone import Cs, CS_ARCH_RISCV, CS_MODE_RISCV64, CS_MODE_RISCV32, CS_MODE_RISCVC
from elftools.elf.elffile import ELFFile
from elftools.elf.constants import SH_FLAGS

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


# TODO: logger


class ELFInstructionFetcher:
    def __init__(self, elf_path):
        self.sections = []
        with open(elf_path, "rb") as f:
            elf = ELFFile(f)
            for section in elf.iter_sections():
                flags = section["sh_flags"]
                if flags & SH_FLAGS.SHF_EXECINSTR:
                    base = section["sh_addr"]
                    size = section["sh_size"]
                    data = section.data()
                    self.sections.append((base, base + size, data))

    def read_word_at_pc(self, pc, size=4):
        """Fetch `size` bytes at `pc` from any executable section."""
        for start, end, data in self.sections:
            if start <= pc < end:
                offset = pc - start
                if offset + size > len(data):
                    return None  # avoid out-of-bounds read
                return int.from_bytes(data[offset : offset + size], "little")
        return None  # pc not found


def load_instr_trace(sess: Session, input_files: Union[Path, List[Path]], force: bool = False, operands: bool = False):
    if operands:
        raise NotImplementedError("operands not included in pctrace")
    if not isinstance(input_files, list):
        input_files = [input_files]
    assert len(input_files) > 0
    name = input_files[0].name

    artifacts = sess.artifacts
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1, "TGC pctrace needs ELF artifact to lookup instructions"
    elf_artifact = elf_artifacts[0]

    # sort input files by name
    sorted_files = sorted(input_files, key=lambda x: x.name)
    # df = pd.read_csv(input_file, sep=":", names=["pc", "rest"])
    dfs = []
    with open(elf_artifact.path, "rb") as f:
        elf = ELFFile(f)
    xlen = elf.elfclass
    mode = CS_MODE_RISCV32 if xlen == 32 else CS_MODE_RISCV64

    md = Cs(CS_ARCH_RISCV, mode | CS_MODE_RISCVC)
    md.detail = False

    def disassemble_word(pc, word, size=4, endian="little", operands: bool = False):
        # print("disassemble_word", pc, word, size)
        try:
            code = word.to_bytes(size, endian)
            # print("code", code)
            insn = next(md.disasm(code, pc))
            # print("insn", insn)
            # input("%1")
            if operands:
                return insn.mnemonic, insn.op_str
            return insn.mnemonic
        except Exception as e:
            # print(e)
            # input("%2")
            if operands:
                return "unknown", ""
            return "unknown"

    fetcher = ELFInstructionFetcher(elf_artifact.path)

    # pc2bytecode = {}

    # def lookup_bytecode(pc, size):
    #     pc = int(pc)
    #     bytecode = pc2bytecode.get(pc)
    #     if bytecode is None:  # not in cache
    #         bytecode = fetcher.read_word_at_pc(pc, size=size)
    #         assert bytecode is not None
    #         pc2bytecode[pc] = bytecode
    #     return bytecode

    # bytecode2name = {}

    # def lookup_name(bytecode, pc=None, size=None):
    #     bytecode = int(bytecode)
    #     name = bytecode2name.get(bytecode)
    #     if name is None:  # not in cache
    #         assert pc is not None
    #         assert size is not None
    #         name = disassemble_word(pc, bytecode, size=size)
    #         assert name is not None
    #         bytecode2name[bytecode] = name
    #     # print("name", name)
    #     # input("?")
    #     return name

    def check_compressed(p):
        with open(p, "rb") as test_f:
            head = test_f.read(4)

        is_gzip = head.startswith(b"\x1f\x8b")
        is_lz4 = head == b"\x04\x22\x4D\x18"

        if is_gzip:
            return "gzip"
        if is_lz4:
            return "lz4"
        return "none"

    @contextmanager
    def smart_open(path, mode="rb"):
        fmt = check_compressed(path)
        if fmt == "gzip":
            import gzip

            f = gzip.open(path, mode)
        elif fmt == "lz4":
            import lz4.frame

            f = lz4.frame.open(path, mode)
        else:
            f = open(path, mode)

        try:
            yield f
        finally:
            f.close()

    for input_file in sorted_files:
        assert input_file.is_file()
        # print("file", input_file)
        # with pd.read_csv(input_file, sep=",", chunksize=2**20, header=None) as reader:
        with smart_open(input_file, "rb") as f:
            # with pd.read_csv(input_file, sep=",", chunksize=2**22, header=None) as reader:
            with pd.read_csv(f, sep=",", chunksize=2**22, header=None) as reader:
                for df in tqdm(reader, disable=False):
                    assert len(df.columns) == 4, "Excpected 4 columns"
                    df = df.rename(columns={0: "pc", 1: "?", 2: "is_branch", 3: "size"})
                    df["is_branch"] = df["is_branch"].astype("category")
                    # print("A", time.time())
                    df["pc"] = df["pc"].apply(lambda x: int(x, 0))
                    df["pc"] = pd.to_numeric(df["pc"])
                    df["size"] = df["size"].astype(int)
                    df["size"] = df["size"].astype("category")
                    df.drop(columns=["?"], inplace=True)
                    # df.drop(columns=["is_branch"], inplace=True)

                    # df["bytecode"] = df[["pc", "size"]].apply(lambda x: lookup_bytecode(x["pc"], x["size"]), axis=1)
                    # df["instr"] = df[["pc", "bytecode", "size"]].apply(
                    #     lambda x: lookup_name(x["bytecode"], pc=x["pc"], size=x["size"]), axis=1
                    # )
                    # df["instr"] = df["instr"].astype("category")
                    # print("A")
                    unique_pc_size = df[["pc", "size"]].drop_duplicates()
                    unique_pc_size["bytecode"] = unique_pc_size.apply(
                        lambda x: fetcher.read_word_at_pc(x["pc"], size=x["size"]), axis=1
                    )
                    # print("B")

                    def disassemble_row(row):
                        return disassemble_word(
                            int(row["pc"]), int(row["bytecode"]), size=int(row["size"]), operands=False
                        )

                    unique_pc_size["instr"] = unique_pc_size.apply(disassemble_row, axis=1)
                    # print("unique_pc_size", unique_pc_size)
                    # input(">>")
                    # print("C")
                    df = df.merge(unique_pc_size, on=["pc", "size"], how="left")
                    df["bytecode"] = pd.to_numeric(df["bytecode"], downcast="unsigned")
                    df["instr"] = df["instr"].astype("category")
                    # print("D")

                    # print("pc2bytecode", len(pc2bytecode))
                    # print("df", df.head(), df.columns, df.dtypes, df.memory_usage())
                    dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df["instr"] = df["instr"].astype("category")
    df["size"] = df["size"].astype("category")
    df["is_branch"] = df["is_branch"].astype("category")
    df["pc"] = pd.to_numeric(df["pc"], downcast="unsigned")
    df["bytecode"] = pd.to_numeric(df["bytecode"], downcast="unsigned")
    df.reset_index(drop=True, inplace=True)
    # print("df", df.head(), df.columns, df.dtypes, df.memory_usage())
    # input(">")

    attrs = {
        "simulator": "tgc",
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.instr_trace.tgc",
    }
    artifact = InstrTraceArtifact(name, df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_files = list(map(Path, args.files))
    load_instr_trace(sess, input_files, force=args.force, operands=args.operands)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
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
