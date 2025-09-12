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

# import time
import sys
import time
import argparse
import multiprocessing
from typing import List, Union, Optional
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from capstone import Cs, CS_ARCH_RISCV, CS_MODE_RISCV64, CS_MODE_RISCV32, CS_MODE_RISCVC
from elftools.elf.elffile import ELFFile
from elftools.elf.constants import SH_FLAGS

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


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


def disassemble_word(md, pc, word, size=4, endian="little", operands: bool = False):
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


def get_disassembler(elf_path):
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
    xlen = elf.elfclass
    mode = CS_MODE_RISCV32 if xlen == 32 else CS_MODE_RISCV64

    md = Cs(CS_ARCH_RISCV, mode | CS_MODE_RISCVC)
    md.detail = False
    return md


from concurrent.futures import ProcessPoolExecutor


def build_entries(rows, elf_path):
    fetcher = ELFInstructionFetcher(elf_path)
    md = get_disassembler(elf_path)

    out = []
    for pc, size in rows:
        bytecode = fetcher.read_word_at_pc(int(pc), size=int(size))
        instr = disassemble_word(md, int(pc), int(bytecode), size=int(size), operands=False)
        out.append(((pc, size), (bytecode, instr)))
    return out


def worker(row_chunk, elf_path):
    fetcher = ELFInstructionFetcher(elf_path)
    md = get_disassembler(elf_path)
    out = {}
    for pc, size in row_chunk:
        bytecode = fetcher.read_word_at_pc(pc, size=size)
        instr = disassemble_word(md, pc, bytecode, size=size, operands=False)
        out[(pc, size)] = (bytecode, instr)
    return out


import itertools


def parallel_dict_builder(rows, elf_path, nprocs=16):
    chunks = np.array_split(rows, nprocs)
    with ProcessPoolExecutor(max_workers=nprocs) as pool:
        results = pool.map(worker, chunks, itertools.repeat(elf_path))
    merged = {}
    for r in results:
        merged.update(r)
    return merged


def normalize_trace(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    force: bool = False,
):
    artifacts = sess.artifacts
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    elf_artifact = None
    if len(elf_artifacts) > 0:
        assert len(elf_artifacts) == 1
        elf_artifact = elf_artifacts[0]

    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    trace_df = trace_artifact.df
    # print("trace_df", len(trace_df), trace_df.memory_usage(deep=True), trace_df.memory_usage(deep=True).sum())

    # pc2bytecode = {}

    # def lookup_bytecode(pc, size):
    #     pc = int(pc)
    #     bytecode = pc2bytecode.get(pc)
    #     if bytecode is None:  # not in cache
    #         bytecode = fetcher.read_word_at_pc(pc, size=size)
    #         assert bytecode is not None
    #         pc2bytecode[pc] = bytecode
    #     return bytecode
    #
    # bytecode2name = {}
    #
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

    md = None

    def disassemble_row(row):
        return disassemble_word(md, int(row["pc"]), int(row["bytecode"]), size=int(row["size"]), operands=False)

    cols = trace_df.columns
    # print("cols", cols)
    has_pc = "pc" in cols
    has_bytecode = "bytecode" in cols
    has_size = "size" in cols
    has_instr = "instr" in cols
    # print("has", has_pc, has_bytecode, has_size, has_instr)
    assert has_pc, "Missing required column in trace_df: pc"

    inplace = True
    if not inplace:
        trace_df = trace_df.copy()

    if not has_size:
        logger.info("Filling missing column: size")
        # TODO: detect from major opcode or elf?
        raise NotImplementedError("size")
        has_size = True

    if not has_instr:
        assert elf_artifact is not None
        md = get_disassembler(elf_artifact.path)

    if not has_bytecode:
        logger.info("Filling missing column: bytecode")
        assert elf_artifact is not None
        fetcher = ELFInstructionFetcher(elf_artifact.path)
        assert has_size
        # unique_pcs = trace_df["pc"].unique()
        unique_pc_size = trace_df.drop_duplicates(subset=["pc"])[["pc", "size"]]
        unique_pc_size["bytecode"] = unique_pc_size.apply(
            lambda x: fetcher.read_word_at_pc(x["pc"], size=x["size"]), axis=1
        )
        # print("unique_pc_size", unique_pc_size)
        assert all(unique_pc_size["bytecode"].notna()), "Unable to get all bytecodes"
        unique_pc_size["bytecode"] = pd.to_numeric(unique_pc_size["bytecode"], downcast="unsigned")
        unique_pc_size["bytecode"] = unique_pc_size["bytecode"].astype("category")
        if not has_instr:
            logger.info("Filling missing column: instr")
            unique_pc_size["instr"] = unique_pc_size.apply(disassemble_row, axis=1)
            unique_pc_size["instr"] = unique_pc_size["instr"].astype("category")
            has_instr = True
        # pc_size_to_bytecode = dict(zip(zip(unique_pc_size["pc"], unique_pc_size["size"]), unique_pc_size["bytecode"]))
        # pc_size_to_instr = dict(zip(zip(unique_pc_size["pc"], unique_pc_size["size"]), unique_pc_size["instr"]))
        # trace_df["bytecode"] = list(map(lambda k: pc_size_to_bytecode[k], zip(trace_df["pc"], trace_df["size"])))
        # trace_df["instr"] = list(map(lambda k: pc_size_to_instr[k], zip(trace_df["pc"], trace_df["size"])))
        unique_pc_size.drop(columns=["size"], inplace=True)
        trace_df = trace_df.merge(unique_pc_size, on=["pc"], how="left")
        has_bytecode = True
        # rows = list(zip(unique_pc_size["pc"], unique_pc_size["size"]))
        # pc_size_dict = parallel_dict_builder(rows, elf_artifact.path, nprocs=16)
        # print("pc_size_dict", pc_size_dict)

        # trace_df["bytecode"] = [pc_size_dict[(pc, size)][0] for pc, size in zip(trace_df["pc"], trace_df["size"])]
        # trace_df["instr"] = [pc_size_dict[(pc, size)][1] for pc, size in zip(trace_df["pc"], trace_df["size"])]

    if not has_instr:

        logger.info("Filling missing column: instr")
        unique_pc_size = trace_df[["pc", "size"]].drop_duplicates()
        unique_pc_size["instr"] = unique_pc_size.apply(disassemble_row, axis=1)
        unique_pc_size["instr"] = unique_pc_size["instr"].astype("category")
        trace_df = trace_df.merge(unique_pc_size, on=["pc", "size"], how="left")
        has_instr = True

    # TODO: normalize dtypes! (check overhead)
    from pandas.api.types import is_unsigned_integer_dtype

    if not is_unsigned_integer_dtype(trace_df["pc"].dtype):
        logger.info("Fixing dtype of column: pc")
        trace_df["pc"] = pd.to_numeric(trace_df["pc"], downcast="unsigned")

    if not isinstance(trace_df["instr"].dtype, pd.CategoricalDtype):
        logger.info("Fixing dtype of column: instr")
        trace_df["instr"] = trace_df["instr"].astype("category")

    if not isinstance(trace_df["size"].dtype, pd.CategoricalDtype):
        logger.info("Fixing dtype of column: size")
        trace_df["size"] = trace_df["size"].astype("category")

    # TODO: handle is_branch/is_return

    if not isinstance(trace_df["bytecode"].dtype, pd.CategoricalDtype):
        logger.info("Fixing dtype of column: bytecode")
        if not is_unsigned_integer_dtype(trace_df["bytecode"].dtype):
            trace_df["bytecode"] = pd.to_numeric(trace_df["bytecode"], downcast="unsigned")
        trace_df["bytecode"] = trace_df["bytecode"].astype("category")

    # TODO: check if saved?
    # print(
    #     "trace_df",
    #     len(trace_df),
    #     trace_df.memory_usage(deep=True),
    #     trace_df.memory_usage(deep=True).sum(),
    #     trace_df.dtypes,
    # )

    if not inplace:
        artifact = InstrTraceArtifact(trace_artifact.name, trace_df, attrs=trace_artifact.attrs)
        sess.add_artifact(artifact, override=force)
    else:
        trace_artifact._data = trace_df
        trace_artifact.update()  # TODO: pass new data via arg


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    normalize_trace(
        sess,
        force=args.force,  # TODO: check if update requires force?
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
