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
import io
import sys
import argparse
import multiprocessing
from typing import List, Union, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm
from elftools.elf.elffile import ELFFile
from elftools.elf.constants import SH_FLAGS

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact

from .helper import process_df

# from pandarallel import pandarallel

# pandarallel.initialize()

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
    except Exception:
        # print(e)
        # input("%2")
        if operands:
            return "unknown", ""
        return "unknown"


def parse_and_process(chunk_bytes):
    df = pd.read_csv(io.BytesIO(chunk_bytes), header=None, sep=",")
    return process_df(df)


def chunk_iter(path, chunk_size=2**22):
    with open(path, "rb") as f:
        buf = b""
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            buf += data
            last_nl = buf.rfind(b"\n")
            if last_nl == -1:
                continue
            yield buf[: last_nl + 1]
            buf = buf[last_nl + 1 :]
        if buf:
            yield buf


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


def chunk_iter_compressed(path, chunk_size=2**22):
    print("chunk_iter_compressed")
    fmt = check_compressed(path)
    if fmt == "gzip":
        import gzip

        f = gzip.open(path, "rb")
    elif fmt == "lz4":
        import lz4.frame

        f = lz4.frame.open(path, "rb")
    else:
        f = open(path, "rb")

    with f:
        buf = b""
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            buf += data
            last_nl = buf.rfind(b"\n")
            if last_nl == -1:
                continue
            yield buf[: last_nl + 1]
            buf = buf[last_nl + 1 :]
        if buf:
            yield buf


def load_instr_trace(
    sess: Session,
    input_files: Union[Path, List[Path]],
    force: bool = False,
    operands: bool = False,
    num_workers: Optional[int] = None,
    executor: str = "process_pool",
    chunk_size: int = 2**22,
):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    if operands:
        raise NotImplementedError("operands not included in pctrace")
    if not isinstance(input_files, list):
        input_files = [input_files]
    assert len(input_files) > 0
    name = input_files[0].name

    # elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    # assert len(elf_artifacts) == 1, "TGC pctrace needs ELF artifact to lookup instructions"
    # elf_artifact = elf_artifacts[0]

    # sort input files by name
    sorted_files = sorted(input_files, key=lambda x: x.name)
    # df = pd.read_csv(input_file, sep=":", names=["pc", "rest"])
    dfs = []
    # with open(elf_artifact.path, "rb") as f:
    #     elf = ELFFile(f)
    # xlen = elf.elfclass
    # mode = CS_MODE_RISCV32 if xlen == 32 else CS_MODE_RISCV64

    # md = Cs(CS_ARCH_RISCV, mode | CS_MODE_RISCVC)
    # md.detail = False

    # fetcher = ELFInstructionFetcher(elf_artifact.path)
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

    # print("A", time.time())
    # t0 = time.time()
    # executor =
    executor_map = {
        "thread_pool": ThreadPoolExecutor,
        "process_pool": ProcessPoolExecutor,
    }
    executor_cls = executor_map.get(executor)
    assert executor_cls is not None, f"Unsupported Executor: {executor}"
    with executor_cls(max_workers=num_workers) as executor:  # tune workers
        # futures = []
        for input_file in sorted_files:
            assert input_file.is_file()
            # print("file", input_file)
            # with pd.read_csv(input_file, sep=",", chunksize=2**20, header=None) as reader:
            # with smart_open(input_file, "rb") as f:
            if True:
                # with pd.read_csv(input_file, sep=",", chunksize=2**22, header=None) as reader:
                # with pd.read_csv(f, sep=",", chunksize=2**22, header=None) as reader:
                # dfs_ = list(executor.map(parse_and_process, chunk_iter(f)))
                dfs_ = list(
                    tqdm(
                        executor.map(parse_and_process, chunk_iter_compressed(input_file, chunk_size=chunk_size)),
                        disable=False,
                    )
                )
                dfs += dfs_
                # with pd.read_csv(f, sep=",", chunksize=2**22, header=None) as reader:
                #     # for df in tqdm(reader, disable=False):
                #     for df in reader:
                #         # print("a")
                #         # future = executor.submit(process_df, df, fetcher, md)
                #         # dfs.append(process_df(df))
                #         future = executor.submit(process_df, df)
                #         futures.append(future)
        # for future in tqdm(as_completed(futures), total=len(futures)):
        #     # print("b")
        #     dfs.append(future.result())
    df = pd.concat(dfs, axis=0)
    # df["instr"] = df["instr"].astype("category")
    df["size"] = df["size"].astype("category")
    df["is_branch"] = df["is_branch"].astype("category")
    df["pc"] = pd.to_numeric(df["pc"], downcast="unsigned")
    # df["bytecode"] = pd.to_numeric(df["bytecode"], downcast="unsigned")
    df.reset_index(drop=True, inplace=True)
    # print("B", time.time() - t0)
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
    load_instr_trace(
        sess,
        input_files,
        force=args.force,
        operands=args.operands,
        num_workers=args.num_workers,
        executor=args.executor,
        chunk_size=args.chunk_size,
    )
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
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--chunk-size", type=int, default=2**22)
    parser.add_argument("--executor", choices=["process_pool", "thread_pool"], type=str, default="process_pool")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
