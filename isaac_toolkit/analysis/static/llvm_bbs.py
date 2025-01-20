#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
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
import io
import sys
import leb128
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from capstone import Cs, CS_ARCH_RISCV, CS_MODE_RISCV32, CS_MODE_RISCV64, CS_MODE_RISCVC

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("llvm_bbs")


def parse_elf(elf_path):
    GISEL = True  # TODO: make arg
    func_to_addrs = defaultdict(list)
    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)

        # extract disassembly to count instructions per bb
        code = elffile.get_section_by_name(".text")
        ops = code.data()
        addr = code["sh_addr"]
        mode = CS_MODE_RISCV32 if xlen == 32 else CS_MODE_RISCV64
        md = Cs(CS_ARCH_RISCV, mode | CS_MODE_RISCVC)
        valid_pcs = set(x.address for x in md.disasm(ops, addr))

        section = elffile.get_section_by_name(".symtab")
        xlen = elffile.elfclass
        assert xlen is not None
        addr_bytes = int(xlen / 8)

        if not section:
            pass
            # print('  No symbol table found. Perhaps this ELF has been stripped?')  # TODO: make err?

        if isinstance(section, SymbolTableSection):
            num_symbols = section.num_symbols()
            # print("  It's a symbol section with %s symbols" % num_symbols)
            for i in range(num_symbols):
                sym = section.get_symbol(i)
                if sym["st_info"]["type"] != "STT_FUNC":
                    continue
                # print(i, sym.entry, sym.name)
                name = sym.name
                # print("name", name)
                # assert name not in func_to_addr, f"Conflict! {name} already in map: {func_to_addr}"
                func_to_addrs[name].append((sym["st_value"], sym["st_size"]))

        addr_to_func = {}
        aliases = defaultdict(list)
        for func_name, funcs in func_to_addrs.items():
            for func in funcs:
                func_addr, func_sz = func
                # print(func_name, ":", func_sz,  hex(func_addr), "-", hex(func_addr + func_sz))
                if func_addr in addr_to_func:
                    if addr_to_func[func_addr] != func_name:
                        aliases[func_name].append(addr_to_func[func_addr])
                else:
                    addr_to_func[func_addr] = func_name
        llvm_bb_addr_map_raw = elffile.get_section_by_name(".llvm_bb_addr_map")
        assert llvm_bb_addr_map_raw is not None, "Missing: .llvm_bb_addr_map"
        llvm_bb_addr_map_raw = llvm_bb_addr_map_raw.data()

        def decode_map(data, addr_to_func):
            ret = {}
            unknown_count = 0
            with io.BytesIO(data) as reader:
                while True:
                    # print("reader", reader)
                    # print("dir(reader)", dir(reader))
                    version = int.from_bytes(reader.read(1), byteorder="little")
                    # print("version", version)
                    # assert version == 2
                    if version != 2:
                        # print("!")
                        # print(reader.read(100))
                        break
                    features = int.from_bytes(reader.read(1), byteorder="little")
                    # print("features", features)
                    assert features == 0
                    func_addr = int.from_bytes(reader.read(addr_bytes), byteorder="little")
                    # print("func_addr", func_addr)
                    func_name = addr_to_func.get(func_addr, None)
                    # print("func_name", func_name)
                    # assert func_name is not None
                    if func_name is None:
                        func_name = f"unknown_func_{unknown_count}"
                        unknown_count += 1
                    # TODO: leb128?
                    # num_bbs = int.from_bytes(reader.read(1), byteorder="little")
                    num_bbs = leb128.u.decode_reader(reader)[0]
                    # print("num_bbs", num_bbs)
                    # assert num_bbs > 0
                    if GISEL:
                        tmp = {}
                    else:
                        tmp = [None] * num_bbs
                    cur = func_addr
                    for i in range(num_bbs):
                        # print("i", i)
                        # bb_id = int.from_bytes(reader.read(1), byteorder="little")
                        bb_id = leb128.u.decode_reader(reader)[0]
                        if not GISEL:
                            assert bb_id < num_bbs
                        start_offset = leb128.u.decode_reader(reader)[0]
                        # print("start_offset", start_offset)
                        assert start_offset >= 0
                        end_offset = leb128.u.decode_reader(reader)[0]
                        # print("end_offset", end_offset)
                        assert end_offset >= 0
                        # TODO: leb128?
                        # metadata = int.from_bytes(reader.read(1), byteorder="little")
                        # metadata = leb128.u.decode_reader(reader)[0]
                        _ = leb128.u.decode_reader(reader)[0]
                        # print("metadata", metadata)
                        # TODO: decode metadata (is_return, is_call, ...)
                        # rest = reader.read(1000)
                        # print("rest", rest)
                        cur += start_offset
                        sz = end_offset - start_offset
                        assert sz >= 0
                        start = cur
                        end = cur + sz
                        pcs = [pc for pc in range(start, end + 2, 2) if pc in valid_pcs]
                        num_instrs = len(pcs)
                        cur += sz
                        if GISEL:
                            tmp[str(bb_id)] = (start, end, sz, num_instrs)
                        else:
                            tmp[bb_id] = (start, end, sz, num_instrs)
                    ret[func_name] = tmp
            return ret

        llvm_bb_addr_map = decode_map(llvm_bb_addr_map_raw, addr_to_func)
        # print("llvm_bb_addr_map", llvm_bb_addr_map)
        VERBOSE = False
        if VERBOSE:
            for func_name in func_to_addrs.keys():
                print(f"{func_name}:")
                bbs = llvm_bb_addr_map.get(func_name, None)
                # PRINT_MISSING = False
                PRINT_MISSING = True
                if bbs is None:
                    if PRINT_MISSING:
                        print("> no bb addr info found")
                    continue
                if GISEL:
                    bbs = dict(sorted(bbs.items(), key=lambda x: int(x[0]))).values()
                for i, bb in enumerate(bbs):
                    start, end, sz, num_instrs = bb
                    print(
                        f"> bb{i}",
                        ":",
                        hex(start),
                        "-",
                        hex(end),
                        f"(len={sz}B, num={num_instrs})",
                    )
                    print()
    return llvm_bb_addr_map


def analyze_llvm_bbs(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]
    trace_pc2bb_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb"
    )  # TODO: optional or different pass
    if len(trace_pc2bb_artifacts) > 0:
        assert len(trace_pc2bb_artifacts) == 1
        trace_pc2bb_artifact = trace_pc2bb_artifacts[0]
        # trace_pc2bb_df = trace_pc2bb_artifact.df
    else:
        pass
        # trace_pc2bb_df = None

    llvm_bbs = parse_elf(elf_artifact.path)
    # print("llvm_bbs", llvm_bbs)
    df_data = []
    for func_name, func_data in llvm_bbs.items():
        # print("fn", func_name)
        for bb_name, bb_data in func_data.items():
            # print("bn", bb_name)
            bb_name = f"%bb.{bb_name}"
            start, end, sz, num_instrs = bb_data
            new = {
                "func_name": func_name,
                "bb_name": bb_name,
                "pcs": (start, end),
                "size": sz,
                "num_instrs": num_instrs,
            }
            # if trace_pc2bb_df is not None:
            #     print("trace_pc2bb_df", trace_pc2bb_df)
            #     # trace_pc2bb_df[["start", "end"]] = trace_pc2bb_df["bb"].apply(pd.Series)
            #     # matches = trace_pc2bb_df.where(lambda x: x["start"] >= start and x["end"] <= end)
            #     matches = trace_pc2bb_df.where(lambda x: x["start"] >= start).dropna()
            #     matches = matches.where(lambda x: x["end"] <= end).dropna()
            #     if len(matches) > 0:
            #         print("matches", matches)
            #         weights = matches["weight"].sum()
            #         rel_weights = matches["rel_weight"].sum()
            #         # print("weights", weights)
            #         # print("rel_weights", rel_weights)
            #         # input("p")
            #         new["num_trace_bbs"] = len(matches)
            #         new["weight"] = weights
            #         new["rel_weight"] = rel_weights
            #     else:
            #         print("not found")
            #         print("start", start)
            #         print("end", end)
            #         print("sz", sz)
            #         # if start == 85388:
            #         if True:
            #             matches2 = trace_pc2bb_df.where(lambda x: x["start"] == start).dropna()
            #             print("matches2", matches2)
            #             matches2 = matches2.where(lambda x: x["end"] > end).dropna()
            #             print("matches2_", matches2)
            #             if len(matches2) > 0:
            #                 assert len(matches2) == 1
            #                 idx = matches2.index[0]
            #                 row = matches2.iloc[0]
            #                 start_ = row["start"]
            #                 end_ = row["end"]
            #                 size_ = row["size"]
            #                 freq_ = row["freq"]
            #                 weight_ = row["weight"]
            #                 rel_weight_ = row["rel_weight"]
            #                 num_instrs_ = row["num_instrs"]
            #                 print("idx", idx)
            #                 print("row", row)
            #                 print("start_", start_)
            #                 print("end_", end_)
            #                 print("size_", size_)
            #                 print("freq_", freq_)
            #                 print("weight_", weight_)
            #                 print("rel_weight_", rel_weight_)
            #                 print("num_instrs", num_instrs_)
            #                 default_enc_size = 4  # TODO: do not hardcode
            #                 trace_pc2bb_df.loc[idx, "end"] = end
            #                 trace_pc2bb_df.loc[idx, "size"] = sz
            #                 trace_pc2bb_df.loc[idx, "weight"] = weight_ * (sz / size_)
            #                 trace_pc2bb_df.loc[idx, "rel_weight"] = rel_weight_ * (sz / size_)
            #                 trace_pc2bb_df.loc[idx, "num_instrs"] = sz / default_enc_size
            #                 # new2 = {"start": end + default_enc_size, "end": end_, "freq": freq_, "size": size_ - sz, "weight": weight_ * (1 - sz / size_), "rel_weight": rel_weight_ * (1 - sz / size_), "num_instrs": (size_ - sz) / default_enc_size}
            #                 new2 = {"start": end, "end": end_, "freq": freq_, "size": size_ - sz, "weight": weight_ * (1 - sz / size_), "rel_weight": rel_weight_ * (1 - sz / size_), "num_instrs": (size_ - sz) / default_enc_size}
            #                 trace_pc2bb_df = pd.concat([trace_pc2bb_df, pd.DataFrame([new2])])
            #                 # TODO: export as updated artifact?
            #                 new["num_trace_bbs"] = 1
            #                 new["weight"] = weight_ * (sz / size_)
            #                 new["rel_weight"] = rel_weight_ * (sz / size_)
            #                 print("new", new)
            #                 print("new2", new2)

            #                 # input("ooo")
            #             else:
            #                 matches3 = trace_pc2bb_df.where(lambda x: x["start"] <= start).dropna()
            #                 print("matches3", matches3)
            #                 matches3 = matches2.where(lambda x: x["end"] >= end).dropna()
            #                 print("matches3_", matches3)
            #                 if func_name == "tvmgen_default_fused_nn_contrib_conv2d_NCHWc" and bb_name == "%bb.33":
            #                     input("lll")
            #                 if len(matches3) > 0:
            #                     input("uuu")
            #             # if llvm bbs is shorter than trace bbs, we may have untaken backwards branches in the trace
            #             # to split the trace bf into multiple ones, do a reverse search
            # if (start, end) in trace_pc2bb_df["bb"]:
            #     input("yes")
            df_data.append(new)
    # pc2locs_df = pd.DataFrame(pc2locs.items(), columns=["pc", "locs"])
    llvm_bbs_df = pd.DataFrame(df_data)
    # llvm_bbs_df.sort_values("rel_weight", inplace=True, ascending=False)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mapping",
        "by": "isaac_toolkit.analysis.static.llvm_bbs",
    }

    llvm_bbs_artifact = TableArtifact("llvm_bbs", llvm_bbs_df, attrs=attrs)
    # print("llvm_bbs_artifact", llvm_bbs_artifact)
    sess.add_artifact(llvm_bbs_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_llvm_bbs(sess, force=args.force)
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
