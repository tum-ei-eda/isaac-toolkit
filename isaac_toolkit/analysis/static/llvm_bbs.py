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
import sys
import argparse
from pathlib import Path

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from capstone import Cs, CS_ARCH_RISCV, CS_MODE_RISCV32, CS_MODE_RISCV64, CS_MODE_RISCVC

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.utils.bb_identity import file_digest

logger = get_logger()


def parse_elf(elf_path):
    """Read fixed BB IDs from linked RISC-V ELFs produced by LLVM 19–22."""
    from isaac_toolkit.utils.llvm_bb_addr_map import decode_bb_addr_map

    with open(elf_path, "rb") as stream:
        elf = ELFFile(stream)
        if elf["e_type"] == "ET_REL":
            raise ValueError("BB address maps require a linked ELF, not an unrelocated object")
        if not elf.little_endian:
            raise NotImplementedError("Big-endian BB address maps are not supported")
        if elf["e_machine"] != "EM_RISCV":
            raise ValueError("BB instruction counting currently requires a RISC-V ELF")
        sections = [section for section in elf.iter_sections()
                    if section["sh_flags"] & 4 and section["sh_type"] == "SHT_PROGBITS"]
        code_bytes = {id(section): section.data() for section in sections}
        symbols = elf.get_section_by_name(".symtab")
        if not isinstance(symbols, SymbolTableSection):
            raise ValueError("BB identity matching requires the ELF symbol table")
        functions = {}
        for symbol in symbols.iter_symbols():
            if symbol["st_info"]["type"] == "STT_FUNC" and symbol["st_shndx"] != "SHN_UNDEF":
                address = symbol["st_value"]
                previous_name, previous_size = functions.get(address, (symbol.name, 0))
                functions[address] = (previous_name, max(previous_size, symbol["st_size"]))
        maps = [section for section in elf.iter_sections()
                if section.name == ".llvm_bb_addr_map" or section["sh_type"] == "SHT_LLVM_BB_ADDR_MAP"]
        if not maps:
            raise ValueError("Missing .llvm_bb_addr_map")
        mode = CS_MODE_RISCV32 if elf.elfclass == 32 else CS_MODE_RISCV64
        disassembler = Cs(CS_ARCH_RISCV, mode | CS_MODE_RISCVC)
        result = {}
        for section in maps:
            for record in decode_bb_addr_map(section.data(), elf.elfclass // 8):
                func_address = record["ranges"][0][0]
                if func_address not in functions:
                    raise ValueError(f"No function symbol for BB map address {func_address:#x}")
                name, func_size = functions[func_address]
                if name in result:
                    raise ValueError(f"Ambiguous BB map for function {name}")
                blocks = {}
                for base, entries in record["ranges"]:
                    for bb_id, start, size in entries:
                        end = start + size
                        # Split cold ranges need not lie inside the entry symbol.
                        if len(record["ranges"]) == 1 and func_size and end > func_address + func_size:
                            raise ValueError(f"BB {bb_id} exceeds function {name}'s symbol range")
                        containing = [sec for sec in sections if sec["sh_addr"] <= start
                                      and end <= sec["sh_addr"] + sec["sh_size"]]
                        if len(containing) != 1:
                            raise ValueError(f"BB {bb_id} of {name} has no unique executable section")
                        sec = containing[0]
                        offset = start - sec["sh_addr"]
                        code = code_bytes[id(sec)][offset:offset + size]
                        instructions = list(disassembler.disasm(code, start))
                        if sum(insn.size for insn in instructions) != size:
                            raise ValueError(f"Cannot fully disassemble BB {bb_id} of {name}; instruction counts are not trustworthy")
                        blocks[str(bb_id)] = (start, end, size, len(instructions))
                result[name] = blocks
        return result


def analyze_llvm_bbs(sess: Session, force: bool = False):
    logger.info("Analyzing LLVM BBs...")
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]
    # trace_pc2bb_artifacts = filter_artifacts(
    #     artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb"
    # )  # TODO: optional or different pass
    # if len(trace_pc2bb_artifacts) > 0:
    #     assert len(trace_pc2bb_artifacts) == 1
    #     trace_pc2bb_artifact = trace_pc2bb_artifacts[0]
    #     # trace_pc2bb_df = trace_pc2bb_artifact.df
    # else:
    #     pass
    #     # trace_pc2bb_df = None

    elf_hash = file_digest(elf_artifact.path)
    llvm_bbs = parse_elf(elf_artifact.path)
    if file_digest(elf_artifact.path) != elf_hash:
        raise ValueError("ELF changed during BB analysis")
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
                "bb_id": int(bb_name.removeprefix("%bb.")),
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
            #                 # new2 = {"start": end + default_enc_size, "end": end_, "freq": freq_,
            #                     "size": size_ - sz, "weight": weight_ * (1 - sz / size_),
            #                     "rel_weight": rel_weight_ * (1 - sz / size_),
            #                     "num_instrs": (size_ - sz) / default_enc_size}
            #                 new2 = {"start": end, "end": end_, "freq": freq_, "size": size_ - sz,
            #                     "weight": weight_ * (1 - sz / size_),
            #                     "rel_weight": rel_weight_ * (1 - sz / size_),
            #                     "num_instrs": (size_ - sz) / default_enc_size}
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
        "elf_sha256": elf_hash,
        "bb_map_decoder": 2,
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
    set_log_level(console_level=args.log, file_level=args.log)
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
