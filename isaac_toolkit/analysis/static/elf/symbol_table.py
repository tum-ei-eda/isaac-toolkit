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
import sys
import logging
import argparse
import posixpath
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("symbol_table")


# TODO: move dwarf.py in elf subdir


def parse_elf(elf_path):
    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)

        data = []

        section = elffile.get_section_by_name(".symtab")

        assert section, "Symbol Table not found!"
        assert isinstance(section, SymbolTableSection)
        for i, sym in enumerate(section.iter_symbols()):
            ty = sym.entry["st_info"]["type"]
            # if ty != "STT_FUNC":
            #     continue
            name = sym.name
            sz = sym.entry["st_size"]
            val = sym.entry["st_value"]
            print("ty", ty)
            print("name", name)
            print("sz", sz)
            print("val", val)
            new = {"name": name, "type": ty, "size": sz, "value": val}
            data.append(new)
        symbol_table_df = pd.DataFrame(data)
        # print("footprint_df", footprint_df)
        # print("  The name of the last symbol in the section is: %s" % (section.get_symbol(num_symbols - 1).name))

        # mapping function symbol to pc range
        # for section in elffile.iter_sections():
        #     if section.name == ".symtab":
        #         symbol_table = section
        #         break

        # for symbol in symbol_table.iter_symbols():
        #     symbol_type = symbol["st_info"]["type"]
        #     if symbol_type == "STT_FUNC":
        #         start_pc = symbol["st_value"]
        #         end_pc = start_pc + symbol["st_size"] - 1
        #         range = (start_pc, end_pc)
        #         # mapping[symbol.name] = range
        #         new = (symbol.name, (start_pc, end_pc))
        #         func2pcs_data.append(new)
        #     # Warning: this mapping uses mangled func names

        # ## mapping source file to function
        # if not elffile.has_dwarf_info():
        #     logger.error("ELF file has no DWARF info!")
        #     return func2pcs_data, None, None

        # dwarfinfo = elffile.get_dwarf_info()

        # def lpe_filename(line_program, file_index):
        #     # Retrieving the filename associated with a line program entry
        #     # involves two levels of indirection: we take the file index from
        #     # the LPE to grab the file_entry from the line program header,
        #     # then take the directory index from the file_entry to grab the
        #     # directory name from the line program header. Finally, we
        #     # join the (base) filename from the file_entry to the directory
        #     # name to get the absolute filename.
        #     lp_header = line_program.header
        #     file_entries = lp_header["file_entry"]

        #     # File and directory indices are 1-indexed.
        #     file_entry = file_entries[file_index] if line_program.header.version >= 5 else file_entries[file_index - 1]
        #     dir_index = file_entry["dir_index"] if line_program.header.version >= 5 else file_entry["dir_index"] - 1
        #     assert dir_index >= 0

        #     # A dir_index of 0 indicates that no absolute directory was recorded during
        #     # compilation; return just the basename.
        #     if dir_index == 0:
        #         return file_entry.name.decode()

        #     directory = lp_header["include_directory"][dir_index]
        #     return posixpath.join(directory, file_entry.name).decode()

        # for CU in dwarfinfo.iter_CUs():
        #     line_program = dwarfinfo.line_program_for_CU(CU)
        #     if line_program is None:
        #         logger.warning("DWARF info is missing a line program for this CU")
        #         continue

        #     for DIE in CU.iter_DIEs():
        #         if (
        #             DIE.tag == "DW_TAG_subprogram"
        #             and "DW_AT_decl_file" in DIE.attributes
        #             and "DW_AT_low_pc" in DIE.attributes
        #             and "DW_AT_high_pc" in DIE.attributes
        #         ):
        #             if "DW_AT_name" in DIE.attributes:
        #                 func_name = DIE.attributes["DW_AT_name"].value.decode()
        #             else:
        #                 func_name = "???"
        #             if "DW_AT_linkage_name" in DIE.attributes:
        #                 linkage_name = DIE.attributes["DW_AT_linkage_name"].value.decode()
        #                 from cpp_demangle import demangle

        #                 unmangled_linkage_name = demangle(linkage_name)
        #             else:
        #                 linkage_name = "???"
        #                 unmangled_linkage_name = "???"
        #             if "DW_AT_decl_file" in DIE.attributes:
        #                 file_index = DIE.attributes["DW_AT_decl_file"].value
        #                 filename = lpe_filename(line_program, file_index)
        #             else:
        #                 file_name = "???"

        #             srcFile_func_dict[filename][0].add(func_name)
        #             srcFile_func_dict[filename][1].add(linkage_name)
        #             srcFile_func_dict[filename][2].add(unmangled_linkage_name)

        # for CU in dwarfinfo.iter_CUs():
        #     line_program = dwarfinfo.line_program_for_CU(CU)
        #     if line_program is None:
        #         logger.warning("  DWARF info is missing a line program for this CU")
        #         continue

        #     CU_name = CU.get_top_DIE().attributes["DW_AT_name"].value.decode("utf-8")

        #     for entry in line_program.get_entries():
        #         if entry.state:
        #             pc = entry.state.address
        #             line = entry.state.line
        #             pc_to_source_line_mapping[CU_name].append((pc, line))

        #     if CU_name in pc_to_source_line_mapping:
        #         pc_to_source_line_mapping[CU_name].sort(key=lambda x: x[0])
    return symbol_table_df


def analyze_dwarf(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]

    symbol_table_df = parse_elf(elf_artifact.path)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mapping",
        "by": "isaac_toolkit.analysis.static.elf.symbol_table",
    }

    symbol_table_artifact = TableArtifact("symbol_table", symbol_table_df, attrs=attrs)
    sess.add_artifact(symbol_table_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_dwarf(sess, force=args.force)
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
