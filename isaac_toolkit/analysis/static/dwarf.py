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
import os
import sys
import logging
import argparse
import posixpath
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("dwarf")


def parse_dwarf(elf_path):
    # func2pcs_data = []
    func_ranges = {}
    # the mapping between source file and its function
    srcFile_func_dict = defaultdict(lambda: [set(), set(), set()])
    # the mapping between program counter and source line
    pc_to_source_line_mapping = defaultdict(list)
    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)

        # mapping function symbol to pc range
        symtab = elffile.get_section_by_name(".symtab")
        if symtab is None:
            raise RuntimeError("No symbol table found (.symtab missing)")

        funcs_by_section = defaultdict(list)  # shndx -> list of (start, size, name, symbol)
        for symbol in symtab.iter_symbols():
            try:
                st_type = symbol["st_info"]["type"]
            except Exception:
                continue
            if st_type != "STT_FUNC":
                continue
            start = symbol["st_value"]
            size = symbol["st_size"]
            shndx = symbol["st_shndx"]
            name = symbol.name or "<anon>"
            funcs_by_section[shndx].append((start, size, name, symbol))

        # 3) For each section, sort by start addr and compute ranges
        for shndx, syms in funcs_by_section.items():
            # skip special indices (we still handle them but there might be no section)
            section = None
            try:
                if isinstance(shndx, int):
                    section = elffile.get_section(shndx)
                else:
                    # PyElfTools may give SHN_UNDEF as string, leave section None
                    section = None
            except Exception:
                section = None

            syms_sorted = sorted(syms, key=lambda x: x[0])
            for idx, (start, size, name, symbol) in enumerate(syms_sorted):

                if size and size != 0:
                    end = start + size - 1
                    func_ranges[name] = (start, end)
                    continue

                # size == 0: try to infer from next symbol in same section
                end = None
                # find next symbol with start > this start
                next_start = None
                for j in range(idx + 1, len(syms_sorted)):
                    candidate_start = syms_sorted[j][0]
                    if candidate_start > start:
                        next_start = candidate_start
                        break
                if next_start is not None:
                    end = next_start - 1
                elif section is not None:
                    # fallback to section end
                    sec_start = section["sh_addr"]
                    sec_size = section["sh_size"]
                    end = sec_start + sec_size - 1
                else:
                    # last resort: set end == start (or None)
                    end = start

                func_ranges[name] = (start, end)

        func2pcs_data = list(func_ranges.items())
        # mapping source file to function
        if not elffile.has_dwarf_info():
            logger.error("ELF file has no DWARF info!")
            return func2pcs_data, None, None

        dwarfinfo = elffile.get_dwarf_info()

        def lpe_filename(line_program, file_index):
            # Retrieving the filename associated with a line program entry
            # involves two levels of indirection: we take the file index from
            # the LPE to grab the file_entry from the line program header,
            # then take the directory index from the file_entry to grab the
            # directory name from the line program header. Finally, we
            # join the (base) filename from the file_entry to the directory
            # name to get the absolute filename.
            lp_header = line_program.header
            file_entries = lp_header["file_entry"]

            # File and directory indices are 1-indexed.
            file_entry = file_entries[file_index] if line_program.header.version >= 5 else file_entries[file_index - 1]
            dir_index = file_entry["dir_index"] if line_program.header.version >= 5 else file_entry["dir_index"] - 1
            assert dir_index >= 0

            # A dir_index of 0 indicates that no absolute directory was recorded during
            # compilation; return just the basename.
            if dir_index == 0:
                return file_entry.name.decode()

            directory = lp_header["include_directory"][dir_index]
            # TODO: try out actual_path = op.normpath(CU.get_top_DIE().get_full_path())?
            return posixpath.join(directory, file_entry.name).decode()

        for CU in dwarfinfo.iter_CUs():
            line_program = dwarfinfo.line_program_for_CU(CU)
            if line_program is None:
                logger.warning("DWARF info is missing a line program for this CU")
                continue

            for DIE in CU.iter_DIEs():
                if (
                    DIE.tag == "DW_TAG_subprogram"
                    and "DW_AT_decl_file" in DIE.attributes
                    and "DW_AT_low_pc" in DIE.attributes
                    and "DW_AT_high_pc" in DIE.attributes
                ):
                    if "DW_AT_name" in DIE.attributes:
                        func_name = DIE.attributes["DW_AT_name"].value.decode()
                    else:
                        func_name = "???"
                    if "DW_AT_linkage_name" in DIE.attributes:
                        linkage_name = DIE.attributes["DW_AT_linkage_name"].value.decode()
                        from cpp_demangle import demangle

                        unmangled_linkage_name = demangle(linkage_name)
                    else:
                        linkage_name = "???"
                        unmangled_linkage_name = "???"
                    if "DW_AT_decl_file" in DIE.attributes:
                        file_index = DIE.attributes["DW_AT_decl_file"].value
                        filename = lpe_filename(line_program, file_index)
                    else:
                        filename = "???"

                    srcFile_func_dict[filename][0].add(func_name)
                    srcFile_func_dict[filename][1].add(linkage_name)
                    srcFile_func_dict[filename][2].add(unmangled_linkage_name)

        for CU in dwarfinfo.iter_CUs():
            line_program = dwarfinfo.line_program_for_CU(CU)
            if line_program is None:
                logger.warning("  DWARF info is missing a line program for this CU")
                continue

            # CU_name = CU.get_top_DIE().attributes["DW_AT_name"].value.decode("utf-8")
            actual_path = os.path.normpath(CU.get_top_DIE().get_full_path())
            # print("actual_path", actual_path)

            for entry in line_program.get_entries():
                if entry.state:
                    pc = entry.state.address
                    line = entry.state.line
                    # print("line", line)
                    # pc_to_source_line_mapping[CU_name].append((pc, line))
                    pc_to_source_line_mapping[actual_path].append((pc, line))

            # if CU_name in pc_to_source_line_mapping:
            if actual_path in pc_to_source_line_mapping:
                # pc_to_source_line_mapping[CU_name].sort(key=lambda x: x[0])
                pc_to_source_line_mapping[actual_path].sort(key=lambda x: x[0])
    return func2pcs_data, srcFile_func_dict, pc_to_source_line_mapping


def analyze_dwarf(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]

    func2pc, file2funcs, file_pc2line = parse_dwarf(elf_artifact.path)
    file2funcs_data = [(file_name, vals[0], vals[1], vals[2]) for file_name, vals in file2funcs.items()]
    # print("func2pc", func2pc)
    # print("file2funcs", file2funcs)
    # print("file_func2line", file_pc2line)
    pc2locs = defaultdict(set)
    for file, pc_lines in file_pc2line.items():
        # print("file", file)
        for pc_line in list(set(pc_lines)):
            # print("pc_line", pc_line)
            assert len(pc_line) == 2
            pc, line = pc_line
            loc = f"{file}:{line}"
            pc2locs[pc].add(loc)
    func2pc_df = pd.DataFrame(func2pc, columns=["func", "pc_range"])
    file2funcs_df = pd.DataFrame(
        file2funcs_data,
        columns=["file", "func_names", "linkage_names", "unmangled_linkage_names"],
    )
    pc2locs_df = pd.DataFrame(pc2locs.items(), columns=["pc", "locs"])
    # print("func2pc_df", func2pc_df)
    # print("file2funcs_df", file2funcs_df)
    # print("pc2locs_df", pc2locs_df)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mapping",
        "by": "isaac_toolkit.analysis.static.dwarf",
    }

    func2pc_artifact = TableArtifact("func2pc", func2pc_df, attrs=attrs)
    file2funcs_artifact = TableArtifact("file2funcs", file2funcs_df, attrs=attrs)
    pc2locs_artifact = TableArtifact("pc2locs", pc2locs_df, attrs=attrs)
    # print("artifact1", func2pc_artifact)
    # print("artifact2", file2funcs_artifact)
    # print("artifact3", pc2locs_artifact)
    sess.add_artifact(func2pc_artifact, override=force)
    sess.add_artifact(file2funcs_artifact, override=force)
    sess.add_artifact(pc2locs_artifact, override=force)


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
