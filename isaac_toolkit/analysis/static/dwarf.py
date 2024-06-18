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
    mapping = defaultdict(tuple)
    # the mapping between source file and its function
    srcFile_func_dict = defaultdict(set)
    # the mapping between program counter and source line
    pc_to_source_line_mapping = defaultdict(list)
    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)
        #### ###
        #### from elftools.elf.sections import SymbolTableSection
        #### # print('  %s sections' % elffile.num_sections())
        #### section = elffile.get_section_by_name('.symtab')

        #### assert section, "Symbol Table not found!"
        #### # print('  Section name: %s, type: %s' %(section.name, section['sh_type']))
        #### if isinstance(section, SymbolTableSection):
        ####    num_symbols = section.num_symbols()
        ####    # print("  It's a symbol section with %s symbols" % num_symbols)
        ####    start_symbol = section.get_symbol_by_name("_start")
        ####    assert len(start_symbol) == 1
        ####    start_symbol = start_symbol[0]
        ####    # print("start_symbol", start_symbol, start_symbol.entry, start_symbol.name)
        ####    start_addr = start_symbol.entry["st_value"]
        ####    # print("start_addr", start_addr)
        ####    total_footprint = 0
        ####    func_footprint = {}
        ####    for i, sym in enumerate(section.iter_symbols()):
        ####        # print("i", s]ym.entry)
        ####        ty = sym.entry["st_info"]["type"]
        ####        if ty != "STT_FUNC":
        ####            continue
        ####        func = sym.name
        ####        sz = sym.entry["st_size"]
        ####        # print("ty", ty)
        ####        # print("sz", sz)
        ####        func_footprint[func] = sz
        ####        total_footprint += sz
        ####    # print("total_footprint", total_footprint)
        ####    # print("func_footprint", func_footprint)
        ####    footprint_df = pd.DataFrame(func_footprint.items(), columns=["func", "bytes"])
        ####    footprint_df.sort_values("bytes", inplace=True, ascending=False)
        ####    footprint_df["rel_bytes"] = footprint_df["bytes"] / total_footprint
        ####    # print("footprint_df", footprint_df)
        ####    # print("  The name of the last symbol in the section is: %s" % (section.get_symbol(num_symbols - 1).name))
        #### # input("123")
        #### ###

        hard_coded_mapping = {}
        # hard_coded_mapping = {
        #     "_start": (int("0x12c", 0), int("0x183", 0))
        # }

        # mapping function symbol to pc range
        for section in elffile.iter_sections():
            if section.name == ".symtab":
                symbol_table = section
                break

        for symbol in symbol_table.iter_symbols():
            symbol_type = symbol["st_info"]["type"]
            if symbol_type == "STT_FUNC":
                start_pc = symbol["st_value"]
                end_pc = start_pc + symbol["st_size"] - 1
                range = (start_pc, end_pc)
                mapping[symbol.name] = range
            elif symbol.name in hard_coded_mapping:
                mapping[symbol.name] = hard_coded_mapping[symbol.name]

        ## mapping source file to function
        if not elffile.has_dwarf_info():
            logger.error("ELF file has no DWARF info!")
            return mappinf, None, None

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

            # A dir_index of 0 indicates that no absolute directory was recorded during
            # compilation; return just the basename.
            if dir_index == 0:
                return file_entry.name.decode()

            directory = lp_header["include_directory"][dir_index]
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
                    func_name = DIE.attributes["DW_AT_name"].value.decode()
                    file_index = DIE.attributes["DW_AT_decl_file"].value

                    filename = lpe_filename(line_program, file_index)
                    if func_name not in srcFile_func_dict[filename]:
                        srcFile_func_dict[filename].add(func_name)

        for CU in dwarfinfo.iter_CUs():
            line_program = dwarfinfo.line_program_for_CU(CU)
            if line_program is None:
                logger.warning("  DWARF info is missing a line program for this CU")
                continue

            CU_name = CU.get_top_DIE().attributes["DW_AT_name"].value.decode("utf-8")

            for entry in line_program.get_entries():
                if entry.state:
                    pc = entry.state.address
                    line = entry.state.line
                    pc_to_source_line_mapping[CU_name].append((pc, line))

            if CU_name in pc_to_source_line_mapping:
                pc_to_source_line_mapping[CU_name].sort(key=lambda x: x[0])
    return mapping, srcFile_func_dict, pc_to_source_line_mapping


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    override = args.force
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]

    func2pc, file2funcs, file_pc2line = parse_dwarf(elf_artifact.path)
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
    func2pc_df = pd.DataFrame(func2pc.items(), columns=["func", "pc_range"])
    file2funcs_df = pd.DataFrame(file2funcs.items(), columns=["file", "func_names"])
    pc2locs_df = pd.DataFrame(pc2locs.items(), columns=["pc", "locs"])
    # print("func2pc_df", func2pc_df)
    # print("file2funcs_df", file2funcs_df)
    # print("pc2locs_df", pc2locs_df)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mapping",
        "by": "isaac_toolkit.analysis.static.dwarf",
    }

    func2pc_artifact = TableArtifact(f"func2pc", func2pc_df, attrs=attrs)
    file2funcs_artifact = TableArtifact(f"file2funcs", file2funcs_df, attrs=attrs)
    pc2locs_artifact = TableArtifact(f"pc2locs", pc2locs_df, attrs=attrs)
    # print("artifact1", func2pc_artifact)
    # print("artifact2", file2funcs_artifact)
    # print("artifact3", pc2locs_artifact)
    sess.add_artifact(func2pc_artifact, override=override)
    sess.add_artifact(file2funcs_artifact, override=override)
    sess.add_artifact(pc2locs_artifact, override=override)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
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
