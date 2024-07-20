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


def parse_elf(elf_path):
    mapping = defaultdict(tuple)
    # the mapping between source file and its function
    srcFile_func_dict = defaultdict(set)
    # the mapping between program counter and source line
    pc_to_source_line_mapping = defaultdict(list)
    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)
        ###
        from elftools.elf.sections import SymbolTableSection

        # print('  %s sections' % elffile.num_sections())
        section = elffile.get_section_by_name(".symtab")

        assert section, "Symbol Table not found!"
        # print('  Section name: %s, type: %s' %(section.name, section['sh_type']))
        if isinstance(section, SymbolTableSection):
            num_symbols = section.num_symbols()
            # print("  It's a symbol section with %s symbols" % num_symbols)
            ### TODO: extract somewhere else
            ### start_symbol = section.get_symbol_by_name("_start")
            ### assert len(start_symbol) == 1
            ### start_symbol = start_symbol[0]
            ### # print("start_symbol", start_symbol, start_symbol.entry, start_symbol.name)
            ### start_addr = start_symbol.entry["st_value"]
            ### # print("start_addr", start_addr)
            total_footprint = 0
            func_footprint = {}
            for i, sym in enumerate(section.iter_symbols()):
                # print("i", s]ym.entry)
                ty = sym.entry["st_info"]["type"]
                if ty != "STT_FUNC":
                    continue
                func = sym.name
                sz = sym.entry["st_size"]
                # print("ty", ty)
                # print("sz", sz)
                func_footprint[func] = sz
                total_footprint += sz
            # print("total_footprint", total_footprint)
            # print("func_footprint", func_footprint)
            footprint_df = pd.DataFrame(func_footprint.items(), columns=["func", "bytes"])
            footprint_df.sort_values("bytes", inplace=True, ascending=False)
            footprint_df["rel_bytes"] = footprint_df["bytes"] / total_footprint
            # print("footprint_df", footprint_df)
            # print("  The name of the last symbol in the section is: %s" % (section.get_symbol(num_symbols - 1).name))
        # input("123")
        ###
        return footprint_df


def analyze_mem_footprint(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]

    footprint_df = parse_elf(elf_artifact.path)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mem_footprint",
        "by": "isaac_toolkit.analysis.static.mem_footprint",
    }

    artifact = TableArtifact(f"mem_footprint", footprint_df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_mem_footprint(sess, force=args.force)
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
