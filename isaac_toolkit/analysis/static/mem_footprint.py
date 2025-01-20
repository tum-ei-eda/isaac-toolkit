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
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("dwarf")


def parse_elf(elf_path):
    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)
        ###
        from elftools.elf.sections import SymbolTableSection

        # print('  %s sections' % elffile.num_sections())
        section = elffile.get_section_by_name(".symtab")

        assert section, "Symbol Table not found!"
        # print('  Section name: %s, type: %s' %(section.name, section['sh_type']))
        if isinstance(section, SymbolTableSection):
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

    artifact = TableArtifact("mem_footprint", footprint_df, attrs=attrs)
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
