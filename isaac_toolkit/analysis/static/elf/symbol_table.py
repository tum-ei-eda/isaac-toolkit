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
import sys
import logging
import argparse
from pathlib import Path

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
        for sym in section.iter_symbols():
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
