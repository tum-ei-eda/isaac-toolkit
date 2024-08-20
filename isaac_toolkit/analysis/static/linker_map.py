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


logger = logging.getLogger("linker_map")


def analyze_linker_map(mapFile):
    ret = []
    data = mapFile.toJson(humanReadable=False)
    segments = data["segments"]
    for segment in segments:
        segment_name = segment["name"]
        files = segment["files"]
        for file in files:
            filepath = file["filepath"]
            if "(" in filepath:  # TODO: use regex instead
                library, obj = filepath[:-1].split("(", 1)
            else:
                library = None
                obj = filepath
            obj_short = Path(obj).name
            library_short = Path(library).name if library is not None else library
            section_type = file["sectionType"]
            symbols = file["symbols"]
            for symbol in symbols:
                symbol_name = symbol["name"]
                new = {"segment": segment_name, "section": section_type, "symbol": symbol_name, "library": library_short, "library_full": library, "object": obj_short, "object_full": obj}
                ret.append(new)
    return ret


def analyze_dwarf(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    linker_map_artifacts = filter_artifacts(artifacts, lambda x: x.name == "linker.map" and x.flags & ArtifactFlag.PYTHON)
    assert len(linker_map_artifacts) == 1
    linker_map_artifact = linker_map_artifacts[0]
    mapFile = linker_map_artifact.data

    symbol_map = analyze_linker_map(mapFile)
    symbol_map_df = pd.DataFrame(symbol_map, columns=["segment", "section", "symbol", "object", "object_full", "library", "library_full"])

    attrs = {
        "kind": "mapping",
        "by": __name__,
    }

    symbol_map_artifact = TableArtifact("symbol_map", symbol_map_df, attrs=attrs)
    sess.add_artifact(symbol_map_artifact, override=force)


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
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
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
