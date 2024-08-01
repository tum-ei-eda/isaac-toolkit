import io
import sys
import leb128
import logging
import argparse
import posixpath
import subprocess
from typing import Optional, Union
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("llvm_bbs")


def query_candidates_from_db(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    label: Optional[str] = None,
    stage: int = 8,
    force: bool = False,
):
    artifacts = sess.artifacts
    choices_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "choices")
    assert len(choices_artifacts) == 1
    choices_artifact = choices_artifacts[0]
    choices_df = choices_artifact.df
    print("choices_df", choices_df)
    if workdir is None:
        raise NotImplementedError("automatic workdir not supported yet!")
    if isinstance(workdir, str):
        workdir = Path(workdir)
    assert isinstance(workdir, Path)
    workdir = workdir.resolve()
    assert label is not None
    index_files = []
    for index, row in choices_df.iterrows():
        # print("index", index)
        # print("row", row)
        func_name = row["func_name"]
        bb_name = row["bb_name"]
        print("func_name", func_name)
        print("bb_name", bb_name)
        out_name = f"{func_name}_{bb_name}_0"
        out_dir = workdir / out_name
        print("out_dir", out_dir)
        out_dir.mkdir(exist_ok=True)
        index_file = out_dir / "index.yml"
        index_files.append(index_file)
        # TODO: to not hardcode this, call directly via python
        args = [
            "python3",
            "-m",
            "tool.main",
        ]
        LIMIT_RESULTS = None
        MIN_INPUTS = 1
        MAX_INPUTS = 3
        MIN_OUTPUTS = 0
        MAX_OUTPUTS = 1
        MAX_NODES = None
        MIN_NODES = 1
        MIN_PATH_LENGTH = 1
        MAX_PATH_LENGTH = 3
        MAX_PATH_WIDTH = 1
        INSTR_PREDICATES = 511  # ALL?
        IGNORE_NAMES = None
        IGNORE_OP_TYPES = None
        XLEN = 64  # TODO: do not hardcode
        args += [
            *["--progress"],
            *["--log", "info"],
            *["--session", label],
            *["--function", func_name],
            *["--basic-block", bb_name],
            *["--stage", str(stage)],
            *["--output-dir", out_dir],
            *["--ignore-const-inputs"],
            *(["--limit-results", str(LIMIT_RESULTS)] if LIMIT_RESULTS is not None else []),
            *(["--min-inputs", str(MIN_INPUTS)] if MIN_INPUTS is not None else []),
            *(["--max-inputs", str(MAX_INPUTS)] if MAX_INPUTS is not None else []),
            *(["--min-outputs", str(MIN_OUTPUTS)] if MIN_OUTPUTS is not None else []),
            *(["--max-outputs", str(MAX_OUTPUTS)] if MAX_OUTPUTS is not None else []),
            *(["--max-nodes", str(MAX_NODES)] if MAX_NODES is not None else []),
            *(["--min-nodes", str(MIN_NODES)] if MIN_NODES is not None else []),
            *(["--min-path-length", str(MIN_PATH_LENGTH)] if MIN_PATH_LENGTH is not None else []),
            *(["--max-path-length", str(MAX_PATH_LENGTH)] if MAX_PATH_LENGTH is not None else []),
            *(["--max-path-width", str(MAX_PATH_WIDTH)] if MAX_PATH_WIDTH is not None else []),
            *(["--instr-predicates", str(INSTR_PREDICATES)] if INSTR_PREDICATES is not None else []),
            *(["--ignore-names", IGNORE_NAMES] if IGNORE_NAMES is not None else []),
            *(["--ignore-op-types", str(IGNORE_OP_TYPES)] if IGNORE_OP_TYPES is not None else []),
            *(["--xlen", str(XLEN)] if XLEN is not None else []),
            *["--write-func"],
            # *["--write-func-fmt", WRITE_FUNC_FMT],
            # *["--write-func-flt", WRITE_FUNC_FLT],
            *["--write-sub"],
            # *["--write-sub-fmt", WRITE_SUB_FMT],
            # *["--write-sub-flt", WRITE_SUB_FLT],
            *["--write-io-sub"],
            # *["--write-io-sub-fmt", WRITE_IO_SUB_FMT],
            # *["--write-io-sub-flt", WRITE_IO_SUB_FLT],
            *["--write-gen"],
            # *["--write-gen-fmt", WRITE_GEN_FMT],
            # *["--write-gen-flt", WRITE_GEN_FLT],
            *["--write-pie"],
            # *["--write-pie-fmt", WRITE_PIE_FMT],
            # *["--write-pie-flt", WRITE_PIE_FLT],
            *["--write-df"],
            # *["--write-df-fmt", WRITE_DF_FMT],
            # *["--write-df-flt", WRITE_DF_FLT],
            *["--write-index"],
            # *["--write-index-fmt", WRITE_INDEX_FMT],
            # *["--write-index-flt", WRITE_INDEX_FLT],
            *["--write-queries"],
        ]
        # args += ["--help"]
        print("args", args)
        subprocess.run(args, check=True)
    combined_index_file = workdir / "combined_index.yml"
    combine_args = [
        "python3",
        "-m",
        "tool.combine_index",
        *index_files,
        "--drop",
        "--out",
        combined_index_file,
    ]
    if len(index_files) in [2, 3]:
        venn_diagram_file = workdir / "venn.pdf"
        combine_args += ["--venn", venn_diagram_file]
    print("combine_args", combine_args)
    subprocess.run(combine_args, check=True)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    query_candidates_from_db(sess, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
