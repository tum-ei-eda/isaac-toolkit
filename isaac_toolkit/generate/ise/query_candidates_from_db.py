import sys
import logging
import argparse
import subprocess
from typing import Optional, Union
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


logger = logging.getLogger("llvm_bbs")


def query_candidates_from_db(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    label: Optional[str] = None,
    stage: int = 8,
    force: bool = False,
    LIMIT_RESULTS: Optional[int] = None,
    # LIMIT_RESULTS: Optional[int] = 5000,
    # LIMIT_RESULTS: Optional[int] = 500,
    MIN_INPUTS: Optional[int] = 1,
    # MAX_INPUTS: Optional[int] = 4,
    MAX_INPUTS: Optional[int] = 3,
    MIN_OUTPUTS: Optional[int] = 0,
    # MAX_OUTPUTS: Optional[int] = 4,
    MAX_OUTPUTS: Optional[int] = 1,
    # TODO: implement topk!
    MAX_NODES: Optional[int] = int(1e3),
    # MAX_NODES: Optional[int] = int(5),
    MAX_ENC_FOOTPRINT: Optional[float] = 1.0,
    MAX_ENC_WEIGHT: Optional[float] = 1.0,
    MIN_ENC_BITS_LEFT: Optional[int] = 5,
    # MIN_NODES: Optional[Union[int, str]] = "auto",
    MIN_NODES: Optional[Union[int, str]] = None,
    MIN_PATH_LENGTH: Optional[int] = 1,
    # MAX_PATH_LENGTH = 3,
    MAX_PATH_LENGTH=5,
    MAX_PATH_WIDTH=2,
    # MAX_PATH_WIDTH = 4,
    INSTR_PREDICATES=511,  # ALL?
    IGNORE_NAMES=None,
    IGNORE_OP_TYPES=None,
    ALLOWED_ENC_SIZES=[32],
    MIN_ISO_WEIGHT=0.05,
    MAX_LOADS=1,
    MAX_STORES=1,
    MAX_MEMS: Optional[int] = 0,  # TODO
    MAX_BRANCHES: Optional[int] = 1,
    # XLEN: Optional[int] = 64,  # TODO: do not hardcode
    XLEN: Optional[int] = 32,  # TODO: do not hardcode
    ENABLE_VARIATION_REUSE_IO=False,
    # ENABLE_VARIATION_REUSE_IO=True,
    HALT_ON_ERROR: bool = True,
    SORT_BY: Optional[str] = "IsoWeight",
    TOPK: Optional[int] = 100,
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
        # if index != 1:
        #     continue
        # input(">")
        func_name = row["func_name"]
        bb_name = row["bb_name"]
        rel_weight = row["rel_weight"]
        num_instrs = row["num_instrs"]
        print("func_name", func_name)
        print("bb_name", bb_name)
        out_name = f"{func_name}_{bb_name}_0"
        out_dir = workdir / out_name
        print("out_dir", out_dir)
        out_dir.mkdir(exist_ok=True)
        index_file = out_dir / "index.yml"
        index_files.append(index_file)
        # TODO: to not hardcode this, call directly via python
        SCALE_WEIGHT = True

        min_iso_weight = MIN_ISO_WEIGHT
        if SCALE_WEIGHT:
            if min_iso_weight is not None:
                min_iso_weight *= rel_weight
                # TODO: div by ?
                min_iso_weight = min(max(0.0, min_iso_weight), 1.0)

        # dynamically determine min_nodes
        min_nodes = MIN_NODES
        if isinstance(min_nodes, str):
            assert min_nodes == "auto"
            assert num_instrs != 0
            assert rel_weight > 0.0
            weight_per_instr = rel_weight / num_instrs
            assert weight_per_instr > 0.0
            min_nodes = None
            if min_iso_weight is not None:
                required_instrs = min_iso_weight / weight_per_instr
                # TODO: div by 2 to consider ISOs
                required_instrs = int(required_instrs + 0.5)  # TODO: round?
                required_instrs = max(1, required_instrs)
                if MAX_NODES is not None:
                    required_instrs = min(required_instrs, MAX_NODES)
                min_nodes = required_instrs

        args = [
            "python3",
            "-m",
            "tool.main",
        ]
        args += [
            *["--progress"],
            *["--times"],
            *["--log", "info"],
            *["--session", label],
            *["--function", func_name],
            *["--basic-block", bb_name],
            *["--stage", str(stage)],
            *["--output-dir", out_dir],
            # *["--ignore-const-inputs"],
            *(["--limit-results", str(LIMIT_RESULTS)] if LIMIT_RESULTS is not None else []),
            *(["--min-inputs", str(MIN_INPUTS)] if MIN_INPUTS is not None else []),
            *(["--max-inputs", str(MAX_INPUTS)] if MAX_INPUTS is not None else []),
            *(["--min-outputs", str(MIN_OUTPUTS)] if MIN_OUTPUTS is not None else []),
            *(["--max-outputs", str(MAX_OUTPUTS)] if MAX_OUTPUTS is not None else []),
            *(["--max-nodes", str(MAX_NODES)] if MAX_NODES is not None else []),
            *(["--min-nodes", str(min_nodes)] if min_nodes is not None else []),
            *(["--max-loads", str(MAX_LOADS)] if MAX_LOADS is not None else []),
            *(["--max-loads", str(MAX_STORES)] if MAX_STORES is not None else []),
            *(["--max-mems", str(MAX_MEMS)] if MAX_MEMS is not None else []),
            *(["--max-branches", str(MAX_BRANCHES)] if MAX_BRANCHES is not None else []),
            *(["--max-enc-footprint", str(MAX_ENC_FOOTPRINT)] if MAX_ENC_FOOTPRINT is not None else []),
            *(["--max-enc-weight", str(MAX_ENC_WEIGHT)] if MAX_ENC_WEIGHT is not None else []),
            *(["--min-enc-bits-left", str(MIN_ENC_BITS_LEFT)] if MIN_ENC_BITS_LEFT is not None else []),
            *(["--min-path-length", str(MIN_PATH_LENGTH)] if MIN_PATH_LENGTH is not None else []),
            *(["--max-path-length", str(MAX_PATH_LENGTH)] if MAX_PATH_LENGTH is not None else []),
            *(["--max-path-width", str(MAX_PATH_WIDTH)] if MAX_PATH_WIDTH is not None else []),
            *(["--min-iso-weight", str(min_iso_weight)] if min_iso_weight is not None else []),
            *(["--instr-predicates", str(INSTR_PREDICATES)] if INSTR_PREDICATES is not None else []),
            *(["--ignore-names", IGNORE_NAMES] if IGNORE_NAMES is not None else []),
            *(["--ignore-op-types", str(IGNORE_OP_TYPES)] if IGNORE_OP_TYPES is not None else []),
            *(["--allowed-enc-sizes", " ".join(map(str, ALLOWED_ENC_SIZES))] if ALLOWED_ENC_SIZES is not None else []),
            *(["--xlen", str(XLEN)] if XLEN is not None else []),
            *(["--enable-variation-reuse-io"] if ENABLE_VARIATION_REUSE_IO else []),  # TODO: use FLT instead?
            *(["--halt-on-error"] if HALT_ON_ERROR else []),  # TODO: use FLT instead?
            *["--write-func"],
            # *["--write-func-fmt", WRITE_FUNC_FMT],
            # *["--write-func-flt", WRITE_FUNC_FLT],
            *["--write-sub"],
            # *["--write-sub-fmt", WRITE_SUB_FMT],
            # *["--write-sub-flt", WRITE_SUB_FLT],
            *["--write-io-sub"],
            # *["--write-io-sub-fmt", WRITE_IO_SUB_FMT],
            # *["--write-io-sub-flt", WRITE_IO_SUB_FLT],
            *["--write-tree"],
            # *["--write-tree-fmt", WRITE_TREE_FMT],
            # *["--write-treee-flt", WRITE_TREE_FLT],
            # *["--write-gen"],
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
        subprocess.run(args, check=True)
    combined_index_file = workdir / "combined_index.yml"
    combine_args = [
        "python3",
        "-m",
        "tool.combine_index",
        *index_files,
        "--drop",
        *(["--sort-by", SORT_BY] if SORT_BY is not None else []),
        *(["--topk", str(TOPK)] if TOPK is not None else []),
        "--out",
        combined_index_file,
    ]
    if len(index_files) in [2, 3]:
        venn_diagram_file = workdir / "venn.jpg"
        combine_args += ["--venn", venn_diagram_file]
    sankey_diagram_file = workdir / "sankey.md"
    combine_args += ["--sankey", sankey_diagram_file]
    overlaps_file = workdir / "overlaps.csv"
    combine_args += ["--overlaps", overlaps_file]
    # print("combine_args", combine_args)
    subprocess.run(combine_args, check=True)
    gen_dir = workdir / "gen"
    gen_dir.mkdir(exist_ok=True)
    generate_args = [
        combined_index_file,
        "--output",
        gen_dir,
        "--split",
        "--split-files",
        "--progress",
        "--inplace",  # TODO use gen/index.yml instead!
    ]
    generate_cdsl_args = [
        "python3",
        "-m",
        "tool.gen.cdsl",
        *generate_args,
    ]
    generate_flat_args = [
        "python3",
        "-m",
        "tool.gen.flat",
        *generate_args,
    ]
    generate_fuse_cdsl_args = [
        "python3",
        "-m",
        "tool.gen.fuse_cdsl",
        *generate_args,
    ]
    print("generate_cdsl_args", generate_cdsl_args)
    print("generate_flat_args", generate_flat_args)
    print("generate_fuse_cdsl_args", generate_fuse_cdsl_args)
    subprocess.run(generate_cdsl_args, check=True)
    subprocess.run(generate_flat_args, check=True)
    subprocess.run(generate_fuse_cdsl_args, check=True)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    query_candidates_from_db(sess, workdir=args.workdir, label=args.label, stage=args.stage, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--stage", type=int, default=8)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
