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
import yaml
import argparse
import subprocess
from typing import Optional, Union
from pathlib import Path
from collections import defaultdict

import pandas as pd
from neo4j import GraphDatabase, Query
import networkx as nx
import networkx.algorithms.isomorphism as iso


from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts, TableArtifact
from isaac_toolkit.utils.graph_utils import memgraph_to_nx
from isaac_toolkit.algorithm.ise.identification.maxmiso import maxmiso_algo
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


def get_unique_maxmisos(maxmisos):
    isos_map = defaultdict(list)
    covered = set()
    for i, m1 in enumerate(maxmisos):
        if i in covered:
            continue
        for j, m2 in enumerate(maxmisos):

            if j <= i:
                continue

            def node_match(*args):
                ret = iso.categorical_node_match("label", None)(*args)
                return ret

            def edge_match(*args):
                return iso.categorical_edge_match("label", None)(*args)

            is_iso = nx.is_isomorphic(
                m1, m2, node_match=node_match, edge_match=edge_match
            )
            if is_iso:
                isos_map[i].append(j)
                covered.add(j)

    logger.debug("isos_map", isos_map, len(isos_map))
    isos_size = {k: len(v) for k, v in isos_map.items()}
    logger.debug("isos_size", isos_size)
    logger.debug("covered", covered, len(covered))
    non_isos = [i for i in range(len(maxmisos)) if i not in covered]
    logger.debug("non_isos", non_isos, len(non_isos))
    unique_maxmisos = [maxmisos[i] for i in non_isos]
    duplicate_maxmisos = [maxmisos[i] for i in covered]
    logger.debug("unique_maxmisos", list(map(str, unique_maxmisos)))
    factors = [isos_size.get(i, 1) for i in non_isos]
    return unique_maxmisos, factors, duplicate_maxmisos


def get_func_query(session, stage, func_name):
    return Query(
        f"""MATCH p0=(n00:INSTR)-[r01:DFG]->(n01:INSTR)
        WHERE n00.func_name = '{func_name}'
        AND n00.session = "{session}"
        AND n00.stage = {stage}
        AND n01.name != "PHI" AND n01.name != "G_PHI"
        RETURN p0
        """
    )


def get_bbs_query(session, stage, func_name):
    return Query(
        f"""MATCH (n00:INSTR)
        WHERE n00.func_name = '{func_name}'
        AND n00.session = "{session}"
        AND n00.stage = {stage}
        RETURN n00.bb_id as bb_id, count(*) as num_instrs
        ORDER BY bb_id
        """
    )


def get_update_nodes_query(maxmiso_idx, maxmiso_nodes, factor=1):
    size = len(maxmiso_nodes)
    return Query(
        f"""
        MATCH (n:INSTR)
        WHERE id(n) IN {maxmiso_nodes}
        SET n.maxmiso_idx = {maxmiso_idx}, n.maxmiso_factor = {factor}, n.maxmiso_size = {size};
        """
    )


def query_candidates_from_db(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    label: Optional[str] = None,
    stage: int = 8,
    force: bool = False,
    progress: bool = False,
    query_config_yaml: Optional[Union[str, Path]] = None,
    limit_results: Optional[int] = None,
    # LIMIT_RESULTS: Optional[int] = 5000,
    # LIMIT_RESULTS: Optional[int] = 500,
    MIN_INPUTS: Optional[int] = 1,
    MAX_INPUTS: Optional[int] = 4,
    # MAX_INPUTS: Optional[int] = 3,
    MIN_OUTPUTS: Optional[int] = 1,
    MAX_OUTPUTS: Optional[int] = 2,
    # MAX_OUTPUTS: Optional[int] = 1,
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
    MAX_PATH_LENGTH=9,
    MAX_PATH_WIDTH=2,
    # MAX_PATH_WIDTH = 4,
    INSTR_PREDICATES=511,  # ALL?
    IGNORE_NAMES=None,
    IGNORE_OP_TYPES=None,
    ALLOWED_ENC_SIZES=[32],
    min_iso_weight=0.05,
    scale_iso_weight: Union[bool, str] = "auto",
    MAX_LOADS=1,
    MAX_STORES=1,
    MAX_MEMS: Optional[int] = 0,  # TODO
    MAX_BRANCHES: Optional[int] = 1,
    xlen: Optional[int] = 32,  # TODO: do not hardcode
    ENABLE_VARIATION_REUSE_IO=False,
    # ENABLE_VARIATION_REUSE_IO=True,
    HALT_ON_ERROR: bool = True,
    sort_by: Optional[str] = "IsoWeight",
    topk: Optional[int] = None,
    partition_with_maxmiso: Union[str, bool] = "auto",
):
    logger.info("Querying candidates from DB...")
    artifacts = sess.artifacts
    choices_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "choices"
    )
    assert len(choices_artifacts) == 1
    choices_artifact = choices_artifacts[0]
    choices_df = choices_artifact.df
    logger.debug("choices_df", choices_df)
    if workdir is None:
        raise NotImplementedError("automatic workdir not supported yet!")
    if isinstance(workdir, str):
        workdir = Path(workdir)
    assert isinstance(workdir, Path)
    workdir = workdir.resolve()
    assert label is not None
    index_files = []
    # FUNC_ONLY = True
    FUNC_ONLY = False
    if FUNC_ONLY:
        # TODO: for real!
        funcs_df = choices_df.groupby("func_name", as_index=False)[
            ["rel_weight", "num_instrs", "freq"]
        ].sum()
        funcs_df["bb_name"] = None
        choices_df = funcs_df

    logger.debug("choices_df", choices_df)

    combined_query_metrics_df = pd.DataFrame()
    missing_bb_ids = set()
    for index, row in choices_df.iterrows():
        # logger.debug("index", index)
        # logger.debug("row", row)
        # if index != 1:
        #     continue
        # input(">")
        func_name = row["func_name"]
        bb_name = row["bb_name"]
        rel_weight = row["rel_weight"]
        num_instrs = row["num_instrs"]
        logger.debug("func_name", func_name)
        logger.debug("bb_name", bb_name)

        bbs_query = get_bbs_query(label, stage, func_name)

        memgraph_config = sess.config.memgraph
        hostname = memgraph_config.hostname
        port = memgraph_config.port
        user = memgraph_config.user
        password = memgraph_config.password

        driver = GraphDatabase.driver(
            f"bolt://{hostname}:{port}", auth=(user, password)
        )
        session = driver.session()
        try:
            func_query = get_func_query(label, stage, func_name)
            # logger.debug("func_query", func_query)
            func_results = session.run(func_query)
            func = memgraph_to_nx(func_results)
            # logger.debug("func", func)
            # input(">>")
            bbs_results = session.run(bbs_query)
            bbs_df = bbs_results.to_df()
        finally:
            session.close()
        # logger.debug("bbs_df", bbs_df)
        # input(">")
        bb_id = int(bb_name.split(".", 1)[1])
        assert len(bbs_df) > 0
        db_bb_ids = bbs_df["bb_id"].unique()
        if bb_id not in db_bb_ids:
            logger.warning(f"BB ID {bb_id} not found in DB!")
            missing_bb_ids.add((func_name, bb_id))
            continue
        maxmiso_idxs = []
        # TODO: skip maxmisos which are unsuitable?
        if isinstance(partition_with_maxmiso, str):
            # TODO: Handle 1/0/true/false/...
            # TODO: if numeric parse number!
            assert partition_with_maxmiso == "auto"
            num_instrs_threshold = 250
            partition_with_maxmiso = num_instrs > num_instrs_threshold
        if partition_with_maxmiso:
            bb_nodes = [
                node
                for node in func.nodes
                if func.nodes[node]["properties"]["bb_id"] == bb_id
            ]
            # logger.debug("bb_nodes", bb_nodes)
            bb = func.subgraph(bb_nodes)
            # logger.debug("bb", bb)
            maxmisos = maxmiso_algo(bb)
            # logger.debug("maxmisos", maxmisos)
            # input(">")
            unique_maxmisos, factors, duplicate_maxmisos = get_unique_maxmisos(maxmisos)
            # TODO: write maxmisos to file?
            # maxmiso_node_ids = [[maxmiso.nodes[n]["key"] for n in maxmiso.nodes] for maxmiso in maxmisos]
            unique_maxmiso_node_ids = [
                [maxmiso.nodes[n]["key"] for n in maxmiso.nodes]
                for maxmiso in unique_maxmisos
            ]
            duplicate_maxmiso_node_ids = [
                [maxmiso.nodes[n]["key"] for n in maxmiso.nodes]
                for maxmiso in duplicate_maxmisos
            ]
            remaining_nodes = set(bb_nodes)
            total_idx = 0
            session = driver.session()
            try:
                # TODO: make unique optional!
                # for i, node_ids in enumerate(maxmiso_node_ids):
                for i, node_ids in enumerate(duplicate_maxmiso_node_ids):
                    factor = 0
                    idx = -1
                    # Negative -> ignore
                    query = get_update_nodes_query(idx, node_ids, factor=0)
                    _ = session.run(query)
                    remaining_nodes -= set(node_ids)
                for i, node_ids in enumerate(unique_maxmiso_node_ids):
                    factor = factors[i]
                    query = get_update_nodes_query(total_idx, node_ids, factor=factor)
                    maxmiso_idxs.append(total_idx)
                    # logger.debug("query", query)
                    _ = session.run(query)
                    # logger.debug("results3", results3)
                    remaining_nodes -= set(node_ids)
                    total_idx += 1
                logger.debug("remaining_nodes", remaining_nodes)
                if len(remaining_nodes) > 0:
                    query = get_update_nodes_query(
                        total_idx, list(remaining_nodes), factor=1
                    )
                    _ = session.run(query)
                    maxmiso_idxs.append(total_idx)
                total_idx += 1
            finally:
                session.close()

        out_name = f"{func_name}_{bb_name}_0"
        out_dir = workdir / out_name
        logger.debug("out_dir", out_dir)
        out_dir.mkdir(exist_ok=True)
        index_file = out_dir / "index.yml"
        index_files.append(index_file)
        # TODO: to not hardcode this, call directly via python

        if scale_iso_weight is not None:
            if isinstance(scale_iso_weight, str):
                assert scale_iso_weight == "auto"
                scale_iso_weight = rel_weight
            assert isinstance(scale_iso_weight, float)
            assert scale_iso_weight > 0

            if min_iso_weight is not None:
                min_iso_weight *= scale_iso_weight
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
            *(["--progress"] if progress else []),
            *["--times"],
            *["--log", "info"],
            *(["--yaml", query_config_yaml] if query_config_yaml is not None else []),
            *["--session", label],
            *["--function", func_name],
            *(["--basic-block", bb_name] if bb_name is not None else []),
            *["--stage", str(stage)],
            *["--output-dir", out_dir],
            # *["--ignore-const-inputs"],
            *(
                ["--limit-results", str(limit_results)]
                if limit_results is not None
                else []
            ),
            *(["--min-inputs", str(MIN_INPUTS)] if MIN_INPUTS is not None else []),
            *(["--max-inputs", str(MAX_INPUTS)] if MAX_INPUTS is not None else []),
            *(["--min-outputs", str(MIN_OUTPUTS)] if MIN_OUTPUTS is not None else []),
            *(["--max-outputs", str(MAX_OUTPUTS)] if MAX_OUTPUTS is not None else []),
            *(["--max-nodes", str(MAX_NODES)] if MAX_NODES is not None else []),
            *(["--min-nodes", str(min_nodes)] if min_nodes is not None else []),
            *(["--max-loads", str(MAX_LOADS)] if MAX_LOADS is not None else []),
            *(["--max-loads", str(MAX_STORES)] if MAX_STORES is not None else []),
            *(["--max-mems", str(MAX_MEMS)] if MAX_MEMS is not None else []),
            *(
                ["--max-branches", str(MAX_BRANCHES)]
                if MAX_BRANCHES is not None
                else []
            ),
            *(
                ["--max-enc-footprint", str(MAX_ENC_FOOTPRINT)]
                if MAX_ENC_FOOTPRINT is not None
                else []
            ),
            *(
                ["--max-enc-weight", str(MAX_ENC_WEIGHT)]
                if MAX_ENC_WEIGHT is not None
                else []
            ),
            *(
                ["--min-enc-bits-left", str(MIN_ENC_BITS_LEFT)]
                if MIN_ENC_BITS_LEFT is not None
                else []
            ),
            *(
                ["--min-path-length", str(MIN_PATH_LENGTH)]
                if MIN_PATH_LENGTH is not None
                else []
            ),
            *(
                ["--max-path-length", str(MAX_PATH_LENGTH)]
                if MAX_PATH_LENGTH is not None
                else []
            ),
            *(
                ["--max-path-width", str(MAX_PATH_WIDTH)]
                if MAX_PATH_WIDTH is not None
                else []
            ),
            *(
                ["--min-iso-weight", str(min_iso_weight)]
                if min_iso_weight is not None
                else []
            ),
            *(
                ["--instr-predicates", str(INSTR_PREDICATES)]
                if INSTR_PREDICATES is not None
                else []
            ),
            *(["--ignore-names", IGNORE_NAMES] if IGNORE_NAMES is not None else []),
            *(
                ["--ignore-op-types", str(IGNORE_OP_TYPES)]
                if IGNORE_OP_TYPES is not None
                else []
            ),
            *(
                ["--allowed-enc-sizes", " ".join(map(str, ALLOWED_ENC_SIZES))]
                if ALLOWED_ENC_SIZES is not None
                else []
            ),
            *(["--xlen", str(xlen)] if xlen is not None else []),
            *(
                ["--enable-variation-reuse-io"] if ENABLE_VARIATION_REUSE_IO else []
            ),  # TODO: use FLT instead?
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
            *["--write-query-metrics"],
        ]
        if partition_with_maxmiso and len(maxmiso_idxs) > 0:
            maxmiso_idxs_str = ",".join(map(str, maxmiso_idxs))
            args += ["--maxmisos", maxmiso_idxs_str]
        # args += ["--help"]
        logger.debug(">", " ".join(map(str, args)))
        subprocess.run(args, check=True)
        query_metrics_file = out_dir / "query_metrics.csv"
        query_metrics_df = pd.read_csv(query_metrics_file)
        query_metrics_df["func"] = func_name
        query_metrics_df["basic_block"] = bb_name
        combined_query_metrics_df = pd.concat(
            [combined_query_metrics_df, query_metrics_df]
        )

    # allow_missing_bb_id = False
    allow_missing_bb_id = True
    if len(missing_bb_ids) > 0:
        assert len(missing_bb_ids) < len(choices_df), "No matching BB IDs found in DB!"
        assert allow_missing_bb_id, f"Missing BB ID DB entries: {missing_bb_ids}"

    combined_query_metrics_file = workdir / "combined_query_metrics.csv"
    combined_query_metrics_df.to_csv(combined_query_metrics_file, index=False)
    combined_index_file = workdir / "combined_index.yml"
    combine_args = [
        "python3",
        "-m",
        "tool.combine_index",
        *index_files,
        "--drop-duplicates",
        "--drop-name-isos",  # NEW
        *(["--progress"] if progress else []),
        *(["--sort-by", sort_by] if sort_by is not None else []),
        *(["--topk", str(topk)] if topk is not None else []),
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
    # logger.debug("combine_args", combine_args)
    subprocess.run(combine_args, check=True)

    # TODO: index and cdsl should use the same instruction names?
    # names = [f"CUSTOM{i}" for i, candidate in enumerate(combined_index_data["candidates"])]
    # num_fused_instrs = [
    #     candidate["properties"]["#Instrs"] for i, candidate in enumerate(combined_index_data["candidates"])
    # ]
    # names_df = pd.DataFrame({"instr": names, "num_fused_instrs": num_fused_instrs})
    # names_df["instr_lower"] = names_df["instr"].apply(lambda x: x.lower())
    names_csv = workdir / "names.csv"
    assign_args = [
        "python3",
        "scripts/assign_names.py",  # TODO: move into toolkit
        combined_index_file,
        "--inplace",
        "--csv",
        names_csv,
    ]
    subprocess.run(assign_args, check=True)

    names_df = pd.read_csv(names_csv)

    # Extract names
    with open(combined_index_file, "r") as f:
        combined_index_data = yaml.safe_load(f)

    name2num_fused_instrs = {
        candidate["properties"]["InstrName"]: candidate["properties"]["#Instrs"]
        for i, candidate in enumerate(combined_index_data["candidates"])
    }
    names_df["num_fused_instrs"] = names_df["instr"].apply(
        lambda x: name2num_fused_instrs[x]
    )
    # logger.debug("names_df", names_df)
    # input(">>>")
    attrs = {}
    ise_instrs_artifact = TableArtifact("ise_instrs", names_df, attrs=attrs)
    sess.add_artifact(ise_instrs_artifact, override=force)

    # gen_dir = Path(gen_dir) if gen_dir is not None else workdir / "gen"
    # gen_dir.mkdir(exist_ok=True)
    # generate_args = [
    #     combined_index_file,
    #     "--output",
    #     gen_dir,
    #     "--split",
    #     "--split-files",
    #     "--progress",
    #     "--inplace",  # TODO use gen/index.yml instead!
    # ]
    # generate_cdsl_args = [
    #     "python3",
    #     "-m",
    #     "tool.gen.cdsl",
    #     *generate_args,
    # ]
    # generate_flat_args = [
    #     "python3",
    #     "-m",
    #     "tool.gen.flat",
    #     *generate_args,
    # ]
    # generate_fuse_cdsl_args = [
    #     "python3",
    #     "-m",
    #     "tool.gen.fuse_cdsl",
    #     *generate_args,
    # ]
    # logger.debug("generate_cdsl_args", generate_cdsl_args)
    # logger.debug("generate_flat_args", generate_flat_args)
    # logger.debug("generate_fuse_cdsl_args", generate_fuse_cdsl_args)
    # subprocess.run(generate_cdsl_args, check=True)
    # subprocess.run(generate_flat_args, check=True)
    # subprocess.run(generate_fuse_cdsl_args, check=True)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    query_candidates_from_db(
        sess,
        workdir=args.workdir,
        label=args.label,
        stage=args.stage,
        force=args.force,
        progress=args.progress,
        query_config_yaml=args.query_config_yaml,
        limit_results=args.limit_results,
        xlen=args.xlen,
        min_iso_weight=args.min_iso_weight,
        scale_iso_weight=args.scale_iso_weight,
        sort_by=args.sort_by,
        topk=args.topk,
        partition_with_maxmiso=args.partition_with_maxmiso,
    )
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
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--progress", action="store_true", default=None)
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--limit-results", type=int, default=None)
    parser.add_argument("--query-config-yaml", type=str, default=None)
    parser.add_argument("--xlen", type=int, default=32)
    parser.add_argument("--min-iso-weight", type=float, default=None)
    parser.add_argument("--scale-iso-weight", type=str, default="auto")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--sort-by", type=str, default=None)
    parser.add_argument("--partition-with-maxmiso", type=str, default="auto")
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
