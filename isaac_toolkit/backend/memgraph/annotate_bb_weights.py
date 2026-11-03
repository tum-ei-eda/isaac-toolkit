#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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
import argparse
import math
from pathlib import Path

from neo4j import GraphDatabase, Query
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.utils.bb_identity import (BBIdentityError, block_id, verified_imports,
    verify_graph_receipts, resolve_block)

logger = get_logger()


class BBMismatchError(ValueError):
    """ELF blocks and the imported CDFG cannot safely share profile weights."""


def graph_bb_counts(driver, label="default", timeout=120, stage=None, module_name=None, compilation_id=None,
                    func_name=None, bb_id=None):
    # Count nodes directly, including instructions with no DFG edges. Input and
    # constant/label nodes describe operands, not ELF instructions.
    query = Query(
        """
        MATCH (b:BB) WHERE b.session = $label
          AND ($func_name IS NULL OR b.func_name = $func_name)
          AND ($bb_id IS NULL OR b.bb_id = $bb_id)
          AND ($stage IS NULL OR b.stage = $stage)
          AND ($module_name IS NULL OR b.module_name = $module_name)
          AND ($compilation_id IS NULL OR b.compilation_id = $compilation_id)
        OPTIONAL MATCH (i:INSTR)
        WHERE i.session = $label AND i.func_name = b.func_name
          AND i.bb_id = b.bb_id AND i.compilation_id = b.compilation_id
          AND i.module_name = b.module_name AND i.stage = b.stage
          AND NOT coalesce(i.op_type, 'instruction') IN ['input', 'constant', 'label']
        RETURN b.func_name AS func_name, b.bb_id AS bb_id,
               '%bb.' + toString(b.bb_id) AS bb_name,
               count(DISTINCT b) AS graph_blocks, count(DISTINCT i) AS graph_nodes,
               collect(DISTINCT b.module_name) AS graph_modules,
               collect(DISTINCT b.stage) AS graph_stages,
               collect(DISTINCT b.name) AS graph_names,
               collect(DISTINCT b.compilation_id) AS graph_compilations,
               collect(DISTINCT id(b)) AS graph_block_ids
    """,
        timeout=timeout,
    )
    with driver.session() as session:
        return [dict(record) for record in session.run(query, label=label, stage=stage,
                module_name=module_name, compilation_id=compilation_id, func_name=func_name, bb_id=bb_id)]


def compare_bb_counts(elf_bbs, graph_counts, *, relative_tolerance=0.10, absolute_tolerance=5, selected=()):
    """Compare block identities and counts; tolerance is abs + rel * ELF count.

    Blocks outside the imported compilation units need not exist in Memgraph.
    Missing selected blocks are errors; other missing blocks are coverage data.
    """
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance < 0
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
    ):
        raise ValueError("BB count tolerances must be finite and nonnegative")
    selected = set(selected)
    graph = {}
    for row in graph_counts:
        fixed_id = block_id(row["bb_name"], row.get("bb_id"))
        key = (row["func_name"], f"%bb.{fixed_id}")
        if key in graph:
            # Never overwrite a second module/stage with the same function/ID.
            old = graph[key]
            old["graph_blocks"] += row["graph_blocks"]
            old["graph_nodes"] += row["graph_nodes"]
            for field in ("graph_modules", "graph_stages", "graph_names", "graph_compilations", "graph_block_ids"):
                old[field] = old.get(field, []) + row.get(field, [])
        else:
            graph[key] = dict(row)
    elf_groups = {}
    for row in elf_bbs.to_dict("records"):
        fixed_id = block_id(row["bb_name"], row.get("bb_id"))
        elf_groups.setdefault((row["func_name"], f"%bb.{fixed_id}"), []).append(row)
    rows = []
    for key in sorted(elf_groups.keys() | graph.keys() | selected):
        elf_rows = elf_groups.get(key, [])
        db = graph.get(key)
        n = elf_rows[0]["num_instrs"] if elf_rows else None
        count = db["graph_nodes"] if db else None
        difference = allowance = None
        severe = False
        if not elf_rows:
            status, severe = "missing_elf", key in selected
        elif db is None:
            status, severe = ("missing_graph", True) if key in selected else ("not_exported", False)
        elif len(elf_rows) > 1 or db["graph_blocks"] != 1:
            status, severe = "ambiguous", True
        elif pd.isna(n) or not math.isfinite(n) or n < 0 or n != int(n):
            status, severe = "invalid_elf_count", True
        elif count == 0 and n > 0:
            difference = n
            allowance = absolute_tolerance + relative_tolerance * n
            status, severe = "empty_graph", True
        else:
            difference = abs(count - n)
            allowance = absolute_tolerance + relative_tolerance * n
            status = "exact" if difference == 0 else ("tolerated" if difference <= allowance else "mismatch")
            severe = status == "mismatch"
        rows.append(
            dict(
                func_name=key[0],
                bb_name=key[1],
                elf_instrs=n,
                graph_nodes=count,
                graph_blocks=db["graph_blocks"] if db else None,
                difference=difference,
                allowance=allowance,
                status=status,
                severe=severe,
                selected=key in selected,
                graph_modules=db.get("graph_modules", []) if db else [],
                graph_stages=db.get("graph_stages", []) if db else [],
                graph_names=db.get("graph_names", []) if db else [],
                graph_compilations=db.get("graph_compilations", []) if db else [],
                graph_block_ids=db.get("graph_block_ids", []) if db else [],
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "func_name",
            "bb_name",
            "elf_instrs",
            "graph_nodes",
            "graph_blocks",
            "difference",
            "allowance",
            "status",
            "severe",
            "selected",
            "graph_modules",
            "graph_stages",
            "graph_names",
            "graph_compilations",
            "graph_block_ids",
        ],
    )


def enforce_bb_report(report, mismatch_action="error"):
    if mismatch_action not in ("error", "warn"):
        raise ValueError("mismatch_action must be 'error' or 'warn'")
    failures = report.loc[report["severe"].astype(bool)]
    matched = report[report["status"].isin(["exact", "tolerated", "mismatch"])]
    logger.info("ELF/CDFG BB count validation: %s", report["status"].value_counts().to_dict())
    missing_elf = int((report["status"] == "missing_elf").sum())
    if missing_elf:
        logger.warning(
            "%d CDFG blocks have no ELF block with the same function/BB identity; "
            "see bb_cdfg_validation for coverage details",
            missing_elf,
        )
    if not failures.empty or matched.empty:
        details = failures.head(10).to_string(index=False)
        message = (
            f"ELF/CDFG BB validation failed: {len(failures)} invalid blocks, "
            f"{len(matched)} comparable blocks. Check the CDFG stage, ELF, and session label."
            f"\n{details}"
        )
        if mismatch_action == "error":
            raise BBMismatchError(message)
        logger.warning(message)


def anonotate_helper(
    driver, func_name, bb_name, num_instrs, freq, rel_weight, label="default", check=True, timeout=120,
    stage=None, module_name=None, compilation_id=None,
):
    if not math.isfinite(num_instrs) or num_instrs <= 0:
        raise ValueError("Cannot annotate weights without a positive ELF instruction count")
    bb_id = block_id(bb_name)
    counts = [r for r in graph_bb_counts(driver, label, timeout, stage, module_name, compilation_id, func_name, bb_id)
              if r["func_name"] == func_name and r["bb_id"] == bb_id]
    if len(counts) != 1 or counts[0]["graph_blocks"] != 1:
        raise BBIdentityError(f"Ambiguous or missing fixed BB identity for {func_name}, {bb_id}")
    count = counts[0]
    if any(len(count[k]) != 1 for k in ("graph_modules", "graph_stages", "graph_compilations")):
        raise BBIdentityError("Annotation requires one verified module/stage/compilation")
    context = dict(module_name=count["graph_modules"][0], stage=count["graph_stages"][0],
                   compilation_id=count["graph_compilations"][0])
    resolved = resolve_block(driver, label, func_name, bb_id, context, timeout)
    if check:
        enforce_bb_report(compare_bb_counts(
            pd.DataFrame([dict(func_name=func_name, bb_name=bb_name, num_instrs=num_instrs)]),
            counts, selected=[(func_name, bb_name)]))
    params = dict(label=label, func_name=func_name, bb_id=bb_id, **context, **resolved,
                  num_instrs=int(num_instrs), freq=float(freq), rel_weight=float(rel_weight),
                  instr_weight=float(rel_weight / num_instrs))
    with driver.session() as session:
        with session.begin_transaction(timeout=timeout) as tx:
            updated = tx.run("""MATCH (a:BB) WHERE id(a) = $graph_block_id
                AND a.session = $label AND a.func_name = $func_name AND a.bb_id = $bb_id
                AND a.module_name = $module_name AND a.stage = $stage
                AND a.compilation_id = $compilation_id
                SET a.bb_freq = $freq, a.bb_instrs = $num_instrs, a.bb_rel_weight = $rel_weight
                RETURN count(a) AS updated""", **params).single()
            if updated is None or updated["updated"] != 1:
                raise BBIdentityError("BB changed while annotating it")
            tx.run("""MATCH (a:INSTR) WHERE a.session = $label
                AND a.func_name = $func_name AND a.bb_id = $bb_id
                AND a.module_name = $module_name AND a.stage = $stage
                AND a.compilation_id = $compilation_id AND a.basic_block = $mir_name
                AND NOT coalesce(a.op_type, 'instruction') IN ['input', 'constant', 'label']
                SET a.instr_freq = $freq, a.instr_rel_weight = $instr_weight""", **params).consume()


def annotate_bb_weights(
    sess: Session,
    label: str = "default",
    force: bool = False,
    relative_tolerance: float = 0.10,
    absolute_tolerance: float = 5,
    mismatch_action: str = "error",
    stage: int = None,
):
    if mismatch_action not in ("error", "warn"):
        raise ValueError("mismatch_action must be 'error' or 'warn'")
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance < 0
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
    ):
        raise ValueError("BB count tolerances must be finite and nonnegative")
    logger.info("Annotating BB weights...")
    memgraph_config = sess.config.memgraph
    memgraph_host = os.environ.get("MEMGRAPH_HOST")
    memgraph_port = os.environ.get("MEMGRAPH_PORT")
    user = ""
    password = ""
    if memgraph_config is not None:
        memgraph_host = memgraph_host or memgraph_config.hostname
        memgraph_port = memgraph_port or memgraph_config.port
        user = memgraph_config.user
        password = memgraph_config.password

    driver = GraphDatabase.driver(f"bolt://{memgraph_host}:{memgraph_port}", auth=(user, password))
    try:

        imports = verified_imports(sess, label, stage)
        verify_graph_receipts(driver, imports, label)
        artifacts = sess.artifacts
        llvm_bbs_artifacts = filter_artifacts(
            artifacts,
            lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs_new",
        )  # TODO: optional or different pass
        assert len(llvm_bbs_artifacts) == 1
        llvm_bbs_artifact = llvm_bbs_artifacts[0]
        llvm_bbs_df = llvm_bbs_artifact.df
        choices = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "choices")
        selected = set()
        if choices:
            selected = set(zip(choices[0].df["func_name"], choices[0].df["bb_name"]))
        report = compare_bb_counts(
            llvm_bbs_df,
            [row for receipt in imports for row in graph_bb_counts(driver, label,
                stage=int(receipt["stage"]), module_name=receipt["module_name"],
                compilation_id=receipt["compilation_id"])],
            selected=selected,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        sess.add_artifact(
            TableArtifact(
                "bb_cdfg_validation",
                report,
                attrs={
                    "kind": "validation",
                    "by": __name__,
                    "label": label,
                    "relative_tolerance": relative_tolerance,
                    "absolute_tolerance": absolute_tolerance,
                    "mismatch_action": mismatch_action,
                },
            ),
            override=force,
        )
        # Validate every block before writing any weights, including cold blocks.
        enforce_bb_report(report, mismatch_action)
        graph_keys = {
            (row["func_name"], row["bb_name"])
            for row in report.to_dict("records")
            if row["status"] in ("exact", "tolerated", "mismatch")
        }
        identities = {(r["func_name"], r["bb_name"]): r for r in report.to_dict("records")}
        # print("llvm_bbs_df", llvm_bbs_df, llvm_bbs_df.columns)
        # input("?")
        for index, row in llvm_bbs_df.iterrows():
            # print("index", index)
            # print("row", row)
            func_name = row["func_name"]
            bb_name = row["bb_name"]
            if (func_name, bb_name) not in graph_keys:
                continue
            num_instrs = row["num_instrs"]
            rel_weight = row["rel_weight"]
            freq = row["freq"]
            if not num_instrs or pd.isna(num_instrs):
                continue
            if not rel_weight or pd.isna(rel_weight):
                continue
            anonotate_helper(
                driver,
                func_name,
                bb_name,
                num_instrs,
                freq,
                rel_weight,
                label=label,
                check=False,
                stage=identities[(func_name, bb_name)]["graph_stages"][0],
                module_name=identities[(func_name, bb_name)]["graph_modules"][0],
                compilation_id=identities[(func_name, bb_name)]["graph_compilations"][0],
            )
    finally:
        driver.close()


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    try:
        annotate_bb_weights(
            sess,
            label=args.label,
            force=args.force,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
            mismatch_action=args.mismatch_action,
            stage=args.stage,
        )
    except BBMismatchError:
        sess.save()  # Preserve the diagnostic table even when the stage stops.
        raise
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--label", default="default", required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--stage", type=int, default=None, help="Select one imported CDFG stage")
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.10,
        help="Allowed count difference as a fraction of ELF instructions (default: 0.10)",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=5,
        help="Additional allowed instruction/node difference (default: 5)",
    )
    parser.add_argument(
        "--mismatch-action",
        choices=("error", "warn"),
        default="error",
        help="Stop before annotating weights, or warn and continue (default: error)",
    )
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
