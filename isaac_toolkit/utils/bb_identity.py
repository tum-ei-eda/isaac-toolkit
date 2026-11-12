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
"""Fixed LLVM block identities and receipts for CDFG recompilation."""

import hashlib
import json
import numbers
import re
import shlex
import shutil
from pathlib import Path

from neo4j import Query

from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


class BBIdentityError(ValueError):
    """The requested ELF block cannot be identified safely in the CDFG."""


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_id(bb_name, explicit=None):
    match = re.fullmatch(r"%bb\.(0|[1-9][0-9]*)", str(bb_name))
    if not match:
        raise BBIdentityError(f"Expected a fixed ELF BB name, got {bb_name!r}")
    value = int(match[1])
    if explicit is not None and (
        isinstance(explicit, bool) or not isinstance(explicit, numbers.Integral) or int(explicit) != value
    ):
        raise BBIdentityError(f"Conflicting BB identity: {bb_name!r}, bb_id={explicit!r}")
    return value


def elf_digest(sess):
    artifacts = filter_artifacts(sess.artifacts, lambda a: a.flags & ArtifactFlag.ELF)
    if len(artifacts) != 1:
        raise BBIdentityError("Exactly one ELF artifact is required for CDFG identity verification")
    return file_digest(artifacts[0].path)


def command_source(row):
    path = Path(row["file"])
    return str((path if path.is_absolute() else Path(row["directory"]) / path).resolve())


def verified_imports(sess, label, stage=None):
    receipts = filter_artifacts(sess.artifacts, lambda a: a.name == "cdfg_imports" and a.flags & ArtifactFlag.TABLE)
    if len(receipts) != 1:
        raise BBIdentityError("No CDFG import receipt. Regenerate the CDFG with this toolkit before querying it.")
    rows = [
        r
        for r in receipts[0].df.to_dict("records")
        if r["label"] == label and (stage is None or int(r["stage"]) == int(stage))
    ]
    if not rows:
        raise BBIdentityError(f"No verified CDFG import for label={label!r}, stage={stage!r}")
    if len({int(r["stage"]) for r in rows}) != 1:
        raise BBIdentityError("Multiple imported stages: select an explicit stage")
    if len({(r["module_name"], int(r["stage"])) for r in rows}) != len(rows):
        raise BBIdentityError("Duplicate CDFG import receipts; regenerate the CDFG")
    elf = elf_digest(sess)
    require_bb_tables(sess, elf)
    commands = filter_artifacts(
        sess.artifacts, lambda a: a.flags & ArtifactFlag.TABLE and a.attrs.get("kind") == "compile_commands"
    )
    if len(commands) != 1:
        raise BBIdentityError("Missing or ambiguous compile commands for the CDFG receipt")
    cache = {}
    for row in rows:
        if row["elf_sha256"] != elf:
            raise BBIdentityError("ELF changed since CDFG import; regenerate the CDFG")
        matches = [r for r in commands[0].df.to_dict("records") if command_source(r) == row["source_file"]]
        if len(matches) != 1 or hashlib.sha256(matches[0]["command"].encode()).hexdigest() != row["command_sha256"]:
            raise BBIdentityError(f'Compile command changed or is ambiguous for {row["source_file"]}')
        if str(Path(matches[0]["directory"]).resolve()) != row["directory"]:
            raise BBIdentityError(f'Compile directory changed for {row["source_file"]}')
        compiler_arg = shlex.split(matches[0]["command"])[0]
        compiler = Path(compiler_arg)
        if not compiler.is_absolute():
            compiler = (
                Path(row["directory"]) / compiler
                if "/" in compiler_arg
                else Path(shutil.which(compiler_arg) or compiler_arg)
            )
        if str(compiler.resolve()) != row["compiler"]:
            raise BBIdentityError(f'Compiler selection changed for {row["source_file"]}')
        dependencies = json.loads(row["dependencies_json"])
        dependencies[row["compiler"]] = row["compiler_sha256"]
        for path, expected in dependencies.items():
            if path not in cache:
                try:
                    cache[path] = file_digest(path)
                except OSError as exc:
                    raise BBIdentityError(f"Cannot verify CDFG dependency {path}: {exc}") from exc
            if cache[path] != expected:
                raise BBIdentityError(f"CDFG input changed: {path}; regenerate the CDFG")
    return rows


def context_for_choice(imports, choice):
    source = choice.get("file")
    rows = imports
    if source is not None:
        rows = [r for r in rows if r["source_file"] == str(Path(source).resolve())]
    scopes = {(r["module_name"], int(r["stage"]), r["compilation_id"]) for r in rows}
    if len(scopes) != 1:
        raise BBIdentityError(f'Ambiguous or missing compilation context for {choice["func_name"]}')
    module, stage, compilation = next(iter(scopes))
    return dict(module_name=module, stage=stage, compilation_id=compilation)


def resolve_block(driver, label, func_name, bb_id, context, timeout=120):
    params = dict(label=label, func=func_name, bb_id=bb_id, **context)
    query = Query(
        """MATCH (b:BB)
        WHERE b.session = $label AND b.func_name = $func AND b.bb_id = $bb_id
          AND b.module_name = $module_name AND b.stage = $stage
          AND b.compilation_id = $compilation_id
        RETURN id(b) AS graph_block_id, b.name AS mir_name,
               b.bb_clone_id AS clone_id, b.bb_identity_version AS identity_version""",
        timeout=timeout,
    )
    with driver.session() as session:
        rows = [dict(r) for r in session.run(query, **params)]
    if len(rows) != 1 or rows[0]["clone_id"] != 0 or rows[0]["identity_version"] != 1:
        raise BBIdentityError(f"Expected one unambiguous BB for {func_name}, fixed ID {bb_id}, {context}; got {rows}")
    return rows[0]


_IDENTITY_PROPERTIES = (
    "bb_id",
    "bb_clone_id",
    "bb_identity_version",
    "name",
    "basic_block",
    "inst",
    "func_name",
    "module_name",
    "stage",
    "op_type",
    "kind",
    "label_type",
    "out_reg_name",
    "out_reg_type",
    "out_reg_class",
    "out_reg_size",
    "mayLoad",
    "mayStore",
    "isPseudo",
    "isReturn",
    "isCall",
    "isTerminator",
    "isBranch",
    "hasUnmodeledSideEffects",
    "isCommutable",
)


def graph_receipt(driver, label, module_name, stage, compilation_id=None):
    query = """MATCH (n) WHERE (n:BB OR n:INSTR) AND n.session = $label
        AND n.module_name = $module_name AND n.stage = $stage
        AND ($compilation_id IS NULL OR n.compilation_id = $compilation_id)
        RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props"""
    params = dict(label=label, module_name=module_name, stage=stage, compilation_id=compilation_id)
    with driver.session() as session:
        rows = [dict(r) for r in session.run(Query(query, timeout=120), **params)]
        edges = [
            dict(r)
            for r in session.run(
                Query(
                    """MATCH (n)-[r]->(m)
            WHERE (n:BB OR n:INSTR) AND (m:BB OR m:INSTR)
              AND n.session = $label AND m.session = $label
              AND n.module_name = $module_name AND m.module_name = $module_name
              AND n.stage = $stage AND m.stage = $stage
              AND ($compilation_id IS NULL OR
                   (n.compilation_id = $compilation_id AND m.compilation_id = $compilation_id))
            RETURN id(r) AS id, id(n) AS src, id(m) AS dst, type(r) AS type, properties(r) AS props""",
                    timeout=120,
                ),
                **params,
            )
        ]
    canonical = dict(
        nodes=sorted(
            (r["id"], sorted(r["labels"]), {k: r["props"].get(k) for k in _IDENTITY_PROPERTIES}) for r in rows
        ),
        edges=sorted(edges, key=lambda r: r["id"]),
    )
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def verify_graph_receipts(driver, imports, label):
    for row in imports:
        actual = graph_receipt(driver, label, row["module_name"], int(row["stage"]), row["compilation_id"])
        if actual != row["graph_sha256"]:
            raise BBIdentityError(
                f'CDFG contents changed or import is missing for {row["module_name"]}, '
                f'stage {row["stage"]}; regenerate it'
            )


def require_bb_tables(sess, expected_elf):
    """Old padded ranges and choices must be recomputed with the fixed decoder."""
    for name in ("llvm_bbs", "llvm_bbs_new", "choices"):
        tables = filter_artifacts(sess.artifacts, lambda a: a.name == name and a.flags & ArtifactFlag.TABLE)
        if (
            len(tables) != 1
            or tables[0].attrs.get("bb_map_decoder") != 2
            or tables[0].attrs.get("elf_sha256") != expected_elf
        ):
            raise BBIdentityError(
                f"{name} is stale or from the old decoder; rerun BB analysis, trace mapping and choices"
            )
