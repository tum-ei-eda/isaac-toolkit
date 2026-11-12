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
import subprocess
import hashlib
import json
import shlex
import shutil
import tempfile
import uuid

from neo4j import GraphDatabase, Query
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.utils.bb_identity import (
    BBIdentityError,
    file_digest,
    elf_digest,
    graph_receipt,
    require_bb_tables,
    command_source,
)

logger = get_logger()


def get_command_for_file(compile_commands_df: pd.DataFrame, file: str):
    # print("compile_commands_df", compile_commands_df)
    # print("files", compile_commands_df["file"].unique())
    matches = compile_commands_df[compile_commands_df["file_resolved"] == file]
    # print("matches", matches)
    if len(matches) == 0:
        matches = compile_commands_df[compile_commands_df["file_name"] == file]
    assert len(matches) > 0, f"Could not find compile commands for file: {file}"
    assert len(matches) == 1, f"Too many matches during compile commands lookup for file: {file}"
    command = matches["command"].values[0]
    directory = matches["directory"].values[0]
    return command, directory


def generate_memgraph_cdfg_via_compile_commands(
    sess: Session,
    label: str,
    stage: int = 32,
    force: bool = False,
):
    if not label or any(c in label for c in "'\\\n\r"):
        raise BBIdentityError("CDFG label must be nonempty and contain no quotes, backslashes or newlines")
    if isinstance(stage, bool) or not isinstance(stage, int) or stage <= 0:
        raise BBIdentityError("A positive CDFG stage mask is required")
    artifacts = sess.artifacts
    choices = filter_artifacts(artifacts, lambda a: a.flags & ArtifactFlag.TABLE and a.name == "choices")
    commands = filter_artifacts(
        artifacts, lambda a: a.flags & ArtifactFlag.TABLE and a.attrs.get("kind") == "compile_commands"
    )
    if len(choices) != 1 or len(commands) != 1 or choices[0].df.empty:
        raise BBIdentityError("Choices and unambiguous compile commands are required")
    command_df = commands[0].df.copy()
    command_df["file_resolved"] = command_df.apply(command_source, axis=1)
    command_df["file_name"] = command_df["file"].map(lambda p: Path(p).name)
    previous = filter_artifacts(artifacts, lambda a: a.flags & ArtifactFlag.TABLE and a.name == "cdfg_imports")
    if len(previous) > 1:
        raise BBIdentityError("Ambiguous CDFG import receipts")
    receipts = previous[0].df.to_dict("records") if previous else []
    elf_hash = elf_digest(sess)
    require_bb_tables(sess, elf_hash)
    config = sess.config.memgraph
    host = os.environ.get("MEMGRAPH_HOST") or (config.hostname if config else "localhost")
    port = os.environ.get("MEMGRAPH_PORT") or (config.port if config else 7687)
    driver = GraphDatabase.driver(f"bolt://{host}:{port}", auth=(config.user, config.password) if config else ("", ""))
    try:
        for source in choices[0].df["file"].unique():
            if source is None:
                raise BBIdentityError("Cannot import a choice without its source file")
            source = str(Path(source).resolve())
            command, cwd = get_command_for_file(command_df, source)
            args = shlex.split(command)
            if Path(args[0]).name in ("ccache", "sccache", "env"):
                raise BBIdentityError("CDFG provenance requires a direct compiler command, without a wrapper")
            compiler = args[0]
            if not Path(compiler).is_absolute():
                compiler = str(Path(cwd) / compiler) if "/" in compiler else shutil.which(compiler)
            if not compiler:
                raise BBIdentityError(f"Compiler not found: {args[0]}")
            compiler = str(Path(compiler).resolve())
            compiler_hash = file_digest(compiler)
            source_hash = file_digest(source)
            compilation = uuid.uuid4().hex
            staging = "isaac-import-" + compilation
            try:
                with tempfile.TemporaryDirectory(prefix="isaac-cdfg-") as temp:
                    # Never overwrite the original object or dependency file.
                    cleaned = [compiler]
                    i = 1
                    while i < len(args):
                        arg = args[i]
                        if arg in ("-o", "-MF", "-MT", "-MQ"):
                            i += 2
                            continue
                        if arg in ("-MD", "-MMD", "-MP"):
                            i += 1
                            continue
                        if arg.startswith(("-MF", "-MT", "-MQ")):
                            i += 1
                            continue
                        cleaned.append(arg)
                        i += 1
                    depfile = Path(temp) / "dependencies.d"
                    cleaned += [
                        "-o",
                        str(Path(temp) / "cdfg.o"),
                        "-MD",
                        "-MF",
                        str(depfile),
                        "-mllvm",
                        "-cdfg-enable=1",
                        "-mllvm",
                        "-cdfg-memgraph-session=" + staging,
                        "-mllvm",
                        "-cdfg-memgraph-purge=0",
                        "-mllvm",
                        f"-cdfg-stage-mask={stage}",
                        "-mllvm",
                        f"-cdfg-memgraph-host={host}",
                        "-mllvm",
                        f"-cdfg-memgraph-port={port}",
                    ]
                    subprocess.run(cleaned, check=True, cwd=cwd)
                    dependency_text = depfile.read_text().replace("\\\n", "")
                    dependencies = {}
                    for dependency in shlex.split(dependency_text.split(":", 1)[1]):
                        path = Path(dependency)
                        path = path if path.is_absolute() else Path(cwd) / path
                        path = str(path.resolve())
                        dependencies[path] = file_digest(path)
                    if dependencies.get(source) != source_hash or file_digest(compiler) != compiler_hash:
                        raise BBIdentityError("Compiler or source changed during CDFG import")
                with driver.session() as db:
                    inventory = [
                        dict(r)
                        for r in db.run(
                            Query(
                                """MATCH (b:BB) WHERE b.session = $staging
                        RETURN b.func_name AS func, b.bb_id AS bb_id, b.name AS name,
                               b.module_name AS module, b.stage AS stage,
                               b.bb_clone_id AS clone, b.bb_identity_version AS version""",
                                timeout=120,
                            ),
                            staging=staging,
                        )
                    ]
                modules = {r["module"] for r in inventory}
                if len(modules) != 1 or not all(isinstance(m, str) and m for m in modules):
                    raise BBIdentityError("CDFG module is missing or ambiguous")
                module = next(iter(modules))
                module_path = Path(module)
                if str((module_path if module_path.is_absolute() else Path(cwd) / module_path).resolve()) != source:
                    raise BBIdentityError("CDFG module does not match the selected source file")
                identities = set()
                for row in inventory:
                    key = (row["func"], row["bb_id"], row["stage"])
                    if (
                        row["version"] != 1
                        or row["clone"] != 0
                        or not isinstance(row["bb_id"], int)
                        or row["bb_id"] < 0
                        or not isinstance(row["stage"], int)
                        or row["stage"] <= 0
                        or row["stage"] & (row["stage"] - 1)
                        or not row["stage"] & stage
                        or key in identities
                    ):
                        raise BBIdentityError(
                            "Invalid/ambiguous CDFG identity. Rebuild and use the updated LLVM compiler."
                        )
                    identities.add(key)
                with driver.session() as db:
                    invalid = db.run(
                        Query(
                            """MATCH (i:INSTR) WHERE i.session = $staging
                        OPTIONAL MATCH (b:BB) WHERE b.session = $staging
                          AND b.func_name = i.func_name AND b.module_name = i.module_name
                          AND b.stage = i.stage AND b.bb_id = i.bb_id AND b.name = i.basic_block
                        WITH i, count(b) AS owners WHERE owners <> 1
                        RETURN count(i) AS invalid""",
                            timeout=120,
                        ),
                        staging=staging,
                    ).single()
                    if invalid is None or invalid["invalid"]:
                        raise BBIdentityError("CDFG instructions have missing or ambiguous BB ownership")
                stages = sorted({r["stage"] for r in inventory})
                if any(
                    r["label"] == label
                    and r["module_name"] == module
                    and int(r["stage"]) in stages
                    and r["source_file"] != source
                    for r in receipts
                ):
                    raise BBIdentityError(
                        "CDFG module name collision; use unambiguous source paths in compile commands"
                    )
                new_receipts = []
                for actual_stage in stages:
                    new_receipts.append(
                        dict(
                            label=label,
                            module_name=module,
                            source_file=source,
                            stage=actual_stage,
                            compilation_id=compilation,
                            elf_sha256=elf_hash,
                            compiler=compiler,
                            compiler_sha256=compiler_hash,
                            directory=str(Path(cwd).resolve()),
                            command_sha256=hashlib.sha256(command.encode()).hexdigest(),
                            dependencies_json=json.dumps(dependencies, sort_keys=True),
                            graph_sha256=graph_receipt(driver, staging, module, actual_stage),
                        )
                    )
                if elf_digest(sess) != elf_hash:
                    raise BBIdentityError("ELF changed during CDFG import")
                # Only publish a complete import. Failed compilation leaves the
                # old public graph and receipt intact; stale nodes cannot merge.
                replaced_modules = sorted(
                    {module}
                    | {
                        r["module_name"]
                        for r in receipts
                        if r["label"] == label and r["source_file"] == source and int(r["stage"]) in stages
                    }
                )
                with driver.session() as db:
                    with db.begin_transaction() as tx:
                        tx.run(
                            """MATCH (n) WHERE (n:BB OR n:INSTR) AND n.session = $label
                            AND n.module_name IN $modules AND n.stage IN $stages DETACH DELETE n""",
                            label=label,
                            modules=replaced_modules,
                            stages=stages,
                        ).consume()
                        tx.run(
                            """MATCH (n) WHERE n.session = $staging
                            SET n.session = $label, n.compilation_id = $compilation""",
                            staging=staging,
                            label=label,
                            compilation=compilation,
                        ).consume()
                receipts = [
                    r
                    for r in receipts
                    if not (r["label"] == label and r["module_name"] in replaced_modules and int(r["stage"]) in stages)
                ] + new_receipts
                sess.add_artifact(
                    TableArtifact("cdfg_imports", pd.DataFrame(receipts), attrs={"kind": "provenance", "by": __name__}),
                    override=True,
                )
                sess.save()
            finally:
                with driver.session() as db:
                    db.run("MATCH (n) WHERE n.session = $staging DETACH DELETE n", staging=staging).consume()
    finally:
        driver.close()


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    generate_memgraph_cdfg_via_compile_commands(sess, label=args.label, stage=args.stage, force=args.force)
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
