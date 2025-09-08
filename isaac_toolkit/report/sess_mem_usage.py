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
import argparse
from pathlib import Path
from typing import Optional, List

import humanize
import pandas as pd


from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts, TableArtifact, FileArtifact

from .report_utils import (
    save_pdf_report,
    JUPYTER_CSS,
)
from .sess_disk_usage import size_df_to_markdown, size_df_to_html, get_file_size


def get_artifact_size(artifact):
    print("get_artifact_size", artifact, type(artifact).__name__)
    if isinstance(artifact, FileArtifact):
        path = artifact.path
        mem = get_file_size(path)
        return mem
    if isinstance(artifact, TableArtifact):
        df = artifact.df
        print("df", df)
        mem = df.memory_usage(deep=True)
        print("mem", mem)
        total_mem = mem.sum() if not isinstance(mem, int) else mem
        print("total_mem", total_mem)
        return total_mem
    raise NotImplementedError(f"Unhandled type: {type(artifact).__name__}")


def format_size(size: float, raw: bool = False):
    if raw:
        return str(size)  # No suffix?
    natural_size = humanize.naturalsize(size)
    return natural_size


def generate_sess_mem_usage(
    sess, output=None, fmt="md", detailed=False, portable=False, style=False, topk=10, force=False, raw=False
):
    # Determine output dir
    out_dir = Path(output) if output else (sess.directory / "reports")
    out_dir.mkdir(exist_ok=True)

    artifacts = sess.artifacts
    artifacts_data = []
    for artifact in artifacts:
        artifact_name = artifact.name
        artifact_kind = str(type(artifact).__name__)
        artifact_size = get_artifact_size(artifact)
        artifacts_data.append((artifact_name, artifact_kind, artifact_size))
    artifacts_df = pd.DataFrame(artifacts_data, columns=["Artifact", "Kind", "Size"])
    total_artifacts_size = artifacts_df["Size"].sum()
    total_artifacts_size_str = format_size(total_artifacts_size, raw=raw)
    artifacts_df = artifacts_df.sort_values("Size", ascending=False)
    if topk is not None:
        artifacts_df = artifacts_df.iloc[:topk]

    print("Sorted Artifacts DF", artifacts_df)

    if fmt in ("md", "txt"):
        content = "# Session Memory Usage Report\n\n"
        content = "## Overview\n\n"
        content += f"Total Artifacts Mem: {total_artifacts_size_str}\n\n"
        content += "## Artifacts\n\n"
        if topk is not None:
            content += f"TOPK={topk}\n\n"
        content += size_df_to_markdown(artifacts_df, cols=["Size"], raw=raw)
        content += "\n"
        body = content
        ext = "md" if fmt == "md" else "txt"
        outfile = out_dir / f"sess_mem_usage_report.{ext}"
        outfile.write_text(body, encoding="utf-8")
    elif fmt in ("html", "pdf"):
        parts = []
        parts.append("<html><head>")
        if style:
            parts.append(JUPYTER_CSS)
        parts.append("</head><body>")
        parts.append("<h1>Session Memory Usage Report</h1>")
        parts.append("<h2>Overview</h2>")
        parts.append(f"<p>Total Artifacts Size: {total_artifacts_size_str}</p>")
        parts.append("<h2>Artifacts</h2>")
        if topk is not None:
            parts.append(f"TOPK={topk}")
        table_content = size_df_to_html(artifacts_df, cols=["Size"], raw=raw)
        table_content = table_content.replace("<table id=", "<table border='1' class='dataframe dataframe' id=")
        parts.append(table_content)
        parts.append("</body></html>")
        body = "\n".join(parts)
        ext = "html" if fmt == "html" else "pdf"
        outfile = out_dir / f"sess_mem_usage_report.{ext}"
        if fmt == "html":
            outfile.write_text(body, encoding="utf-8")
        else:
            save_pdf_report(body, outfile)
    else:
        assert False, f"Unsupported fmt: {fmt}"

    print(f"[isaac_toolkit.report] Runtime report written to {outfile}")  # TODO: logging!


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_sess_mem_usage(
        sess,
        output=args.out,
        fmt=args.fmt,
        detailed=args.detailed,
        portable=args.portable,
        style=args.style,
        topk=args.topk,
        force=args.force,
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser(description="Generate runtime report from ISAAC session.")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--out", help="Custom output directory (default: SESSION/reports)")
    parser.add_argument("--fmt", choices=["md", "txt", "html", "pdf"], default="md", help="Report format")
    parser.add_argument("--detailed", action="store_true", help="Include detailed breakdowns")
    parser.add_argument("--portable", action="store_true", help="Embed plots as base64 (only for HTML output)")
    parser.add_argument("--style", action="store_true", help="Use custom CSS for HTML output")
    parser.add_argument("--raw", action="store_true", help="Output raw sizes instead of humanized ones")
    parser.add_argument("--topk", type=int, default=10, help="Limit number of table rows")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
