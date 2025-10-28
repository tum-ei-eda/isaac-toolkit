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

from .report_utils import (
    save_pdf_report,
    JUPYTER_CSS,
)


def get_directory_size(path):
    return sum(file.stat().st_size for file in Path(path).rglob("*"))


def get_file_size(path):
    return path.stat().st_size


def format_size(size: float, raw: bool = False):
    if raw:
        return str(size)  # No suffix?
    natural_size = humanize.naturalsize(size)
    return natural_size


def size_df_to_markdown(df: pd.DataFrame, cols: List[str] = ["Size"], raw: bool = False) -> str:
    df_temp = df.copy()
    for column in df_temp.columns:
        if "size" in column.lower():
            df_temp[column] = df_temp[column].apply(lambda x: format_size(x, raw=raw))

    return df_temp.to_markdown(index=False, tablefmt="github")


def size_df_to_html(
    df: pd.DataFrame, cols: List[str] = ["Size"], raw: bool = False, title: Optional[str] = None
) -> str:
    df_display = df.copy()

    def colormap_helper(s):
        # Normalize to 0-1
        norm = (s - s.min()) / (s.max() - s.min())
        # Compute RGB as a gradient from red to green
        colors = [f"rgb({int(255*(1-v))},{int(255*v)},0)" for v in norm]
        # print("colors", colors, len(colors))
        return colors

    # Apply Pandas Styler for a nicer HTML table
    styler = df_display.style
    if title is not None:
        styler = styler.set_caption(title)
    if len(cols) > 0:
        styler = styler.bar(subset=cols, cmap="coolwarm")
        styler = styler.format({col: lambda x: format_size(x, raw=raw) for col in cols})
    return styler.to_html()


def generate_sess_disk_usage(
    sess, output=None, fmt="md", detailed=False, portable=False, style=False, topk=10, force=False, raw=False
):
    # Determine output dir
    out_dir = Path(output) if output else (sess.directory / "reports")
    out_dir.mkdir(exist_ok=True)

    sess_dir = sess.directory
    sess_dir_size = get_directory_size(sess_dir)
    sess_dir_size_str = format_size(sess_dir_size, raw=raw)
    artifacts = sess.artifacts
    artifacts_data = []
    for artifact in artifacts:
        artifact_name = artifact.name
        artifact_kind = str(type(artifact).__name__)
        artifact_size = get_file_size(artifact.path)
        artifacts_data.append((artifact_name, artifact_kind, artifact_size))
    artifacts_df = pd.DataFrame(artifacts_data, columns=["Artifact", "Kind", "Size"])
    total_artifacts_size = artifacts_df["Size"].sum()
    total_artifacts_size_str = format_size(total_artifacts_size, raw=raw)
    non_artifacts_size = sess_dir_size - total_artifacts_size
    non_artifacts_size_str = format_size(non_artifacts_size, raw=raw)
    # print("Artifacts DF", artifacts_df)
    artifacts_df = artifacts_df.sort_values("Size", ascending=False)
    if topk is not None:
        artifacts_df = artifacts_df.iloc[:topk]

    print("Sorted Artifacts DF", artifacts_df)

    if fmt in ("md", "txt"):
        content = "# Session Disk Usage Report\n\n"
        content = "## Overview\n\n"
        content += f"Session directory size: {sess_dir_size_str}\n\n"
        content += f"Non Artifacts Size: {non_artifacts_size_str}\n\n"
        content += f"Total Artifacts Size: {total_artifacts_size_str}\n\n"
        content += "## Artifacts\n\n"
        if topk is not None:
            content += f"TOPK={topk}\n\n"
        content += size_df_to_markdown(artifacts_df, cols=["Size"], raw=raw)
        content += "\n"
        body = content
        ext = "md" if fmt == "md" else "txt"
        outfile = out_dir / f"sess_disk_usage_report.{ext}"
        outfile.write_text(body, encoding="utf-8")
    elif fmt in ("html", "pdf"):
        parts = []
        parts.append("<html><head>")
        if style:
            parts.append(JUPYTER_CSS)
        parts.append("</head><body>")
        parts.append("<h1>Session Disk Usage Report</h1>")
        parts.append("<h2>Overview</h2>")
        parts.append(f"<p>Session directory size: {sess_dir_size_str}</p>")
        parts.append(f"<p>Non Artifacts Size: {non_artifacts_size_str}</p>")
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
        outfile = out_dir / f"sess_disk_usage_report.{ext}"
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
    generate_sess_disk_usage(
        sess,
        output=args.out,
        fmt=args.fmt,
        detailed=args.detailed,
        portable=args.portable,
        style=args.style,
        topk=args.topk,
        raw=args.raw,
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
