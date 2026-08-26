#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Generate code-size reports from static-analysis artifacts."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts

from .report_sess_disk_usage import size_df_to_html, size_df_to_markdown
from .report_utils import JUPYTER_CSS, save_pdf_report

SECTION_GROUPS = {
    "Code": (".text", ".init", ".fini"),
    "Read-only data": (".rodata", ".srodata"),
    "Initialized data": (".data", ".sdata"),
    "Zero-initialized data": (".bss", ".sbss"),
}


def _load_table(sess: Session, name: str):
    artifacts = filter_artifacts(
        sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.TABLE and artifact.name == name
    )
    return artifacts[0].df.copy() if len(artifacts) == 1 else None


def load_code_size_dfs(sess: Session):
    return _load_table(sess, "mem_sections"), _load_table(sess, "mem_footprint")


def classify_section(name: str) -> str:
    for category, prefixes in SECTION_GROUPS.items():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            return category
    return "Other"


def generate_code_size_summary(sections: pd.DataFrame, functions: pd.DataFrame) -> pd.DataFrame:
    values = []
    totals = {category: 0 for category in SECTION_GROUPS}
    if sections is not None and not sections.empty:
        categorized = sections.assign(Category=sections["name"].fillna("").map(classify_section))
        grouped = categorized.groupby("Category", observed=True)["data_size"].sum()
        totals.update({category: int(grouped.get(category, 0)) for category in totals})
        values.extend((category, size) for category, size in totals.items())
        values.extend(
            (
                ("Estimated ROM", totals["Code"] + totals["Read-only data"] + totals["Initialized data"]),
                ("Estimated RAM", totals["Initialized data"] + totals["Zero-initialized data"]),
            )
        )
    if functions is not None and not functions.empty:
        values.append(("Function symbols", int(functions["bytes"].sum())))
    return pd.DataFrame(values, columns=["Metric", "Bytes"])


def generate_section_table(sections: pd.DataFrame, topk: int = 10) -> pd.DataFrame:
    if sections is None or sections.empty:
        return pd.DataFrame(columns=["Section", "Category", "Bytes", "Share"])
    result = sections.rename(columns={"name": "Section", "data_size": "Bytes"}).copy()
    result = result[result["Bytes"] > 0]
    result["Category"] = result["Section"].map(classify_section)
    total = result["Bytes"].sum()
    result["Share"] = result["Bytes"] / total if total else 0.0
    return result.sort_values("Bytes", ascending=False).head(topk).reset_index(drop=True)


def generate_function_table(functions: pd.DataFrame, topk: int = 10) -> pd.DataFrame:
    if functions is None or functions.empty:
        return pd.DataFrame(columns=["Function", "Bytes", "Share"])
    result = functions.rename(columns={"func": "Function", "bytes": "Bytes", "rel_bytes": "Share"}).copy()
    if "Share" not in result:
        total = result["Bytes"].sum()
        result["Share"] = result["Bytes"] / total if total else 0.0
    return result[["Function", "Bytes", "Share"]].sort_values("Bytes", ascending=False).head(topk).reset_index(drop=True)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    display = df.copy()
    if "Share" in display:
        display["Share"] = display["Share"].map(lambda value: f"{value:.2%}")
    return size_df_to_markdown(display, cols=[column for column in display if column == "Bytes"])


def _html_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return "<p><em>No data available.</em></p>"
    display = df.copy()
    if "Share" in display:
        display["Share"] = display["Share"].map(lambda value: f"{value:.2%}")
    table = size_df_to_html(display, cols=[column for column in display if column == "Bytes"], title=title)
    return table.replace("<table id=", "<table border='1' class='dataframe dataframe' id=")


def generate_mem_footprint_report(sess, output=None, fmt="md", detailed=False, style=False, topk=10, force=False):
    del force
    sections, functions = load_code_size_dfs(sess)
    if sections is None and functions is None:
        return None
    summary = generate_code_size_summary(sections, functions)
    section_table = generate_section_table(sections, topk=topk)
    function_table = generate_function_table(functions, topk=topk)
    out_dir = Path(output) if output else sess.directory / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt in ("md", "txt"):
        parts = ["# Code Size Report", "## Summary", _markdown_table(summary), "## Largest Functions", _markdown_table(function_table)]
        if detailed:
            parts.extend(("## Largest ELF Sections", _markdown_table(section_table)))
        body = "\n\n".join(parts) + "\n"
        outfile = out_dir / f"mem_footprint_report.{fmt if fmt == 'txt' else 'md'}"
        outfile.write_text(body, encoding="utf-8")
    elif fmt in ("html", "pdf"):
        parts = ["<html><head>", JUPYTER_CSS if style else "", "</head><body>", "<h1>Code Size Report</h1>", "<h2>Summary</h2>", _html_table(summary, "Code-size summary"), "<h2>Largest Functions</h2>", _html_table(function_table, "Largest functions")]
        if detailed:
            parts.extend(("<h2>Largest ELF Sections</h2>", _html_table(section_table, "Largest ELF sections")))
        parts.append("</body></html>")
        body = "\n".join(parts)
        outfile = out_dir / f"mem_footprint_report.{fmt}"
        if fmt == "html":
            outfile.write_text(body, encoding="utf-8")
        else:
            save_pdf_report(body, outfile)
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")
    print(f"[isaac_toolkit.report] Code-size report written to {outfile}")
    return outfile


def get_parser():
    parser = argparse.ArgumentParser(description="Generate a code-size report from an ISAAC session.")
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--out", help="Custom output directory (default: SESSION/reports)")
    parser.add_argument("--fmt", choices=["md", "txt", "html", "pdf"], default="md")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--style", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    return parser


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    session_dir = Path(args.session)
    if not session_dir.is_dir():
        parser.error(f"session directory does not exist: {session_dir}")
    sess = Session.from_dir(session_dir)
    generate_mem_footprint_report(sess, args.out, args.fmt, args.detailed, args.style, args.topk, args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
