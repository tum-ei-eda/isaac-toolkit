#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Generate reports from dynamic pipeline/basic-block cost artifacts."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts

from .report_utils import JUPYTER_CSS, save_pdf_report


def _load_table(sess: Session, name: str):
    artifacts = filter_artifacts(
        sess.artifacts, lambda artifact: artifact.flags & ArtifactFlag.TABLE and artifact.name == name
    )
    return artifacts[0].df.copy() if len(artifacts) == 1 else None


def load_perf_dfs(sess: Session):
    """Load the pipeline metric artifacts used by the performance report."""
    return (
        _load_table(sess, "bb_cost"),
        _load_table(sess, "bb_cost_stats"),
        _load_table(sess, "bb_cost_distribution"),
        _load_table(sess, "unique_bbs"),
    )


def generate_perf_summary(costs: pd.DataFrame) -> pd.DataFrame:
    """Return whole-trace totals and ratios for available pipeline metrics."""
    if costs is None or costs.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    values = []
    ir = costs["Ir"].sum() if "Ir" in costs else None
    cycles = costs["Cycles"].sum() if "Cycles" in costs else None
    for metric in ("Ir", "Cycles", "StallCycles", "BranchMispredicts", "L1IMisses", "L1DMisses"):
        if metric in costs:
            values.append((metric, costs[metric].sum()))
    if ir:
        if cycles is not None:
            values.append(("CPI", cycles / ir))
        if "Latency" in costs:
            values.append(("AverageLatency", (costs["Latency"] * costs["Ir"]).sum() / ir))
    return pd.DataFrame(values, columns=["Metric", "Value"])


def generate_top_bb_perf(stats: pd.DataFrame, unique_bbs: pd.DataFrame, topk: int = 10) -> pd.DataFrame:
    """Rank static BBs by their total dynamic cycle contribution."""
    if stats is None or unique_bbs is None or stats.empty:
        return pd.DataFrame()

    bbs = unique_bbs.reset_index(names="bb_idx")
    columns = [column for column in ("bb_idx", "first_pc", "last_pc", "func", "num_instrs") if column in bbs]
    result = stats.merge(bbs[columns], on="bb_idx", how="left")
    result["TotalCycles"] = result["Cycles_mean"] * result["invocations"]
    total_cycles = result["TotalCycles"].sum()
    result["CycleShare"] = result["TotalCycles"] / total_cycles if total_cycles else 0.0
    result.sort_values("TotalCycles", ascending=False, inplace=True)
    display = [
        column
        for column in (
            "bb_idx",
            "func",
            "first_pc",
            "last_pc",
            "num_instrs",
            "invocations",
            "TotalCycles",
            "CycleShare",
            "Cycles_mean",
            "CPI_mean",
            "StallCycles_mean",
            "Latency_mean",
            "BranchMispredicts_mean",
            "L1IMisses_mean",
            "L1DMisses_mean",
            "cost_patterns",
        )
        if column in result
    ]
    return result[display].head(topk).reset_index(drop=True)


def _format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ("first_pc", "last_pc"):
        if column in result:
            result[column] = result[column].map(lambda value: hex(int(value)) if pd.notna(value) else "?")
    return result


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return _format_for_display(df).to_markdown(index=False, tablefmt="github", floatfmt=".4f")


def _html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>No data available.</em></p>"
    return _format_for_display(df).to_html(index=False, float_format=lambda value: f"{value:.4f}", classes="dataframe")


def generate_perf_report(sess, output=None, fmt="md", detailed=False, style=False, topk=10, force=False):
    """Write a pipeline performance report, or return ``None`` if no costs exist."""
    del force  # Reports are regular output files and are always regenerated.
    costs, stats, distribution, unique_bbs = load_perf_dfs(sess)
    if costs is None:
        return None

    summary = generate_perf_summary(costs)
    top_bbs = generate_top_bb_perf(stats, unique_bbs, topk=topk)
    patterns = pd.DataFrame()
    if detailed and distribution is not None:
        patterns = distribution.sort_values(["occurrences", "probability"], ascending=False).head(topk)

    out_dir = Path(output) if output else sess.directory / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("md", "txt"):
        parts = [
            "# Pipeline Performance Report",
            "## Summary",
            _markdown_table(summary),
            "## Top Basic Blocks",
            _markdown_table(top_bbs),
        ]
        if detailed:
            parts.extend(("## Most Frequent BB Cost Patterns", _markdown_table(patterns)))
        body = "\n\n".join(parts) + "\n"
        ext = "md" if fmt == "md" else "txt"
        outfile = out_dir / f"perf_report.{ext}"
        outfile.write_text(body, encoding="utf-8")
    elif fmt in ("html", "pdf"):
        parts = [
            "<html><head>",
            JUPYTER_CSS if style else "",
            "</head><body>",
            "<h1>Pipeline Performance Report</h1>",
            "<h2>Summary</h2>",
            _html_table(summary),
            "<h2>Top Basic Blocks</h2>",
            _html_table(top_bbs),
        ]
        if detailed:
            parts.extend(("<h2>Most Frequent BB Cost Patterns</h2>", _html_table(patterns)))
        parts.append("</body></html>")
        body = "\n".join(parts)
        outfile = out_dir / f"perf_report.{fmt}"
        if fmt == "html":
            outfile.write_text(body, encoding="utf-8")
        else:
            save_pdf_report(body, outfile)
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")

    print(f"[isaac_toolkit.report] Performance report written to {outfile}")
    return outfile


def get_parser():
    parser = argparse.ArgumentParser(description="Generate a pipeline performance report from an ISAAC session.")
    parser.add_argument("--session", "--sess", "-s", required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--out", help="Custom output directory (default: SESSION/reports)")
    parser.add_argument("--fmt", choices=["md", "txt", "html", "pdf"], default="md")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--style", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    session_dir = Path(args.session)
    if not session_dir.is_dir():
        get_parser().error(f"session directory does not exist: {session_dir}")
    sess = Session.from_dir(session_dir)
    generate_perf_report(
        sess,
        output=args.out,
        fmt=args.fmt,
        detailed=args.detailed,
        style=args.style,
        topk=args.topk,
        force=args.force,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
