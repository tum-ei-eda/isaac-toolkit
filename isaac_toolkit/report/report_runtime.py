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
from typing import Optional

import pandas as pd


from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts

from .report_utils import (
    save_pdf_report,
    JUPYTER_CSS,
    encode_image_base64,
    format_hex,
    monospace_addresses_html,
    monospace_addresses_md,
)


def generate_top_bb_table_html(df: pd.DataFrame, top_n: int = 10) -> str:
    """
    Generate a styled HTML table of the top-N basic blocks sorted by rel_weight.
    """

    def helper(x):
        if x is None:
            return "?"
        if isinstance(x, set):
            return list(x)[0]
        return x

    df["func_name"] = df["func_name"].apply(helper)

    df_sorted = df.sort_values("rel_weight", ascending=False).head(top_n).copy()

    # Format addresses as hex
    df_sorted["start"] = df_sorted["start"].apply(format_hex)
    df_sorted["end"] = df_sorted["end"].apply(format_hex)

    # Select useful columns
    df_display = df_sorted[["func_name", "start", "end", "freq", "size", "num_instrs", "weight", "rel_weight"]]

    def colormap_helper(s):
        # Normalize to 0-1
        norm = (s - s.min()) / (s.max() - s.min())
        # Compute RGB as a gradient from red to green
        colors = [f"rgb({int(255*(1-v))},{int(255*v)},0)" for v in norm]
        # print("colors", colors, len(colors))
        return colors

    # Apply Pandas Styler for a nicer HTML table
    styler = (
        df_display.style.set_caption(f"Top {top_n} Basic Blocks by Relative Weight")
        # .bar(subset=["rel_weight"], color="#4CAF50")
        # .bar(subset=["rel_weight"], color=colormap_helper)
        .bar(subset=["rel_weight"], cmap="coolwarm")
        .format(monospace_addresses_html, subset=["start", "end"])
        .format({"rel_weight": "{:.2%}", "weight": "{:,}", "freq": "{:,}", "size": "{:,}", "num_instrs": "{:,}"})
    )
    return styler.to_html()


def generate_top_bb_table_md(df: pd.DataFrame, top_n: int = 10) -> str:
    """
    Generate a Markdown table of the top-N basic blocks sorted by rel_weight.
    """

    def helper(x):
        if x is None:
            return "?"
        if isinstance(x, set):
            return list(x)[0]
        return x

    df["func_name"] = df["func_name"].apply(helper)

    df_sorted = df.sort_values("rel_weight", ascending=False).head(top_n).copy()

    # Format addresses as hex
    df_sorted["start"] = df_sorted["start"].apply(format_hex)
    df_sorted["end"] = df_sorted["end"].apply(format_hex)

    # Select useful columns
    df_display = df_sorted[["func_name", "start", "end", "freq", "size", "num_instrs", "weight", "rel_weight"]]

    # Format rel_weight as percentage
    df_display["rel_weight"] = df_display["rel_weight"].map(lambda x: f"{x:.2%}")
    df_display[["start", "end"]] = df_display[["start", "end"]].applymap(monospace_addresses_md)

    return df_display.to_markdown(index=False, tablefmt="github")


def collect_plot_files(sess: Session):
    plots_dir = sess.directory / "plots"
    known = [
        "runtime_per_func.jpg",
        "runtime_per_instr.jpg",
        "runtime_per_opcode.jpg",
        "runtime_per_library.jpg",
        "runtime_per_object.jpg",
        "runtime_per_llvm_bb.jpg",
    ]
    return {name: plots_dir / name for name in known if (plots_dir / name).is_file()}


def load_runtime_dfs(sess: Session):
    """Load runtime-related dataframes from artifacts."""
    artifacts = sess.artifacts

    pc2bb_df = None
    instrs_hist_df = None
    opcodes_hist_df = None

    pc2bb = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb")
    if pc2bb:
        pc2bb_df = pc2bb[0].df.copy()

    instrs_hist = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "instrs_hist")
    if instrs_hist:
        instrs_hist_df = instrs_hist[0].df.copy()

    opcodes_hist = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "opcodes_hist")
    if opcodes_hist:
        opcodes_hist_df = opcodes_hist[0].df.copy()

    return pc2bb_df, instrs_hist_df, opcodes_hist_df


def generate_runtime_summary(pc2bb_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate runtime per function."""
    if pc2bb_df is None:
        return pd.DataFrame()

    df = pc2bb_df.copy()

    def helper(x):
        if x is None:
            return "?"
        if isinstance(x, set):
            return list(x)[0]
        return x

    df["func_name"] = df["func_name"].apply(helper)
    df = df[["func_name", "rel_weight"]]
    df = df.groupby("func_name", as_index=False).sum()
    df = df.sort_values("rel_weight", ascending=False).reset_index(drop=True)
    return df


def render_report_text(summary_df, instrs_hist_df, opcodes_hist_df, plots, detailed=False) -> str:
    lines = []
    lines.append("# Runtime Report\n")

    if summary_df is not None and not summary_df.empty:
        lines.append("## Runtime per Function\n")
        for _, row in summary_df.iterrows():
            lines.append(f"- {row['func_name']}: {row['rel_weight']:.2%}")
        if "runtime_per_func.jpg" in plots:
            lines.append(f"![Runtime per Function]({plots['runtime_per_func.jpg']})")

    if detailed:
        if instrs_hist_df is not None and not instrs_hist_df.empty:
            lines.append("\n## Runtime per Instruction\n")
            for _, row in instrs_hist_df.sort_values("rel_count", ascending=False).iterrows():
                lines.append(f"- {row['instr']}: {row['rel_count']:.2%}")
            if "runtime_per_instr.jpg" in plots:
                lines.append(f"![Runtime per Instruction]({plots['runtime_per_instr.jpg']})")

        if opcodes_hist_df is not None and not opcodes_hist_df.empty:
            lines.append("\n## Runtime per Opcode\n")
            for _, row in opcodes_hist_df.sort_values("rel_count", ascending=False).iterrows():
                lines.append(f"- {row['opcode']}: {row['rel_count']:.2%}")
            if "runtime_per_opcode.jpg" in plots:
                lines.append(f"![Runtime per Opcode]({plots['runtime_per_opcode.jpg']})")

    return "\n".join(lines)


def render_report_html(
    summary_df, instrs_hist_df, opcodes_hist_df, plots, detailed=False, portable=False, style=False
) -> str:
    parts = []
    parts.append("<html><head>")
    if style:
        parts.append(JUPYTER_CSS)
    parts.append("</head><body>")
    parts.append("<h1>Runtime Report</h1>")

    def img_tag(path: Path, alt: str):
        if portable:
            b64 = encode_image_base64(path)
            return f'<img src="data:image/jpeg;base64,{b64}" alt="{alt}" style="max-width:600px;">'
        else:
            return f'<img src="{path}" alt="{alt}" style="max-width:600px;">'

    if summary_df is not None and not summary_df.empty:
        parts.append("<h2>Runtime per Function</h2>")
        parts.append(summary_df.to_html(index=False, float_format="%.4f", classes="dataframe"))
        if "runtime_per_func.jpg" in plots:
            parts.append(img_tag(plots["runtime_per_func.jpg"], "Runtime per Function"))

    if detailed:
        if instrs_hist_df is not None and not instrs_hist_df.empty:
            parts.append("<h2>Runtime per Instruction</h2>")
            parts.append(instrs_hist_df.to_html(index=False, float_format="%.4f", classes="dataframe"))
            if "runtime_per_instr.jpg" in plots:
                parts.append(img_tag(plots["runtime_per_instr.jpg"], "Runtime per Instruction"))

        if opcodes_hist_df is not None and not opcodes_hist_df.empty:
            parts.append("<h2>Runtime per Opcode</h2>")
            parts.append(opcodes_hist_df.to_html(index=False, float_format="%.4f", classes="dataframe"))
            if "runtime_per_opcode.jpg" in plots:
                parts.append(img_tag(plots["runtime_per_opcode.jpg"], "Runtime per Opcode"))

    parts.append("</body></html>")
    return "\n".join(parts)


def generate_runtime_report(
    sess, output=None, fmt="md", detailed=False, portable=False, style=False, topk=10, force=False
):
    pc2bb_df, instrs_hist_df, opcodes_hist_df = load_runtime_dfs(sess)
    summary_df = generate_runtime_summary(pc2bb_df)

    # Determine output dir
    out_dir = Path(output) if output else (sess.directory / "reports")
    out_dir.mkdir(exist_ok=True)
    plots = collect_plot_files(sess)

    if fmt in ("md", "txt"):
        body = render_report_text(summary_df, instrs_hist_df, opcodes_hist_df, plots, detailed=detailed)
        if pc2bb_df is not None:
            new = "\n\n## Runtime per Trace BB\n\n"
            new += generate_top_bb_table_md(pc2bb_df, top_n=topk)
            body += new
        ext = "md" if fmt == "md" else "txt"
        outfile = out_dir / f"runtime_report.{ext}"
        outfile.write_text(body, encoding="utf-8")
    elif fmt in ("html", "pdf"):
        body = render_report_html(
            summary_df, instrs_hist_df, opcodes_hist_df, plots, detailed=detailed, portable=portable, style=style
        )
        if pc2bb_df is not None:
            lines = body.splitlines()
            pre = "\n".join(lines[:-1])
            # print("pre", pre, len(pre))
            post = lines[-1]
            # print("post", post, len(post))
            new = "\n<h2>Runtime per Trace BB</h2>\n\n"
            new += generate_top_bb_table_html(pc2bb_df, top_n=topk)
            new = new.replace("<table id=", "<table border='1' class='dataframe dataframe' id=")
            body = pre + new + post
        ext = "html" if fmt == "html" else "pdf"
        outfile = out_dir / f"runtime_report.{ext}"
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
    generate_runtime_report(
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
    parser.add_argument("--topk", type=int, default=10, help="Limit number of table rows")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
