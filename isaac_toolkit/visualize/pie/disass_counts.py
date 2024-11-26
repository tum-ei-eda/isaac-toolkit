import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


logger = logging.getLogger(__name__)


# TODO: share with other pie scripts
def plot_pie_data(series, y, threshold: float = 0.1, title: str = "Pie Chart", legend: bool = True):

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{p:.2f}%\n({v:d})".format(p=pct, v=val)

        return my_autopct

    plot = series.plot.pie(
        y=y,
        autopct="%1.1f%%",
        # autopct=make_autopct(series[y].values),
        # legend=legend,
        title=title,
        labeldistance=1 if not legend else None,
    )
    if legend:
        # Matplotlibs hides legend labels starting with an '_'...
        labels = [f" {x}" if str(x).startswith("_") else x for x in series.index]
        plt.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return plot


# TODO: share with other pie scripts
def generate_pie_data(df, x: str, y: str, topk: int = 9):
    ret = df.copy()
    ret.set_index(x, inplace=True)
    ret.sort_values(y, inplace=True, ascending=False)
    a = ret.iloc[:topk]
    b = ret.iloc[topk:].agg(others=(y, "sum"))

    ret = pd.concat([a, b])
    ret = ret[y]

    return ret


def create_disass_counts_pie_plots(
    sess: Session, threshold: float = 0.05, topk: int = 9, fmt: str = "jpg", legend: bool = True, force: bool = False
):
    artifacts = sess.artifacts
    # TODO: allow missing files!
    disass_instrs_hist_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "disass_instrs_hist"
    )
    disass_instrs_hist_df = None
    if len(disass_instrs_hist_artifacts) > 0:
        assert len(disass_instrs_hist_artifacts) == 1
        disass_instrs_hist_artifact = disass_instrs_hist_artifacts[0]
        disass_instrs_hist_df = disass_instrs_hist_artifact.df

    disass_opcodes_hist_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "disass_opcodes_hist"
    )
    disass_opcodes_hist_df = None
    if len(disass_opcodes_hist_artifacts) > 0:
        assert len(disass_opcodes_hist_artifacts) == 1
        disass_opcodes_hist_artifact = disass_opcodes_hist_artifacts[0]
        disass_opcodes_hist_df = disass_opcodes_hist_artifact.df

    plots_dir = sess.directory / "plots"
    plots_dir.mkdir(exist_ok=True)
    # TODO: use threshold

    if disass_instrs_hist_df is not None:
        pie_data = generate_pie_data(disass_instrs_hist_df, x="instr", y="rel_count", topk=topk)
        pie_plot = plot_pie_data(
            pie_data, "rel_count", threshold=threshold, legend=legend, title="Counts (static) per Instruction"
        )
        pie_plot_file = plots_dir / f"disass_counts_per_instr.{fmt}"
        if pie_plot_file.is_file():
            assert force, f"File already exists: {pie_plot_file}"
        pie_plot.get_figure().savefig(
            pie_plot_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    if disass_opcodes_hist_df is not None:
        pie_data = generate_pie_data(disass_opcodes_hist_df, x="opcode", y="rel_count", topk=topk)
        pie_plot = plot_pie_data(
            pie_data, "rel_count", threshold=threshold, legend=legend, title="Instruction Counts (static) per Opcode"
        )
        pie_plot_file = plots_dir / f"disass_counts_per_opcode.{fmt}"
        if pie_plot_file.is_file():
            assert force, f"File already exists: {pie_plot_file}"
        pie_plot.get_figure().savefig(
            pie_plot_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    create_disass_counts_pie_plots(
        sess, threshold=args.threshold, topk=args.topk, legend=args.legend, force=args.force, fmt=args.fmt
    )
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--fmt", type=str, choices=["jpg", "png", "pdf"], default="jpg")
    parser.add_argument("--topk", type=int, default=9)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
