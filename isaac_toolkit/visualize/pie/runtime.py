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
import logging
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


logger = logging.getLogger(__name__)


# TODO: share with other pie scripts
def plot_pie_data(
    series, y, threshold: float = 0.1, title: str = "Pie Chart", legend: bool = True
):

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


# TODO: share with other pie scripts
def agg_library_runtime(
    runtime_df, symbol_map_df, by: str = "library", col: str = "rel_weight"
):
    ret = runtime_df.set_index("func_name").join(
        symbol_map_df.set_index("symbol"), how="left"
    )
    ret = ret[[by, col]]
    ret = ret.groupby(by, as_index=False, dropna=False).sum()
    return ret


def create_runtime_pie_plots(
    sess: Session,
    threshold: float = 0.05,
    topk: int = 9,
    fmt: str = "jpg",
    legend: bool = True,
    force: bool = False,
):
    artifacts = sess.artifacts
    # TODO: allow missing files!
    pc2bb_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb"
    )
    assert len(pc2bb_artifacts) == 1
    pc2bb_artifact = pc2bb_artifacts[0]
    pc2bb_df = pc2bb_artifact.df

    symbol_map_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "symbol_map"
    )
    symbol_map_df = None
    if len(symbol_map_artifacts) > 0:
        assert len(symbol_map_artifacts) == 1
        symbol_map_artifact = symbol_map_artifacts[0]
        symbol_map_df = symbol_map_artifact.df

    instrs_hist_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "instrs_hist"
    )
    instrs_hist_df = None
    if len(instrs_hist_artifacts) > 0:
        assert len(instrs_hist_artifacts) == 1
        instrs_hist_artifact = instrs_hist_artifacts[0]
        instrs_hist_df = instrs_hist_artifact.df

    opcodes_hist_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "opcodes_hist"
    )
    opcodes_hist_df = None
    if len(opcodes_hist_artifacts) > 0:
        assert len(opcodes_hist_artifacts) == 1
        opcodes_hist_artifact = opcodes_hist_artifacts[0]
        opcodes_hist_df = opcodes_hist_artifact.df

    llvm_bbs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs_new"
    )
    llvm_bbs_df = None
    if len(llvm_bbs_artifacts) > 0:
        assert len(llvm_bbs_artifacts) == 1
        llvm_bbs_artifact = llvm_bbs_artifacts[0]
        llvm_bbs_df = llvm_bbs_artifact.df.copy()

    plots_dir = sess.directory / "plots"
    plots_dir.mkdir(exist_ok=True)
    # TODO: use threshold

    if pc2bb_df is not None:
        runtime_df = pc2bb_df.copy()

        def helper(x):
            if x is None:
                return "?"
            if isinstance(x, set):
                # assert len(x) == 1
                assert len(x) > 0
                # Only pick first element if alias exists
                return list(x)[0]
            return x

        runtime_df["func_name"] = runtime_df["func_name"].apply(helper)
        runtime_df = runtime_df[["func_name", "rel_weight"]]
        runtime_df = runtime_df.groupby("func_name", as_index=False, dropna=False).sum()
        runtime_per_func_data = generate_pie_data(
            runtime_df, x="func_name", y="rel_weight", topk=topk
        )
        runtime_per_func_plot = plot_pie_data(
            runtime_per_func_data,
            "rel_weight",
            threshold=threshold,
            legend=legend,
            title="Runtime per Func",
        )
        runtime_per_func_plot_file = plots_dir / f"runtime_per_func.{fmt}"
        if runtime_per_func_plot_file.is_file():
            assert force, f"File already exists: {runtime_per_func_plot_file}"
        runtime_per_func_plot.get_figure().savefig(
            runtime_per_func_plot_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        if llvm_bbs_df is not None:
            llvm_bbs_df["func_bb"] = (
                llvm_bbs_df["func_name"] + "-" + llvm_bbs_df["bb_name"]
            )
            runtime_per_llvm_bb_data = generate_pie_data(
                llvm_bbs_df, x="func_bb", y="rel_weight", topk=topk
            )
            runtime_per_llvm_bb_plot = plot_pie_data(
                runtime_per_llvm_bb_data,
                "rel_weight",
                threshold=threshold,
                legend=legend,
                title="Runtime per LLVM BB",
            )
            runtime_per_llvm_bb_plot_file = plots_dir / f"runtime_per_llvm_bb.{fmt}"
            if runtime_per_llvm_bb_plot_file.is_file():
                assert force, f"File already exists: {runtime_per_llvm_bb_plot_file}"
            runtime_per_llvm_bb_plot.get_figure().savefig(
                runtime_per_llvm_bb_plot_file,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
        if symbol_map_df is not None:
            # library
            library_runtime_df = agg_library_runtime(
                runtime_df, symbol_map_df, by="library", col="rel_weight"
            )
            runtime_per_library_data = generate_pie_data(
                library_runtime_df, x="library", y="rel_weight", topk=topk
            )
            runtime_per_library_plot = plot_pie_data(
                runtime_per_library_data,
                "rel_weight",
                threshold=threshold,
                legend=legend,
                title="Runtime per Library",
            )
            runtime_per_library_plot_file = plots_dir / f"runtime_per_library.{fmt}"
            if runtime_per_library_plot_file.is_file():
                assert force, f"File already exists: {runtime_per_library_plot_file}"
            runtime_per_library_plot.get_figure().savefig(
                runtime_per_library_plot_file,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
            # object
            object_runtime_df = agg_library_runtime(
                runtime_df, symbol_map_df, by="object", col="rel_weight"
            )
            runtime_per_object_data = generate_pie_data(
                object_runtime_df, x="object", y="rel_weight", topk=topk
            )
            runtime_per_object_plot = plot_pie_data(
                runtime_per_object_data,
                "rel_weight",
                threshold=threshold,
                legend=legend,
                title="Runtime per Object",
            )
            runtime_per_object_plot_file = plots_dir / f"runtime_per_object.{fmt}"
            if runtime_per_object_plot_file.is_file():
                assert force, f"File already exists: {runtime_per_object_plot_file}"
            runtime_per_object_plot.get_figure().savefig(
                runtime_per_object_plot_file,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
    if instrs_hist_df is not None:
        instrs_hist_data = generate_pie_data(
            instrs_hist_df, x="instr", y="rel_count", topk=topk
        )
        instrs_hist_plot = plot_pie_data(
            instrs_hist_data,
            "rel_count",
            threshold=threshold,
            legend=legend,
            title="Runtime per Instr",
        )
        instrs_hist_plot_file = plots_dir / f"runtime_per_instr.{fmt}"
        if instrs_hist_plot_file.is_file():
            assert force, f"File already exists: {instrs_hist_plot_file}"
        instrs_hist_plot.get_figure().savefig(
            instrs_hist_plot_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    if opcodes_hist_df is not None:
        opcodes_hist_data = generate_pie_data(
            opcodes_hist_df, x="opcode", y="rel_count", topk=topk
        )
        opcodes_hist_plot = plot_pie_data(
            opcodes_hist_data,
            "rel_count",
            threshold=threshold,
            legend=legend,
            title="Runtime per Opcode",
        )
        opcodes_hist_plot_file = plots_dir / f"runtime_per_opcode.{fmt}"
        if opcodes_hist_plot_file.is_file():
            assert force, f"File already exists: {opcodes_hist_plot_file}"
        opcodes_hist_plot.get_figure().savefig(
            opcodes_hist_plot_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    # attrs = {
    #     # "elf_file": elf_artifact.name,
    #     "kind": "plot",
    #     "by": __name__,
    # }

    # artifact = TableArtifact("choices", choices_df, attrs=attrs)
    # sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    create_runtime_pie_plots(
        sess,
        threshold=args.threshold,
        topk=args.topk,
        legend=args.legend,
        force=args.force,
        fmt=args.fmt,
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
