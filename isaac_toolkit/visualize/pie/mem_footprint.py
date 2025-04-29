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

    # series.fillna("?", inplace=True)
    # print("series", series.head())
    # input()

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


def generate_pie_data(df, x: str, y: str, topk: int = 9):
    ret = df.copy()
    ret.set_index(x, inplace=True)
    ret.sort_values(y, inplace=True, ascending=False)
    a = ret.iloc[:topk]
    b = ret.iloc[topk:].agg(others=(y, "sum"))

    ret = pd.concat([a, b])
    ret = ret[y]

    return ret


def agg_library_footprint(mem_footprint_df, symbol_map_df, by: str = "library", col: str = "rel_bytes"):
    # ret = mem_footprint_df.copy()
    ret = mem_footprint_df.set_index("func").join(symbol_map_df.set_index("symbol"), how="left")
    ret = ret[[by, col]]
    ret = ret.groupby(by, as_index=False, dropna=False).sum()
    return ret


def create_mem_footprint_pie_plots(
    sess: Session,
    threshold: float = 0.05,
    topk: int = 9,
    fmt: str = "jpg",
    legend: bool = True,
    force: bool = False,
):
    artifacts = sess.artifacts
    # TODO: allow missing files!
    mem_footprint_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "mem_footprint"
    )
    assert len(mem_footprint_artifacts) == 1
    mem_footprint_artifact = mem_footprint_artifacts[0]
    mem_footprint_df = mem_footprint_artifact.df

    effective_mem_footprint_artifacts = filter_artifacts(
        artifacts,
        lambda x: x.flags & ArtifactFlag.TABLE and x.name == "effective_mem_footprint",
    )
    effective_mem_footprint_df = None
    if len(effective_mem_footprint_artifacts) > 0:
        assert len(effective_mem_footprint_artifacts) == 1
        effective_mem_footprint_artifact = effective_mem_footprint_artifacts[0]
        effective_mem_footprint_df = effective_mem_footprint_artifact.df

    symbol_map_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "symbol_map"
    )
    symbol_map_df = None
    if len(symbol_map_artifacts) > 0:
        assert len(symbol_map_artifacts) == 1
        symbol_map_artifact = symbol_map_artifacts[0]
        symbol_map_df = symbol_map_artifact.df

    plots_dir = sess.directory / "plots"
    plots_dir.mkdir(exist_ok=True)
    # TODO: use threshold

    if mem_footprint_df is not None:
        mem_footprint_per_func_data = generate_pie_data(mem_footprint_df, x="func", y="rel_bytes", topk=topk)
        mem_footprint_per_func_plot = plot_pie_data(
            mem_footprint_per_func_data,
            "rel_bytes",
            threshold=threshold,
            legend=legend,
            title="Memory Footprint per Func",
        )
        mem_footprint_per_func_plot_file = plots_dir / f"mem_footprint_per_func.{fmt}"
        if mem_footprint_per_func_plot_file.is_file():
            assert force, f"File already exists: {mem_footprint_per_func_plot_file}"
        mem_footprint_per_func_plot.get_figure().savefig(
            mem_footprint_per_func_plot_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        # input(">")
        if symbol_map_df is not None:
            # library
            library_footprint_df = agg_library_footprint(mem_footprint_df, symbol_map_df, by="library", col="rel_bytes")
            mem_footprint_per_library_data = generate_pie_data(
                library_footprint_df, x="library", y="rel_bytes", topk=topk
            )
            mem_footprint_per_library_plot = plot_pie_data(
                mem_footprint_per_library_data,
                "rel_bytes",
                threshold=threshold,
                legend=legend,
                title="Memory Footprint per Library",
            )
            mem_footprint_per_library_plot_file = plots_dir / f"mem_footprint_per_library.{fmt}"
            if mem_footprint_per_library_plot_file.is_file():
                assert force, f"File already exists: {mem_footprint_per_library_plot_file}"
            mem_footprint_per_library_plot.get_figure().savefig(
                mem_footprint_per_library_plot_file,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
            # object
            object_footprint_df = agg_library_footprint(mem_footprint_df, symbol_map_df, by="object", col="rel_bytes")
            mem_footprint_per_object_data = generate_pie_data(object_footprint_df, x="object", y="rel_bytes", topk=topk)
            mem_footprint_per_object_plot = plot_pie_data(
                mem_footprint_per_object_data,
                "rel_bytes",
                threshold=threshold,
                legend=legend,
                title="Memory Footprint per Object",
            )
            mem_footprint_per_object_plot_file = plots_dir / f"mem_footprint_per_object.{fmt}"
            if mem_footprint_per_object_plot_file.is_file():
                assert force, f"File already exists: {mem_footprint_per_object_plot_file}"
            mem_footprint_per_object_plot.get_figure().savefig(
                mem_footprint_per_object_plot_file, bbox_inches="tight", dpi=300
            )
            plt.close()
    if effective_mem_footprint_df is not None:
        effective_mem_footprint_per_func_data = generate_pie_data(
            effective_mem_footprint_df, x="func", y="eff_rel_bytes", topk=topk
        )
        effective_mem_footprint_per_func_plot = plot_pie_data(
            effective_mem_footprint_per_func_data,
            "eff_rel_bytes",
            threshold=threshold,
            legend=legend,
            title="Eff. Memory Footprint per Func",
        )
        effective_mem_footprint_per_func_plot_file = plots_dir / f"effective_mem_footprint_per_func.{fmt}"
        if effective_mem_footprint_per_func_plot_file.is_file():
            assert force, f"File already exists: {effective_mem_footprint_per_func_plot_file}"
        effective_mem_footprint_per_func_plot.get_figure().savefig(
            effective_mem_footprint_per_func_plot_file,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        if symbol_map_df is not None:
            # library
            library_footprint_df = agg_library_footprint(
                effective_mem_footprint_df,
                symbol_map_df,
                by="library",
                col="eff_rel_bytes",
            )
            effective_mem_footprint_per_library_data = generate_pie_data(
                library_footprint_df, x="library", y="eff_rel_bytes", topk=topk
            )
            effective_mem_footprint_per_library_plot = plot_pie_data(
                effective_mem_footprint_per_library_data,
                "eff_rel_bytes",
                threshold=threshold,
                legend=legend,
                title="Eff. Memory Footprint per Library",
            )
            effective_mem_footprint_per_library_plot_file = plots_dir / f"effective_mem_footprint_per_library.{fmt}"
            if effective_mem_footprint_per_library_plot_file.is_file():
                assert force, f"File already exists: {effective_mem_footprint_per_library_plot_file}"
            effective_mem_footprint_per_library_plot.get_figure().savefig(
                effective_mem_footprint_per_library_plot_file,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
            # object
            object_footprint_df = agg_library_footprint(
                effective_mem_footprint_df,
                symbol_map_df,
                by="object",
                col="eff_rel_bytes",
            )
            effective_mem_footprint_per_object_data = generate_pie_data(
                object_footprint_df, x="object", y="eff_rel_bytes", topk=topk
            )
            effective_mem_footprint_per_object_plot = plot_pie_data(
                effective_mem_footprint_per_object_data,
                "eff_rel_bytes",
                threshold=threshold,
                legend=legend,
                title="Eff. Memory Footprint per Object",
            )
            effective_mem_footprint_per_object_plot_file = plots_dir / f"effective_mem_footprint_per_object.{fmt}"
            if effective_mem_footprint_per_object_plot_file.is_file():
                assert force, f"File already exists: {effective_mem_footprint_per_object_plot_file}"
            effective_mem_footprint_per_object_plot.get_figure().savefig(
                effective_mem_footprint_per_object_plot_file,
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
    create_mem_footprint_pie_plots(
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
