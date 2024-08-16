import io
import sys
import leb128
import logging
import argparse
import posixpath
from typing import Optional
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("llvm_bbs")


# TODO: share with other pie scripts
def plot_pie_data(series, y, threshold: float = 0.1):
    plot = series.plot.pie(y=y)
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
def agg_library_runtime(runtime_df, symbol_map_df, by: str = "library", col: str = "rel_weight"):
    ret = runtime_df.set_index("func_name").join(symbol_map_df.set_index("symbol"), how="left")
    ret = ret[[by, col]]
    ret = ret.groupby(by, as_index=False, dropna=False).sum()
    return ret


def create_pie_plots(sess: Session, threshold: float = 0.05, topk: int = 9, fmt: str = "jpg", force: bool = False):
    artifacts = sess.artifacts
    # TODO: allow missing files!
    pc2bb_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb")
    assert len(pc2bb_artifacts) == 1
    pc2bb_artifact = pc2bb_artifacts[0]
    pc2bb_df = pc2bb_artifact.df
    symbol_map_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "symbol_map"
    )
    assert len(symbol_map_artifacts) == 1
    symbol_map_artifact = symbol_map_artifacts[0]
    symbol_map_df = symbol_map_artifact.df
    plots_dir = sess.directory / "plots"
    plots_dir.mkdir(exist_ok=True)
    # TODO: use threshold

    if pc2bb_df is not None:
        runtime_df = pc2bb_df.copy()

        def helper(x):
            if x is None:
                return "?"
            if isinstance(x, set):
                assert len(x) == 1
                return list(x)[0]
            return x

        runtime_df["func_name"] = runtime_df["func_name"].apply(helper)
        runtime_df = runtime_df[["func_name", "rel_weight"]]
        runtime_df = runtime_df.groupby("func_name", as_index=False, dropna=False).sum()
        print("runtime_df", runtime_df)
        input(">>>")
        runtime_per_func_data = generate_pie_data(runtime_df, x="func_name", y="rel_weight", topk=topk)
        runtime_per_func_plot = plot_pie_data(runtime_per_func_data, "rel_weight", threshold=threshold)
        runtime_per_func_plot_file = plots_dir / f"runtime_per_func.{fmt}"
        if runtime_per_func_plot_file.is_file():
            assert force, "File already exists: {runtime_per_func_plot_file}"
        runtime_per_func_plot.get_figure().savefig(runtime_per_func_plot_file, bbox_inches="tight")
        plt.close()
        if symbol_map_df is not None:
            # library
            library_runtime_df = agg_library_runtime(runtime_df, symbol_map_df, by="library", col="rel_weight")
            runtime_per_library_data = generate_pie_data(library_runtime_df, x="library", y="rel_weight", topk=topk)
            runtime_per_library_plot = plot_pie_data(runtime_per_library_data, "rel_weight", threshold=threshold)
            runtime_per_library_plot_file = plots_dir / f"runtime_per_library.{fmt}"
            if runtime_per_library_plot_file.is_file():
                assert force, "File already exists: {runtime_per_library_plot_file}"
            runtime_per_library_plot.get_figure().savefig(runtime_per_library_plot_file, bbox_inches="tight")
            plt.close()
            # object
            object_runtime_df = agg_library_runtime(runtime_df, symbol_map_df, by="object", col="rel_weight")
            runtime_per_object_data = generate_pie_data(object_runtime_df, x="object", y="rel_weight", topk=topk)
            runtime_per_object_plot = plot_pie_data(runtime_per_object_data, "rel_weight", threshold=threshold)
            runtime_per_object_plot_file = plots_dir / f"runtime_per_object.{fmt}"
            if runtime_per_object_plot_file.is_file():
                assert force, "File already exists: {runtime_per_object_plot_file}"
            runtime_per_object_plot.get_figure().savefig(runtime_per_object_plot_file, bbox_inches="tight")
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
    create_pie_plots(sess, threshold=args.threshold, topk=args.topk, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=9)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
