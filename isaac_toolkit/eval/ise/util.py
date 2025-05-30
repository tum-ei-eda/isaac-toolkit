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

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import TableArtifact, filter_artifacts, ArtifactFlag


logger = logging.getLogger("ise_util")


# TODO: get names from artifacts?


def check_util(
    sess: Session,
    force: bool = False,
    # names_csv: str = None,
):
    artifacts = sess.artifacts

    # instruction names
    # assert names_csv is not None
    # names_csv = Path(names_csv)
    # assert names_csv.is_file()
    # names_df = pd.read_csv(names_csv)
    # ise_instrs_df = names_df
    ise_instrs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "ise_instrs"
    )
    assert len(ise_instrs_artifacts) == 1
    ise_instrs_artifact = ise_instrs_artifacts[0]
    ise_instrs_df = ise_instrs_artifact.df
    ise_instr_names = ise_instrs_df["instr_lower"].values
    # print("ise_instr_names", ise_instr_names)

    # static counts
    disass_hist_artifacts = filter_artifacts(artifacts, lambda x: x.name == "disass_instrs_hist")
    assert len(disass_hist_artifacts) == 1
    disass_hist_artifact = disass_hist_artifacts[0]
    disass_hist_df = disass_hist_artifact.df
    # print("disass_hist_df", disass_hist_df)
    disass_hist_custom_df = disass_hist_df[disass_hist_df["instr"].apply(lambda x: x.lower() in ise_instr_names)]
    # print("disass_hist_custom_df", disass_hist_custom_df)
    total_disass_instrs = disass_hist_df["count"].sum()

    merged_disass_hist_custom_df = pd.merge(
        ise_instrs_df,
        disass_hist_custom_df,
        how="outer",
        left_on="instr_lower",
        right_on="instr",
        suffixes=("", "_y"),
    )

    # static_count_sum = disass_hist_df["count"].sum()
    # static_count_max = disass_hist_df["count"].max()
    # print("static_count", static_count_sum, static_count_max)
    # static_custom_count_sum = merged_disass_hist_custom_df["count"].sum()
    # static_custom_count_max = merged_disass_hist_custom_df["count"].max()
    # print("static_custom_count", static_custom_count_sum, static_custom_count_max)

    # dynamic counts
    instrs_hist_artifacts = filter_artifacts(artifacts, lambda x: x.name == "instrs_hist")
    assert len(instrs_hist_artifacts) == 1
    instrs_hist_artifact = instrs_hist_artifacts[0]
    instrs_hist_df = instrs_hist_artifact.df
    # print("instrs_hist_df", instrs_hist_df)
    instrs_hist_custom_df = instrs_hist_df[instrs_hist_df["instr"].apply(lambda x: x.lower() in ise_instr_names)]
    # print("instrs_hist_custom_df", instrs_hist_custom_df)
    total_instrs = instrs_hist_df["count"].sum()

    merged_instrs_hist_custom_df = pd.merge(
        ise_instrs_df, instrs_hist_custom_df, how="outer", left_on="instr_lower", right_on="instr", suffixes=("", "_y")
    )
    # dynamic_count_sum = instrs_hist_df["count"].sum()
    # dynamic_count_max = instrs_hist_df["count"].max()
    # print("dynamic_instr_count", dynamic_count_sum, dynamic_count_max)
    # dynamic_custom_count_sum = merged_instrs_hist_custom_df["count"].sum()
    # dynamic_custom_count_max = merged_instrs_hist_custom_df["count"].max()
    # print("dynamic_custom_count", dynamic_custom_count_sum, dynamic_custom_count_max)

    # combine
    # TODO: move counts to counts.py?
    merged_disass_hist_custom_df["estimated_reduction"] = merged_disass_hist_custom_df["count"] * (
        merged_disass_hist_custom_df["num_fused_instrs"] - 1
    )
    # TODO: consider instruction size to get memory footprints?
    estimated_total_disass_reduction = merged_disass_hist_custom_df["estimated_reduction"].sum()
    estimated_total_disass_without_ise = total_disass_instrs + estimated_total_disass_reduction
    rel_custom_count = merged_disass_hist_custom_df["rel_count"].sum()
    merged_disass_hist_custom_df["estimated_reduction_rel_scaled"] = (
        merged_disass_hist_custom_df["estimated_reduction"] / estimated_total_disass_reduction
    )
    merged_disass_hist_custom_df["estimated_reduction_rel"] = (
        merged_disass_hist_custom_df["estimated_reduction"] / estimated_total_disass_without_ise
    )
    static_agg_df = pd.DataFrame(
        [
            {
                "instr": None,
                "count": merged_disass_hist_custom_df["count"].sum(),
                "rel_count": merged_disass_hist_custom_df["rel_count"].sum(),
                "estimated_reduction": estimated_total_disass_reduction,
                "estimated_reduction_rel": merged_disass_hist_custom_df["estimated_reduction_rel"].sum(),
                "estimated_reduction_rel_scaled": merged_disass_hist_custom_df["estimated_reduction_rel_scaled"].sum(),
            }
        ]
    )
    static_counts_custom_df = pd.concat(
        [
            static_agg_df,
            merged_disass_hist_custom_df[
                [
                    "instr",
                    "count",
                    "rel_count",
                    "estimated_reduction",
                    "estimated_reduction_rel_scaled",
                    "estimated_reduction_rel",
                ]
            ],
        ]
    )
    merged_disass_hist_custom_df["used"] = merged_disass_hist_custom_df["count"] > 0

    merged_instrs_hist_custom_df["estimated_reduction"] = merged_instrs_hist_custom_df["count"] * (
        merged_instrs_hist_custom_df["num_fused_instrs"] - 1
    )
    estimated_total_reduction = merged_instrs_hist_custom_df["estimated_reduction"].sum()
    estimated_total_instrs_without_ise = total_instrs + estimated_total_reduction
    rel_custom_count = merged_instrs_hist_custom_df["rel_count"].sum()
    merged_instrs_hist_custom_df["estimated_reduction_rel_scaled"] = (
        merged_instrs_hist_custom_df["estimated_reduction"] / estimated_total_reduction
    )
    merged_instrs_hist_custom_df["estimated_reduction_rel"] = (
        merged_instrs_hist_custom_df["estimated_reduction"] / estimated_total_instrs_without_ise
    )
    dynamic_agg_df = pd.DataFrame(
        [
            {
                "instr": None,
                "count": merged_instrs_hist_custom_df["count"].sum(),
                "rel_count": rel_custom_count,
                "estimated_reduction": estimated_total_reduction,
                "estimated_reduction_rel": merged_instrs_hist_custom_df["estimated_reduction_rel"].sum(),
                "estimated_reduction_rel_scaled": merged_instrs_hist_custom_df["estimated_reduction_rel_scaled"].sum(),
            }
        ]
    )
    dynamic_counts_custom_df = pd.concat(
        [
            dynamic_agg_df,
            merged_instrs_hist_custom_df[
                [
                    "instr",
                    "count",
                    "rel_count",
                    "estimated_reduction",
                    "estimated_reduction_rel_scaled",
                    "estimated_reduction_rel",
                ]
            ],
        ]
    )
    merged_instrs_hist_custom_df["used"] = merged_instrs_hist_custom_df["count"] > 0

    ise_util_df = pd.merge(
        merged_disass_hist_custom_df, merged_instrs_hist_custom_df, on="instr", suffixes=("_static", "_dynamic")
    )
    ise_util_df["used_only_static"] = ise_util_df["used_static"] & ~ise_util_df["used_dynamic"]
    n_instrs = len(ise_instrs_df)
    n_used_static = ise_util_df["used_static"].sum()
    n_used_dynamic = ise_util_df["used_dynamic"].sum()
    n_used_static_rel = n_used_static / n_instrs
    n_used_dynamic_rel = n_used_dynamic / n_instrs
    ise_util_agg_df = pd.DataFrame(
        [
            {
                "instr": None,
                "n_total": n_instrs,
                "n_used_static": n_used_static,
                "n_used_dynamic": n_used_dynamic,
                "n_used_static_rel": n_used_static_rel,
                "n_used_dynamic_rel": n_used_dynamic_rel,
            }
        ]
    )
    ise_util_df = pd.concat(
        [ise_util_agg_df, ise_util_df[["instr", "used_static", "used_dynamic", "used_only_static"]]]
    )

    attrs = {}
    static_counts_custom_artifact = TableArtifact("static_counts_custom", static_counts_custom_df, attrs=attrs)
    dynamic_counts_custom_artifact = TableArtifact("dynamic_counts_custom", dynamic_counts_custom_df, attrs=attrs)
    ise_util_artifact = TableArtifact("ise_util", ise_util_df, attrs=attrs)

    sess.add_artifact(static_counts_custom_artifact, override=force)
    sess.add_artifact(dynamic_counts_custom_artifact, override=force)
    sess.add_artifact(ise_util_artifact, override=force)

    # TODO
    # pc2bb_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb")
    # assert len(pc2bb_artifacts) == 1
    # pc2bb_artifact = pc2bb_artifacts[0]
    # pc2bb_df = pc2bb_artifact.df

    # symbol_map_artifacts = filter_artifacts(
    #     artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "symbol_map"
    # )
    # symbol_map_df = None
    # if len(symbol_map_artifacts) > 0:
    #     assert len(symbol_map_artifacts) == 1
    #     symbol_map_artifact = symbol_map_artifacts[0]
    #     symbol_map_df = symbol_map_artifact.df

    # llvm_bbs_artifacts = filter_artifacts(
    #     artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs_new"
    # )
    # llvm_bbs_df = None
    # if len(llvm_bbs_artifacts) > 0:
    #     assert len(llvm_bbs_artifacts) == 1
    #     llvm_bbs_artifact = llvm_bbs_artifacts[0]
    #     llvm_bbs_df = llvm_bbs_artifact.df.copy()

    # plots_dir = sess.directory / "plots"
    # plots_dir.mkdir(exist_ok=True)
    # # TODO: use threshold
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    # print("elf_artifacts", elf_artifacts)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    trace_df = trace_artifact.df.copy()
    trace_len = len(trace_df)

    func2pc_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "func2pc")
    assert len(func2pc_artifacts) == 1
    func2pc_artifact = func2pc_artifacts[0]
    func2pc_df = func2pc_artifact.df.copy()
    func2pc_df[["start", "end"]] = func2pc_df["pc_range"].apply(pd.Series)

    def helper(pc):
        # print("x", x)
        # print("!", func2pc_df["start"] >= x[0] & func2pc_df["end"] <= x[1])
        matches = func2pc_df[func2pc_df["start"] <= pc]
        matches = matches[matches["end"] >= pc]
        # print("matches", matches)
        if len(matches) == 0:
            return None
        # print("len(matches)", len(matches))
        # assert len(matches) == 1
        # match_ = matches.iloc[0]
        # func = match_["func"]
        # TODO: assert that alias!
        funcs = set(matches["func"].values)
        FIRST_ONLY = True
        if FIRST_ONLY:
            if len(funcs) > 0:
                funcs = list(funcs)[0]
            else:
                funcs = None
        # return func
        return funcs

    def helper2(pc):
        # print("x", x)
        # print("!", func2pc_df["start"] >= x[0] & func2pc_df["end"] <= x[1])
        matches = trace_df[trace_df["pc"] == pc]
        # print("matches", matches)
        if len(matches) == 0:
            return None
        # print("len(matches)", len(matches))
        # assert len(matches) == 1
        # match_ = matches.iloc[0]
        # func = match_["func"]
        # TODO: assert that alias!
        instrs = set(matches["instr"].values)
        # print("instrs", instrs)
        assert len(instrs) == 1
        instr = list(instrs)[0]
        return instr

    pc_df = trace_df["pc"].value_counts().to_frame("count").reset_index()
    # trace_df["func_name"] = trace_df["pc"].apply(helper)
    pc_df["instr"] = pc_df["pc"].apply(helper2)
    pc_df["func_name"] = pc_df["pc"].apply(helper)
    CUSTOM_ONLY = True
    if CUSTOM_ONLY:
        pc_df = pc_df[pc_df["instr"].apply(lambda x: x.lower() in ise_instr_names)]

    # instrs_per_func = pc_df.groupby("func_name")["instr"].value_counts().to_dict()
    # print("instrs_per_func", instrs_per_func)
    func_ise_counts = {}
    for func_name, group_df in pc_df.groupby("func_name"):
        counts = group_df.groupby("instr")["count"].sum()
        # counts = group_df["instr"].value_counts().to_dict()
        # print("counts", counts)
        func_ise_counts[func_name] = counts.to_dict()
    # print("func_ise_counts", func_ise_counts)
    func_ise_rel_counts = {
        func_name: {instr: count / trace_len for instr, count in counts.items()}
        for func_name, counts in func_ise_counts.items()
    }
    # print("func_ise_rel_counts", func_ise_rel_counts)
    dynamic_counts_custom_per_func_df = pd.DataFrame(
        [{"func_name": func_name, **counts} for func_name, counts in func_ise_counts.items()]
    ).fillna(0)
    # print("dynamic_counts_custom_per_func_df", dynamic_counts_custom_per_func_df)
    dynamic_rel_counts_custom_per_func_df = pd.DataFrame(
        [{"func_name": func_name, **counts} for func_name, counts in func_ise_rel_counts.items()]
    ).fillna(0)
    # print("dynamic_rel_counts_custom_per_func_df", dynamic_rel_counts_custom_per_func_df)

    dynamic_counts_custom_per_func_artifact = TableArtifact(
        "dynamic_counts_custom_per_func", dynamic_counts_custom_per_func_df, attrs=attrs
    )
    dynamic_rel_counts_custom_per_func_artifact = TableArtifact(
        "dynamic_rel_counts_custom_per_func", dynamic_rel_counts_custom_per_func_df, attrs=attrs
    )

    sess.add_artifact(dynamic_counts_custom_per_func_artifact, override=force)
    sess.add_artifact(dynamic_rel_counts_custom_per_func_artifact, override=force)

    # static per_func
    disass_table_artifacts = filter_artifacts(artifacts, lambda x: x.name == "disass")
    assert len(disass_table_artifacts) == 1
    disass_df = disass_table_artifacts[0].df.copy()
    disass_len = len(disass_df)
    CUSTOM_ONLY = True
    if CUSTOM_ONLY:
        disass_df = disass_df[disass_df["instr"].apply(lambda x: x.lower() in ise_instr_names)]

    static_pc_df = disass_df
    static_pc_df["func_name"] = static_pc_df["pc"].apply(helper)
    # print("static_pc_df", static_pc_df)

    static_func_ise_counts = {}
    for func_name, group_df in static_pc_df.groupby("func_name"):
        # print("func_name", func_name)
        counts = group_df["instr"].value_counts()
        # counts = group_df["instr"].value_counts().to_dict()
        # print("counts", counts)
        static_func_ise_counts[func_name] = counts.to_dict()
    # print("static_func_ise_counts", static_func_ise_counts)
    static_func_ise_rel_counts = {
        func_name: {instr: count / disass_len for instr, count in counts.items()}
        for func_name, counts in static_func_ise_counts.items()
    }
    # print("static_func_ise_rel_counts", static_func_ise_rel_counts)
    static_counts_custom_per_func_df = pd.DataFrame(
        [{"func_name": func_name, **counts} for func_name, counts in static_func_ise_counts.items()]
    ).fillna(0)
    # print("static_counts_custom_per_func_df", static_counts_custom_per_func_df)
    static_rel_counts_custom_per_func_df = pd.DataFrame(
        [{"func_name": func_name, **counts} for func_name, counts in static_func_ise_rel_counts.items()]
    ).fillna(0)
    # print("static_rel_counts_custom_per_func_df", static_rel_counts_custom_per_func_df)

    static_counts_custom_per_func_artifact = TableArtifact(
        "static_counts_custom_per_func", static_counts_custom_per_func_df, attrs=attrs
    )
    static_rel_counts_custom_per_func_artifact = TableArtifact(
        "static_rel_counts_custom_per_func", static_rel_counts_custom_per_func_df, attrs=attrs
    )

    sess.add_artifact(static_counts_custom_per_func_artifact, override=force)
    sess.add_artifact(static_rel_counts_custom_per_func_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    check_util(
        sess,
        force=args.force,
        # names_csv=args.names_csv,
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
    # parser.add_argument("--names-csv", required=True)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
