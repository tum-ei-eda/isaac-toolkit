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
import argparse
from typing import Optional
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts
from isaac_toolkit.logging import get_logger, set_log_level

logger = get_logger()


# TODO: share codes
def generate_pie_data(df, x: str, y: str, topk: Optional[int] = None):
    ret = df.copy()
    ret.set_index(x, inplace=True)
    ret.sort_values(y, inplace=True, ascending=False)
    if topk is not None:
        a = ret.iloc[:topk]
        b = ret.iloc[topk:].agg(others=(y, "sum"))
        ret = pd.concat([a, b])
    ret = ret[y]

    return ret


def compare_with_sess(
    sess: Session,
    other_sess: Session,
    force: bool = False,
):
    logger.info("Comparing ISEs between sessions...")
    artifacts = sess.artifacts
    artifacts_ = other_sess.artifacts
    pc2bb_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb")
    assert len(pc2bb_artifacts) == 1
    pc2bb_artifact = pc2bb_artifacts[0]
    pc2bb_df = pc2bb_artifact.df
    pc2bb_artifacts_ = filter_artifacts(artifacts_, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb")
    assert len(pc2bb_artifacts_) == 1
    pc2bb_artifact_ = pc2bb_artifacts_[0]
    pc2bb_df_ = pc2bb_artifact_.df

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
    llvm_bbs_artifacts_ = filter_artifacts(
        artifacts_, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs_new"
    )
    llvm_bbs_df_ = None
    if len(llvm_bbs_artifacts_) > 0:
        assert len(llvm_bbs_artifacts_) == 1
        llvm_bbs_artifact_ = llvm_bbs_artifacts_[0]
        llvm_bbs_df_ = llvm_bbs_artifact_.df.copy()

    plots_dir = sess.directory / "plots"
    plots_dir.mkdir(exist_ok=True)
    # TODO: use threshold

    if pc2bb_df is not None and pc2bb_df is not None:
        # TODO: share codes with pie
        runtime_df = pc2bb_df.copy()
        runtime_df_ = pc2bb_df_.copy()
        runtime = runtime_df["weight"].sum()
        runtime_ = runtime_df_["weight"].sum()
        runtime_rel = runtime / runtime_
        # print("runtime_rel", runtime_rel)
        # input("!")

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
        runtime_df_["func_name"] = runtime_df_["func_name"].apply(helper)

        if runtime_rel <= 1:
            temp = pd.DataFrame([{"func_name": "DIFF", "weight": (1 - runtime_rel) * runtime_}])
            runtime_df = pd.concat([runtime_df, temp])
            runtime_df["rel_weight"] = runtime_df["weight"] / runtime_
            temp_ = pd.DataFrame([{"func_name": "DIFF", "weight": 0, "rel_weight": 0}])
            runtime_df_ = pd.concat([runtime_df_, temp_])
        else:
            # TODO: check if ok?
            temp_ = pd.DataFrame([{"func_name": "DIFF", "weight": (1 - (1 / runtime_rel)) * runtime}])
            runtime_df_ = pd.concat([runtime_df_, temp_])
            runtime_df_["rel_weight"] = runtime_df_["weight"] / runtime
            temp = pd.DataFrame([{"func_name": "DIFF", "weight": 0, "rel_weight": 0}])
            runtime_df = pd.concat([runtime_df, temp])

        # print("runtime_df", runtime_df)
        # print("runtime_df_", runtime_df_)

        runtime_df = runtime_df[["func_name", "rel_weight"]]
        runtime_df = runtime_df.groupby("func_name", as_index=False, dropna=False).sum()
        runtime_per_func_data = generate_pie_data(runtime_df, x="func_name", y="rel_weight")
        runtime_df_ = runtime_df_[["func_name", "rel_weight"]]
        runtime_df_ = runtime_df_.groupby("func_name", as_index=False, dropna=False).sum()
        runtime_per_func_data_ = generate_pie_data(runtime_df_, x="func_name", y="rel_weight")
        # print("runtime_per_func_data", runtime_per_func_data)
        # print("runtime_per_func_data_", runtime_per_func_data_)
        merged_runtime_per_func_data = runtime_per_func_data.to_frame(name="rel_weight").merge(
            runtime_per_func_data_.to_frame(name="rel_weight"),
            on="func_name",
            how="outer",
            suffixes=("", "_"),
            # indicator=True,
        )
        # print("merged_runtime_per_func_data", merged_runtime_per_func_data)
        merged_runtime_per_func_data.sort_values("rel_weight", inplace=True, ascending=False)
        merged_runtime_per_func_data["diff"] = (
            (merged_runtime_per_func_data["rel_weight"] - merged_runtime_per_func_data["rel_weight_"])
            .fillna(0)
            .round(5)
        )
        merged_runtime_per_func_data["diff_rel"] = (
            (merged_runtime_per_func_data["diff"] / merged_runtime_per_func_data["rel_weight_"]).fillna(0).round(5)
        )
        merged_runtime_per_func_data = merged_runtime_per_func_data[merged_runtime_per_func_data["diff"] != 0]
        # print("merged_runtime_per_func_data", merged_runtime_per_func_data)
        # input(">>>")
        attrs = {}
        artifact = TableArtifact("compare_runtime_per_func", merged_runtime_per_func_data, attrs=attrs)
        sess.add_artifact(artifact, override=force)
        if llvm_bbs_df is not None and llvm_bbs_df_ is not None:
            llvm_bbs_df["func_bb"] = llvm_bbs_df["func_name"] + "-" + llvm_bbs_df["bb_name"]
            llvm_bbs_df_["func_bb"] = llvm_bbs_df_["func_name"] + "-" + llvm_bbs_df_["bb_name"]
            if runtime_rel <= 1:
                temp = pd.DataFrame([{"func_bb": "DIFF", "weight": (1 - runtime_rel) * runtime_}])
                llvm_bbs_df = pd.concat([llvm_bbs_df, temp])
                llvm_bbs_df["rel_weight"] = llvm_bbs_df["weight"] / runtime_
                temp_ = pd.DataFrame([{"func_bb": "DIFF", "weight": 0, "rel_weight": 0}])
                llvm_bbs_df_ = pd.concat([llvm_bbs_df_, temp_])
            else:
                # TODO: check if ok?
                temp_ = pd.DataFrame([{"func_bb": "DIFF", "weight": (1 - (1 / runtime_rel)) * runtime}])
                llvm_bbs_df_ = pd.concat([llvm_bbs_df_, temp_])
                llvm_bbs_df_["rel_weight"] = llvm_bbs_df_["weight"] / runtime
                temp = pd.DataFrame([{"func_bb": "DIFF", "weight": 0, "rel_weight": 0}])
                llvm_bbs_df = pd.concat([llvm_bbs_df, temp])
            runtime_per_llvm_bb_data = generate_pie_data(
                llvm_bbs_df,
                x="func_bb",
                y="rel_weight",
            )
            runtime_per_llvm_bb_data_ = generate_pie_data(
                llvm_bbs_df_,
                x="func_bb",
                y="rel_weight",
            )
            # print("runtime_per_llvm_bb_data", runtime_per_llvm_bb_data)
            # print("runtime_per_llvm_bb_data_", runtime_per_llvm_bb_data_)
            merged_runtime_per_llvm_bb_data = runtime_per_llvm_bb_data.to_frame(name="rel_weight").merge(
                runtime_per_llvm_bb_data_.to_frame(name="rel_weight"),
                on="func_bb",
                how="outer",
                suffixes=("", "_"),
                # indicator=True,
            )
            merged_runtime_per_llvm_bb_data.sort_values("rel_weight", inplace=True, ascending=False)
            merged_runtime_per_llvm_bb_data["diff"] = (
                (merged_runtime_per_llvm_bb_data["rel_weight"] - merged_runtime_per_llvm_bb_data["rel_weight_"])
                .fillna(0)
                .round(5)
            )
            merged_runtime_per_llvm_bb_data["diff_rel"] = (
                (merged_runtime_per_llvm_bb_data["diff"] / merged_runtime_per_llvm_bb_data["rel_weight_"])
                .fillna(0)
                .round(5)
            )
            merged_runtime_per_llvm_bb_data = merged_runtime_per_llvm_bb_data[
                merged_runtime_per_llvm_bb_data["diff"] != 0
            ]
            # print("merged_runtime_per_llvm_bb_data", merged_runtime_per_llvm_bb_data)
            # input(">>>")
            attrs = {}
            artifact = TableArtifact("compare_runtime_per_llvm_bb", merged_runtime_per_llvm_bb_data, attrs=attrs)
            sess.add_artifact(artifact, override=force)

    mem_footprint_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "mem_footprint"
    )
    assert len(mem_footprint_artifacts) == 1
    mem_footprint_artifact = mem_footprint_artifacts[0]
    mem_footprint_df = mem_footprint_artifact.df
    mem_footprint_artifacts_ = filter_artifacts(
        artifacts_, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "mem_footprint"
    )
    assert len(mem_footprint_artifacts_) == 1
    mem_footprint_artifact_ = mem_footprint_artifacts_[0]
    mem_footprint_df_ = mem_footprint_artifact_.df

    effective_mem_footprint_artifacts = filter_artifacts(
        artifacts,
        lambda x: x.flags & ArtifactFlag.TABLE and x.name == "effective_mem_footprint",
    )
    eff_mem_footprint_df = None
    if len(effective_mem_footprint_artifacts) > 0:
        assert len(effective_mem_footprint_artifacts) == 1
        effective_mem_footprint_artifact = effective_mem_footprint_artifacts[0]
        eff_mem_footprint_df = effective_mem_footprint_artifact.df
    effective_mem_footprint_artifacts_ = filter_artifacts(
        artifacts_,
        lambda x: x.flags & ArtifactFlag.TABLE and x.name == "effective_mem_footprint",
    )
    eff_mem_footprint_df_ = None
    if len(effective_mem_footprint_artifacts_) > 0:
        assert len(effective_mem_footprint_artifacts_) == 1
        effective_mem_footprint_artifact_ = effective_mem_footprint_artifacts_[0]
        eff_mem_footprint_df_ = effective_mem_footprint_artifact_.df
    if mem_footprint_df is not None and mem_footprint_df_ is not None:
        mem_footprint = mem_footprint_df["bytes"].sum()
        mem_footprint_ = mem_footprint_df_["bytes"].sum()
        mem_footprint_rel = mem_footprint / mem_footprint_
        # print("mem_footprint", mem_footprint)
        # print("mem_footprint_", mem_footprint_)
        # print("mem_footprint_rel", mem_footprint_rel)
        # input("?")
        # print("mem_footprint_df", mem_footprint_df)
        # print("mem_footprint_df_", mem_footprint_df_)
        if mem_footprint_rel <= 1:
            temp = pd.DataFrame([{"func": "DIFF", "bytes": (1 - mem_footprint_rel) * mem_footprint_}])
            mem_footprint_df = pd.concat([mem_footprint_df, temp])
            mem_footprint_df["rel_bytes"] = mem_footprint_df["bytes"] / mem_footprint_
            temp_ = pd.DataFrame([{"func": "DIFF", "bytes": 0, "rel_bytes": 0}])
            mem_footprint_df_ = pd.concat([mem_footprint_df_, temp_])
        else:
            # TODO: check if ok?
            temp_ = pd.DataFrame([{"func": "DIFF", "bytes": (1 - (1 / mem_footprint_rel)) * mem_footprint}])
            mem_footprint_df_ = pd.concat([mem_footprint_df_, temp_])
            mem_footprint_df_["rel_bytes"] = mem_footprint_df_["bytes"] / mem_footprint
            temp = pd.DataFrame([{"func": "DIFF", "bytes": 0, "rel_bytes": 0}])
            mem_footprint_df = pd.concat([mem_footprint_df, temp])
        # print("mem_footprint_df2", mem_footprint_df)
        # print("mem_footprint_df_2", mem_footprint_df_)
        mem_footprint_per_func_data = generate_pie_data(
            mem_footprint_df,
            x="func",
            y="rel_bytes",
        )
        mem_footprint_per_func_data_ = generate_pie_data(
            mem_footprint_df_,
            x="func",
            y="rel_bytes",
        )
        merged_mem_footprint_per_func_data = mem_footprint_per_func_data.to_frame(name="rel_bytes").merge(
            mem_footprint_per_func_data_.to_frame(name="rel_bytes"),
            on="func",
            how="outer",
            suffixes=("", "_"),
            # indicator=True,
        )
        # print("merged_mem_footprint_per_func_data", merged_mem_footprint_per_func_data)
        merged_mem_footprint_per_func_data.sort_values("rel_bytes", inplace=True, ascending=False)
        merged_mem_footprint_per_func_data["diff"] = merged_mem_footprint_per_func_data[
            "rel_bytes"
        ] - merged_mem_footprint_per_func_data["rel_bytes_"].fillna(0).round(3)
        merged_mem_footprint_per_func_data["diff_rel"] = merged_mem_footprint_per_func_data[
            "diff"
        ] / merged_mem_footprint_per_func_data["rel_bytes_"].fillna(0).round(3)
        merged_mem_footprint_per_func_data = merged_mem_footprint_per_func_data[
            merged_mem_footprint_per_func_data["diff"] != 0
        ]
        # print("merged_mem_footprint_per_func_data", merged_mem_footprint_per_func_data)
        # input(">>>")
        attrs = {}
        artifact = TableArtifact("compare_mem_footprint_per_func", merged_mem_footprint_per_func_data, attrs=attrs)
        sess.add_artifact(artifact, override=force)
    if eff_mem_footprint_df is not None and eff_mem_footprint_df_ is not None:
        eff_mem_footprint = eff_mem_footprint_df["bytes"].sum()
        eff_mem_footprint_ = eff_mem_footprint_df_["bytes"].sum()
        eff_mem_footprint_rel = eff_mem_footprint / eff_mem_footprint_
        # print("eff_mem_footprint", eff_mem_footprint)
        # print("eff_mem_footprint_", eff_mem_footprint_)
        # print("eff_mem_footprint_rel", eff_mem_footprint_rel)
        # input("?")
        # print("eff_mem_footprint_df", eff_mem_footprint_df)
        # print("eff_mem_footprint_df_", eff_mem_footprint_df_)
        if eff_mem_footprint_rel <= 1:
            temp = pd.DataFrame([{"func": "DIFF", "bytes": (1 - eff_mem_footprint_rel) * eff_mem_footprint_}])
            eff_mem_footprint_df = pd.concat([eff_mem_footprint_df, temp])
            eff_mem_footprint_df["eff_rel_bytes"] = eff_mem_footprint_df["bytes"] / eff_mem_footprint_
            temp_ = pd.DataFrame([{"func": "DIFF", "bytes": 0, "eff_rel_bytes": 0}])
            eff_mem_footprint_df_ = pd.concat([eff_mem_footprint_df_, temp_])
        else:
            # TODO: check if ok?
            temp_ = pd.DataFrame([{"func": "DIFF", "bytes": (1 - (1 / eff_mem_footprint_rel)) * eff_mem_footprint}])
            eff_mem_footprint_df_ = pd.concat([eff_mem_footprint_df_, temp_])
            eff_mem_footprint_df_["eff_rel_bytes"] = eff_mem_footprint_df_["bytes"] / eff_mem_footprint
            temp = pd.DataFrame([{"func": "DIFF", "bytes": 0, "eff_rel_bytes": 0}])
            eff_mem_footprint_df = pd.concat([eff_mem_footprint_df, temp])
        # print("eff_mem_footprint_df2", eff_mem_footprint_df)
        # print("eff_mem_footprint_df_2", eff_mem_footprint_df_)

        eff_mem_footprint_per_func_data = generate_pie_data(
            eff_mem_footprint_df,
            x="func",
            y="eff_rel_bytes",
        )
        eff_mem_footprint_per_func_data_ = generate_pie_data(
            eff_mem_footprint_df_,
            x="func",
            y="eff_rel_bytes",
        )
        merged_eff_mem_footprint_per_func_data = eff_mem_footprint_per_func_data.to_frame(name="eff_rel_bytes").merge(
            eff_mem_footprint_per_func_data_.to_frame(name="eff_rel_bytes"),
            on="func",
            how="outer",
            suffixes=("", "_"),
            # indicator=True,
        )
        # print("merged_eff_mem_footprint_per_func_data", merged_eff_mem_footprint_per_func_data)
        merged_eff_mem_footprint_per_func_data.sort_values("eff_rel_bytes", inplace=True, ascending=False)
        merged_eff_mem_footprint_per_func_data["diff"] = merged_eff_mem_footprint_per_func_data[
            "eff_rel_bytes"
        ] - merged_eff_mem_footprint_per_func_data["eff_rel_bytes_"].fillna(0).round(3)
        merged_eff_mem_footprint_per_func_data["diff_rel"] = merged_eff_mem_footprint_per_func_data[
            "diff"
        ] / merged_eff_mem_footprint_per_func_data["eff_rel_bytes_"].fillna(0).round(3)
        merged_eff_mem_footprint_per_func_data = merged_eff_mem_footprint_per_func_data[
            merged_eff_mem_footprint_per_func_data["diff"] != 0
        ]
        # print("merged_eff_mem_footprint_per_func_data", merged_eff_mem_footprint_per_func_data)
        # input(">>>")
        attrs = {}
        artifact = TableArtifact(
            "compare_eff_mem_footprint_per_func", merged_eff_mem_footprint_per_func_data, attrs=attrs
        )
        sess.add_artifact(artifact, override=force)

    # attrs = {}

    # artifact = TableArtifact("compare_sess", compare_sess_df, attrs=attrs)
    # sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    assert args.with_session is not None
    session_dir_ = Path(args.with_session)
    assert session_dir_.is_dir(), f"Session dir does not exist: {session_dir}"
    sess_ = Session.from_dir(session_dir_)
    compare_with_sess(
        sess,
        other_sess=sess_,
        force=args.force,
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
    parser.add_argument("--with-session", required=True)
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
