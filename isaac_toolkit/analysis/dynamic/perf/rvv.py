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
import numpy as np

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts, TraceArtifact


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def classify(instr):
    instr = instr.lower().replace(".", "_")
    if instr[0] != "v":
        return None
    instr_base = instr.split("_", 1)[0]
    if instr_base in ["vwmacc"]:
        return "MAC"
    if instr_base in ["vmul"]:
        return "MUL"
    if instr_base in ["vadd", "vmin", "vmax", "vxor", "vsll", "vwadd"]:
        return "ARITH"
    if instr.startswith("vset"):
        return "CFG"
    if instr.startswith("vmvr"):
        return "MV Reg"
    if instr == "vmv_s_x":
        return "MV VPR[0] <- GPR"
    if instr == "vmv_v_i":
        return "MV VPR <- IMM"
    if instr == "vmv_v_x":
        return "MV VPR <- GPR"
    if instr == "vmv_x_s":
        return "MV GPR <- VPR[0]"
    if instr.startswith("vmv"):
        return "MV"
    if instr.startswith("vred"):
        return "REDUCE"
    if instr.startswith("vl"):
        return "LOAD"
    if instr.startswith("vsr"):
        return "STORE VREG"
    if instr.startswith("vse") or instr.startswith("vss"):
        return "STORE"
    if instr.startswith("vslide"):
        return "SLIDE"
    # return f"OTHER ({instr})"
    return "OTHER"


def classify_rvv_instrs(instr_trace_df):
    instr_trace_df["instr_alt"] = instr_trace_df["instr"].apply(lambda x: x.lower().replace(".", "_"))
    instr_trace_df["category"] = instr_trace_df["instr_alt"].apply(classify)
    instr_trace_df.drop(columns=["instr_alt"])
    return instr_trace_df


def decode_vsew(x):
    if pd.isna(x):
        return x
    assert int(x) == x
    x = int(x)
    return 8 * 2**x


LMUL_MAP = {
    0b000: "1",
    0b001: "2",
    0b010: "4",
    0b011: "8",
    0b100: "reserved",
    0b101: "1/8",
    0b110: "1/4",
    0b111: "1/2",
}


def decode_vtype(x):
    # print("decode_vtype", x)
    vtype = x["vtype"]
    # print("vtype", vtype)
    if pd.isna(vtype):
        # print("nan")
        return vtype, vtype, vtype, vtype, vtype
    assert int(vtype) == vtype
    vtype = int(vtype)
    vill = (vtype >> 31) & 0b1
    vma = (vtype >> 7) & 0b1
    vta = (vtype >> 6) & 0b1
    vsew = decode_vsew((vtype >> 3) & 0b111)
    vlmul = LMUL_MAP[vtype & 0b111]
    ret = vill, vma, vta, vsew, vlmul
    # print("ret", ret)
    return ret


def decode_rvv_cols(
    perf_trace_df,
):
    rvv_trace_df = perf_trace_df[["vtype", "vl", "vlenb"]]
    # rvv_trace_df[["vill", "vma", "vta", "vsew", "vlmul"]] =
    rvv_trace_df[["vill", "vma", "vta", "vsew", "vlmul"]] = rvv_trace_df.apply(
        decode_vtype, axis=1, result_type="expand"
    )
    rvv_trace_df["vl"] = rvv_trace_df["vl"].replace(0, np.nan)
    rvv_trace_df["vl_util"] = rvv_trace_df["vl"] / rvv_trace_df["vlenb"]
    # print("temp", temp.head())
    # print("rvv_trace_df")
    # print(rvv_trace_df.head())
    vsew_vlmul_hist_df = rvv_trace_df[["vsew", "vlmul"]].value_counts().to_frame()
    vsew_vlmul_hist_df["count_rel"] = vsew_vlmul_hist_df["count"] / len(rvv_trace_df.dropna())
    vl_hist_df = rvv_trace_df["vl"].value_counts().to_frame()
    vl_hist_df["count_rel"] = vl_hist_df["count"] / len(rvv_trace_df.dropna())
    return rvv_trace_df, vsew_vlmul_hist_df, vl_hist_df


def collect_rvv_metrics(
    rvv_trace_df,
    # verbose: bool = False,
):
    num_total_instrs = len(rvv_trace_df)
    num_rvv_instrs = len(rvv_trace_df["vl"].dropna())
    num_rvv_instrs_rel = num_rvv_instrs / num_total_instrs
    mean_vl = rvv_trace_df["vl"].mean()
    min_vl = rvv_trace_df["vl"].min()
    max_vl = rvv_trace_df["vl"].max()
    vlenb = rvv_trace_df["vlenb"].dropna().unique()
    assert len(vlenb) == 1
    vlenb = vlenb[0]
    metrics = {
        "num_total_instrs": num_total_instrs,
        "num_rvv_instrs": num_rvv_instrs,
        "num_rvv_instrs_rel": num_rvv_instrs_rel,
        "mean_vl": mean_vl,
        "min_vl": min_vl,
        "max_vl": max_vl,
        "vlenb": vlenb,
    }

    rvv_metrics_df = pd.DataFrame([metrics])
    return rvv_metrics_df


def get_rvv_metrics(
    sess: Session,
    force: bool = False,
    # verbose: bool = False,
):
    artifacts = sess.artifacts

    instr_trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(instr_trace_artifacts) == 1
    instr_trace_artifact = instr_trace_artifacts[0]
    assert instr_trace_artifact.attrs.get("simulator") in ["etiss_perf", "etiss"]
    instr_trace_df = instr_trace_artifact.df
    perf_trace_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TRACE and x.attrs.get("kind") == "perf_trace"
    )
    assert len(perf_trace_artifacts) == 1
    perf_trace_artifact = perf_trace_artifacts[0]
    assert perf_trace_artifact.attrs.get("simulator") in ["etiss_perf", "etiss"]
    perf_trace_df = perf_trace_artifact.df

    instr_trace_df = classify_rvv_instrs(instr_trace_df.copy())
    categories_hist_df = instr_trace_df["category"].value_counts().to_frame()
    categories_hist_df["count_rel"] = categories_hist_df["count"] / categories_hist_df["count"].sum()
    print("categories", categories_hist_df)
    rvv_trace_df, vsew_vlmul_hist_df, vl_hist_df = decode_rvv_cols(perf_trace_df)
    print(vsew_vlmul_hist_df)
    print(vl_hist_df)

    rvv_metrics_df = collect_rvv_metrics(
        rvv_trace_df,
    )
    print("rvv_metrics_df", rvv_metrics_df)

    attrs = {
        "perf_trace": perf_trace_artifact.name,
        "kind": "table",
        "by": __name__,
    }
    attrs_hist = {
        "perf_trace": perf_trace_artifact.name,
        "kind": "hist",
        "by": __name__,
    }
    attrs2 = {
        "perf_trace": perf_trace_artifact.name,
        "kind": "metrics",
        "by": __name__,
    }

    rvv_trace_artifact = TraceArtifact("rvv_trace", rvv_trace_df, attrs=attrs)
    rvv_metrics_artifact = TableArtifact("rvv_metrics", rvv_metrics_df, attrs=attrs2)
    vsew_vlmul_hist_artifact = TraceArtifact("vsew_vlmul_hist", vsew_vlmul_hist_df, attrs=attrs_hist)
    vl_hist_artifact = TraceArtifact("vl_hist", vl_hist_df, attrs=attrs_hist)
    sess.add_artifact(rvv_trace_artifact, override=force)
    sess.add_artifact(rvv_metrics_artifact, override=force)
    sess.add_artifact(vsew_vlmul_hist_artifact, override=force)
    sess.add_artifact(vl_hist_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    get_rvv_metrics(sess, force=args.force)
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
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
