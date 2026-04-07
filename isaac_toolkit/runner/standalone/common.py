#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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

from typing import Union, List
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import MetricsArtifact
from isaac_toolkit.frontend.instr_trace.tgc import (
    load_instr_trace as load_tgc_instr_trace,
)
from isaac_toolkit.frontend.instr_trace.etiss import (
    load_instr_trace as load_etiss_instr_trace,
)
from isaac_toolkit.frontend.instr_trace.etiss_new import (
    load_instr_trace as load_etiss_perf_instr_trace,
)
from isaac_toolkit.frontend.instr_trace.spike import (
    load_instr_trace as load_spike_instr_trace,
)

from isaac_toolkit.logging import get_logger

logger = get_logger()


# def load_run_artifacts(
#     sess: Session,
#     dest_dir: Union[str, Path],
#     program: str,
#     force: bool = False,
#     # progress: bool = False,
# ):
#     dest_dir = Path(dest_dir)
#     assert dest_dir.is_dir(), f"Missing: {dest_dir}"
#
#     # Do nothing...


def load_trace_artifacts(
    sess: Session,
    dest_dir: Union[str, Path],
    program: str,
    simulator: str,
    force: bool = False,
    # progress: bool = False,
):
    del program
    dest_dir = Path(dest_dir)
    assert dest_dir.is_dir(), f"Missing: {dest_dir}"

    out_dir = dest_dir / "out"
    assert out_dir.is_dir(), f"Missing: {out_dir}"

    instr_trace_path = out_dir / f"{simulator}_instrs.log"
    assert instr_trace_path.exists()  # can be file or dir!

    instr_trace_frontends = {
        "tgc": load_tgc_instr_trace,
        "etiss": load_etiss_instr_trace,
        "etiss_perf": load_etiss_perf_instr_trace,
        "etiss_perf_vicuna": load_etiss_perf_instr_trace,
        "spike": load_spike_instr_trace,
        "spike_rv32": load_spike_instr_trace,
        "spike_rv64": load_spike_instr_trace,
        "spike_bm": load_spike_instr_trace,
    }
    load_instr_trace = instr_trace_frontends.get(simulator)
    assert load_instr_trace is not None
    operands = False  # TODO: store operands in extra artifact!
    load_instr_trace(
        # sess, instr_trace_file, force=force, progress=progress, operands=operands
        sess,
        instr_trace_path,
        force=force,
        operands=operands,
    )


def load_sim_metrics(
    sess: Session,
    sim_metrics: Union[dict, List[dict]],
    program: str,
    simulator: str,
    force: bool = False,
    # progress: bool = False,
):
    print("load_sim_metrics")
    if isinstance(sim_metrics, dict):
        metrics_df = pd.DataFrame([sim_metrics])
    else:
        assert isinstance(sim_metrics, list)
        assert len(sim_metrics) > 0
        assert isinstance(sim_metrics[0], dict)
        metrics_df = pd.DataFrame(sim_metrics)

    attrs = {
        "simulator": simulator,
        "program": program,
        "kind": "sim_metrics",
        "by": __name__,
    }
    metrics_artifact = MetricsArtifact("sim_metrics", metrics_df, attrs=attrs)
    print("metrics_artifact", metrics_artifact)
    sess.add_artifact(metrics_artifact, override=force)
