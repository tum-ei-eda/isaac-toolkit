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
import pandas as pd
import pytest

from isaac_toolkit.analysis.dynamic.histogram.loc import collect_loc_histograms as collect_dynamic
from isaac_toolkit.analysis.static.histogram.loc import collect_loc_histograms as collect_static


def inputs():
    pc2locs = pd.DataFrame(
        {
            "pc": [0x100, 0x104, 0x108],
            "locs": [{"foo.c:1"}, {"foo.c:1", "foo.c:2"}, {"foo.c:2"}],
        }
    )
    disass = pd.DataFrame(
        {
            "pc": [0x100, 0x104, 0x108],
            "bytecode": [0x13, 0x13, 0x23],
            "instr": ["addi", "addi", "sw"],
        }
    )
    trace = pd.DataFrame(
        {
            "pc": [0x100, 0x100, 0x104, 0x108, 0x108, 0x108],
            "bytecode": [0x13, 0x13, 0x13, 0x23, 0x23, 0x23],
            "instr": ["addi", "addi", "addi", "sw", "sw", "sw"],
        }
    )
    return pc2locs, disass, trace


def test_static_and_dynamic_histograms_are_collected_separately():
    pc2locs, disass, trace = inputs()
    static = collect_static(pc2locs, disass)["count"].set_index("loc")
    dynamic = collect_dynamic(pc2locs, trace)["count"].set_index("loc")
    assert static.loc["foo.c:1", "count"] == 2
    assert static.loc["foo.c:2", "count"] == 2
    assert dynamic.loc["foo.c:1", "count"] == 3
    assert dynamic.loc["foo.c:2", "count"] == 4
    assert static["rel_count"].sum() == pytest.approx(1.0)
    assert dynamic["rel_count"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("collector,input_index", [(collect_static, 1), (collect_dynamic, 2)])
def test_optional_instruction_and_opcode_histograms(collector, input_index):
    data = inputs()
    histograms = collector(data[0], data[input_index], metrics=["per_instr", "per_opcode"])
    assert set(histograms) == {"instr", "opcode"}
    assert set(histograms["opcode"]["opcode"]) == {"OP-IMM", "STORE"}


def test_unknown_metric_is_rejected():
    pc2locs, _, trace = inputs()
    with pytest.raises(ValueError, match="Unsupported LOC histogram metrics"):
        collect_dynamic(pc2locs, trace, metrics=["cycles"])
