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

from .etiss import ETISSStandaloneBuilder, invoke_etiss_builder, parse_etiss_args
from .etiss_perf import ETISSPerfStandaloneBuilder, invoke_etiss_perf_builder, parse_etiss_perf_args
from .etiss_perf_vicuna import ETISSPerfVicunaStandaloneBuilder, invoke_etiss_perf_vicuna_builder, parse_etiss_perf_vicuna_args
from .vicuna import VicunaStandaloneBuilder, invoke_vicuna_builder, parse_vicuna_args
from .spike import SpikeStandaloneBuilder, invoke_spike_builder, parse_spike_args

# from .etiss_perf import invoke_etiss_perf_builder, parse_etiss_perf_args

invoke_lookup = {}


def register_simulator(name, parse_func, invoke_func):
    invoke_lookup[name] = (parse_func, invoke_func)


def lookup_simulator(name):
    return invoke_lookup.get(name)


register_simulator("etiss", parse_etiss_args, invoke_etiss_builder)
register_simulator("etiss_perf", parse_etiss_perf_args, invoke_etiss_perf_builder)
register_simulator("etiss_perf_vicuna", parse_etiss_perf_vicuna_args, invoke_etiss_perf_vicuna_builder)
register_simulator("vicuna", parse_vicuna_args, invoke_vicuna_builder)
register_simulator("spike", parse_spike_args, invoke_spike_builder)
# register_simulator("spike_bm", parse_spike_bm_args, invoke_spike_bm_builder)
# register_simulator("tgc", parse_tgc_args, invoke_tgc_builder)
# register_simulator("tgc", parse_dbt_args, invoke_dbt_builder)
# register_simulator("tgc", parse_vicuna_args, invoke_vicuna_builder)


STANDALONE_BUILDERS = {
    "etiss": ETISSStandaloneBuilder,
    "spike": SpikeStandaloneBuilder,
}
