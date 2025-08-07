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
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, fields, replace
from typing import Union, Optional, Dict, List, Any

import yaml
import dacite
from dacite import from_dict, Config

from isaac_toolkit.logging import get_logger

logger = get_logger()

DEFAULT_CONFIG = {
    "logging": {
        "console": {
            "level": "INFO",
        },
        "file": {
            "level": "DEBUG",
            "rotate": False,
            "limit": 1000,
        },
    },
    "memgraph": {
        "hostname": "localhost",
        "port": 7687,
        "user": "",
        "password": "",
        "database": "memgraph",
    },
    "flow": {
        "demo": {
            "stages": None,
            "target": None,
            "mlonmcu": None,
            # "llvm": None,
            "cdfg": None,
            # "sim": None,
            "choose": None,
            "query": None,
            "coredsl": None,
            "hls": None,
            "asip_syn": None,
            "fpga_syn": None,
        },
    },
}


def check_supported_types(data):
    ALLOWED_TYPES = (int, float, str, bool)
    if isinstance(data, dict):
        for value in data.values():
            check_supported_types(value)
    elif isinstance(data, list):
        for x in data:
            check_supported_types(x)
    else:
        if data is not None:
            assert isinstance(data, ALLOWED_TYPES), f"Unsupported type: {type(data)}"


class YAMLSettings:
    @classmethod
    def from_dict(cls, data: dict):
        """Convert dict into instance of YAMLSettings."""
        try:
            return from_dict(data_class=cls, data=data, config=Config(strict=True))
        except dacite.exceptions.UnexpectedDataError as err:
            logger.error("Unexpected key in ISAACConfig. Check for missmatch between ISAAC Toolkit versions!")
            raise err

    @classmethod
    def from_yaml(cls, text: str):
        data = yaml.safe_load(text)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: Path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data=data)

    def to_yaml(self):
        data = asdict(self)
        check_supported_types(data)
        text = yaml.dump(data)
        return text

    def to_yaml_file(self, path: Path):
        text = self.to_yaml()
        with open(path, "w") as file:
            file.write(text)

    def merge(self, other: "YAMLSettings", overwrite: bool = False, inplace: bool = False):
        """Merge two instances of YAMLSettings."""
        if not inplace:
            ret = replace(self)  # Make a copy of self
        for f1 in fields(other):
            k1 = f1.name
            v1 = getattr(other, k1)
            if v1 is None:
                continue
            t1 = type(v1)
            found = False
            for f2 in fields(self):
                k2 = f2.name
                v2 = getattr(self, k2)
                if k2 == k1:
                    found = True
                    if v2 is None:
                        if inplace:
                            setattr(self, k2, v1)
                        else:
                            setattr(ret, k2, v1)
                    else:
                        t2 = type(v2)
                        assert t1 is t2, "Type conflict"
                        if isinstance(v1, YAMLSettings):
                            v2.merge(v1, overwrite=overwrite, inplace=True)
                        elif isinstance(v1, dict):
                            if overwrite:
                                v2.clear()
                                v2.update(v1)
                            else:
                                for dict_key, dict_val in v1.items():
                                    if dict_key in v2:
                                        if isinstance(dict_val, YAMLSettings):
                                            assert isinstance(v2[dict_key], YAMLSettings)
                                            v2[dict_key].merge(dict_val, overwrite=overwrite, inplace=True)
                                        elif isinstance(dict_val, dict):
                                            v2[dict_key].update(dict_val)
                                        else:
                                            v2[dict_key] = dict_val
                                    else:
                                        v2[dict_key] = dict_val
                        elif isinstance(v1, list):
                            if overwrite:
                                v2.clear()
                            new = [x for x in v1 if x not in v2]
                            v2.extend(new)
                        else:
                            assert isinstance(
                                v2, (int, float, str, bool, Path)
                            ), f"Unsupported field type for merge {t1}"
                            if inplace:
                                setattr(self, k1, v1)
                            else:
                                setattr(ret, k1, v1)
                    break
            assert found
        if not inplace:
            return ret


@dataclass
class ConsoleLoggingSettings(YAMLSettings):
    level: Union[int, str] = logging.INFO


@dataclass
class FileLoggingSettings(YAMLSettings):
    level: Union[int, str] = logging.INFO
    limit: Optional[int] = None  # TODO: implement
    rotate: bool = False  # TODO: implement


@dataclass
class LoggingSettings(YAMLSettings):
    console: ConsoleLoggingSettings
    file: FileLoggingSettings


@dataclass
class MemgraphSettings(YAMLSettings):
    hostname: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = "memgraph"


@dataclass
class OL2Settings(YAMLSettings):
    config_template: Optional[str] = None
    until_step: Optional[str] = None
    target_freq: Optional[int] = None
    target_util: Optional[int] = None


@dataclass
class SynopsysSettings(YAMLSettings):
    pdk: Optional[str] = None
    clock_ns: Optional[float] = None
    core_name: Optional[str] = None


@dataclass
class VivadoSettings(YAMLSettings):
    part: Optional[str] = None
    clock_ns: Optional[float] = None
    core_name: Optional[str] = None


@dataclass
class ASIPSynSettings(YAMLSettings):
    tool: Optional[str] = None
    skip_baseline: Optional[bool] = None
    skip_default: Optional[bool] = None
    skip_shared: Optional[bool] = None
    baseline_use: Optional[str] = None
    ol2: Optional[OL2Settings] = None
    synopsys: Optional[SynopsysSettings] = None


@dataclass
class FPGASynSettings(YAMLSettings):
    tool: Optional[str] = None
    skip_baseline: Optional[bool] = None
    skip_default: Optional[bool] = None
    skip_shared: Optional[bool] = None
    baseline_use: Optional[str] = None
    vivado: Optional[VivadoSettings] = None


@dataclass
class NailgunSettings(YAMLSettings):
    core_name: Optional[str] = None
    ilp_solver: Optional[str] = None
    resource_model: Optional[str] = None
    clock_ns: Optional[float] = None
    schedule_timeout: Optional[float] = None
    refine_timeout: Optional[float] = None
    cell_library: Optional[str] = None
    enable_ol2: Optional[bool] = None
    sched_algo_ms: Optional[bool] = None
    sched_algo_pa: Optional[bool] = None
    sched_algo_mi: Optional[bool] = None
    sched_algo_ra: Optional[bool] = None
    share_resources: Optional[bool] = None  # TODO: partition?
    # label


@dataclass
class HLSSettings(YAMLSettings):
    tool: Optional[str] = None
    nailgun: Optional[NailgunSettings] = None


@dataclass
class StageSettings(YAMLSettings):
    enable: Optional[bool] = None
    defaults: Optional[Dict[str, Any]] = None


@dataclass
class RISCVSettings(YAMLSettings):
    arch: Optional[str] = None
    abi: Optional[str] = None
    xlen: Optional[int] = None  # TODO: auto?


@dataclass
class MlonmcuSettings(YAMLSettings):
    global_isel: Optional[bool] = None
    toolchain: Optional[str] = None
    optimize: Optional[str] = None
    unroll: Optional[bool] = None
    target: Optional[str] = None


# @dataclass
# class SimSettings(YAMLSettings):
#     iss: Optional[str] = None
#     iss_perf: Optional[str] = None
#     rtl: Optional[str] = None
#     fpga: Optional[str] = None


@dataclass
class CDFGSettings(YAMLSettings):
    stage: Optional[int] = None
    force_purge_db: Optional[bool] = None


# @dataclass
# class LLVMSettings(YAMLSettings):
#     global_isel: Optional[bool] = None


@dataclass
class ChooseSettings(YAMLSettings):
    check_potential_min_supported: Optional[float] = None
    bb_threshold: Optional[float] = None
    bb_min_weight: Optional[float] = None
    bb_min_supported_weight: Optional[float] = None
    bb_max_num: Optional[int] = None
    bb_min_instrs: Optional[int] = None
    allow_mem: Optional[bool] = None
    allow_loads: Optional[bool] = None
    allow_stores: Optional[bool] = None
    allow_branches: Optional[bool] = None
    allow_compressed: Optional[bool] = None
    allow_custom: Optional[bool] = None
    allow_fp: Optional[bool] = None
    allow_system: Optional[bool] = None


@dataclass
class QuerySettings(YAMLSettings):
    config_file: Optional[str] = None
    limit_results: Optional[int] = None
    # ...


@dataclass
class CoredslSettings(YAMLSettings):
    set_name: Optional[str] = None
    core_name: Optional[str] = None
    splitted: Optional[bool] = None
    base_extensions: Optional[List[str]] = None


@dataclass
class ProgramSettings(YAMLSettings):
    name: Optional[str] = None
    # ...


@dataclass
class ExperimentSettings(YAMLSettings):
    label: Optional[str] = None
    program: Optional[ProgramSettings] = None
    datetime: Optional[str] = None
    # directory: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class DockerSettings(YAMLSettings):
    etiss_image: Optional[str] = None
    seal5_image: Optional[str] = None
    hls_image: Optional[str] = None
    enable: Optional[bool] = None


# @dataclass
# class RetargetingSettings(YAMLSettings):
#     iss: Optional[ISSRetargetingSettings] = None
#     llvm: Optional[LLVMRetargetingSettings] = None


@dataclass
class ETISSSettings(YAMLSettings):
    core_name: Optional[str] = None


@dataclass
class DemoSettings(YAMLSettings):
    stages: Optional[Dict[str, StageSettings]] = None
    riscv: Optional[RISCVSettings] = None
    mlonmcu: Optional[MlonmcuSettings] = None
    # llvm: Optional[LLVMSettings] = None
    etiss: Optional[ETISSSettings] = None
    cdfg: Optional[CDFGSettings] = None
    # sim: Optional[SimSettings] = None
    choose: Optional[ChooseSettings] = None
    query: Optional[QuerySettings] = None
    coredsl: Optional[CoredslSettings] = None
    hls: Optional[HLSSettings] = None
    asip_syn: Optional[ASIPSynSettings] = None
    fpga_syn: Optional[FPGASynSettings] = None
    experiment: Optional[ExperimentSettings] = None
    docker: Optional[DockerSettings] = None
    # retargeting: Optional[RetargetingSettings] = None


@dataclass
class FlowSettings(YAMLSettings):
    demo: Optional[DemoSettings] = None


@dataclass
class IsaacConfig(YAMLSettings):
    logging: Optional[LoggingSettings] = None
    memgraph: Optional[MemgraphSettings] = None
    flow: Optional[FlowSettings] = None
