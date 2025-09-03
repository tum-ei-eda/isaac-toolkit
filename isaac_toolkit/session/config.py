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
from dataclasses import dataclass, asdict, fields
from typing import Union, Optional

import yaml
from dacite import from_dict

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
    "artifacts": {
        "instr_trace": {
            "fmt": "pickle",
            "engine": None,
            "compression_method": "zstd",
            "compression_level": None,
        },
        "table": {
            "fmt": "pickle",
            "engine": None,
            "compression_method": None,
            "compression_level": None,
        }
    }
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
        # print("from_dict", data)
        return from_dict(data_class=cls, data=data)

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

    def merge(self, other: "YAMLSettings", overwrite: bool = False):
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
                        setattr(self, k2, v1)
                    else:
                        t2 = type(v2)
                        assert t1 is t2, "Type conflict"
                        if isinstance(v1, YAMLSettings):
                            v2.merge(v1, overwrite=overwrite)
                        elif isinstance(v1, dict):
                            if overwrite:
                                v2.clear()
                                v2.update(v1)
                            else:
                                for dict_key, dict_val in v1.items():
                                    if dict_key in v2:
                                        if isinstance(dict_val, YAMLSettings):
                                            assert isinstance(v2[dict_key], YAMLSettings)
                                            v2[dict_key].merge(dict_val, overwrite=overwrite)
                                    else:
                                        v2[dict_key] = dict_val
                        elif isinstance(v1, list):
                            if overwrite:
                                v2.clear()
                            # duplicates are dropped here
                            new = [x for x in v1 if x not in v2]
                            v2.extend(new)
                        else:
                            assert isinstance(
                                v2, (int, float, str, bool, Path)
                            ), f"Unsupported field type for merge {t1}"
                            setattr(self, k1, v1)
                    break
            assert found


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
class TableArtifactsSettings(YAMLSettings):
    # compression: Optional[CompressionSettings] = None
    fmt: Optional[str] = None
    engine: Optional[str] = None
    compression_method: Optional[str] = None
    compression_level: Optional[int] = None


@dataclass
class ArtifactsSettings(YAMLSettings):
    instr_trace: Optional[TableArtifactsSettings] = None
    table: Optional[TableArtifactsSettings] = None


@dataclass
class IsaacConfig(YAMLSettings):
    logging: Optional[LoggingSettings] = None
    memgraph: Optional[MemgraphSettings] = None
    artifacts: Optional[ArtifactsSettings] = None
