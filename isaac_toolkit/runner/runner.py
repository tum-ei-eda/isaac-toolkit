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
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path

from isaac_toolkit.logging import get_logger

logger = get_logger()


class ISAACRunner(ABC):

    def __init__(self):
        self.metrics = []

    @abstractmethod
    def run(self, dest_dir: Union[str, Path], verbose: bool = False, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def trace(self, dest_dir: Union[str, Path], verbose: bool = False, **kwargs) -> dict:
        raise NotImplementedError

    # TODO: builder metrics
    def add_metrics(self, metrics):
        if isinstance(metrics, dict):
            self.metrics.append(metrics)
        else:
            self.metrics.extend(metrics)

    def get_metrics(self, latest: bool = False, allow_none: bool = False):
        if latest:
            if len(self.metrics) > 0:
                return self.metrics[-1]
            assert allow_none
            return None
        if len(self.metrics) == 0:
            assert allow_none
        return self.metrics
