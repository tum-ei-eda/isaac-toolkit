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
import subprocess
from typing import Optional


def demangle_fallback(func_name: str) -> str:
    try:
        return subprocess.check_output(["c++filt", name], text=True).strip()
    except Exception as ex:
        return name  # TODO: err
        raise RuntimeError(f"Could not unmangle func name: {func_name}") from ex


def unmangle_helper(func_name: Optional[str]) -> Optional[str]:
    if func_name is None:
        return None
    if not func_name.startswith("_Z"):
        return func_name
    try:
        from cpp_demangle import demangle

        return demangle(func_name)
    except ImportError:
        # Fallback to c++filt
        name = demangle_fallback(func_name)
        return name
