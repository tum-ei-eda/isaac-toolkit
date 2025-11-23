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
import pytest

from isaac_toolkit.session.artifact import (
    ArtifactFlag,
    PythonArtifact,
    filter_artifacts,
)


FOO_ARTIFACT = PythonArtifact("foo", data=[])
BAR_ARTIFACT = PythonArtifact("bar", data=[], attrs={"key": "val"})
BAZ_ARTIFACT = PythonArtifact("baz", data=[], flags=ArtifactFlag.TABLE)

ALL_ARTIFACTS = [FOO_ARTIFACT, BAR_ARTIFACT, BAZ_ARTIFACT]


@pytest.mark.parametrize(
    "artifacts,func,expected",
    [
        (ALL_ARTIFACTS, lambda _: True, ALL_ARTIFACTS),
        (ALL_ARTIFACTS, lambda _: False, []),
        (ALL_ARTIFACTS, lambda x: x.name == "foo", [FOO_ARTIFACT]),
        (ALL_ARTIFACTS, lambda x: x.attrs.get("key") == "val", [BAR_ARTIFACT]),
        (ALL_ARTIFACTS, lambda x: x.flags & ArtifactFlag.PYTHON, ALL_ARTIFACTS),
        (ALL_ARTIFACTS, lambda x: x.flags & ArtifactFlag.TABLE, [BAZ_ARTIFACT]),
        ([], lambda x: x.name in ["foo", "bar", "baz"], []),
    ],
)
def test_artifact_filter_artifacts(artifacts, func, expected):
    assert filter_artifacts(artifacts, func) == expected
