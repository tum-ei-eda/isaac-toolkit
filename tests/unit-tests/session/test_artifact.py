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
        ([], lambda x: name in ["foo", "bar", "baz"], []),
    ],
)
def test_artifact_filter_artifacts(artifacts, func, expected):
    assert filter_artifacts(artifacts, func) == expected
