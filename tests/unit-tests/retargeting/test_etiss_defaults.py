"""Verify fill mode propagation to local scripts and Docker for both ISS paths."""
from unittest.mock import patch
import yaml
import pytest
from isaac_toolkit.retargeting.iss.etiss import retarget_etiss_iss
from isaac_toolkit.retargeting.iss.etiss_perf import retarget_etiss_perf


@pytest.mark.parametrize("performance", [False, True])
@pytest.mark.parametrize("docker", [False, True])
@pytest.mark.parametrize("fill_mode", [None, "empty"])
def test_fill_mode_reaches_backend(tmp_path, monkeypatch, performance, docker, fill_mode):
    script = tmp_path / "backend.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    key = "ETISS_PERF_SCRIPT_LOCAL" if performance else "ETISS_SCRIPT_LOCAL"
    monkeypatch.setenv(key, str(script))
    if performance:
        hls = tmp_path / "hls"
        hls.mkdir()
        (tmp_path / "default_index.yml").write_text(yaml.safe_dump(
            {"global": {"artifacts": {"HLS_DIR": str(hls)}}}))
    kwargs = {"workdir": tmp_path}
    if docker:
        kwargs["docker_image"] = "test-etiss"
    if fill_mode is not None:
        kwargs["fill_mode"] = fill_mode
    function = retarget_etiss_perf if performance else retarget_etiss_iss
    module = function.__module__
    with patch(f"{module}.subprocess.run") as run:
        function(None, **kwargs)
    expected = fill_mode or "auto"
    if docker:
        assert f"-e ISAAC_M2ISAR_FILL_MODE={expected}" in run.call_args.args[0]
    else:
        assert run.call_args.kwargs["env"]["ISAAC_M2ISAR_FILL_MODE"] == expected
