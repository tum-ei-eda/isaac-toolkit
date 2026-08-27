from pathlib import Path

import yaml

from isaac_toolkit.frontend.mlonmcu import run_initializer


def test_enable_etiss_tracing_does_not_mutate_source():
    source = {"runs": [{"target_name": "etiss_rv32", "feature_names": ["cfu_wca"], "config": {}}]}
    traced = run_initializer._enable_tracing(source)
    assert source["runs"][0]["feature_names"] == ["cfu_wca"]
    assert traced["runs"][0]["feature_names"] == ["cfu_wca", "log_instrs", "trace"]
    assert traced["runs"][0]["config"]["log_instrs.to_file"] is True
    assert traced["runs"][0]["config"]["etiss.experimental_print_to_file"] is True


def test_run_initializer_imports_outputs(tmp_path, monkeypatch):
    model = tmp_path / "model.tar"
    model.write_text("model", encoding="utf-8")
    initializer = tmp_path / "initializer.yml"
    initializer.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {
                        "target_name": "etiss_rv32",
                        "model_name": "model.tar",
                        "feature_names": [],
                        "config": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    imported = []

    def fake_run(command, check, cwd):
        assert check is True
        assert cwd == Path.cwd()
        effective_initializer = Path(command[command.index("--initializer") + 1])
        effective_data = yaml.safe_load(effective_initializer.read_text(encoding="utf-8"))
        assert effective_data["runs"][0]["model_name"] == str(model.resolve())
        output = Path(command[command.index("--dest") + 1])
        run_dir = output / "runs" / "0"
        run_dir.mkdir(parents=True)
        for name in ("generic_mlonmcu", "generic_mlonmcu.dump", "compile_commands.json", "generic_mlonmcu.map"):
            (run_dir / name).write_text("[]" if name.endswith(".json") else "data", encoding="utf-8")

    monkeypatch.setattr(run_initializer.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_initializer,
        "_import_static_artifacts",
        lambda sess, run_dir, force: imported.append((run_dir, force)),
    )

    class FakeSession:
        def __init__(self):
            self.saved = False

        def save(self):
            self.saved = True

    session = FakeSession()
    monkeypatch.chdir(tmp_path)
    run_initializer.run_mlonmcu_initializer(session, initializer.resolve(), force=True)
    assert session.saved
    assert len(imported) == 1
    assert imported[0][1] is True


def test_resolve_model_paths_preserves_symbolic_model_names(tmp_path):
    source = {"runs": [{"model_name": "mobilenet_v1"}]}
    resolved = run_initializer._resolve_model_paths(source, tmp_path)
    assert resolved["runs"][0]["model_name"] == "mobilenet_v1"
    assert resolved is not source


def test_find_run_directory_ignores_latest_symlink(tmp_path):
    run_dir = tmp_path / "runs" / "0"
    run_dir.mkdir(parents=True)
    (tmp_path / "runs" / "latest").symlink_to(run_dir, target_is_directory=True)
    assert run_initializer._find_run_directory(tmp_path) == run_dir
