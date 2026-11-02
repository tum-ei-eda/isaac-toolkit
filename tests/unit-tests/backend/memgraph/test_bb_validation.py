"""Guard profile annotation against ELF/CDFG identity and count drift."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

m = importlib.import_module("isaac_toolkit.backend.memgraph.annotate_bb_weights")


def elf(*counts):
    return pd.DataFrame(
        [dict(func_name="f", bb_name=f"%bb.{i}", num_instrs=n, freq=0, rel_weight=0.0) for i, n in enumerate(counts)],
        columns=["func_name", "bb_name", "num_instrs", "freq", "rel_weight"],
    )


def graph(*counts):
    return [dict(func_name="f", bb_name=f"%bb.{i}", graph_nodes=n, graph_blocks=1) for i, n in enumerate(counts)]


def test_counts_are_checked_in_both_directions_with_combined_tolerance():
    report = m.compare_bb_counts(elf(100, 100, 100, 100, 100), graph(100, 115, 85, 116, 84))
    assert report.status.tolist() == ["exact", "tolerated", "tolerated", "mismatch", "mismatch"]
    assert report.allowance.tolist() == [15] * 5
    with pytest.raises(m.BBMismatchError, match="2 invalid blocks"):
        m.enforce_bb_report(report)
    m.enforce_bb_report(report, "warn")


def test_missing_selected_blocks_and_empty_graph_are_errors():
    report = m.compare_bb_counts(elf(3, 0, 10, 8), graph(0, 0), selected=[("f", "%bb.2")])
    assert report.status.tolist() == ["empty_graph", "exact", "missing_graph", "not_exported"]
    assert report.severe.tolist() == [True, False, True, False]


def test_unmatched_graph_blocks_are_reported_and_selected_missing_elf_fails():
    report = m.compare_bb_counts(elf(10), graph(10, 3), selected=[("f", "%bb.1")])
    assert report.status.tolist() == ["exact", "missing_elf"]
    assert report.severe.tolist() == [False, True]


def test_duplicate_elf_or_database_block_identity_is_rejected():
    db = graph(10)
    db[0]["graph_blocks"] = 2
    assert m.compare_bb_counts(elf(10), db).status.tolist() == ["ambiguous"]
    assert m.compare_bb_counts(pd.concat([elf(10), elf(10)]), graph(10)).status.tolist() == ["ambiguous"]


@pytest.mark.parametrize("count", [float("nan"), float("inf"), -1, 2.5])
def test_invalid_elf_count_is_not_silently_accepted(count):
    report = m.compare_bb_counts(elf(count), graph(10))
    assert report.status.tolist() == ["invalid_elf_count"]
    with pytest.raises(m.BBMismatchError):
        m.enforce_bb_report(report)


@pytest.mark.parametrize("counts", [[], graph(10)])
def test_no_comparable_blocks_cannot_pass(counts):
    with pytest.raises(m.BBMismatchError, match="0 comparable blocks"):
        m.enforce_bb_report(m.compare_bb_counts(elf(), counts))


@pytest.mark.parametrize(
    "options",
    [
        dict(relative_tolerance=-1),
        dict(absolute_tolerance=float("nan")),
        dict(relative_tolerance=float("inf")),
        dict(mismatch_action="ignore"),
    ],
)
def test_invalid_options_fail_before_database_access(monkeypatch, options):
    driver = MagicMock()
    monkeypatch.setattr(m.GraphDatabase, "driver", driver)
    with pytest.raises(ValueError):
        m.annotate_bb_weights(None, **options)
    driver.assert_not_called()


def test_cold_block_mismatch_stops_all_writes_and_keeps_report(monkeypatch):
    sess = SimpleNamespace(
        artifacts=[m.TableArtifact("llvm_bbs_new", elf(10, 100))],
        config=SimpleNamespace(memgraph=None),
        add_artifact=MagicMock(),
    )
    driver = MagicMock()
    monkeypatch.setattr(m.GraphDatabase, "driver", lambda *a, **kw: driver)
    monkeypatch.setattr(m, "graph_bb_counts", lambda *a, **kw: graph(10, 200))
    annotate = MagicMock()
    monkeypatch.setattr(m, "anonotate_helper", annotate)
    with pytest.raises(m.BBMismatchError):
        m.annotate_bb_weights(sess)
    annotate.assert_not_called()
    report = sess.add_artifact.call_args.args[0]
    assert report.name == "bb_cdfg_validation"
    assert report.df.status.tolist() == ["exact", "mismatch"]
    driver.close.assert_called_once()


def test_count_query_uses_session_scope_and_excludes_operand_nodes():
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value = graph(7)
    label = "a'quoted-session"
    assert m.graph_bb_counts(driver, label) == graph(7)
    query = session.run.call_args.args[0]
    assert label not in query.text
    assert session.run.call_args.kwargs == {"label": label}
    assert "OPTIONAL MATCH (i:INSTR)" in query.text
    assert "['input', 'constant', 'label']" in query.text
    assert "i.module_name = b.module_name AND i.stage = b.stage" in query.text


def test_cli_saves_report_when_validation_stops(tmp_path, monkeypatch):
    sess = MagicMock()
    monkeypatch.setattr(m.Session, "from_dir", lambda path: sess)
    monkeypatch.setattr(m, "annotate_bb_weights", MagicMock(side_effect=m.BBMismatchError("mismatch")))
    args = m.get_parser().parse_args(["--session", str(tmp_path), "--label", "test"])
    with pytest.raises(m.BBMismatchError):
        m.handle(args)
    sess.save.assert_called_once()


@pytest.mark.parametrize("action,count", [("error", 11), ("warn", 100)])
def test_tolerated_or_explicit_warning_mode_can_annotate(monkeypatch, action, count):
    blocks = elf(10)
    blocks["freq"] = 3
    blocks["rel_weight"] = 0.25
    sess = SimpleNamespace(
        artifacts=[m.TableArtifact("llvm_bbs_new", blocks)],
        config=SimpleNamespace(memgraph=None),
        add_artifact=MagicMock(),
    )
    monkeypatch.setattr(m.GraphDatabase, "driver", MagicMock())
    monkeypatch.setattr(m, "graph_bb_counts", lambda *a, **kw: graph(count))
    annotate = MagicMock()
    monkeypatch.setattr(m, "anonotate_helper", annotate)
    m.annotate_bb_weights(sess, mismatch_action=action)
    annotate.assert_called_once()
    assert annotate.call_args.kwargs["check"] is False


def test_weight_writes_bind_names_as_parameters_and_consume_results():
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    m.anonotate_helper(driver, 'f"quoted', "%bb.0", 10, 3, 0.2, label="test", check=False)
    assert session.run.call_count == 2
    assert session.run.return_value.consume.call_count == 2
    for call in session.run.call_args_list:
        assert 'f"quoted' not in call.args[0].text
        assert call.kwargs["func_name"] == 'f"quoted'
        assert call.kwargs["instr_weight"] == 0.02
