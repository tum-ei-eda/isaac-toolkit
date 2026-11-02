"""Verify the toolkit forwards candidate options to the query backend."""
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pandas as pd
import pytest

module = importlib.import_module('isaac_toolkit.generate.ise.query_candidates_from_db')


def test_parser_options_and_yaml_delegation():
    parser = module.get_parser()
    defaults = parser.parse_args(['--session', 's'])
    for name in ('query_timeout', 'min_path_length', 'max_path_length', 'max_path_width',
                 'ignore_names', 'ignore_op_types', 'allowed_reg_classes', 'limit_results'):
        assert getattr(defaults, name) is None
    args = parser.parse_args(['--session', 's', '--query-timeout', '2.5', '--max-path-width', '4',
                              '--min-path-length', '0', '--max-path-length', '5',
                              '--ignore-names', 'PHI,COPY', '--ignore-op-types', 'input',
                              '--allowed-reg-classes', '3', '--limit-results', '0'])
    assert (args.query_timeout, args.max_path_width, args.min_path_length, args.max_path_length) == (2.5, 4, 0, 5)
    assert (args.ignore_names, args.ignore_op_types, args.allowed_reg_classes, args.limit_results) == ('PHI,COPY', 'input', 3, 0)


def test_subprocess_receives_options(tmp_path, monkeypatch):
    choices = SimpleNamespace(df=pd.DataFrame([dict(func_name='f', bb_name='%bb.0', rel_weight=1.0, num_instrs=4)]))
    monkeypatch.setattr(module, 'filter_artifacts', lambda *args: [choices])
    monkeypatch.setattr(module, 'memgraph_to_nx', lambda result: nx.DiGraph())
    driver = MagicMock()
    driver.session.return_value.run.return_value.to_df.return_value = pd.DataFrame({'bb_id': [0]})
    monkeypatch.setattr(module.GraphDatabase, 'driver', lambda *args, **kwargs: driver)
    sess = SimpleNamespace(artifacts=[], config=SimpleNamespace(memgraph=None))
    captured = []

    class StopAfterCapture(Exception):
        pass

    def capture(args, check):
        captured.extend(map(str, args))
        assert check
        raise StopAfterCapture
    monkeypatch.setattr(module.subprocess, 'run', capture)
    with pytest.raises(StopAfterCapture):
        module.query_candidates_from_db(
            sess, workdir=tmp_path, label='s', partition_with_maxmiso=False, scale_iso_weight=None,
            query_timeout=2.5, limit_results=0, MIN_PATH_LENGTH=0, MAX_PATH_LENGTH=5, MAX_PATH_WIDTH=4,
            IGNORE_NAMES=['PHI', 'COPY'], IGNORE_OP_TYPES=['input', 'label'], allowed_reg_classes=3,
            ALLOWED_ENC_SIZES=[16, 32],
        )
    for option, value in [('--query-timeout', '2.5'), ('--limit-results', '0'), ('--max-path-width', '4'),
                          ('--min-path-length', '0'), ('--max-path-length', '5'), ('--ignore-names', 'PHI,COPY'),
                          ('--ignore-op-types', 'input,label'), ('--allowed_reg_classes', '3')]:
        assert captured[captured.index(option) + 1] == value
    idx = captured.index('--allowed-enc-sizes')
    assert captured[idx + 1:idx + 3] == ['16', '32']
    for call in driver.session.return_value.run.call_args_list:
        assert call.args[0].timeout == 2.5


def test_lookup_queries_escape_names():
    for make_query in (module.get_func_query, module.get_bbs_query):
        q = make_query('s"\\\n', 8, 'f"\\\n', timeout=3)
        assert q.timeout == 3
        assert '"s\\"\\\\\\n"' in q.text
        assert '"f\\"\\\\\\n"' in q.text


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), True])
def test_invalid_timeout_rejected_before_db_access(timeout):
    with pytest.raises(ValueError, match='query_timeout'):
        module.query_candidates_from_db(None, query_timeout=timeout)


def test_handle_forwards_cli_options(tmp_path, monkeypatch):
    sess = MagicMock()
    monkeypatch.setattr(module.Session, 'from_dir', lambda path: sess)
    captured = {}
    monkeypatch.setattr(module, 'query_candidates_from_db', lambda session, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(module, 'set_log_level', lambda **kwargs: None)
    args = module.get_parser().parse_args([
        '--session', str(tmp_path), '--query-timeout', '4', '--max-path-width', '3',
        '--min-path-length', '1', '--max-path-length', '2', '--ignore-names', 'PHI',
        '--ignore-op-types', 'input', '--allowed-reg-classes', '2',
    ])
    module.handle(args)
    assert captured['query_timeout'] == 4
    assert captured['MAX_PATH_WIDTH'] == 3
    assert captured['MIN_PATH_LENGTH'] == 1
    assert captured['MAX_PATH_LENGTH'] == 2
    assert captured['IGNORE_NAMES'] == 'PHI'
    assert captured['IGNORE_OP_TYPES'] == 'input'
    assert captured['allowed_reg_classes'] == 2
    sess.save.assert_called_once()
