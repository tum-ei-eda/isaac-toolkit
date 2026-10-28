from pathlib import Path
import pandas as pd
import pytest
from isaac_toolkit.frontend.instr_trace.spike import parse_and_process, load_instr_trace


TRACE = b'''Spike startup
core   0: 0x0000000080000000 (0x00000001) unknown
core   0: 3 0x0000000080000000 (0x00000001) x1 0x10
core   1: 0x0000000100000004 (0x003100b3) add ra, sp, gp
core   0: exception trap_illegal_instruction, epc 0x80000008
core   0: tval 0x0
core   0: 0x0000000080000008 (0x00008067) ret
'''


def test_parse_spike_disassembly_without_double_counting_commit_log():
    df = parse_and_process(TRACE)
    assert list(df.pc) == [0x80000000, 0x100000004, 0x80000008]
    assert list(df['size']) == [2, 4, 4]
    assert list(df.instr) == ['unknown', 'add', 'ret']
    assert list(df.core) == [0, 1, 0]


@pytest.mark.parametrize('executor', ['thread_pool', 'process_pool'])
@pytest.mark.parametrize('content', [TRACE, b'', b'noise only\n'])
def test_load_spike_without_optional_operands(tmp_path, executor, content):
    path = tmp_path / 'spike_bm_instrs.log'
    path.write_bytes(content)
    class Session:
        def __init__(self): self.artifacts = []
        def add_artifact(self, artifact, **kwargs): self.artifacts.append(artifact)
    sess = Session()
    load_instr_trace(sess, path, operands=True, executor=executor, num_workers=1, chunk_size=19)
    assert len(sess.artifacts) == 1
    df = sess.artifacts[0].df
    assert len(df) == (3 if content == TRACE else 0)
    assert 'is_trap' not in df  # High HTIF addresses are normal program code.
    assert 'operands' not in df


@pytest.mark.parametrize('target', ['spike', 'spike_pk', 'spike_rv32', 'spike_rv64', 'spike_rv32_min'])
def test_proxy_kernel_addresses_are_marked_as_traps(tmp_path, target):
    path = tmp_path / f'{target}_instrs.log'
    path.write_text('core 0: 0x7ffffffc (0x00008067) ret\n'
                    'core 0: 0x80000000 (0x00008067) ret\n'
                    'core 0: 0x100000004 (0x00008067) ret\n')

    class Session:
        def add_artifact(self, artifact, **kwargs):
            self.artifact = artifact

    sess = Session()
    load_instr_trace(sess, path, executor='thread_pool', num_workers=1)
    assert list(sess.artifact.df.is_trap) == [False, True, True]
    load_instr_trace(sess, path, executor='thread_pool', num_workers=1, proxy_kernel=False)
    assert 'is_trap' not in sess.artifact.df


def test_proxy_kernel_mode_can_be_explicit_for_custom_trace_name(tmp_path):
    path = tmp_path / 'trace.log'
    path.write_bytes(TRACE)

    class Session:
        def add_artifact(self, artifact, **kwargs):
            self.artifact = artifact

    sess = Session()
    load_instr_trace(sess, path, executor='thread_pool', num_workers=1, proxy_kernel=True)
    assert sess.artifact.df.is_trap.all()
