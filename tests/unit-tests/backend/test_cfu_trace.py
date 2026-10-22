import json

import pytest

pytest.importorskip("m2isar")
from isaac_toolkit.frontend.instr_trace.cfu import read_trace


def test_retirement_trace_checks_words_and_order(tmp_path):
    dump = tmp_path / "firmware.dump"
    dump.write_text("Disassembly of section .text:\n 0: 00b5060b .4byte 0xb5060b\n 4: 00000013 nop\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "policy": "cfu", "opcode": 11, "xlen": 32,
        "instructions": {"XIsaac.CUSTOM0": {"function_id": 0, "code": 11, "mask": 0xfe00707f,
            "operands": {"rd": "rd", "rs1": "rs1", "rs2": "rs2"}, "source_sha256": "a" * 64}}}))
    trace = tmp_path / "trace.csv"
    header = "cycle,pc,bytecode\n"
    trace.write_text(header + "10,0x0,0x00b5060b\n12,0x4,0x00000013\n")
    result = read_trace(trace, dump, manifest, chunk_size=1)
    assert result.instr.tolist() == ["custom0", "nop"]
    assert result.cycle.tolist() == [10, 12]
    for bad in ["10,0x0,0x00b5060b\n10,0x4,0x00000013\n",
                "10,0x0,0x00b5060b\n12,0x4,0x00000093\n",
                "10,0x8,0x00000013\n"]:
        trace.write_text(header + bad)
        with pytest.raises(ValueError):
            read_trace(trace, dump, manifest, chunk_size=1)


def test_baseline_branch_names_match_analysis(tmp_path):
    from isaac_toolkit.arch.riscv import riscv_branch_instrs, riscv_return_instrs
    dump = tmp_path / "firmware.dump"
    dump.write_text(" 0:\t0080006f\tj 8\n 8:\t00008067\tret\n")
    trace = tmp_path / "trace.csv"
    trace.write_text("cycle,pc,bytecode\n1,0x0,0x0080006f\n2,0x8,0x00008067\n")
    frame = read_trace(trace, dump)
    assert frame.instr.iloc[0] in riscv_branch_instrs
    assert frame.instr.iloc[1] in riscv_return_instrs


def test_cfu_discovery_load_dispatch_avoids_mlonmcu(tmp_path, monkeypatch):
    from isaac_toolkit.flow.demo.stage import load
    from isaac_toolkit.frontend.instr_trace import cfu
    metadata = {"schema": 1, "runner": "verilator", "start_func": "benchmark",
                "end_func": "cfu_benchmark_end"}
    (tmp_path / "cfu_discovery.json").write_text(json.dumps(metadata))
    calls = []
    def unexpected(*args, **kwargs):
        pytest.fail("CFU discovery must not use the MLonMCU/ETISS loader")
    monkeypatch.setattr(load, "load_mlonmcu_exported_run", unexpected)
    for name in ("load_elf", "load_linker_map", "load_disass", "load_compile_commands_json"):
        monkeypatch.setattr(load, name, lambda sess, path, **kwargs: calls.append(path.name))
    def trace_loader(sess, trace, disassembly, **kwargs):
        assert kwargs["start_func"] == "benchmark"
        assert kwargs["end_func"] == "cfu_benchmark_end"
        calls.append(trace.name)
    monkeypatch.setattr(cfu, "load_instr_trace", trace_loader)
    load.load_artifacts(object(), tmp_path, force=True)
    assert set(calls) == {"firmware.elf", "firmware.map", "firmware.dump", "instruction_trace.csv", "compile_commands.json"}
    metadata["runner"] = "unknown"
    (tmp_path / "cfu_discovery.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported CFU discovery"):
        load.load_artifacts(object(), tmp_path)
