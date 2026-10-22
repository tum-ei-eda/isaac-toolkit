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
    assert result.instr.tolist() == ["CUSTOM0", "NOP"]
    assert result.cycle.tolist() == [10, 12]
    for bad in ["10,0x0,0x00b5060b\n10,0x4,0x00000013\n",
                "10,0x0,0x00b5060b\n12,0x4,0x00000093\n",
                "10,0x8,0x00000013\n"]:
        trace.write_text(header + bad)
        with pytest.raises(ValueError):
            read_trace(trace, dump, manifest, chunk_size=1)
