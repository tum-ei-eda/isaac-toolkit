"""Exercise actual CoreDSL parsing/writing and persisted CFU assignments."""

import json
from pathlib import Path

import pytest

pytest.importorskip("m2isar")
from m2isar.backends.coredsl2_set.writer import gen_cdsl_code
from m2isar.frontends.coredsl2_set.parser import parse_cdsl2_set
from m2isar.metamodel import M2Model, M2_METAMODEL_VERSION
from isaac_toolkit.generate.iss.cfu import CFUEncoder, validate_candidate


BASE = Path(__file__).resolve().parents[4] / "etiss_arch_riscv" / "rv_base"
needs_arch_sources = pytest.mark.skipif(not BASE.is_dir(), reason="requires sibling etiss_arch_riscv sources")


def parse(tmp_path, name="ADD_SHIFT", unary=False, immediate=False):
    src = tmp_path / f"{name}.core_desc"
    second = ('unsigned<5> imm [[is_imm]] [[in]];' if immediate else
              'unsigned<5> rs2 [[is_reg]] [[reg_class="GPR"]] [[reg_type="s32"]] [[in]];')
    if unary:
        second = ""
    rhs = "X[rs1] << 1" if unary else ("X[rs1] + imm" if immediate else "(X[rs1] + X[rs2]) << 1")
    asm = "{name(rd)}, {name(rs1)}" + ("" if unary else (", {imm}" if immediate else ", {name(rs2)}"))
    src.write_text(f'''import "RVI.core_desc"
InstructionSet XTest extends RV32I {{ instructions {{ {name} {{
operands: {{
unsigned<5> rd [[is_reg]] [[reg_class="GPR"]] [[reg_type="s32"]] [[out]];
unsigned<5> rs1 [[is_reg]] [[reg_class="GPR"]] [[reg_type="s32"]] [[in]];
{second}
}}
encoding: auto;
assembly: {{"{name.lower()}", "{asm}"}};
behavior: {{ X[rd] = {rhs}; }}
}} }} }}''')
    model = parse_cdsl2_set(src, [str(BASE)])
    instr_set = model["XTest"]
    return src, instr_set.unencoded_instructions[name]


@needs_arch_sources
def test_roundtrip_and_stable_filtered_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    enc = CFUEncoder(manifest)
    source, instr = parse(tmp_path)
    enc.encode(instr, "XTest", source)
    unary_source, unary = parse(tmp_path, "SHIFT", unary=True)
    enc.encode(unary, "XTest", unary_source)
    enc.save()
    assert instr.code == 0x0B and instr.mask == 0xFE00707F
    assert unary.code == 0x100B and unary.mask == 0xFFF0707F
    from m2isar.metamodel import arch
    s = arch.InstructionSet("XTest", ["RV32I"], {}, {}, {}, {(instr.code, instr.mask): instr}, {})
    path = tmp_path / "encoded.core_desc"
    path.write_text('import "RVI.core_desc"\n' + gen_cdsl_code(
        M2Model(M2_METAMODEL_VERSION, {}, {"XTest": s}, {}), with_includes=False))
    parsed = parse_cdsl2_set(path, [str(BASE)])["XTest"]
    roundtrip = next(iter(parsed.instructions.values()))
    assert (roundtrip.code, roundtrip.mask) == (instr.code, instr.mask)
    # A filtered list containing only the second operation keeps its original ID.
    again = CFUEncoder(manifest, allow_new=False)
    _, unary2 = parse(tmp_path, "SHIFT", unary=True)
    unary_source.write_text("// DateTime: changed, Sub: 0\n" + unary_source.read_text())
    again.encode(unary2, "XTest", unary_source)
    assert unary2.code == unary.code
    assert json.loads(manifest.read_text())["instructions"]["XTest.SHIFT"]["function_id"] == 1
    new_source, new_instr = parse(tmp_path, "UNKNOWN")
    with pytest.raises(ValueError, match="No existing"):
        again.encode(new_instr, "XTest", new_source)


@needs_arch_sources
def test_reject_immediate_and_changed_semantics(tmp_path):
    enc = CFUEncoder(tmp_path / "manifest.json")
    source, instr = parse(tmp_path, immediate=True)
    with pytest.raises(ValueError, match="immediates"):
        enc.encode(instr, "XTest", source)
    source, instr = parse(tmp_path)
    enc.encode(instr, "XTest", source)
    source.write_text(source.read_text().replace("<< 1", "<< 2"))
    changed = parse_cdsl2_set(source, [str(BASE)])["XTest"].unencoded_instructions["ADD_SHIFT"]
    with pytest.raises(ValueError, match="changed"):
        enc.encode(changed, "XTest", source)


@needs_arch_sources
def test_reject_corrupt_or_exhausted_namespace(tmp_path):
    manifest = tmp_path / "manifest.json"
    data = {"schema": 1, "policy": "cfu", "opcode": 11, "xlen": 32,
            "instructions": {str(i): {"function_id": i, "source_sha256": "a" * 64,
                "operands": {"rd": "rd", "rs1": "rs1", "rs2": "rs2"},
                "code": ((i >> 3) << 25) | ((i & 7) << 12) | 11, "mask": 0xFE00707F}
                for i in range(1024)}}
    manifest.write_text(json.dumps(data))
    enc = CFUEncoder(manifest)
    source, instr = parse(tmp_path)
    with pytest.raises(ValueError, match="exhausted"):
        enc.encode(instr, "XTest", source)
    data["instructions"]["1"]["function_id"] = 0
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="colliding"):
        CFUEncoder(manifest)


@needs_arch_sources
def test_function_id_bit_mapping(tmp_path):
    enc = CFUEncoder(tmp_path / "manifest.json")
    for i in range(10):
        source, instr = parse(tmp_path, f"OP{i}")
        enc.encode(instr, "XTest", source)
        assert (instr.code & 0x7F) == 0x0B
        assert ((instr.code >> 25) << 3) | ((instr.code >> 12) & 7) == i
    enc.save()
    data = json.loads(enc.path.read_text())
    data["instructions"]["XTest.OP0"]["code"] ^= 0x20
    enc.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Malformed"):
        CFUEncoder(enc.path)


def test_candidate_contract():
    p = {"OperandNames": ["rd", "rs1"], "OperandDirs": ["OUT", "IN"],
         "OperandTypes": ["REG", "REG"], "OperandRegClasses": ["GPR", "GPR"],
         "OperandEncBits": [5, 5], **{f: 0 for f in
         ("#Loads", "#Stores", "#Mems", "#Branches", "#Terminators")}}
    validate_candidate(p)
    for field, bad in (("OperandDirs", ["INOUT", "IN"]), ("OperandTypes", ["REG", "IMM"]),
                       ("OperandRegClasses", ["GPR", "FPR"]), ("#Stores", 1)):
        with pytest.raises(ValueError):
            validate_candidate({**p, field: bad})


@needs_arch_sources
def test_core_generation_and_artifact_drift(tmp_path):
    """Verify the production writer outputs, then detect a split-set mutation."""
    from types import ModuleType
    import yaml
    from isaac_toolkit.generate.iss.generate_etiss_core import generate_etiss_core

    source, _ = parse(tmp_path)
    props = {"InstrName": "ADD_SHIFT", "EncodingPolicy": "cfu",
             "OperandNames": ["rd", "rs1", "rs2"], "OperandDirs": ["OUT", "IN", "IN"],
             "OperandTypes": ["REG"] * 3, "OperandRegClasses": ["GPR"] * 3,
             "OperandEncBits": [5] * 3, **{f: 0 for f in
             ("#Loads", "#Stores", "#Mems", "#Branches", "#Terminators")}}
    index = tmp_path / "combined_index.yml"
    index.write_text(yaml.safe_dump({"candidates": [{"properties": props, "artifacts": {"cdsl": str(source)}}]}))
    gen = tmp_path / "gen"
    generate_etiss_core(None, workdir=tmp_path, gen_dir=gen, index_files=[index], set_names=["XTest"],
                        base_dir=str(BASE), tum_dir=str(BASE.parent), skip_errors=False,
                        base_extensions=["i", "m", "c", "zicsr", "zifencei"],
                        encoding_policy="cfu", n_parallel=1, extra_includes=[str(BASE)])
    verifier_path = BASE.parents[1] / "scripts" / "verify_cfu_artifacts.py"
    if not verifier_path.is_file():
        pytest.skip("requires ISAAC Demo artifact verifier")
    verifier = ModuleType("verify_cfu_artifacts")
    exec(compile(verifier_path.read_text(), str(verifier_path), "exec"), verifier.__dict__)
    report = verifier.verify(index, gen, tmp_path / "cfu_manifest.json", "XTest", BASE)
    assert report["active_instruction_count"] == 1
    # Filtered/selected flow stages use --no-auto-encoding; replay the ledger.
    generate_etiss_core(None, workdir=tmp_path, gen_dir=gen, index_files=[index], set_names=["XTest"],
                        base_dir=str(BASE), tum_dir=str(BASE.parent), skip_errors=False,
                        base_extensions=["i", "m", "c", "zicsr", "zifencei"],
                        encoding_policy="cfu", n_parallel=1, extra_includes=[str(BASE)], auto_encoding=False)
    assert verifier.verify(index, gen, tmp_path / "cfu_manifest.json", "XTest", BASE) == report
    split = gen / "XTest.splitted.core_desc"
    split.write_text(split.read_text().replace("7'b0001011", "7'b0101011"))
    with pytest.raises(ValueError, match="split CoreDSL"):
        verifier.verify(index, gen, tmp_path / "cfu_manifest.json", "XTest", BASE)


def test_query_operand_filter_is_opt_in():
    """CFU rejection happens before candidate ranking, without changing legacy mode."""
    from copy import deepcopy
    from types import SimpleNamespace
    import pandas as pd
    stage = pytest.importorskip("tool.stages.filter_subs_operands")
    from tool.enums import ExportFilter

    good = {"OperandNames": ["rd", "rs1", "rs2"], "OperandDirs": ["OUT", "IN", "IN"],
            "OperandTypes": ["REG"] * 3, "OperandRegClasses": ["GPR"] * 3,
            "OperandEncBits": [5] * 3, "Status": ExportFilter.SELECTED.value,
            **{f: 0 for f in ("#Loads", "#Stores", "#Mems", "#Branches", "#Terminators")}}
    rows = [deepcopy(good) for _ in range(5)]
    rows[1]["OperandTypes"][2] = "IMM"
    rows[2]["OperandDirs"][0] = "INOUT"
    rows[3]["OperandRegClasses"][1] = "FPR"
    rows[4]["#Loads"] = 1
    settings = SimpleNamespace(progress=False,
        write=SimpleNamespace(gen_flt=ExportFilter.SELECTED),
        filters=SimpleNamespace(cfu=True, max_in_operands=99, max_out_operands=99, max_inout_operands=99))
    df = pd.DataFrame(deepcopy(rows))
    stage.filter_subs_operands(settings, [None] * len(rows), df)
    assert list(df.Status) == [ExportFilter.SELECTED.value] + [ExportFilter.FILTERED_OPERANDS.value] * 4
    settings.filters.cfu = False
    df = pd.DataFrame(rows)
    stage.filter_subs_operands(settings, [None] * len(rows), df)
    assert list(df.Status) == [ExportFilter.SELECTED.value] * len(rows)


def test_cdsl_emitters_guard_gpr_writeback_only():
    cdsl = pytest.importorskip("tool.gen.cdsl_utils")
    from tool.gen.flat_utils import FlatCodeEmitter
    from tool.gen.tree_utils import Assignment, Declaration, Ref, Register, Statements

    for emitter in (cdsl.CDSLEmitter(32), FlatCodeEmitter(guard_gpr_writes=True)):
        # Output computation precedes the guard, so effects and computations in
        # earlier statements are not skipped when rd=0. FPR0 remains writable.
        tree = Statements(children=[
            Declaration("unsigned<32>", children=[Ref("outp0"), Ref("computed")]),
            Assignment(children=[Register("dest", "gpr", children=[Ref("rd")]), Ref("outp0")]),
            Assignment(children=[Register("dest2", "gpr", children=[Ref("rd2")]), Ref("outp0")]),
            Assignment(children=[Register("fpdest", "fpr", children=[Ref("fd")]), Ref("outp0")]),
        ])
        emitter.visit(tree)
        assert "unsigned<32> outp0=computed;" in emitter.output
        assert "if (rd != 0) { X[rd]=outp0; }" in emitter.output
        assert "if (rd2 != 0) { X[rd2]=outp0; }" in emitter.output
        assert emitter.output.count("if (") == 2
        assert emitter.output.index("outp0=computed") < emitter.output.index("if (rd != 0)")
    intermediate = FlatCodeEmitter()
    intermediate.visit(tree)
    assert "if (" not in intermediate.output
