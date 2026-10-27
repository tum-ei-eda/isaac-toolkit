"""Exercise the default generator with real base ISAs and multiple custom sets."""
import json
from pathlib import Path
import pickle
import pytest
import yaml
from isaac_toolkit.generate.iss.generate_etiss_core import generate_etiss_core
from m2isar.transforms.encode_instructions.planner import overlaps

BASE = Path(__file__).resolve().parents[4] / "etiss_arch_riscv"
pytestmark = pytest.mark.skipif(not BASE.is_dir(), reason="requires sibling etiss_arch_riscv")


@pytest.mark.parametrize("xlen,compressed", [(32, False), (32, True), (64, False)])
def test_default_typed_core_with_shared_encoding_domain(tmp_path, xlen, compressed):
    indices = []
    for set_name in ("Alpha", "Beta"):
        source = tmp_path / f"{set_name}.core_desc"
        source.write_text(f'''import "{BASE / 'rv_base/RVI.core_desc'}"
InstructionSet Candidate extends RV{xlen}I {{
 instructions {{ ADD {{
  operands: {{
   unsigned<5> rd [[out]] [[is_reg]];
   unsigned<5> rs1 [[in]] [[is_reg]];
   unsigned<5> rs2 [[in]] [[is_reg]];
  }}
  encoding: auto;
  assembly: {{"add", "{{rd}}, {{rs1}}, {{rs2}}"}};
  behavior: {{ X[rd] = X[rs1] + X[rs2]; }}
 }} }}
}}
''')
        index = tmp_path / f"{set_name}.yml"
        index.write_text(yaml.safe_dump({"candidates": [{"artifacts": {"cdsl": str(source)},
                                                        "properties": {"InstrName": "ADD"}}]}))
        indices.append(index)
    generate_etiss_core(None, workdir=tmp_path, index_files=indices, set_names=["Alpha", "Beta"],
                        core_name="TypedCore", xlen=xlen, skip_errors=False, n_parallel=1, add_mnemonic_prefix=True,
                        base_extensions=["i", "zicsr"] + (["c"] if compressed else []),
                        base_dir=str(BASE / "rv_base"), tum_dir=str(BASE))
    gen = tmp_path / "gen"
    model = pickle.loads((gen / "TypedCore.m2isarmodel").read_bytes())
    core = model.cores["TypedCore"]
    assert core.instr_classes == ({16, 32} if compressed else {32})
    assert core.main_reg_file.ty.element_type.size == xlen
    assert core.pc_memory.ty.size == xlen
    custom = [instruction for instruction in core.instructions.values() if instruction.ext_name in ("Alpha", "Beta")]
    assert len(custom) == 2
    assert not overlaps((custom[0].code, custom[0].mask), (custom[1].code, custom[1].mask))
    assert len(json.loads((gen / "encoding_plan.json").read_text())["assignments"]) == 2
    for index in indices:
        candidate = yaml.safe_load(index.read_text())["candidates"][0]
        assert Path(candidate["artifacts"]["cdsl_encoded"]).is_file()
