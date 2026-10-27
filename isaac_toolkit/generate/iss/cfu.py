"""CFU-Playground's stateless RV32 R-type encoding contract."""

import hashlib
import json
from pathlib import Path

from m2isar.metamodel import arch, attribute_info, type_info


def reg_bitfield(name):
    return arch.BitField(name, arch.RangeSpec(4, 0), type_info.TypeKind.UINT)


OPCODE = 0x0B
CAPACITY = 1024
MASK = 0xFE00707F


def validate_candidate(properties):
    """Validate metadata after all operand/constant transformations."""
    names = properties["OperandNames"]
    dirs = properties["OperandDirs"]
    kinds = properties["OperandTypes"]
    classes = properties["OperandRegClasses"]
    widths = properties["OperandEncBits"]
    if not len(names) == len(dirs) == len(kinds) == len(classes) == len(widths):
        raise ValueError("CFU operand metadata lengths disagree")
    if len(set(names)) != len(names):
        raise ValueError("CFU operand names must be unique")
    if dirs.count("OUT") != 1 or dirs.count("IN") > 2 or any(d not in ("IN", "OUT") for d in dirs):
        raise ValueError("CFU requires at most two inputs and exactly one output; no tied INOUT operands")
    if any(k != "REG" for k in kinds) or any(c != "GPR" for c in classes):
        raise ValueError("CFU operands must be GPRs, without encoded immediates")
    if any(w != 5 for w in widths):
        raise ValueError("CFU register indices must occupy five bits")
    for field in ("#Loads", "#Stores", "#Mems", "#Branches", "#Terminators"):
        if properties[field] != 0:
            raise ValueError(f"CFU operation has forbidden effects: {field}")


def operands(instruction):
    inputs, outputs = [], []
    for name, operand in instruction.operands.items():
        attrs = operand.attributes
        if attribute_info.OperandAttribute.IS_REG not in attrs or attribute_info.OperandAttribute.IS_IMM in attrs:
            raise ValueError(f"{instruction.name}: CFU does not support instruction immediates")
        reg_class = [getattr(value, "value", value) for value in attrs.get(attribute_info.OperandAttribute.REG_CLASS, [])]
        reg_type = [getattr(value, "value", value) for value in attrs.get(attribute_info.OperandAttribute.REG_TYPE, [])]
        if operand.size != 5 or reg_class != ["GPR"] or reg_type not in (["s32"], ["i32"], ["u32"]):
            raise ValueError(f"{instruction.name}: CFU requires five-bit GPR indices")
        directions = [d for d in (attribute_info.OperandAttribute.IN, attribute_info.OperandAttribute.OUT,
                                  attribute_info.OperandAttribute.INOUT) if d in attrs]
        if directions == [attribute_info.OperandAttribute.IN]:
            inputs.append(name)
        elif directions == [attribute_info.OperandAttribute.OUT]:
            outputs.append(name)
        else:
            raise ValueError(f"{instruction.name}: ambiguous or tied operand direction")
    if len(inputs) > 2 or len(outputs) != 1:
        raise ValueError(f"{instruction.name}: CFU requires <=2 inputs and one output")
    return inputs, outputs[0]


class CFUEncoder:
    """Allocate once per run; filtered sets must reuse the original assignments."""

    def __init__(self, manifest, allow_new=True):
        self.path = Path(manifest)
        self.allow_new = allow_new
        if not allow_new and not self.path.is_file():
            raise ValueError("Reusing CFU encodings requires an existing manifest")
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data.get("schema") != 1 or self.data.get("policy") != "cfu":
                raise ValueError("Unsupported CFU manifest")
        else:
            self.data = {"schema": 1, "policy": "cfu", "opcode": OPCODE, "xlen": 32, "instructions": {}}
        entries = self.data["instructions"]
        ids = [entry["function_id"] for entry in entries.values()]
        if (self.data.get("opcode") != OPCODE or self.data.get("xlen") != 32
                or any(type(i) is not int or not 0 <= i < CAPACITY for i in ids) or len(ids) != len(set(ids))):
            raise ValueError("Invalid or colliding CFU function IDs")
        for key, entry in entries.items():
            mapping = entry.get("operands", {})
            digest = entry.get("source_sha256", "")
            if (set(mapping) != {"rd", "rs1", "rs2"} or not isinstance(mapping["rd"], str)
                    or not mapping["rd"] or any(v is not None and (not isinstance(v, str) or not v)
                                               for v in mapping.values())
                    or (mapping["rs1"] is None and mapping["rs2"] is not None)
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError(f"Malformed persisted CFU instruction: {key}")
            function_id = entry["function_id"]
            mask = MASK | (0x01F00000 if mapping["rs2"] is None else 0)
            mask |= 0x000F8000 if mapping["rs1"] is None else 0
            code = ((function_id >> 3) << 25) | ((function_id & 7) << 12) | OPCODE
            if entry.get("code") != code or entry.get("mask") != mask:
                raise ValueError(f"Malformed persisted CFU encoding: {key}")

    def encode(self, instruction, set_name, source):
        if instruction.has_encoding:
            raise ValueError("CFU allocation requires unencoded source instructions")
        inputs, output = operands(instruction)
        key = f"{set_name}.{instruction.name}"
        # Generation adds a comment with the current candidate-list index.
        # Filtering changes that index without changing instruction semantics.
        code = "\n".join(line for line in Path(source).read_text().splitlines()
                         if not line.lstrip().startswith("//"))
        digest = hashlib.sha256(code.encode()).hexdigest()
        mapping = {"rd": output, "rs1": inputs[0] if inputs else None,
                   "rs2": inputs[1] if len(inputs) == 2 else None}
        entries = self.data["instructions"]
        entry = entries.get(key)
        if entry is not None:
            if entry["source_sha256"] != digest or entry["operands"] != mapping:
                raise ValueError(f"CFU instruction changed after allocation: {key}")
            function_id = entry["function_id"]
        else:
            if not self.allow_new:
                raise ValueError(f"No existing CFU assignment: {key}")
            used = {e["function_id"] for e in entries.values()}
            function_id = next((i for i in range(CAPACITY) if i not in used), None)
            if function_id is None:
                raise ValueError("CFU CUSTOM0 function-ID namespace exhausted")
        instruction.encoding = [
            arch.BitVal(7, function_id >> 3),
            reg_bitfield(mapping["rs2"]) if mapping["rs2"] else arch.BitVal(5, 0),
            reg_bitfield(mapping["rs1"]) if mapping["rs1"] else arch.BitVal(5, 0),
            arch.BitVal(3, function_id & 7), reg_bitfield(output), arch.BitVal(7, OPCODE),
        ]
        instruction.process_encoding()
        expected_mask = MASK | (0x01F00000 if mapping["rs2"] is None else 0)
        expected_mask |= 0x000F8000 if mapping["rs1"] is None else 0
        expected_code = ((function_id >> 3) << 25) | ((function_id & 7) << 12) | OPCODE
        if instruction.mask != expected_mask or instruction.code != expected_code:
            raise ValueError(f"CFU encoding does not match contract: {key}")
        result = {"function_id": function_id, "source_sha256": digest, "operands": mapping,
                  "code": expected_code, "mask": expected_mask}
        if entry is not None and entry != result:
            raise ValueError(f"Malformed persisted CFU encoding: {key}")
        entries[key] = result
        return instruction

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)
