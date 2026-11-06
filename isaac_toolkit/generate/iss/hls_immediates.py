"""Make legacy HLS operand decoding explicit without changing instruction bits."""
from m2isar.metamodel import arch, behav, type_info


def normalize_hls_immediates(instruction_set, xlen):
    """Normalize a private HLS model in place.

    Legacy CoreDSL omits signed operand declarations. Treenail also packs
    encoded field runs densely, dropping implicit low zeros (e.g. B offsets).
    Use zero-based packed fields and reconstruct their signed, scaled values
    in behavior. The fixed encoding code and mask stay identical.
    """
    for instruction in instruction_set.instructions.values():
        replacements = {}
        for name, field in list(instruction.fields.items()):
            parts = [e for e in instruction.encoding if isinstance(e, arch.BitField) and e.name == name]
            if not parts:
                continue
            low = min(e.range.lower for e in parts)
            high = max(e.range.upper for e in parts)
            signed = field.ty.kind == type_info.TypeKind.INT
            if not low and not signed:
                continue
            encoded = {bit for e in parts for bit in range(e.range.lower, e.range.upper + 1)}
            if encoded != set(range(low, high + 1)):
                raise ValueError(f"HLS operand {instruction.name}/{name} has internal encoding holes")
            width = high - low + 1
            if high + 1 != field.ty.size or high + 1 > xlen:
                raise ValueError(f"Unsupported HLS operand width: {instruction.name}/{name}")
            packed_name = name + "_hls_packed"
            if packed_name in instruction.fields:
                raise ValueError(f"HLS operand name collision: {packed_name}")
            packed = arch.BitFieldDescr(packed_name, width, type_info.TypeKind.UINT)
            for part in parts:
                part.name = packed_name
                part.kind = type_info.TypeKind.UINT
                part.ty.kind = type_info.TypeKind.UINT
                part.range = arch.RangeSpec(part.range.upper - low, part.range.lower - low)
            instruction.fields.pop(name)
            instruction.fields[packed_name] = packed
            operand = instruction.operands.pop(name, None)
            if operand is not None:
                operand.name = packed_name
                operand._size = width
                operand.signed = False
                instruction.operands[packed_name] = operand
            # HLS does not consume assembler operands, but keep its placeholders
            # consistent with the renamed encoding field.
            instruction.assembly = (instruction.assembly or "").replace("{" + name + "}", "{" + packed_name + "}")
            kind = type_info.TypeKind.INT if signed else type_info.TypeKind.UINT
            replacements[id(field)] = (packed, kind, width, low)

        def rewrite(node):
            if isinstance(node, behav.IndexedReference) and id(node.reference) in replacements:
                raise ValueError(f"Indexed HLS immediate is unsupported: {instruction.name}")
            if isinstance(node, behav.NamedReference) and id(node.reference) in replacements:
                packed, kind, width, low = replacements[id(node.reference)]
                value = behav.TypeConv(kind, width, behav.NamedReference(packed))
                value = behav.TypeConv(kind, xlen, value)
                if low:
                    value = behav.BinaryOperation(value, behav.Operator("<<"), behav.Literal(low))
                return value
            if isinstance(node, behav.BaseNode):
                for key, value in list(vars(node).items()):
                    if isinstance(value, behav.BaseNode):
                        setattr(node, key, rewrite(value))
                    elif isinstance(value, list):
                        setattr(node, key, [rewrite(v) if isinstance(v, behav.BaseNode) else v for v in value])
                    elif isinstance(value, tuple):
                        setattr(node, key, tuple(rewrite(v) if isinstance(v, behav.BaseNode) else v for v in value))
            return node

        instruction.operation = rewrite(instruction.operation)
        # Recalculate size/code/mask, preserving the explicitly packed fields.
        old_code, old_mask = instruction.code, instruction.mask
        instruction.process_encoding()
        if instruction.code != old_code or instruction.mask != old_mask:
            raise ValueError(f"HLS operand normalization changed encoding: {instruction.name}")
