"""Adapt RISC-V writeback to cores that suppress x0 writes themselves."""
from m2isar.metamodel import arch, attribute_info, behav


def normalize_hls_zero_write_guards(instruction_set):
    """Remove only canonical x0 guards in a private HLS model.

    The caller must establish that the core discards writes to register zero.
    Otherwise predicated writeback requires reading the old destination value,
    which introduces an RdRD port even for instructions without reuse-I/O.
    Genuine destination-register reads are rejected rather than discarded.
    """
    changed = []

    def ungroup(node):
        while isinstance(node, behav.Group):
            node = node.expr
        return node

    def rewrite(node, instruction):
        if isinstance(node, behav.Conditional) and len(node.conds) == len(node.stmts) == 1:
            condition = ungroup(node.conds[0])
            block = node.stmts[0]
            statements = block.statements if isinstance(block, behav.Operation) else [block]
            if len(statements) == 1 and isinstance(statements[0], behav.Assignment):
                target = statements[0].target
                if (isinstance(target, behav.IndexedReference)
                        and isinstance(target.reference, arch.RegisterBank)
                        and attribute_info.RegisterAttribute.IS_MAIN_REG in target.reference.attributes
                        and target.right is None):
                    index = ungroup(target.index)
                    if (isinstance(index, behav.NamedReference)
                            and isinstance(index.reference, arch.BitFieldDescr)
                            and isinstance(condition, behav.BinaryOperation)
                            and condition.op.value == "!="):
                        left, right = ungroup(condition.left), ungroup(condition.right)
                        if (isinstance(left, behav.NamedReference)
                                and left.reference is index.reference
                                and isinstance(right, behav.Literal) and int(right.value) == 0):
                            changed.append(instruction.name)
                            return rewrite(block, instruction)
        if isinstance(node, behav.BaseNode):
            for key, value in list(vars(node).items()):
                if isinstance(value, behav.BaseNode):
                    setattr(node, key, rewrite(value, instruction))
                elif isinstance(value, list):
                    setattr(node, key, [rewrite(v, instruction) for v in value])
        return node

    def check_reads(node, output_fields, instruction):
        if isinstance(node, behav.Assignment):
            check_reads(node.expr, output_fields, instruction)
            return
        if isinstance(node, behav.IndexedReference) and isinstance(node.reference, arch.RegisterBank):
            index = ungroup(node.index)
            if isinstance(index, behav.NamedReference) and index.reference.name in output_fields:
                raise ValueError(f"HLS instruction {instruction.name} genuinely reads its destination")
        if isinstance(node, behav.BaseNode):
            for value in vars(node).values():
                if isinstance(value, behav.BaseNode):
                    check_reads(value, output_fields, instruction)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        check_reads(child, output_fields, instruction)

    for instruction in instruction_set.instructions.values():
        outputs = {name for name, operand in instruction.operands.items()
                   if attribute_info.OperandAttribute.IS_REG in operand.attributes
                   and (attribute_info.OperandAttribute.OUT in operand.attributes
                        or attribute_info.OperandAttribute.INOUT in operand.attributes)}
        instruction.operation = rewrite(instruction.operation, instruction)
        check_reads(instruction.operation, outputs, instruction)
    return changed
