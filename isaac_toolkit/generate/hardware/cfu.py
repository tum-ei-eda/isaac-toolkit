"""Lower pure RV32 candidate trees to a CFU-Playground SystemVerilog module.

The first lowering supports integer arithmetic/logic and shifts. Unsupported
operations fail the export; this is not a replacement for a scheduling HLS tool.
Candidate pickle files must come from the trusted local generation flow.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re

import yaml

from isaac_toolkit.generate.iss.cfu import CFUEncoder, validate_candidate
from tool.gen.tree_utils import Assignment, Cast, Constant, Declaration, Operation, Ref, Register, Statements


def dtype(text):
    match = re.fullmatch(r"(unsigned|signed)(?:<(\d+)>)?", text)
    if not match:
        raise ValueError(f"Unsupported integer type: {text}")
    width = int(match[2] or 32)
    if not 1 <= width <= 64:
        raise ValueError(f"Unsupported integer width: {width}")
    return width, match[1] == "signed"


class Datapath:
    def __init__(self, mapping):
        self.mapping = mapping
        self.names = {}
        self.lines = []
        self.result = None

    def wire(self, expression, type_name):
        width, signed = dtype(type_name)
        name = f"v{len(self.lines)}"
        self.lines.append(f"  wire {'signed ' if signed else ''}[{width-1}:0] {name} = {expression};")
        return name

    def expression(self, node):
        if isinstance(node, Ref):
            if node.name not in self.names:
                raise ValueError(f"Undefined reference: {node.name}")
            return self.names[node.name]
        if isinstance(node, Constant):
            if type(node.value) is not int:
                raise ValueError("CFU requires integer constants")
            return self.wire(str(node.value), node.out_types[0])
        if isinstance(node, Register):
            if node.reg_class != "gpr" or len(node.children) != 1 or not isinstance(node.children[0], Ref):
                raise ValueError("Unsupported register read")
            operand = node.children[0].name
            for port, field in enumerate(("rs1", "rs2")):
                if self.mapping[field] == operand:
                    return f"cmd_payload_inputs_{port}"
            raise ValueError(f"Register is not a mapped CFU input: {operand}")
        if isinstance(node, Cast):
            value = self.expression(node.children[0])
            # The wire assignment extends according to the source signedness,
            # then truncates to the explicit destination width.
            return self.wire(value, node.to)
        if isinstance(node, Operation):
            if len(node.children) != 2:
                raise ValueError(f"Unsupported operation arity: {node.name}")
            lhs, rhs = [self.expression(child) for child in node.children]
            binary = {"ADD": "+", "ADDI": "+", "SUB": "-", "MUL": "*",
                      "AND": "&", "ANDI": "&", "OR": "|", "ORI": "|",
                      "XOR": "^", "XORI": "^"}
            if node.name in binary:
                value = f"({lhs} {binary[node.name]} {rhs})"
            elif node.name in ("SLL", "SLLI", "SRL", "SRLI", "SRA", "SRAI"):
                # These are RV32 instruction operations, not generic LLVM shifts.
                operator = "<<" if node.name.startswith("SLL") else ">>"
                if node.name.startswith("SRA"):
                    lhs, operator = f"$signed(32'({lhs}))", ">>>"
                else:
                    lhs = f"$unsigned(32'({lhs}))"
                value = f"({lhs} {operator} ({rhs} & 32'd31))"
            else:
                raise ValueError(f"Unsupported CFU RTL operation: {node.name}")
            return self.wire(value, node.out_types[0])
        raise ValueError(f"Unsupported CFU expression: {type(node).__name__}")

    def lower(self, tree):
        if not isinstance(tree, Statements):
            raise ValueError("Expected a statement tree")
        for node in tree.children:
            if isinstance(node, Declaration):
                ref, value = node.children
                if not isinstance(ref, Ref) or ref.name in self.names:
                    raise ValueError("Invalid or duplicate declaration")
                self.names[ref.name] = self.wire(self.expression(value), node.decl_type)
            elif isinstance(node, Assignment):
                dest, value = node.children
                if (self.result is not None or not isinstance(dest, Register)
                        or dest.reg_class != "gpr" or len(dest.children) != 1
                        or not isinstance(dest.children[0], Ref)
                        or dest.children[0].name != self.mapping["rd"]):
                    raise ValueError("Expected one mapped architectural result")
                self.result = self.expression(value)
            else:
                raise ValueError(f"Unsupported CFU statement: {type(node).__name__}")
        if self.result is None:
            raise ValueError("Missing CFU result")
        return "\n".join(self.lines), self.result


def generate(index, manifest, output, set_name="XIsaac"):
    ledger = CFUEncoder(manifest, allow_new=False).data["instructions"]
    declarations, cases, records = [], [], {}
    candidates = yaml.safe_load(index.read_text())["candidates"]
    if not candidates:
        raise ValueError("Cannot export an empty CFU")
    for number, candidate in enumerate(candidates):
        props, artifacts = candidate["properties"], candidate["artifacts"]
        validate_candidate(props)
        key = f"{set_name}.{props['InstrName']}"
        if props.get("EncodingPolicy") != "cfu" or key in records:
            raise ValueError("Invalid CFU policy or duplicate instruction")
        entry = ledger[key]
        source = Path(artifacts["cdsl"])
        normalized = "\n".join(line for line in source.read_text().splitlines()
                               if not line.lstrip().startswith("//"))
        if hashlib.sha256(normalized.encode()).hexdigest() != entry["source_sha256"]:
            raise ValueError(f"CoreDSL differs from manifest: {key}")
        tree_path = Path(artifacts["tree"])
        with tree_path.open("rb") as handle:
            tree = pickle.load(handle)
        body, result = Datapath(entry["operands"]).lower(tree)
        declarations.append(f"  // {key}\n  if (1) begin : op{number}\n{body}\n"
                            f"    assign results[{number}] = {result};\n  end")
        cases.append(f"      10'd{entry['function_id']}: selected_result = results[{number}];")
        records[key] = {"function_id": entry["function_id"],
                        "tree_sha256": hashlib.sha256(tree_path.read_bytes()).hexdigest(),
                        "source_sha256": entry["source_sha256"]}
    rtl = '''// Generated ISAAC CFU: one registered response slot, pure RV32 datapaths.
`timescale 1ns/1ps
module Cfu (
  input wire cmd_valid, output wire cmd_ready,
  input wire [9:0] cmd_payload_function_id,
  input wire [31:0] cmd_payload_inputs_0, cmd_payload_inputs_1,
  output reg rsp_valid, input wire rsp_ready,
  output reg [31:0] rsp_payload_outputs_0,
  input wire reset, clk
);
'''
    rtl += f"  wire [31:0] results [0:{len(records)-1}];\n"
    rtl += "\n".join(declarations)
    rtl += "\n  reg [31:0] selected_result;\n  always @* begin\n    selected_result = 32'b0;\n    case (cmd_payload_function_id)\n"
    rtl += "\n".join(cases)
    rtl += '''
      default: selected_result = 32'b0;
    endcase
  end
  assign cmd_ready = !reset && (!rsp_valid || rsp_ready);
  always @(posedge clk) begin
    if (reset) begin
      rsp_valid <= 1'b0;
      rsp_payload_outputs_0 <= 32'b0;
    end else if (cmd_ready) begin
      rsp_valid <= cmd_valid;
      if (cmd_valid) rsp_payload_outputs_0 <= selected_result;
    end
  end
endmodule
'''
    output.mkdir(parents=True, exist_ok=True)
    (output / "cfu.sv").write_text(rtl)
    report = {"instructions": records, "response_latency_cycles": 1,
              "response_capacity": 1, "unknown_function_result": 0,
              "rtl_sha256": hashlib.sha256(rtl.encode()).hexdigest(),
              "verification": "not yet verified"}
    (output / "cfu_rtl.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set-name", default="XIsaac")
    args = parser.parse_args()
    generate(args.index, args.manifest, args.output, args.set_name)
