"""Load CFU-Playground VexRiscv retirement CSV with static-word validation.

The pinned full+cfu+debug configuration and current firmware use RV32IM without
compressed instructions. Runtime operand values are not captured by this trace.
"""

import argparse
from pathlib import Path
import re

import pandas as pd

from isaac_toolkit.generate.iss.cfu import CFUEncoder
from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import InstrTraceArtifact


def read_trace(trace, disassembly, manifest=None, chunk_size=250000):
    static = {}
    for line in disassembly.read_text().splitlines():
        match = re.match(r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\s+([^\s]+)", line)
        if match:
            address, word, mnemonic = match.groups()
            static[int(address, 16)] = (int(word, 16), mnemonic.upper())
    entries = CFUEncoder(manifest, allow_new=False).data["instructions"] if manifest else {}
    names = {}
    for address, (word, mnemonic) in static.items():
        if word & 0x7f == 0x0b:
            matches = [key for key, entry in entries.items() if word & entry["mask"] == entry["code"]]
            if len(matches) != 1:
                raise ValueError(f"Unmapped custom instruction at 0x{address:x}")
            mnemonic = matches[0].split(".")[-1]
        names[address] = mnemonic
    parts, previous_cycle = [], -1
    for frame in pd.read_csv(trace, chunksize=chunk_size):
        if list(frame.columns) != ["cycle", "pc", "bytecode"]:
            raise ValueError("Unexpected VexRiscv trace columns")
        for column in ("pc", "bytecode"):
            frame[column] = frame[column].map(lambda value: int(value, 16))
        if frame.empty:
            continue
        if frame["cycle"].iloc[0] <= previous_cycle or (frame["cycle"].diff().dropna() <= 0).any():
            raise ValueError("Retirement cycles must strictly increase")
        previous_cycle = int(frame["cycle"].iloc[-1])
        expected = frame["pc"].map({address: value[0] for address, value in static.items()})
        if expected.isna().any() or (frame["bytecode"] != expected).any():
            raise ValueError("Retirement word/PC differs from firmware disassembly")
        if ((frame["bytecode"] & 3) != 3).any():
            raise ValueError("Compressed retirement trace is not supported")
        frame["instr"] = frame["pc"].map(names)
        frame["size"] = 4
        parts.append(frame)
    if not parts:
        raise ValueError("Empty retirement trace")
    result = pd.concat(parts, ignore_index=True)
    for column in ("pc", "bytecode"):
        result[column] = pd.to_numeric(result[column], downcast="unsigned")
    for column in ("instr", "size"):
        result[column] = result[column].astype("category")
    return result


def load_instr_trace(sess, input_file, disassembly, manifest=None, force=False,
                     start_func=None, end_func=None):
    frame = read_trace(input_file, disassembly, manifest)
    attrs = {"simulator": "cfu_playground_verilator", "cpu_arch": "rv32im",
             "cpu_variant": "full+cfu+debug", "operand_values": "not captured",
             "by": "isaac_toolkit.frontend.instr_trace.cfu"}
    if start_func:
        attrs["benchmark_start_func"] = start_func
    if end_func:
        attrs["benchmark_end_func"] = end_func
    sess.add_artifact(InstrTraceArtifact(input_file.name, frame, attrs=attrs), override=force)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--disassembly", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    session = Session.from_dir(args.session)
    frame = load_instr_trace(session, args.file, args.disassembly, args.manifest, args.force)
    session.save()
    print(f"Loaded {len(frame)} validated VexRiscv retirements")
