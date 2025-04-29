#
# Copyright (c) 2025 TUM Department of Electrical and Computer Engineering.
#
# This file is part of ISAAC Toolkit.
# See https://github.com/tum-ei-eda/isaac-toolkit.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
import logging
import argparse
import humanize
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)

# Inspired by: https://github.com/tum-ei-eda/etiss/blob/master/src/bare_etiss_processor/get_metrics.py


# TODO: merge mem_trace with instr_trace
# TODO: trunc_trace also for mem_trace?


DEFAULT_STACK_SIZE = 0x4000


def print_sz(sz, unknown_msg=""):
    if sz is None:
        return f"unknown [{unknown_msg}]" if unknown_msg else "unknown"
    return humanize.naturalsize(sz) + " (" + hex(sz) + ")"


class MemRange:
    def __init__(self, name, min, max):
        self.name = name
        self.min = min
        self.max = max
        assert self.min <= self.max, "Invalid MemRange"
        self.num_reads = 0
        self.num_writes = 0
        self.read_bytes = 0
        self.written_bytes = 0
        self.low = 0xFFFFFFFF
        self.high = 0

    def contains(self, adr):
        return adr >= self.min and adr < self.max

    def trace(self, adr, mode, pc, sz):
        self.low = min(adr, self.low)
        self.high = max(adr, self.high)
        if mode == "r":
            self.num_reads += 1
            self.read_bytes += sz
        elif mode == "w":
            self.num_writes += 1
            self.written_bytes += sz
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @property
    def count(self):
        return self.num_reads + self.num_writes

    def usage(self):
        if self.low > self.high:
            return 0
        return self.high - self.low

    def stats(self):
        if self.low > self.high:
            return self.name + "\t[not accessed]"
        return (
            f"{self.name}\t[0x{self.low:x}-0x{self.high:x}] \t({self.count} times, "
            "reads: {self.num_reads} <{self.read_bytes}B>, "
            "writes: {self.num_writes} <{self.written_bytes}B>)"
        )


def process_symbol_table(symbol_table_df):
    # TODO: do not iterate over all symbols?
    heap_start = None
    for sym in symbol_table_df.itertuples(index=False):
        if sym.name == "_heap_start":
            heap_start = sym.value
    return heap_start


def process_sections(mem_sections_df):
    m = {}
    m["rom_rodata"] = 0
    m["rom_code"] = 0
    m["rom_misc"] = 0
    m["ram_data"] = 0
    m["ram_zdata"] = 0

    ignoreSections = [
        "",
        ".stack",
        ".comment",
        ".riscv.attributes",
        ".strtab",
        ".shstrtab",
    ]

    for s in mem_sections_df.itertuples(index=False):
        if s.name.startswith(".text"):
            m["rom_code"] += s.data_size
        elif s.name.startswith(".srodata"):
            m["rom_rodata"] += s.data_size
        elif s.name.startswith(".sdata"):
            m["ram_data"] += s.data_size
        elif s.name == ".rodata":
            m["rom_rodata"] += s.data_size
        elif s.name == ".vectors" or s.name == ".init_array":
            m["rom_misc"] += s.data_size
        elif s.name == ".data":
            m["ram_data"] += s.data_size
        elif s.name == ".bss" or s.name == ".sbss" or s.name == ".shbss":
            m["ram_zdata"] += s.data_size
        elif s.name.startswith(".gcc_except"):
            pass
        elif s.name.startswith(".sdata2"):
            pass
        elif s.name.startswith(".debug_"):
            pass
        elif s.name in ignoreSections:
            pass
        else:
            print("warning: ignored: " + s.name + " / size: " + str(s.data_size))

    return m


def collect_mem_metrics(
    mem_trace_df,
    mem_sections_df,
    symbol_table_df,
    mem_layout_df,
    max_stack: int = DEFAULT_STACK_SIZE,
    verbose: bool = False,
):
    # print("mem_trace_df", len(mem_trace_df))
    # TODO: count per function?

    static_sizes = process_sections(mem_sections_df)
    heap_start = process_symbol_table(symbol_table_df)

    ram_start = mem_layout_df[mem_layout_df["segment"] == "ram"]["start"].iloc[0]
    ram_size = mem_layout_df[mem_layout_df["segment"] == "ram"]["size"].iloc[0]
    # heap_start = mem_layout_df["heap_start"].iloc[0]
    stack_size = max_stack

    d = MemRange("Data", ram_start, heap_start)
    h = MemRange("Heap", heap_start, ram_start + ram_size - stack_size)
    s = MemRange("Stack", ram_start + ram_size - stack_size, ram_start + ram_size)
    mems = [d, h, s]

    for row in mem_trace_df.itertuples(index=False):
        # print("row", row)
        # input(">")
        for mem in mems:
            # print("mem", mem)
            if mem.contains(row.addr):
                # print("trace")
                mem.trace(row.addr, row.mode, row.pc, row.bytes)
    if verbose:
        for mem in mems:
            print(mem.stats())

    rom_size = sum([static_sizes[k] for k in static_sizes if k.startswith("rom_")])
    ram_size = sum([static_sizes[k] for k in static_sizes if k.startswith("ram_")])

    trace_available = True

    results = {
        "rom": rom_size,
        "rom_rodata": static_sizes["rom_rodata"],
        "rom_code": static_sizes["rom_code"],
        "rom_misc": static_sizes["rom_misc"],
        "ram": (ram_size + s.usage() + h.usage()) if trace_available else ram_size,
        "ram_data": static_sizes["ram_data"],
        "ram_zdata": static_sizes["ram_zdata"],
        "ram_stack": s.usage() if trace_available else None,
        "ram_heap": h.usage() if trace_available else None,
    }

    if verbose:
        print("=== Results ===")
        print("ROM usage:        " + print_sz(results["rom"]))
        print("  read-only data: " + print_sz(results["rom_rodata"]))
        print("  code:           " + print_sz(results["rom_code"]))
        print("  other required: " + print_sz(results["rom_misc"]))
        print(
            "RAM usage:        "
            + print_sz(results["ram"])
            + ("" if trace_available else " [stack and heap usage not included]")
        )
        print("  data:           " + print_sz(results["ram_data"]))
        print("  zero-init data: " + print_sz(results["ram_zdata"]))
        print("  stack:          " + print_sz(results["ram_stack"], unknown_msg="missing trace file"))
        print("  heap:           " + print_sz(results["ram_heap"], unknown_msg="missing trace file"))

    mem_metrics_data = [results]
    mem_metrics_df = pd.DataFrame(mem_metrics_data)

    results2 = [
        {
            "name": r.name,
            "low": r.low,
            "high": r.high,
            "count": r.count,
            "num_reads": r.num_reads,
            "num_writes": r.num_writes,
            "read_bytes": r.read_bytes,
            "writted_bytes": r.written_bytes,
        }
        for r in mems
    ]
    mem_access_df = pd.DataFrame(results2)

    return mem_metrics_df, mem_access_df


def analyze_mem_trace(
    sess: Session,
    force: bool = False,
    max_stack: int = DEFAULT_STACK_SIZE,
    verbose: bool = False,
):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)

    # Memory Trace
    mem_trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "mem_trace")
    assert len(mem_trace_artifacts) == 1
    mem_trace_artifact = mem_trace_artifacts[0]
    assert mem_trace_artifact.attrs.get("simulator") in ["etiss"]  # TODO: support spike?

    # Memory Sections
    mem_sections_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "mem_sections"
    )
    assert len(mem_sections_artifacts) == 1
    mem_sections_artifact = mem_sections_artifacts[0]

    # Symbol Table
    symbol_table_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "symbol_table"
    )
    assert len(symbol_table_artifacts) == 1
    symbol_table_artifact = symbol_table_artifacts[0]

    # Memory Layout
    mem_layout_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "mem_layout"
    )
    assert len(mem_layout_artifacts) == 1
    mem_layout_artifact = mem_layout_artifacts[0]

    mem_metrics_df, mem_access_df = collect_mem_metrics(
        mem_trace_artifact.df,
        mem_sections_artifact.df,
        symbol_table_artifact.df,
        mem_layout_artifact.df,
        max_stack=max_stack,
        verbose=verbose,
    )

    attrs = {
        "mem_trace": mem_trace_artifact.name,
        "kind": "metrics",
        "by": __name__,
    }

    mem_metrics_artifact = TableArtifact("mem_metrics", mem_metrics_df, attrs=attrs)
    sess.add_artifact(mem_metrics_artifact, override=force)

    attrs2 = {
        "mem_trace": mem_trace_artifact.name,
        "kind": "table",
        "by": __name__,
    }

    mem_access_artifact = TableArtifact("mem_access", mem_access_df, attrs=attrs2)
    sess.add_artifact(mem_access_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_mem_trace(sess, force=args.force, max_stack=args.max_stack, verbose=args.verbose)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-stack", type=int, default=DEFAULT_STACK_SIZE)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
