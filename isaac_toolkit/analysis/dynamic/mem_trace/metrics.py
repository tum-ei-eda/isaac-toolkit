#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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
from typing import List
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import humanize
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts

logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)

# Inspired by: https://github.com/tum-ei-eda/etiss/blob/master/src/bare_etiss_processor/get_metrics.py


# TODO: merge mem_trace with instr_trace
# TODO: trunc_trace also for mem_trace?


DEFAULT_STACK_SIZE = 0x4000


@dataclass
class TraceItem:
    idx: int
    pc: int
    mode: str
    count: int = 0
    num_bytes: int = 0
    addrs: List[int] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    strides: List[int] = field(default_factory=list)


def helper(x):
    return int(x, 16)


def init_mems(romStart, ramStart, heapStart, ramSize, stackSize):
    # print("init_mems", romStart, ramStart, heapStart, ramSize, stackSize)
    r = MemRange("ROM", romStart, ramStart)
    d = MemRange("Data", ramStart, heapStart)
    h = MemRange("Heap", heapStart, ramStart + ramSize - stackSize)
    s = MemRange("Stack", ramStart + ramSize - stackSize, ramStart + ramSize)
    mems = [r, d, h, s]
    return mems


def worker(args):
    addrs, romStart, ramStart, heapStart, ramSize, stackSize = args
    mems = init_mems(romStart, ramStart, heapStart, ramSize, stackSize)
    last_addr = None
    last_idx = None
    for addr, mode, pc, sz, idx in addrs:
        # addr = int(addr, 16)
        if last_idx == idx:
            assert last_addr is not None
            stride = addr - last_addr
            # print("stride", stride)
        else:
            stride = None
        for mem in mems:
            if mem.contains(addr):
                mem.trace(addr, mode, pc, sz, idx, stride)
        last_addr = addr
        last_idx = idx
    mems_ = []
    for mem in mems:
        mems_.append(mem.freeze())
    return mems_


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
        self.num_multi_reads = 0
        self.read_strides_per_size_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.multi_read_strides_per_size_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.read_strides_per_pc_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.multi_read_strides_per_pc_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.read_strides_per_idx_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.multi_read_strides_per_idx_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.num_reads_per_size = defaultdict(int)
        self.num_multi_reads_per_size = defaultdict(int)
        self.read_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.multi_read_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.num_reads_per_pc = defaultdict(int)
        self.num_multi_reads_per_pc = defaultdict(int)
        self.read_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.multi_read_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.num_reads_per_idx = defaultdict(int)
        self.num_multi_reads_per_idx = defaultdict(int)
        self.read_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.multi_read_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.num_writes = 0
        self.num_multi_writes = 0
        self.write_strides_per_size_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.multi_write_strides_per_size_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.write_strides_per_pc_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.multi_write_strides_per_pc_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.write_strides_per_idx_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.multi_write_strides_per_idx_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.num_writes_per_size = defaultdict(int)
        self.num_multi_writes_per_size = defaultdict(int)
        self.write_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.multi_write_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.num_writes_per_pc = defaultdict(int)
        self.num_multi_writes_per_pc = defaultdict(int)
        self.write_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.multi_write_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.num_writes_per_idx = defaultdict(int)
        self.num_multi_writes_per_idx = defaultdict(int)
        self.write_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.multi_write_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.read_bytes = 0
        self.multi_read_bytes = 0
        self.read_bytes_per_size = defaultdict(int)
        self.multi_read_bytes_per_size = defaultdict(int)
        self.read_bytes_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.multi_read_bytes_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.read_bytes_per_pc = defaultdict(int)
        self.multi_read_bytes_per_pc = defaultdict(int)
        self.read_bytes_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.multi_read_bytes_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.read_bytes_per_idx = defaultdict(int)
        self.multi_read_bytes_per_idx = defaultdict(int)
        self.read_bytes_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.multi_read_bytes_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.written_bytes = 0
        self.multi_written_bytes = 0
        self.written_bytes_per_size = defaultdict(int)
        self.multi_written_bytes_per_size = defaultdict(int)
        self.written_bytes_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.multi_written_bytes_alignments_per_size = defaultdict(lambda: defaultdict(int))
        self.written_bytes_per_pc = defaultdict(int)
        self.multi_written_bytes_per_pc = defaultdict(int)
        self.written_bytes_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.multi_written_bytes_alignments_per_pc = defaultdict(lambda: defaultdict(int))
        self.written_bytes_per_idx = defaultdict(int)
        self.multi_written_bytes_per_idx = defaultdict(int)
        self.written_bytes_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.multi_written_bytes_alignments_per_idx = defaultdict(lambda: defaultdict(int))
        self.low = 0xFFFFFFFF
        self.high = 0
        self.alignments = set()
        self.multi_alignments = set()

    def contains(self, adr):
        return adr >= self.min and adr < self.max

    def update(self, other):
        self.low = min(other.low, self.low)
        self.high = max(other.high, self.high)
        self.num_reads += other.num_reads
        self.num_writes += other.num_writes
        self.read_bytes += other.read_bytes
        self.written_bytes += other.written_bytes
        self.alignments |= other.alignments
        for sz, num in other.num_reads_per_size.items():
            self.num_reads_per_size[sz] += num
        for sz, num in other.num_writes_per_size.items():
            self.num_writes_per_size[sz] += num
        for sz, num in other.read_bytes_per_size.items():
            self.read_bytes_per_size[sz] += num
        for sz, num in other.written_bytes_per_size.items():
            self.written_bytes_per_size[sz] += num
        for sz, num in other.num_reads_per_pc.items():
            self.num_reads_per_pc[sz] += num
        for sz, num in other.num_writes_per_pc.items():
            self.num_writes_per_pc[sz] += num
        for sz, num in other.read_bytes_per_pc.items():
            self.read_bytes_per_pc[sz] += num
        for sz, num in other.written_bytes_per_pc.items():
            self.written_bytes_per_pc[sz] += num
        for sz, num in other.num_reads_per_idx.items():
            self.num_reads_per_idx[sz] += num
        for sz, num in other.num_writes_per_idx.items():
            self.num_writes_per_idx[sz] += num
        for sz, num in other.read_bytes_per_idx.items():
            self.read_bytes_per_idx[sz] += num
        for sz, num in other.written_bytes_per_idx.items():
            self.written_bytes_per_idx[sz] += num
        # TODO: alignment merging missing?
        # TODO: strides, multi,...

    def trace_multi(self, entry: TraceItem):
        strides_hist = dict(Counter(entry.strides))
        # print("strides_hist", strides_hist)
        sizes_hist = dict(Counter(entry.sizes))
        assert self.count > 1
        # print("sizes_hist", sizes_hist)
        # addrs_hist = dict(Counter(entry.addrs))
        # print("addrs_hist", addrs_hist)
        # assert all(self.contains(addr) for addr in set(entry.addrs))
        alignments = []
        for adr, sz, stride in zip(entry.addrs, entry.sizes, entry.strides):
            # print("adr,sz", adr, sz)
            assert self.contains(adr)
            # self.low = min(adr, self.low)
            # self.high = max(adr, self.high)
            alignment = adr % sz
            alignments.append(alignment)
            # self.multi_alignments.add(alignment)
            # if entry.mode == "r":
            #     self.read_alignments_per_size[sz][alignment] += 1
            #     self.read_alignments_per_pc[entry.pc][alignment] += 1
            #     self.read_alignments_per_idx[entry.idx][alignment] += 1
            #     self.read_bytes_per_size[sz] += sz
            #     self.read_bytes_alignments_per_size[sz][alignment] += sz
            #     self.read_bytes_alignments_per_pc[entry.pc][alignment] += sz
            #     self.read_bytes_alignments_per_idx[entry.idx][alignment] += sz
            # elif entry.mode == "w":
            #     self.write_alignments_per_size[sz][alignment] += 1
            #     self.write_alignments_per_pc[entry.pc][alignment] += 1
            #     self.write_alignments_per_idx[entry.idx][alignment] += 1
            #     self.written_bytes_per_size[sz] += sz
            #     self.written_bytes_alignments_per_size[sz][alignment] += sz
            #     self.written_bytes_alignments_per_pc[entry.pc][alignment] += sz
            #     self.written_bytes_alignments_per_idx[entry.idx][alignment] += sz
            # else:
            #     raise ValueError(f"Invalid mode: {entry.mode}")
            # TODO: stride len?
            self.trace(adr, entry.mode, entry.pc, sz, entry.idx, stride)
        alignments_hist = dict(Counter(alignments))
        if len(alignments_hist) == 1:
            common_alignment = list(alignments_hist.keys())[0]
        else:
            # print("alignments_hist", alignments_hist)
            freq_ = 0
            for align, freq in alignments_hist.items():
                if freq > freq_:
                    freq_ = freq
                    common_alignment = align
            # print("common_alignment", common_alignment)
            # raise NotImplementedError("different alignments?")
        self.multi_alignments.add(common_alignment)
        if entry.mode == "r":
            self.num_multi_reads += 1
            if len(sizes_hist) == 1:
                common_size = list(sizes_hist.keys())[0]
                self.num_multi_reads_per_size[common_size] += 1
                self.multi_read_alignments_per_size[common_size][common_alignment] += 1
                self.multi_read_bytes_per_size[common_size] += sum(entry.sizes)
                self.multi_read_bytes_alignments_per_size[common_size][common_alignment] += sum(entry.sizes)
            else:
                raise NotImplementedError("different sizes?")
            if len(strides_hist) == 1:
                common_stride = list(strides_hist.keys())[0]
                # self.multi_read_strides_hist[common_stride] += 1
                self.multi_read_strides_per_size_alignment[common_size][common_alignment][common_stride] += 1
                self.multi_read_strides_per_pc_alignment[entry.pc][common_alignment][common_stride] += 1
                self.multi_read_strides_per_idx_alignment[entry.idx][common_alignment][common_stride] += 1
            else:
                freq_ = 0
                # print("strides_hist", strides_hist)
                for stride, freq in strides_hist.items():
                    if freq > freq_:
                        freq_ = freq
                        common_stride = stride
                # print("common_stride", common_stride)
                self.multi_read_strides_per_size_alignment[common_size][common_alignment][common_stride] += 1
                self.multi_read_strides_per_pc_alignment[entry.pc][common_alignment][common_stride] += 1
                self.multi_read_strides_per_idx_alignment[entry.idx][common_alignment][common_stride] += 1
                # raise NotImplementedError("different strides?")
            self.num_multi_reads_per_pc[entry.pc] += 1
            self.multi_read_alignments_per_pc[entry.pc][common_alignment] += 1
            self.num_multi_reads_per_idx[entry.idx] += 1
            self.multi_read_alignments_per_idx[entry.idx][common_alignment] += 1
            self.multi_read_bytes += sum(entry.sizes)
            self.multi_read_bytes_per_pc[entry.pc] += sum(entry.sizes)
            self.multi_read_bytes_alignments_per_pc[entry.pc][common_alignment] += sum(entry.sizes)
            self.multi_read_bytes_per_idx[entry.idx] += sum(entry.sizes)
            self.multi_read_bytes_alignments_per_idx[entry.idx][common_alignment] += sum(entry.sizes)
        elif entry.mode == "w":
            self.num_multi_writes += 1
            if len(sizes_hist) == 1:
                common_size = list(sizes_hist.keys())[0]
                self.num_multi_writes_per_size[common_size] += 1
                self.multi_write_alignments_per_size[common_size][common_alignment] += 1
                self.multi_written_bytes_per_size[common_size] += sum(entry.sizes)
                self.multi_written_bytes_alignments_per_size[common_size][common_alignment] += sum(entry.sizes)
            else:
                raise NotImplementedError("different sizes?")
            if len(strides_hist) == 1:
                common_stride = list(strides_hist.keys())[0]
                # self.multi_write_strides_hist[common_stride] += 1
                self.multi_write_strides_per_size_alignment[common_size][common_alignment][common_stride] += 1
                self.multi_write_strides_per_pc_alignment[entry.pc][common_alignment][common_stride] += 1
                self.multi_write_strides_per_idx_alignment[entry.idx][common_alignment][common_stride] += 1
            else:
                raise NotImplementedError("different strides?")
            self.num_multi_writes_per_pc[entry.pc] += 1
            self.multi_write_alignments_per_pc[entry.pc][common_alignment] += 1
            self.num_multi_writes_per_idx[entry.idx] += 1
            self.multi_write_alignments_per_idx[entry.idx][common_alignment] += 1
            self.multi_written_bytes += sum(entry.sizes)
            self.multi_written_bytes_per_pc[entry.pc] += sum(entry.sizes)
            self.multi_written_bytes_alignments_per_pc[entry.pc][common_alignment] += sum(entry.sizes)
            self.multi_written_bytes_per_idx[entry.idx] += sum(entry.sizes)
            self.multi_written_bytes_alignments_per_idx[entry.idx][common_alignment] += sum(entry.sizes)
        else:
            raise ValueError(f"Invalid mode: {entry.mode}")
        # input("!")
        # raise NotImplementedError

    def trace(self, adr, mode, pc, sz, idx, stride):
        self.low = min(adr, self.low)
        self.high = max(adr, self.high)
        alignment = adr % sz
        self.alignments.add(alignment)
        if mode == "r":
            self.num_reads += 1
            if stride is not None:
                self.read_strides_per_size_alignment[sz][alignment][stride] += 1
                self.read_strides_per_pc_alignment[pc][alignment][stride] += 1
                self.read_strides_per_idx_alignment[idx][alignment][stride] += 1
            self.num_reads_per_size[sz] += 1
            self.num_reads_per_pc[pc] += 1
            self.num_reads_per_idx[idx] += 1
            self.read_alignments_per_size[sz][alignment] += 1
            self.read_alignments_per_pc[pc][alignment] += 1
            self.read_alignments_per_idx[idx][alignment] += 1
            self.read_bytes += sz
            self.read_bytes_per_size[sz] += sz
            self.read_bytes_per_pc[pc] += sz
            self.read_bytes_per_idx[idx] += sz
            self.read_bytes_alignments_per_size[sz][alignment] += sz
            self.read_bytes_alignments_per_pc[pc][alignment] += sz
            self.read_bytes_alignments_per_idx[idx][alignment] += sz
        elif mode == "w":
            self.num_writes += 1
            if stride is not None:
                self.write_strides_per_size_alignment[sz][alignment][stride] += 1
                self.write_strides_per_pc_alignment[pc][alignment][stride] += 1
                self.write_strides_per_idx_alignment[idx][alignment][stride] += 1
            self.num_writes_per_size[sz] += 1
            self.num_writes_per_pc[pc] += 1
            self.num_writes_per_idx[idx] += 1
            self.write_alignments_per_size[sz][alignment] += 1
            self.write_alignments_per_pc[pc][alignment] += 1
            self.write_alignments_per_idx[idx][alignment] += 1
            self.written_bytes += sz
            self.written_bytes_per_size[sz] += sz
            self.written_bytes_per_pc[pc] += sz
            self.written_bytes_per_idx[idx] += sz
            self.written_bytes_alignments_per_size[sz][alignment] += sz
            self.written_bytes_alignments_per_pc[pc][alignment] += sz
            self.written_bytes_alignments_per_idx[idx][alignment] += sz
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @property
    def sizes(self):
        sizes = sorted(list(set(self.num_reads_per_size.keys()) | set(self.num_writes_per_size.keys())))
        return sizes

    @property
    def multi_sizes(self):
        sizes = sorted(list(set(self.num_reads_per_size.keys()) | set(self.num_writes_per_size.keys())))
        return sizes

    @property
    def read_pcs(self):
        pcs = sorted(list(self.num_reads_per_pc.keys()))
        return pcs

    @property
    def write_pcs(self):
        pcs = sorted(list(self.num_writes_per_pc.keys()))
        return pcs

    @property
    def pcs(self):
        pcs = sorted(list(set(self.read_pcs) | set(self.write_pcs)))
        return pcs

    @property
    def multi_read_pcs(self):
        pcs = sorted(list(self.num_multi_reads_per_pc.keys()))
        return pcs

    @property
    def multi_write_pcs(self):
        pcs = sorted(list(self.num_multi_writes_per_pc.keys()))
        return pcs

    @property
    def multi_pcs(self):
        pcs = sorted(list(set(self.multi_read_pcs) | set(self.multi_write_pcs)))
        return pcs

    @property
    def read_strides(self):
        ret = sorted(
            sum(
                [
                    list(temp2.keys())
                    for size, temp in self.read_strides_per_size_alignment.items()
                    for align, temp2 in temp.items()
                ],
                [],
            )
        )
        return ret

    @property
    def write_strides(self):
        ret = sorted(
            sum(
                [
                    list(temp2.keys())
                    for size, temp in self.write_strides_per_size_alignment.items()
                    for align, temp2 in temp.items()
                ],
                [],
            )
        )
        return ret

    @property
    def strides(self):
        ret = sorted(list(set(self.read_strides) | set(self.write_strides)))
        return ret

    # @property
    # def strides_hist(self):
    #     ret = {
    #         stride: self.read_strides_hist.get(stride, 0) + self.write_strides_hist.get(stride, 0)
    #         for stride in self.strides
    #     }
    #     return ret

    @property
    def multi_read_strides(self):
        ret = sorted(
            sum(
                [
                    list(temp2.keys())
                    for size, temp in self.multi_read_strides_per_size_alignment.items()
                    for align, temp2 in temp.items()
                ],
                [],
            )
        )
        return ret

    @property
    def multi_write_strides(self):
        ret = sorted(
            sum(
                [
                    list(temp2.keys())
                    for size, temp in self.multi_write_strides_per_size_alignment.items()
                    for align, temp2 in temp.items()
                ],
                [],
            )
        )
        return ret

    @property
    def multi_strides(self):
        ret = sorted(list(set(self.multi_read_strides) | set(self.multi_write_strides)))
        return ret

    @property
    def multi_strides_hist(self):
        ret = {
            stride: self.multi_read_strides_hist.get(stride, 0) + self.multi_write_strides_hist.get(stride, 0)
            for stride in self.multi_strides
        }
        return ret

    @property
    def read_idxs(self):
        idxs = sorted(list(self.num_reads_per_idx.keys()))
        return idxs

    @property
    def write_idxs(self):
        idxs = sorted(list(self.num_writes_per_idx.keys()))
        return idxs

    @property
    def idxs(self):
        idxs = sorted(list(set(self.read_idxs) | set(self.write_idxs)))
        return idxs

    @property
    def multi_read_idxs(self):
        idxs = sorted(list(self.num_multi_reads_per_idx.keys()))
        return idxs

    @property
    def multi_write_idxs(self):
        idxs = sorted(list(self.num_multi_writes_per_idx.keys()))
        return idxs

    @property
    def multi_idxs(self):
        idxs = sorted(list(set(self.multi_read_idxs) | set(self.multi_write_idxs)))
        return idxs

    @property
    def count(self):
        return self.num_reads + self.num_writes

    @property
    def multi_count(self):
        return self.num_multi_reads + self.num_multi_writes

    # @property
    # def count_per_size(self):
    #     return {
    #         size: self.num_reads_per_size.get(size, 0) + self.num_writes_per_size.get(size, 0) for size in self.sizes
    #     }
    def count_per_size(self, size):
        return self.num_reads_per_size.get(size, 0) + self.num_writes_per_size.get(size, 0)

    # @property
    # def count_per_size_alignment(self):
    #     return {
    #         size: {
    #             align: self.read_alignments_per_size.get(size, {}).get(align, 0)
    #             + self.write_alignments_per_size.get(size, {}).get(align, 0)
    #             for align in self.alignments
    #         }
    #         for size in self.sizes
    #     }
    def count_per_size_alignment(self, size, align):
        return self.read_alignments_per_size.get(size, {}).get(align, 0) + self.write_alignments_per_size.get(
            size, {}
        ).get(align, 0)

    # @property
    # def multi_count_per_size_alignment(self):
    #     return {
    #         size: {
    #             align: self.multi_read_alignments_per_size.get(size, {}).get(align, 0)
    #             + self.multi_write_alignments_per_size.get(size, {}).get(align, 0)
    #             for align in self.multi_alignments
    #         }
    #         for size in self.multi_sizes
    #     }
    def multi_count_per_size_alignment(self, size, align):
        return self.multi_read_alignments_per_size.get(size, {}).get(
            align, 0
        ) + self.multi_write_alignments_per_size.get(size, {}).get(align, 0)

    def strides_per_size_alignment(self, size, align):
        return sorted(
            list(
                set(self.read_strides_per_size_alignment.get(size, {}).get(align, {}).keys())
                | set(self.write_strides_per_size_alignment.get(size, {}).get(align, {}).keys())
            )
        )

    def multi_strides_per_size_alignment(self, size, align):
        return sorted(
            list(
                set(self.multi_read_strides_per_size_alignment.get(size, {}).get(align, {}).keys())
                | set(self.multi_write_strides_per_size_alignment.get(size, {}).get(align, {}).keys())
            )
        )

    def multi_strides_per_pc_alignment(self, pc, align):
        return sorted(
            list(
                set(self.multi_read_strides_per_pc_alignment.get(pc, {}).get(align, {}).keys())
                | set(self.multi_write_strides_per_pc_alignment.get(pc, {}).get(align, {}).keys())
            )
        )

    def strides_per_pc_alignment(self, pc, align):
        return sorted(
            list(
                set(self.read_strides_per_pc_alignment.get(pc, {}).get(align, {}).keys())
                | set(self.write_strides_per_pc_alignment.get(pc, {}).get(align, {}).keys())
            )
        )

    def multi_strides_per_idx_alignment(self, idx, align):
        return sorted(
            list(
                set(self.multi_read_strides_per_idx_alignment.get(idx, {}).get(align, {}).keys())
                | set(self.multi_write_strides_per_idx_alignment.get(idx, {}).get(align, {}).keys())
            )
        )

    # @property
    # def count_per_pc(self):
    #     return {pc: self.num_reads_per_pc.get(pc, 0) + self.num_writes_per_pc.get(pc, 0) for pc in self.pcs}
    def count_per_pc(self, pc):
        return self.num_reads_per_pc.get(pc, 0) + self.num_writes_per_pc.get(pc, 0)

    # @property
    # def multi_count_per_pc(self):
    #     return {
    #         pc: self.num_multi_reads_per_pc.get(pc, 0) + self.num_multi_writes_per_pc.get(pc, 0)
    #         for pc in self.multi_pcs
    #     }
    def multi_count_per_pc(self, pc):
        return self.num_multi_reads_per_pc.get(pc, 0) + self.num_multi_writes_per_pc.get(pc, 0)

    # @property
    # def count_per_idx(self):
    #     return {idx: self.num_reads_per_idx.get(idx, 0) + self.num_writes_per_idx.get(idx, 0) for idx in self.idxs}
    def count_per_idx(self, idx):
        return self.num_reads_per_idx.get(idx, 0) + self.num_writes_per_idx.get(idx, 0)

    # @property
    # def multi_count_per_idx(self):
    #     return {
    #         idx: self.num_multi_reads_per_idx.get(idx, 0) + self.num_multi_writes_per_idx.get(idx, 0)
    #         for idx in self.multi_idxs
    #     }
    def multi_count_per_idx(self, idx):
        return self.num_multi_reads_per_idx.get(idx, 0) + self.num_multi_writes_per_idx.get(idx, 0)

    # @property
    # def count_per_pc_alignment(self):
    #     return {
    #         pc: {
    #             align: self.read_alignments_per_pc.get(pc, {}).get(align, 0)
    #             + self.write_alignments_per_pc.get(pc, {}).get(align, 0)
    #             for align in self.alignments
    #         }
    #         for pc in self.pcs
    #     }
    def count_per_pc_alignment(self, pc, align):
        return self.read_alignments_per_pc.get(pc, {}).get(align, 0) + self.write_alignments_per_pc.get(pc, {}).get(
            align, 0
        )

    # @property
    # def multi_count_per_pc_alignment(self):
    #     return {
    #         pc: {
    #             align: self.multi_read_alignments_per_pc.get(pc, {}).get(align, 0)
    #             + self.multi_write_alignments_per_pc.get(pc, {}).get(align, 0)
    #             for align in self.multi_alignments
    #         }
    #         for pc in self.multi_pcs
    #     }
    def multi_count_per_pc_alignment(self, pc, align):
        return self.multi_read_alignments_per_pc.get(pc, {}).get(align, 0) + self.multi_write_alignments_per_pc.get(
            pc, {}
        ).get(align, 0)

    # @property
    # def count_per_idx_alignment(self):
    #     return {
    #         idx: {
    #             align: self.read_alignments_per_idx.get(idx, {}).get(align, 0)
    #             + self.write_alignments_per_idx.get(idx, {}).get(align, 0)
    #             for align in self.alignments
    #         }
    #         for idx in self.idxs
    #     }
    def count_per_idx_alignment(self, idx, align):
        return self.read_alignments_per_idx.get(idx, {}).get(align, 0) + self.write_alignments_per_idx.get(idx, {}).get(
            align, 0
        )

    # @property
    # def multi_count_per_idx_alignment(self):
    #     return {
    #         idx: {
    #             align: self.multi_read_alignments_per_idx.get(idx, {}).get(align, 0)
    #             + self.multi_write_alignments_per_idx.get(idx, {}).get(align, 0)
    #             for align in self.multi_alignments
    #         }
    #         for idx in self.multi_idxs
    #     }
    def multi_count_per_idx_alignment(self, idx, align):
        return self.multi_read_alignments_per_idx.get(idx, {}).get(align, 0) + self.multi_write_alignments_per_idx.get(
            idx, {}
        ).get(align, 0)

    # @property
    # def count_per_alignment(self):
    #     return {
    #         align: (
    #             sum(self.read_alignments_per_size[size].get(align, 0) for size in self.read_alignments_per_size)
    #             + sum(self.write_alignments_per_size[size].get(align, 0) for size in self.write_alignments_per_size)
    #         )
    #         for align in self.alignments
    #     }
    def count_per_alignment(self, align):
        return sum(self.read_alignments_per_size[size].get(align, 0) for size in self.read_alignments_per_size) + sum(
            self.write_alignments_per_size[size].get(align, 0) for size in self.write_alignments_per_size
        )

    # @property
    # def multi_count_per_size(self):
    #     return {
    #         size: self.num_multi_reads_per_size.get(size, 0) + self.num_multi_writes_per_size(size, 0)
    #         for size in self.multi_sizes
    #     }
    def multi_count_per_size(self, size):
        return self.num_multi_reads_per_size.get(size, 0) + self.num_multi_writes_per_size(size, 0)

    # @property
    # def multi_count_per_alignment(self):
    #     return {
    #         align: (
    #             sum(
    #                 self.multi_read_alignments_per_size[size].get(align, 0)
    #                 for size in self.multi_read_alignments_per_size
    #             )
    #             + sum(
    #                 self.multi_write_alignments_per_size[size].get(align, 0)
    #                 for size in self.multi_write_alignments_per_size
    #             )
    #         )
    #         for align in self.multi_alignments
    #     }
    def multi_count_per_alignment(self, align):
        return sum(
            self.multi_read_alignments_per_size[size].get(align, 0) for size in self.multi_read_alignments_per_size
        ) + sum(
            self.multi_write_alignments_per_size[size].get(align, 0) for size in self.multi_write_alignments_per_size
        )

    def usage(self):
        if self.low > self.high:
            return 0
        return self.high - self.low

    def stats(self):
        if self.low > self.high:
            return self.name + "\t[not accessed]"
        return (
            f"{self.name}\t[0x{self.low:x}-0x{self.high:x}] \t({self.count} times, "
            f"reads: {self.num_reads} <{self.read_bytes}B>, "
            f"writes: {self.num_writes} <{self.written_bytes}B>)"
        )

    def freeze(self):
        return FrozenMemRange(self)


class FrozenMemRange:
    def __init__(self, mem: MemRange):
        self.name = mem.name
        self.min = mem.min
        self.max = mem.max
        self.num_reads = mem.num_reads
        self.num_reads_per_size = dict(mem.num_reads_per_size)
        self.read_alignments_per_size = {k: dict(v) for k, v in mem.read_alignments_per_size.items()}
        self.num_writes = mem.num_writes
        self.num_writes_per_size = dict(mem.num_writes_per_size)
        self.write_alignments_per_size = {k: dict(v) for k, v in mem.write_alignments_per_size.items()}
        self.read_bytes = mem.read_bytes
        self.read_bytes_per_size = dict(mem.read_bytes_per_size)
        self.read_bytes_alignments_per_size = {k: dict(v) for k, v in mem.read_bytes_alignments_per_size.items()}
        self.written_bytes = mem.written_bytes
        self.written_bytes_per_size = dict(mem.written_bytes_per_size)
        self.written_bytes_alignments_per_size = {k: dict(v) for k, v in mem.written_bytes_alignments_per_size.items()}
        self.low = mem.low
        self.high = mem.high
        self.alignments = mem.alignments

    def usage(self):
        if self.low > self.high:
            return 0
        return self.high - self.low


def process_symbol_table(symbol_table_df, allow_missing: bool = True):
    # TODO: do not iterate over all symbols?
    heap_start = None
    for sym in symbol_table_df.itertuples(index=False):
        if sym.name == "_heap_start":
            heap_start = sym.value
    assert heap_start is not None or allow_missing  # TODO: warning
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
    multi: bool = False,
):
    # print("mem_trace_df", len(mem_trace_df))
    # TODO: count per function?

    static_sizes = process_sections(mem_sections_df)
    heap_start = process_symbol_table(symbol_table_df)

    rom_start = mem_layout_df[mem_layout_df["segment"] == "rom"]["start"].iloc[0]
    ram_start = mem_layout_df[mem_layout_df["segment"] == "ram"]["start"].iloc[0]
    ram_size = mem_layout_df[mem_layout_df["segment"] == "ram"]["size"].iloc[0]
    # heap_start = mem_layout_df["heap_start"].iloc[0]
    stack_size = max_stack
    if heap_start is None:
        heap_start = ram_start + ram_size - stack_size  # workaround: heap=stack

    mems = init_mems(rom_start, ram_start, heap_start, ram_size, stack_size)
    _, _, h, s = mems

    # TODO: what if first pc is actually 0x0?

    mem_trace_df = mem_trace_df[mem_trace_df["idx"] != 0]
    # start = mem_trace_df["pc"].ne(0).idxmax()
    # if start != 0:
    #     # mem_trace_df = mem_trace_df.copy()
    #     print("start", start)
    #     print("mem_trace_df", mem_trace_df.head(), len(mem_trace_df))
    #     mem_trace_df = mem_trace_df.iloc[start:].copy()
    #     print("mem_trace_df_", mem_trace_df.head(), len(mem_trace_df))
    #     # input("!")
    # TODO: don't save!
    addrs = mem_trace_df[["addr", "mode", "pc", "bytes"]].values
    # print("len(addrs)", len(addrs))
    import multiprocessing as mp
    import os

    USE_POOL = False

    if USE_POOL:
        num_threads = os.cpu_count()
        with mp.Pool(num_threads) as p:
            num_chunks = num_threads

            def iter_chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield lst[i : i + n]

            n = len(addrs) // num_chunks
            chunked_addrs = iter_chunks(addrs, n)
            args = [(chunk, rom_start, ram_start, heap_start, ram_size, stack_size) for chunk in chunked_addrs]
            for mems_ in p.imap_unordered(worker, args):
                for i, mem_ in enumerate(mems_):
                    if mem_.usage() == 0:
                        continue
                    mem = mems[i]
                    mem.update(mem_)
                    # mem.trace(mem_.high)
                    # mem.trace(mem_.low)
                    # mem.count += mem_.count
    else:
        last_addr = None
        # last_idx = None
        idx_count = 0
        current = None
        i = 0

        for row in mem_trace_df.itertuples(index=True):
            # print("row", row)
            progress = i / len(mem_trace_df)
            if i % 1000000 == 0:
                print("progress", progress)
            i += 1
            if current is not None and current.idx == row.idx and current.pc == row.pc and current.mode == row.mode:
                assert last_addr is not None
                stride = row.addr - last_addr
                idx_count += 1
                # assert current.pc == row.pc
                # assert current.mode == row.mode
                current.sizes.append(row.bytes)
                current.count += 1
                current.strides.append(stride)
                current.addrs.append(row.addr)
                current.num_bytes += row.bytes
                # print("current", current)
                # print("row", row)
                # print("stride", stride)
                # print("idx_count", idx_count)
                # input("?")
            else:
                if current is not None:
                    if current.count > 1:
                        # print("MERGE")
                        # print("current", current)
                        for mem in mems:
                            # print("mem", mem)
                            if mem.contains(current.addrs[0]):
                                mem.trace_multi(current)
                    else:
                        for mem in mems:
                            # print("mem", mem)
                            if mem.contains(current.addrs[0]):
                                mem.trace(
                                    current.addrs[0], current.mode, current.pc, current.sizes[0], current.idx, None
                                )
                    current = None
                assert current is None

                current = TraceItem(row.idx, row.pc, row.mode)
                current.count += 1
                current.num_bytes += row.bytes
                current.addrs.append(row.addr)
                current.sizes.append(row.bytes)
                stride = None
            last_addr = row.addr
            # last_idx = row.idx

        if current is not None:
            # print("MERGE")
            # print("current", current)
            # input("!")
            if current.count > 1:
                # print("MERGE")
                # print("current", current)
                for mem in mems:
                    # print("mem", mem)
                    if mem.contains(current.addrs[0]):
                        mem.trace_multi(current)
            else:
                for mem in mems:
                    # print("mem", mem)
                    if mem.contains(current.addrs[0]):
                        mem.trace(current.addrs[0], current.mode, current.pc, current.sizes[0], current.idx, None)
            current = None
        # TODO: update process pool too
    if verbose:
        for mem in mems:
            print(mem.stats())

    rom_size = sum([static_sizes[k] for k in static_sizes if k.startswith("rom_")])
    ram_size = sum([static_sizes[k] for k in static_sizes if k.startswith("ram_")])

    trace_available = True

    mem_metrics = {
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
        print("ROM usage:        " + print_sz(mem_metrics["rom"]))
        print("  read-only data: " + print_sz(mem_metrics["rom_rodata"]))
        print("  code:           " + print_sz(mem_metrics["rom_code"]))
        print("  other required: " + print_sz(mem_metrics["rom_misc"]))
        print(
            "RAM usage:        "
            + print_sz(mem_metrics["ram"])
            + ("" if trace_available else " [stack and heap usage not included]")
        )
        print("  data:           " + print_sz(mem_metrics["ram_data"]))
        print("  zero-init data: " + print_sz(mem_metrics["ram_zdata"]))
        print("  stack:          " + print_sz(mem_metrics["ram_stack"], unknown_msg="missing trace file"))
        print("  heap:           " + print_sz(mem_metrics["ram_heap"], unknown_msg="missing trace file"))

    mem_metrics_data = [mem_metrics]
    mem_metrics_df = pd.DataFrame(mem_metrics_data)

    # by mem
    mem_access_by_mem = [
        {
            "name": r.name,
            "low": r.low,
            "high": r.high,
            "count": r.count,
            "num_reads": r.num_reads,
            "num_writes": r.num_writes,
            "read_bytes": r.read_bytes,
            "written_bytes": r.written_bytes,
            "read_bytes_per_read": (r.read_bytes / r.num_reads) if r.num_reads > 0 else None,
            "written_bytes_per_write": (r.written_bytes / r.num_writes) if r.num_writes > 0 else None,
        }
        for r in mems
    ]
    mem_access_by_mem_df = pd.DataFrame(mem_access_by_mem)
    if multi:
        multi_mem_access_by_mem = [
            {
                "name": r.name,
                "low": r.low,
                "high": r.high,
                "count": r.multi_count,
                "num_reads": r.num_multi_reads,
                "num_writes": r.num_multi_writes,
                "read_bytes": r.multi_read_bytes,
                "written_bytes": r.multi_written_bytes,
                "read_bytes_per_read": (r.multi_read_bytes / r.num_multi_reads) if r.num_multi_reads > 0 else None,
                "written_bytes_per_write": (r.multi_written_bytes / r.num_multi_writes) if r.num_multi_writes else None,
                "strides": r.multi_strides,
            }
            for r in mems
        ]
        multi_mem_access_by_mem_df = pd.DataFrame(multi_mem_access_by_mem)

    # by size
    mem_access_by_size = []
    for r in mems:
        temp = {
            "name": r.name,
            "low": r.low,
            "high": r.high,
        }
        for size in r.sizes:
            for align in sorted(list(r.alignments)):
                temp_ = {
                    **temp,
                    "size": size,
                    "alignment": align,
                    # "count": r.count_per_size_alignment[size].get(align, 0),
                    "count": r.count_per_size_alignment(size, align),
                    # "num_reads": r.num_reads_per_size.get(size, 0),
                    "num_reads": r.read_alignments_per_size.get(size, {}).get(align),
                    # "num_writes": r.num_writes_per_size.get(size, 0),
                    "num_writes": r.write_alignments_per_size.get(size, {}).get(align),
                    # "read_bytes": r.read_bytes_per_size.get(size, 0),
                    "read_bytes": r.read_bytes_alignments_per_size.get(size, {}).get(align),
                    "read_bytes_per_read": (
                        (
                            r.read_bytes_alignments_per_size.get(size, {}).get(align)
                            / r.read_alignments_per_size.get(size, {}).get(align)
                        )
                        if r.read_alignments_per_size.get(size, {}).get(align, 0) > 0
                        else None
                    ),
                    "written_bytes_write": (
                        (
                            r.written_bytes_alignments_per_size.get(size, {}).get(align)
                            / r.write_alignments_per_size.get(size, {}).get(align)
                        )
                        if r.write_alignments_per_size.get(size, {}).get(align, 0) > 0
                        else None
                    ),
                }
                mem_access_by_size.append(temp_)
    mem_access_by_size_df = pd.DataFrame(mem_access_by_size)
    if multi:
        multi_mem_access_by_size = []
        for r in mems:
            temp = {
                "name": r.name,
                "low": r.low,
                "high": r.high,
            }
            for size in r.multi_sizes:
                for align in sorted(list(r.multi_alignments)):
                    temp_ = {
                        **temp,
                        "size": size,
                        "alignment": align,
                        # "count": r.multi_count_per_size_alignment[size].get(align, 0),
                        "count": r.multi_count_per_size_alignment(size, align),
                        # "num_reads": r.num_reads_per_size.get(size, 0),
                        "num_reads": r.multi_read_alignments_per_size.get(size, {}).get(align),
                        # "num_writes": r.num_writes_per_size.get(size, 0),
                        "num_writes": r.multi_write_alignments_per_size.get(size, {}).get(align),
                        # "read_bytes": r.read_bytes_per_size.get(size, 0),
                        "read_bytes": r.multi_read_bytes_alignments_per_size.get(size, {}).get(align),
                        # "written_bytes": r.written_bytes_per_size.get(size, 0),
                        "written_bytes": r.multi_written_bytes_alignments_per_size.get(size, {}).get(align),
                        "read_bytes_per_read": (
                            (
                                r.multi_read_bytes_alignments_per_size.get(size, {}).get(align)
                                / r.multi_read_alignments_per_size.get(size, {}).get(align)
                            )
                            if r.multi_read_alignments_per_size.get(size, {}).get(align, 0) > 0
                            else None
                        ),
                        "written_bytes_per_write": (
                            (
                                r.multi_written_bytes_alignments_per_size.get(size, {}).get(align)
                                / r.multi_write_alignments_per_size.get(size, {}).get(align)
                            )
                            if r.multi_write_alignments_per_size.get(size, {}).get(align, 0) > 0
                            else None
                        ),
                        "strides": r.multi_strides_per_size_alignment(size, align),
                    }
                    multi_mem_access_by_size.append(temp_)
        multi_mem_access_by_size_df = pd.DataFrame(multi_mem_access_by_size)

    # by pc
    mem_access_by_pc = []
    # results4_ = []
    # results4__ = []
    # pc_to_idxs = mem_trace_df.groupby("pc")["idx"].apply(lambda x: list(set(list(x)))).to_dict()
    for r in mems:
        # print("mem", r)
        # print("strides", r.strides)
        # print("strides_hist", r.strides_hist)
        # print("mem.num_multi_reads", r.num_multi_reads)
        # print("mem.num_multi_writes", r.num_multi_writes)
        # print("mem.num_multi", r.num_multi)
        # input("o")
        temp = {
            "name": r.name,
            "low": r.low,
            "high": r.high,
        }
        for pc in r.pcs:
            # print("pc", pc)
            # idxs = pc_to_idxs[pc]
            # num_idxs = len(idxs)
            # print("idxs", idxs, len(idxs))
            # TODO: ignore align
            for align in sorted(list(r.alignments)):
                temp_ = {
                    **temp,
                    "pc": pc,
                    # "size": size,
                    "alignment": align,
                    # "count": r.count_per_pc_alignment[pc].get(align, 0),
                    "count": r.count_per_pc_alignment(pc, align),
                    # "num_reads": r.num_reads_per_pc.get(pc, 0),
                    "num_reads": r.read_alignments_per_pc.get(pc, {}).get(align),
                    # "num_writes": r.num_writes_per_pc.get(pc, 0),
                    "num_writes": r.write_alignments_per_pc.get(pc, {}).get(align),
                    # "read_bytes": r.read_bytes_per_pc.get(pc, 0),
                    "read_bytes": r.read_bytes_alignments_per_pc.get(pc, {}).get(align),
                    # "written_bytes": r.written_bytes_per_pc.get(pc, 0),
                    "written_bytes": r.written_bytes_alignments_per_pc.get(pc, {}).get(align),
                    "read_bytes_per_read": (
                        (
                            r.read_bytes_alignments_per_pc.get(pc, {}).get(align)
                            / r.read_alignments_per_pc.get(pc, {}).get(align)
                        )
                        if r.read_alignments_per_pc.get(pc, {}).get(align, 0) > 0
                        else None
                    ),
                    "written_bytes_per_write": (
                        (
                            r.written_bytes_alignments_per_pc.get(pc, {}).get(align)
                            / r.write_alignments_per_pc.get(pc, {}).get(align)
                        )
                        if r.write_alignments_per_pc.get(pc, {}).get(align, 0) > 0
                        else None
                    ),
                }
                mem_access_by_pc.append(temp_)
                # temp__ = {
                #     **temp_,
                #     "num_idxs": num_idxs,
                #     "num_reads_per_idx": temp_["num_reads"] / num_idxs if temp_["num_reads"] is not None else None,
                #     "num_writes_per_idx": temp_["num_writes"] / num_idxs if temp_["num_writes"] is not None else None,
                #     "read_bytes_per_idx": temp_["read_bytes"] / num_idxs if temp_["read_bytes"] is not None else None,
                #     "written_bytes_per_idx": (
                #         temp_["written_bytes"] / num_idxs if temp_["written_bytes"] is not None else None
                #     ),
                # }
                # # TODO: make optional
                # results4_.append(temp__)
                # if (temp__["read_bytes_per_idx"] is not None and temp__["read_bytes_per_idx"] > 5.0) or (
                #     temp__["written_bytes_per_idx"] and temp__["written_bytes_per_idx"] > 5.0
                # ):
                #     results4__.append(temp__)
                # # results4__.append(temp__)
    mem_access_by_pc_df = pd.DataFrame(mem_access_by_pc)
    # mem_access_df3_ = pd.DataFrame(results4_)
    # mem_access_df3__ = pd.DataFrame(results4__)
    # mem_access_df3 = mem_access_df3__
    if multi:
        print("1")
        multi_mem_access_by_pc = []
        for r in mems:
            print("mem", r)
            print("mem.multi_pcs", r.multi_pcs)
            print("mem.multi_alignments", r.multi_alignments)
            temp = {
                "name": r.name,
                "low": r.low,
                "high": r.high,
            }
            for pc in r.multi_pcs:
                # print("pc", pc)
                # idxs = pc_to_idxs[pc]
                # num_idxs = len(idxs)
                # print("idxs", idxs, len(idxs))
                # TODO: ignore align
                for align in sorted(list(r.multi_alignments)):
                    temp_ = {
                        **temp,
                        "pc": pc,
                        # "size": size,
                        "alignment": align,
                        # "count": r.multi_count_per_pc_alignment[pc].get(align, 0),
                        "count": r.multi_count_per_pc_alignment(pc, align),
                        # "num_reads": r.num_reads_per_pc.get(pc, 0),
                        "num_reads": r.multi_read_alignments_per_pc.get(pc, {}).get(align),
                        # "num_writes": r.num_writes_per_pc.get(pc, 0),
                        "num_writes": r.multi_write_alignments_per_pc.get(pc, {}).get(align),
                        # "read_bytes": r.read_bytes_per_pc.get(pc, 0),
                        "read_bytes": r.multi_read_bytes_alignments_per_pc.get(pc, {}).get(align),
                        # "written_bytes": r.written_bytes_per_pc.get(pc, 0),
                        "written_bytes": r.multi_written_bytes_alignments_per_pc.get(pc, {}).get(align),
                        "read_bytes_per_read": (
                            (
                                r.multi_read_bytes_alignments_per_pc.get(pc, {}).get(align)
                                / r.multi_read_alignments_per_pc.get(pc, {}).get(align)
                            )
                            if r.multi_read_alignments_per_pc.get(pc, {}).get(align, 0) > 0
                            else None
                        ),
                        "written_bytes_per_write": (
                            (
                                r.multi_written_bytes_alignments_per_pc.get(pc, {}).get(align)
                                / r.multi_write_alignments_per_pc.get(pc, {}).get(align)
                            )
                            if r.multi_write_alignments_per_pc.get(pc, {}).get(align, 0) > 0
                            else None
                        ),
                        "strides": r.multi_strides_per_pc_alignment(pc, align),
                    }
                    multi_mem_access_by_pc.append(temp_)
        multi_mem_access_by_pc_df = pd.DataFrame(multi_mem_access_by_pc)

    # by idx
    mem_access_by_idx = []
    idx_to_pc = dict(zip(mem_trace_df["idx"], mem_trace_df["pc"]))
    for r in mems:
        temp = {
            "name": r.name,
            "low": r.low,
            "high": r.high,
        }
        # print("len(idxs)", len(r.idxs))
        for idx in r.idxs:
            # print("idx", idx)
            # pc = mem_trace_df[mem_trace_df["idx"] == idx]["pc"].iloc[0]
            pc = idx_to_pc[idx]
            for align in sorted(list(r.alignments)):
                temp_ = {
                    **temp,
                    "idx": idx,
                    "pc": pc,
                    # "size": size,
                    "alignment": align,
                    # "count": r.count_per_idx_alignment[idx].get(align, 0),
                    "count": r.count_per_idx_alignment(idx, align),
                    # "num_reads": r.num_reads_per_idx.get(idx, 0),
                    "num_reads": r.read_alignments_per_idx.get(idx, {}).get(align),
                    # "num_writes": r.num_writes_per_idx.get(idx, 0),
                    "num_writes": r.write_alignments_per_idx.get(idx, {}).get(align),
                    # "read_bytes": r.read_bytes_per_idx.get(idx, 0),
                    "read_bytes": r.read_bytes_alignments_per_idx.get(idx, {}).get(align),
                    # "written_bytes": r.written_bytes_per_idx.get(idx, 0),
                    "written_bytes": r.written_bytes_alignments_per_idx.get(idx, {}).get(align),
                    "read_bytes_per_read": (
                        (
                            r.read_bytes_alignments_per_idx.get(idx, {}).get(align)
                            / r.read_alignments_per_idx.get(idx, {}).get(align)
                        )
                        if r.read_alignments_per_idx.get(idx, {}).get(align, 0) > 0
                        else None
                    ),
                    "written_bytes_per_write": (
                        (
                            r.written_bytes_alignments_per_idx.get(idx, {}).get(align)
                            / r.write_alignments_per_idx.get(idx, {}).get(align)
                        )
                        if r.write_alignments_per_idx.get(idx, {}).get(align, 0) > 0
                        else None
                    ),
                }
                mem_access_by_idx.append(temp_)
    mem_access_by_idx_df = pd.DataFrame(mem_access_by_idx)
    if multi:
        multi_mem_access_by_idx = []
        for r in mems:
            temp = {
                "name": r.name,
                "low": r.low,
                "high": r.high,
            }
            # print("len(idxs)", len(r.idxs))
            for idx in r.multi_idxs:
                # print("idx", idx)
                # pc = multi_mem_trace_df[mem_trace_df["idx"] == idx]["pc"].iloc[0]
                pc = idx_to_pc[idx]
                for align in sorted(list(r.multi_alignments)):
                    temp_ = {
                        **temp,
                        "idx": idx,
                        "pc": pc,
                        # "size": size,
                        "alignment": align,
                        "count": r.multi_count_per_idx_alignment(idx, align),
                        # "num_reads": r.num_reads_per_idx.get(idx, 0),
                        "num_reads": r.multi_read_alignments_per_idx.get(idx, {}).get(align),
                        # "num_writes": r.num_writes_per_idx.get(idx, 0),
                        "num_writes": r.multi_write_alignments_per_idx.get(idx, {}).get(align),
                        # "read_bytes": r.read_bytes_per_idx.get(idx, 0),
                        "read_bytes": r.multi_read_bytes_alignments_per_idx.get(idx, {}).get(align),
                        # "written_bytes": r.written_bytes_per_idx.get(idx, 0),
                        "written_bytes": r.multi_written_bytes_alignments_per_idx.get(idx, {}).get(align),
                        "read_bytes_per_read": (
                            (
                                r.multi_read_bytes_alignments_per_idx.get(idx, {}).get(align)
                                / r.multi_read_alignments_per_idx.get(idx, {}).get(align)
                            )
                            if r.multi_read_alignments_per_idx.get(idx, {}).get(align, 0) > 0
                            else None
                        ),
                        "written_bytes_per_write": (
                            (
                                r.multi_written_bytes_alignments_per_idx.get(idx, {}).get(align)
                                / r.multi_write_alignments_per_idx.get(idx, {}).get(align)
                            )
                            if r.multi_write_alignments_per_idx.get(idx, {}).get(align, 0) > 0
                            else None
                        ),
                        "strides": r.multi_strides_per_idx_alignment(idx, align),
                    }
                    multi_mem_access_by_idx.append(temp_)
        multi_mem_access_by_idx_df = pd.DataFrame(multi_mem_access_by_idx)

    if multi:
        return (
            mem_metrics_df,
            mem_access_by_mem_df,
            mem_access_by_size_df,
            mem_access_by_pc_df,
            mem_access_by_idx_df,
            multi_mem_access_by_mem_df,
            multi_mem_access_by_size_df,
            multi_mem_access_by_pc_df,
            multi_mem_access_by_idx_df,
        )
    else:
        return mem_metrics_df, mem_access_by_mem_df, mem_access_by_size_df, mem_access_by_pc_df, mem_access_by_idx_df


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
    multi = True  # TODO: expose

    dfs = collect_mem_metrics(
        mem_trace_artifact.df,
        mem_sections_artifact.df,
        symbol_table_artifact.df,
        mem_layout_artifact.df,
        max_stack=max_stack,
        verbose=verbose,
        multi=multi,
    )
    if multi:
        (
            mem_metrics_df,
            mem_access_by_mem_df,
            mem_access_by_size_df,
            mem_access_by_pc_df,
            mem_access_by_idx_df,
            # multi_mem_metrics_df,
            multi_mem_access_by_mem_df,
            multi_mem_access_by_size_df,
            multi_mem_access_by_pc_df,
            multi_mem_access_by_idx_df,
        ) = dfs
    else:
        mem_metrics_df, mem_access_by_mem_df, mem_access_by_size_df, mem_access_by_pc_df, mem_access_by_idx_df = dfs

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

    mem_access_by_mem_artifact = TableArtifact("mem_access_by_mem", mem_access_by_mem_df, attrs=attrs2)  # by section
    sess.add_artifact(mem_access_by_mem_artifact, override=force)
    mem_access_by_size_artifact = TableArtifact("mem_access_by_size", mem_access_by_size_df, attrs=attrs2)  # by size
    sess.add_artifact(mem_access_by_size_artifact, override=force)
    mem_access_by_pc_artifact = TableArtifact("mem_access_by_pc", mem_access_by_pc_df, attrs=attrs2)  # by pc
    sess.add_artifact(mem_access_by_pc_artifact, override=force)
    mem_access_by_idx_artifact = TableArtifact("mem_access_by_idx", mem_access_by_idx_df, attrs=attrs2)  # by idx
    sess.add_artifact(mem_access_by_idx_artifact, override=force)
    if multi:
        multi_mem_access_by_mem_artifact = TableArtifact(
            "multi_mem_access_by_mem", multi_mem_access_by_mem_df, attrs=attrs2
        )  # by section
        sess.add_artifact(multi_mem_access_by_mem_artifact, override=force)
        multi_mem_access_by_size_artifact = TableArtifact(
            "multi_mem_access_by_size", multi_mem_access_by_size_df, attrs=attrs2
        )  # by size
        sess.add_artifact(multi_mem_access_by_size_artifact, override=force)
        multi_mem_access_by_pc_artifact = TableArtifact(
            "multi_mem_access_by_pc", multi_mem_access_by_pc_df, attrs=attrs2
        )  # by pc
        sess.add_artifact(multi_mem_access_by_pc_artifact, override=force)
        multi_mem_access_by_idx_artifact = TableArtifact(
            "multi_mem_access_by_idx", multi_mem_access_by_idx_df, attrs=attrs2
        )  # by idx
        sess.add_artifact(multi_mem_access_by_idx_artifact, override=force)


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
