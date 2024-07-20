import io
import sys
import leb128
import logging
import argparse
import posixpath
from pathlib import Path
from collections import defaultdict

import pandas as pd
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("llvm_bbs")


def parse_elf(elf_path):
    GISEL = True  # TODO: make arg
    func_to_addrs = defaultdict(list)
    with open(elf_path, "rb") as f:
        elffile = ELFFile(f)
        section = elffile.get_section_by_name('.symtab')

        if not section:
            print('  No symbol table found. Perhaps this ELF has been stripped?')  # TODO: make err?

        if isinstance(section, SymbolTableSection):
            num_symbols = section.num_symbols()
            print("  It's a symbol section with %s symbols" % num_symbols)
            for i in range(num_symbols):
                sym = section.get_symbol(i)
                if sym["st_info"]["type"] != "STT_FUNC":
                    continue
                print(i, sym.entry, sym.name)
                name = sym.name
                print("name", name)
                # assert name not in func_to_addr, f"Conflict! {name} already in map: {func_to_addr}"
                func_to_addrs[name].append((sym["st_value"], sym["st_size"]))
        print("func_to_addrs", func_to_addrs)

        addr_to_func = {}
        aliases = defaultdict(list)
        for func_name, funcs in func_to_addrs.items():
            for func in funcs:
                func_addr, func_sz = func
                print(func_name, ":", func_sz,  hex(func_addr), "-", hex(func_addr + func_sz))
                if func_addr in addr_to_func:
                    if addr_to_func[func_addr] != func_name:
                       aliases[func_name].append(addr_to_func[func_addr])
                else:
                    addr_to_func[func_addr] = func_name
        print("addr_to_func", addr_to_func)
        llvm_bb_addr_map_raw = elffile.get_section_by_name(".llvm_bb_addr_map")
        assert llvm_bb_addr_map_raw is not None, "Missing: .llvm_bb_addr_map"
        llvm_bb_addr_map_raw = llvm_bb_addr_map_raw.data()
        print("llvm_bb_addr_map_raw", llvm_bb_addr_map_raw)
        def decode_map(data, addr_to_func):
            ret = {}
            with io.BytesIO(data) as reader:
                while True:
                    print("reader", reader)
                    print("dir(reader)", dir(reader))
                    version = int.from_bytes(reader.read(1), byteorder="little")
                    print("version", version)
                    # assert version == 2
                    if version != 2:
                        print("!")
                        print(reader.read(100))
                        break
                    features = int.from_bytes(reader.read(1), byteorder="little")
                    print("features", features)
                    assert features == 0
                    func_addr = int.from_bytes(reader.read(4), byteorder="little")
                    print("func_addr", func_addr)
                    func_name = addr_to_func.get(func_addr, None)
                    print("func_name", func_name)
                    assert func_name is not None
                    num_bbs = int.from_bytes(reader.read(1), byteorder="little")
                    print("num_bbs", num_bbs)
                    assert num_bbs > 0
                    if GISEL:
                        tmp = {}
                    else:
                        tmp = [None] * num_bbs
                    cur = func_addr
                    for i in range(num_bbs):
                        print("i", i)
                        bb_id = int.from_bytes(reader.read(1), byteorder="little")
                        print("bb_id", bb_id)
                        if not GISEL:
                            assert bb_id < num_bbs
                        start_offset = leb128.u.decode_reader(reader)[0]
                        print("start_offset", start_offset)
                        assert start_offset >= 0
                        end_offset = leb128.u.decode_reader(reader)[0]
                        print("end_offset", end_offset)
                        assert end_offset >= 0
                        metadata = int.from_bytes(reader.read(1), byteorder="little")
                        print("metadata", metadata)
                        # TODO: decode metadata (is_return, is_call, ...)
                        # rest = reader.read(1000)
                        # print("rest", rest)
                        cur += start_offset
                        sz = end_offset - start_offset
                        start = cur
                        end = cur + sz
                        print("sz", sz)
                        print("start", start)
                        print("end", end)
                        cur += sz
                        if GISEL:
                            tmp[str(bb_id)] = (start, end, sz)
                        else:
                            tmp[bb_id] = (start, end, sz)
                    print("tmp", tmp)
                    ret[func_name] = tmp
            return ret
        llvm_bb_addr_map = decode_map(llvm_bb_addr_map_raw, addr_to_func)
        print("llvm_bb_addr_map", llvm_bb_addr_map)
        for func_name in func_to_addrs.keys():
            print(f"{func_name}:")
            bbs = llvm_bb_addr_map.get(func_name, None)
            PRINT_MISSING = False
            if bbs is None:
                if PRINT_MISSING:
                    print("> no bb addr info found")
                continue
            if GISEL:
                bbs = dict(sorted(bbs.items(), key=lambda x: int(x[0]))).values()
            for i, bb in enumerate(bbs):
                start, end, sz = bb
                print(f"> bb{i}", ":", hex(start), "-", hex(end), f"(len={sz}B)")
                print()
    return llvm_bb_addr_map


def analyze_llvm_bbs(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    elf_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.ELF)
    # print("elf_artifacts", elf_artifacts)
    assert len(elf_artifacts) == 1
    elf_artifact = elf_artifacts[0]
    trace_pc2bb_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "pc2bb")  # TODO: optional or different pass
    assert len(trace_pc2bb_artifacts) == 1
    trace_pc2bb_artifact = trace_pc2bb_artifacts[0]
    trace_pc2bb_df = trace_pc2bb_artifact.df

    llvm_bbs = parse_elf(elf_artifact.path)
    print("llvm_bbs", llvm_bbs)
    df_data = []
    for func_name, func_data in llvm_bbs.items():
        print("fn", func_name)
        for bb_name, bb_data in func_data.items():
            print("bn", bb_name)
            bb_name = f"%bb.{bb_name}"
            start, end, sz = bb_data
            new = {"func_name": func_name, "bb_name": bb_name, "pcs": (start, end), "size": sz}
            trace_pc2bb_df[["start", "end"]] = trace_pc2bb_df["bb"].apply(pd.Series)
            # matches = trace_pc2bb_df.where(lambda x: x["start"] >= start and x["end"] <= end)
            matches = trace_pc2bb_df.where(lambda x: x["start"] >= start).dropna()
            matches = matches.where(lambda x: x["end"] <= end).dropna()
            if len(matches) > 0:
                print("matches", matches)
                weights = matches["weight"].sum()
                rel_weights = matches["rel_weight"].sum()
                print("weights", weights)
                print("rel_weights", rel_weights)
                # input("p")
                new["num_trace_bbs"] = len(matches)
                new["weight"] = weights
                new["rel_weight"] = rel_weights
            else:
                print("not found")
            # if (start, end) in trace_pc2bb_df["bb"]:
            #     input("yes")
            df_data.append(new)
    # pc2locs_df = pd.DataFrame(pc2locs.items(), columns=["pc", "locs"])
    llvm_bbs_df = pd.DataFrame(df_data)
    llvm_bbs_df.sort_values("rel_weight", inplace=True, ascending=False)
    # print("llvm_bbs_df", llvm_bbs_df)
    # print("trace_pc2bb_df", trace_pc2bb_df)

    attrs = {
        "elf_file": elf_artifact.name,
        "kind": "mapping",
        "by": "isaac_toolkit.analysis.static.llvm_bbs",
    }

    llvm_bbs_artifact = TableArtifact(f"llvm_bbs", llvm_bbs_df, attrs=attrs)
    # print("llvm_bbs_artifact", llvm_bbs_artifact)
    sess.add_artifact(llvm_bbs_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    analyze_llvm_bbs(sess, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
