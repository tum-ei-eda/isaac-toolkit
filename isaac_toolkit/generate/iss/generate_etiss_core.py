import sys
import yaml
import logging
import argparse
from typing import Optional, Union, List
from pathlib import Path

# import m2isar
# import m2isar.metamodel.arch
# from m2isar.frontends.coredsl2_set.parser import parse_cdsl2_set

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logger = logging.getLogger("llvm_bbs")


def get_cdsl_sets(ext: str, xlen: int = 32, compressed: bool = False):
    ext = ext.lower()
    cdsl_sets = {
        "i": [f"RV{xlen}I"] + ([f"RV{xlen}IC"] if compressed else []),
        "m": [f"RV{xlen}M"],
        "a": [f"RV{xlen}A"],
        "f": [f"RV{xlen}F"] + ([f"RV{xlen}FC"] if compressed else []),
        "d": [f"RV{xlen}D"] + ([f"RV{xlen}DC"] if compressed else []),
        "zifencei": ["Zifencei"],
        "zicsr": ["Zicsr"],
    }
    ret = cdsl_sets.get(ext, None)
    assert ret is not None, f"Lookup failed for ext: {ext}"
    return ret


def apply_etiss_overrides(sets: List[str], semihosting: bool = True):
    etiss_overrides = {
        "zicsr": "tum_csr",
        "RV32A": "tum_rva",
        "RV64A": "tum_rva64",
        "RV64M": "tum_rvm",
    }
    ret = []
    new = []
    for orig in sets:
        replaced = etiss_overrides.get(orig, None)
        if replaced:
            new.append(replaced)
        else:
            ret.append(orig)
    if semihosting:
        new += ["tum_semihosting"]
    new += ["tum_ret"]
    return ret + new


def get_cdsl_includes(sets: List[str], base_dir: Union[str, Path] = "rv_base", tum_dir: Union[str, Path] = "."):
    # Reverse lookup required!
    cdsl_includes = {
        "RISCVBase.core_desc": ["RISCVBase"],
        "RV32I.core_desc": ["RV32I", "Zicsr", "Zifencei", "RVNMode", "RVSMode", "RVDebug"],
        "RV64I.core_desc": ["RV64I"],
        "RVA.core_desc": ["RV32A", "RV64A"],
        "RVC.core_desc": ["RV32IC", "RV32FC", "RV32DC", "RV64IC", "RV64FC", "RV64DC", "RV128IC"],
        "RVD.core_desc": ["RV32D", "RV64D"],
        "RVF.core_desc": ["RV32F", "RV64F"],
        # "RVI.core_desc": ["RV32I", "RV64I", "Zicsr", "Zifencei", "RVNMode", "RVSMode", "RVDebug"],
        "RVM.core_desc": ["RV32M", "RV64M"],
        "tum_rvm.core_desc": ["tum_rvm"],
        "tum_rva.core_desc": ["tum_rva", "tum_rva64"],
        "tum_mod.core_desc": ["tum_ret", "tum_csr", "tum_semihosting"],
    }
    ret = set()
    for set_name in sets:
        # print("set_name", set_name)
        matched = None
        for cdsl_file, cdsl_sets in cdsl_includes.items():
            if set_name in cdsl_sets:
                matched = cdsl_file
                break
        assert matched, f"Include could not be resolved for set: {set_name}"
        if "tum" in set_name.lower():
            base = Path(tum_dir)
        else:
            base = Path(base_dir)
        ret.add(str(base / matched))
    return ret


def generate_etiss_core(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    core_name: str = "ISAACCore",
    xlen: int = 32,
    ignore_etiss: bool = False,
    semihosting: bool = True,
    base_extensions: List[str] = ["i", "m", "a", "f", "d", "c", "zifencei"],
    # etiss_overrides: List[str] = ["tum_csr", "tum_ret", "tum_rva", "tum_semihosting"],
    auto_encoding: bool = True,
    split: bool = True,  # One set per new instr
    force: bool = False,
):
    # artifacts = sess.artifacts
    # TODO: get combined_index.yml from artifacts!
    assert workdir is not None
    combined_index_file = workdir / "combined_index.yml"
    with open(combined_index_file, "r") as f:
        index_data = yaml.safe_load(f)
    print("index_data", index_data)
    # Not uses set() here as we want to preserve the order!
    all_cdsl_sets = []
    compressed = "c" in base_extensions
    for ext in base_extensions:
        if ext == "c":
            continue
        cdsl_sets = get_cdsl_sets(ext, xlen=xlen, compressed=compressed)
        for cdsl_set in cdsl_sets:
            if cdsl_set not in all_cdsl_sets:
                all_cdsl_sets.append(cdsl_set)
    if not ignore_etiss:
        all_cdsl_sets = apply_etiss_overrides(all_cdsl_sets, semihosting=semihosting)
    print("all_cdsl_sets", all_cdsl_sets)
    cdsl_includes = get_cdsl_includes(all_cdsl_sets)
    print("cdsl_includes", cdsl_includes)
    constants = {}
    memories = {}
    instructions = {}
    if ignore_etiss:
        constants["XLEN"] = m2isar.metamodel.arch.Constant("XLEN", value=32, attributes={}, size=None, signed=False)
        main_reg = m2isar.metamodel.arch.Memory(
            "X",
            m2isar.metamodel.arch.RangeSpec(32),
            size=32,
            attributes={m2isar.metamodel.arch.MemoryAttribute.IS_MAIN_REG: []},
        )
        main_mem = m2isar.metamodel.arch.Memory(
            "MEM",
            m2isar.metamodel.arch.RangeSpec(1 << 32),
            size=8,
            attributes={m2isar.metamodel.arch.MemoryAttribute.IS_MAIN_MEM: []},
        )
        pc = m2isar.metamodel.arch.Memory(
            "PC",
            m2isar.metamodel.arch.RangeSpec(0),
            size=32,
            attributes={m2isar.metamodel.arch.MemoryAttribute.IS_PC: []},
        )
        memories["X"] = main_reg
        memories["MEM"] = main_mem
        memories["PC"] = pc
    functions = {}
    intrinsics = {}
    generated_sets = []
    contributing_types = all_cdsl_sets
    for set_name, set_file in generated_sets:
        instr_set = parse_cdsl2_set(set_file)
        print("instr_set", instr_set)
        instructions.update(instr_set.instructions)
        contributing_types.append(set_name)
    generated_core = m2isar.metamodel.arch.CoreDef(
        name=core_name,
        contributing_types=contributing_types,
        template=None,  # type: ignore
        constants=constants,
        memories=memories,
        memory_aliases={},
        functions=functions,
        instructions=instructions,
        instr_classes={32},
        intrinsics=intrinsics,
    )
    print("generated_core", generated_core)
    input("!!")


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_etiss_core(sess, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    # TODO: !
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
