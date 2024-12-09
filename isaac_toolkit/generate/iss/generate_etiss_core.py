import sys
import yaml
import logging
import pickle
import argparse
from typing import Optional, Union, List
from pathlib import Path

import m2isar
import m2isar.metamodel.arch
from m2isar.metamodel import M2_METAMODEL_VERSION, M2Model

from m2isar.frontends.coredsl2_set.parser import parse_cdsl2_set
from m2isar.backends.coredsl2_set.writer import gen_cdsl_code
from m2isar.transforms.encode_instructions.encoder import encode_instructions

from isaac_toolkit.session import Session


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
        "Zicsr": "tum_csr",
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


def get_cdsl_includes(
    sets: List[str],
    base_dir: Union[str, Path] = "rv_base",
    tum_dir: Union[str, Path] = ".",
):
    # Reverse lookup required!
    cdsl_includes = {
        "RISCVBase.core_desc": ["RISCVBase"],
        # "RV32I.core_desc": ["RV32I", "Zicsr", "Zifencei", "RVNMode", "RVSMode", "RVDebug"],
        "RV64I.core_desc": ["RV64I"],
        "RVA.core_desc": ["RV32A", "RV64A"],
        "RVC.core_desc": [
            "RV32IC",
            "RV32FC",
            "RV32DC",
            "RV64IC",
            "RV64FC",
            "RV64DC",
            "RV128IC",
        ],
        "RVD.core_desc": ["RV32D", "RV64D"],
        "RVF.core_desc": ["RV32F", "RV64F"],
        "RVI.core_desc": [
            "RV32I",
            "RV64I",
            "Zicsr",
            "Zifencei",
            "RVNMode",
            "RVSMode",
            "RVDebug",
        ],
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


def get_includes_code(includes: List[Union[str, Path]], prefix: str = ""):
    ret = []
    for inc in includes:
        if isinstance(inc, Path):
            inc = str(inc)
        if ".core_desc" not in inc:
            inc = f"{inc}.core_desc"
        if inc[0] != "/":
            inc = f"{prefix}{inc}"
        new = f'import "{inc}"'
        ret.append(new)
    return "\n".join(ret)


# def generate_base_core
#     core_name: str = "IsaacCore",
#     xlen: int = 32,
#     ignore_etiss: bool = False,
#     semihosting: bool = True,
#     base_extensions: List[str] = ["i", "m", "a", "f", "d", "c", "zifencei"],
#     # etiss_overrides: List[str] = ["tum_csr", "tum_ret", "tum_rva", "tum_semihosting"],
# ):


def generate_etiss_core(
    sess: Session,
    workdir: Optional[Union[str, Path]] = None,
    core_name: str = "IsaacCore",
    set_name: str = "XIsaac",
    xlen: int = 32,
    ignore_etiss: bool = False,
    semihosting: bool = True,
    base_extensions: List[str] = ["i", "m", "a", "f", "d", "c", "zifencei", "zicsr"],
    auto_encoding: bool = True,
    split: bool = True,  # One set per new instr
    force: bool = False,
    base_dir: str = "rv_base",
    tum_dir: str = ".",
    skip_errors: bool = True,
):
    # artifacts = sess.artifacts
    # TODO: get combined_index.yml from artifacts!
    assert workdir is not None
    if not isinstance(workdir, Path):
        workdir = Path(workdir)
    assert workdir.is_dir()
    combined_index_file = workdir / "combined_index.yml"
    core_out_model_file = workdir / f"{core_name}.m2isarmodel"
    set_out_model_file = workdir / f"{set_name}.m2isarmodel"
    core_out_cdsl_file = workdir / f"{core_name}.core_desc"
    set_out_cdsl_file = workdir / f"{set_name}.core_desc"
    set_hls_out_cdsl_file = workdir / f"{set_name}.hls.core_desc"
    errs_file = workdir / "errs.txt"
    with open(combined_index_file, "r") as f:
        index_data = yaml.safe_load(f)
    # print("index_data", index_data)
    generated_sets = []
    for i, sub_data in enumerate(index_data["candidates"]):
        # print("i", i)
        # print("sub_data", sub_data)
        sub_artifacts = sub_data["artifacts"]
        # print("sub_artifacts", sub_artifacts)
        # sub_properties = sub_data["properties"]
        # print("sub_properties", sub_properties)
        sub_name = f"C{i}"
        sub_file = Path(sub_artifacts["cdsl"])
        generated_sets.append((sub_name, sub_file))
        # input(">")
    # print("generated_sets", generated_sets)
    # input(">>")
    # Not uses set() here as we want to preserve the order!
    all_cdsl_sets = []
    instr_classes = {32}
    compressed = "c" in base_extensions
    instr_classes.add(16)
    for ext in base_extensions:
        if ext == "c":
            continue
        cdsl_sets = get_cdsl_sets(ext, xlen=xlen, compressed=compressed)
        for cdsl_set in cdsl_sets:
            if cdsl_set not in all_cdsl_sets:
                all_cdsl_sets.append(cdsl_set)
    if not ignore_etiss:
        all_cdsl_sets = apply_etiss_overrides(all_cdsl_sets, semihosting=semihosting)
    # print("all_cdsl_sets", all_cdsl_sets)
    core_includes = get_cdsl_includes(all_cdsl_sets, base_dir=base_dir, tum_dir=tum_dir)
    core_includes.add(set_out_cdsl_file.resolve())
    # print("cdsl_includes", cdsl_includes)
    core_includes_code = get_includes_code(core_includes)
    constants = {}
    memories = {}
    unencoded_instructions = {}
    constants["XLEN"] = m2isar.metamodel.arch.Constant(
        "XLEN", value=xlen, attributes={}, size=None, signed=False
    )
    # if ignore_etiss:
    if True:
        main_reg = m2isar.metamodel.arch.Memory(
            "X",
            m2isar.metamodel.arch.RangeSpec(32),
            size=xlen,
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
    contributing_types = all_cdsl_sets
    # extra_includes = ["/work/git/students/cecil/etiss_arch_riscv/rv_base"]  # TODO: Do not hardcode
    extra_includes = [
        "/mnt/wd8tb/Data/students_archive/cecil/etiss_arch_riscv/rv_base"
    ]  # TODO: Do not hardcode
    # print("SSSS")
    name_idx = 0
    errs = {}
    for set_name_, set_file in generated_sets:
        # print("set_name_", set_name_)
        # print("set_file", set_file)
        # input(">>>")
        try:
            models = parse_cdsl2_set(set_file, extra_includes)
        except Exception as e:
            errs[set_name_] = str(e)
            if skip_errors:
                logger.exception(e)
            else:
                raise e
        instr_sets = list(models.values())
        # for instr_set in instr_sets:
        #     print("instr_set", instr_set)
        #     print("instr_set.instructions", instr_set.instructions)
        #     print("instr_set.unencoded_instructions", instr_set.unencoded_instructions)
        # print("instr_sets", instr_sets)
        instr_sets = [
            instr_set
            for instr_set in instr_sets
            if len(instr_set.instructions) > 0
            or len(instr_set.unencoded_instructions) > 0
        ]
        # print("instr_sets", instr_sets)
        if len(instr_sets) == 1:
            instr_set = instr_sets[0]
        else:
            instr_set = instr_sets[-1]
        # print("instr_set", instr_set)
        assert len(instr_set.instructions) == 0
        assert len(instr_set.unencoded_instructions) > 0
        # instructions.update(instr_set.instructions)
        for instr_def in instr_set.unencoded_instructions.values():
            name = f"CUSTOM{name_idx}"
            name_idx += 1
            instr_def.name = name
            instr_def.mnemonic = name.lower()
            unencoded_instructions[name] = instr_def
        # contributing_types.append(set_name_)
        # break  # TODO
    unencoded_instructions_ = list(unencoded_instructions.values())
    encoded_instructions_ = encode_instructions(unencoded_instructions_)
    print("encoded_instructions_", encoded_instructions_, len(encoded_instructions_))
    encoded_instructions = {
        (instr_def.code, instr_def.mask): instr_def
        for instr_def in encoded_instructions_
    }
    contributing_types.append(set_name)
    print("encoded_instructions", encoded_instructions, len(encoded_instructions))
    instructions = {}
    instructions.update(encoded_instructions)
    # input("1")
    generated_core = m2isar.metamodel.arch.CoreDef(
        name=core_name,
        contributing_types=contributing_types,
        template=None,  # type: ignore
        constants=constants,
        memories=memories,
        memory_aliases={},
        functions=functions,
        # instructions=instructions,
        instructions={},
        unencoded_instructions={},
        instr_classes=instr_classes,
        intrinsics=intrinsics,
    )
    extension = [f"RV{xlen}I"]  # TODO: add include?
    set_includes = get_cdsl_includes(extension, base_dir=base_dir, tum_dir=tum_dir)
    set_includes_code = get_includes_code(set_includes)
    set_hls_includes = []  # TODO: fix duplicate includes
    set_hls_includes_code = get_includes_code(set_hls_includes)
    generated_set = m2isar.metamodel.arch.InstructionSet(
        name=set_name,
        extension=extension,
        constants={},
        memories={},
        functions={},
        instructions=encoded_instructions,
        unencoded_instructions={},
    )
    # print("generated_core", generated_core)
    # print("generated_set", generated_set)
    with open(core_out_model_file, "wb") as f:
        cores = {core_name: generated_core}
        model_obj = M2Model(M2_METAMODEL_VERSION, cores, {}, {})
        pickle.dump(model_obj, f)
    core_cdsl_code = gen_cdsl_code(model_obj, with_includes=False)
    with open(core_out_cdsl_file, "w") as f:
        f.write(core_includes_code)
        f.write("\n\n")
        f.write(core_cdsl_code)
    with open(set_out_model_file, "wb") as f:
        sets = {set_name: generated_set}
        model_obj = M2Model(M2_METAMODEL_VERSION, {}, sets, {})
        pickle.dump(model_obj, f)
    set_cdsl_code = gen_cdsl_code(model_obj, with_includes=False)
    with open(set_out_cdsl_file, "w") as f:
        f.write(set_includes_code)
        f.write("\n\n")
        f.write(set_cdsl_code)
    model_obj.sets[set_name].memories["X"] = main_reg
    model_obj.sets[set_name].extension = []
    set_hls_cdsl_code = gen_cdsl_code(model_obj, with_includes=False, legacy=True)
    with open(set_hls_out_cdsl_file, "w") as f:
        f.write(set_hls_includes_code)
        f.write("\n\n")
        f.write(set_hls_cdsl_code)
    with open(errs_file, "w") as f:
        errs_text = "\n".join([f"{name}: {err}" for name, err in errs.items()])
        f.write(errs_text)
    # input("!!")


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_etiss_core(
        sess,
        force=args.force,
        workdir=args.workdir,
        core_name=args.core_name,
        set_name=args.set_name,
        xlen=args.xlen,
        ignore_etiss=args.ignore_etiss,
        semihosting=args.semihosting,
        base_extensions=args.base_extensions.split(","),
        auto_encoding=args.auto_encoding,
        split=args.split,
        base_dir=args.base_dir,
        tum_dir=args.tum_dir,
        skip_errors=args.skip_errors,
    )
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
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--core-name", type=str, default="IsaacCore")
    parser.add_argument("--set-name", type=str, default="XIsaac")
    parser.add_argument("--xlen", type=int, default=32)
    parser.add_argument("--ignore_etiss", action="store_true")
    parser.add_argument("--semihosting", action="store_true")
    parser.add_argument("--base-extensions", type=str, default="i,m,a,f,d,c,zifencei")
    parser.add_argument("--auto-encoding", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--base-dir", type=str, default="rv_base")
    parser.add_argument("--tum-dir", type=str, default=".")
    parser.add_argument("--skip-errors", action="store_true")

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
