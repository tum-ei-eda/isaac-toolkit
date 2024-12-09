import re
import sys
import argparse
from pathlib import Path

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, DisassArtifact, TableArtifact


def load_disass(sess: Session, input_file: Path, force: bool = False):
    assert input_file.is_file()
    name = input_file.name
    attrs = {
        "target": "riscv",  # TODO: 32/64?
        "kind": "disass",
        "elf": None,  # TODO
        "by": "isaac_toolkit.frontend.disass.objdump",
    }
    artifact = DisassArtifact(name, input_file, attrs=attrs)
    sess.add_artifact(artifact, override=force)

    with open(input_file, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if ":" not in line:
            continue
        addr, rest = line.split(":")
        addr = addr.strip()
        # print("rest", rest)
        splitted = rest.split("\t")
        splitted = list(filter(None, splitted))
        if len(splitted) < 2 or len(splitted) > 3:
            continue
        # print("splitted", splitted)
        addr = int(addr, 16)
        rest = rest.strip()
        # print("addr", addr)
        word = splitted[0]
        word = word.strip()
        word = int(word, 16)
        # print("word", word)
        insn = splitted[1]
        insn = insn.strip()
        # print("insn", insn)
        if len(splitted) == 3:
            args = splitted[2]
            args = args.strip()
        else:
            args = ""
        # print("args", args)
        # input(">")
        new = {"pc": addr, "bytecode": word, "instr": insn, "args": args}
        data.append(new)
    disass_df = pd.DataFrame(data)
    # name_ = f"{name}_table"
    disass_artifact = TableArtifact("disass", disass_df, attrs=attrs)
    sess.add_artifact(disass_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_file = Path(args.file)
    load_disass(sess, input_file, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
