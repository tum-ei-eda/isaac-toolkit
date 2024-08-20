import sys
import pandas as pd
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, InstrTraceArtifact


# TODO: logger


def load_instr_trace(sess: Session, input_file: Path, force: bool = False):
    assert input_file.is_file()
    name = input_file.name
    df = pd.read_csv(input_file, sep=":", names=["pc", "rest"])
    df["pc"] = df["pc"].apply(lambda x: int(x, 0))
    # TODO: normalize instr names
    df[["instr", "rest"]] = df["rest"].str.split(" # ", n=1, expand=True)
    df["instr"] = df["instr"].apply(lambda x: x.strip())
    df[["bytecode", "operands"]] = df["rest"].str.split(" ", n=1, expand=True)
    df["bytecode"] = df["bytecode"].apply(
        lambda x: int(x, 16) if "0x" in x else (int(x, 2) if "0b" in x else int(x, 2))
    )
    MEM_OPTIMIZED = True
    if MEM_OPTIMIZED:
        df["instr"] = df["instr"].astype("category")
        df["pc"] = pd.to_numeric(df["pc"])
        df["bytecode"] = pd.to_numeric(df["bytecode"])

    def convert(x):
        ret = {}
        for y in x:
            if len(y.strip()) == 0:
                continue
            assert "=" in y
            k, v = y.split("=", 1)
            assert k not in ret
            ret[k] = int(v)
        return ret

    df["operands"] = df["operands"].apply(lambda x: convert(x[1:-1].split(" | ")))
    # df.drop(columns=["operands"], inplace=True)
    df.drop(columns=["rest"], inplace=True)

    attrs = {
        "simulator": "etiss",
        "cpu_arch": "unknown",
        "by": "isaac_toolkit.frontend.instr_trace.etiss",
    }
    artifact = InstrTraceArtifact(name, df, attrs=attrs)
    sess.add_artifact(artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_file = Path(args.file)
    load_instr_trace(sess, input_file, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
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
