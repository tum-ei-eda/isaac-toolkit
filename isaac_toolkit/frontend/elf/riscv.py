import sys
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, FileArtifact


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    input_file = Path(args.file)
    assert input_file.is_file()
    name = input_file.name
    attrs = {}  # TODO
    artifact = FileArtifact(name, input_file, flags=ArtifactFlag.INPUT | ArtifactFlag.ELF, attrs=attrs)
    sess.artifacts.append(artifact)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
