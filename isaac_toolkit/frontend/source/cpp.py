import sys
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, SourceArtifact

from .c import get_parser, handle


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args, fmt="cpp")


if __name__ == "__main__":
    main(sys.argv[1:])
