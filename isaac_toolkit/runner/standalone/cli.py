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
import os
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.cli.utils import parse_override_args

# from .etiss import parse_etiss_args
from .registry import lookup_simulator

logger = get_logger()


def add_common_args(parser):
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument(
        "--simulator",
        type=str,
        choices=["etiss", "etiss_perf", "spike", "spike_bm", "tgc", "dbt", "vicuna"],
        default=None,
    )
    parser.add_argument("--session", "--sess", "-s", type=str)
    # parser.add_argument("--simulator", type=str, required=True)
    parser.add_argument(
        "-c",
        "--config",
        "--config-overrides",
        "--overrides",
        dest="config",
        metavar="KEY=VALUE",
        nargs="+",
        action="append",
        help=(
            "Set a number of key-value pairs "
            "(do not put spaces before or after the = sign). "
            "If a value contains spaces, you should define "
            "it with double quotes: "
            'foo="this is a sentence". Note that '
            "values are always treated as strings."
        ),
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--dest", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trace", action="store_true")  # TODO: expose make target?
    parser.add_argument("--load", action="store_true")
    # TODO: no-load?
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")


EXAMPLES_DIR = os.environ.get("EXAMPLES_DIR")
STANDALONE_EXAMPLES_DIR = os.environ.get(
    "STANDALONE_EXAMPLES_DIR", (Path(EXAMPLES_DIR) / "standalone") if EXAMPLES_DIR is not None else None
)


def handle_coremark(args):
    # print("handle_coremark")
    # print("args", args)
    coremark_dir = args.dir
    if coremark_dir is None:
        coremark_dir = os.environ.get(
            "COREMARK_EXAMPLE_DIR",
            (Path(STANDALONE_EXAMPLES_DIR) / "coremark") if STANDALONE_EXAMPLES_DIR is not None else None,
        )
    assert coremark_dir is not None, "Please define COREMARK_EXAMPLE_DIR"
    coremark_dir = Path(coremark_dir)
    assert coremark_dir.is_dir(), f"Missing: {coremark_dir}"
    coremark_overrides = {}
    if args.coremark_iterations is not None:
        coremark_overrides["COREMARK_ITERATIONS"] = args.coremark_iterations
    handle(args, coremark_dir.resolve(), coremark_overrides)


def handle_embench(args):
    # print("handle_embench")
    # print("args", args)
    embench_dir = args.dir
    if embench_dir is None:
        embench_dir = os.environ.get(
            "EMBENCH_EXAMPLE_DIR",
            (Path(STANDALONE_EXAMPLES_DIR) / "embench") if STANDALONE_EXAMPLES_DIR is not None else None,
        )
    assert embench_dir is not None, "Please define EMBENCH_EXAMPLE_DIR"
    embench_dir = Path(embench_dir)
    assert embench_dir.is_dir(), f"Missing: {embench_dir}"
    embench_overrides = {}
    handle(args, embench_dir.resolve(), embench_overrides)


def handle_custom(args):
    # print("handle_custom")
    # print("args", args)
    custom_dir = args.dir
    custom_dir = Path(custom_dir)
    assert custom_dir.is_dir(), f"Missing: {custom_dir}"
    custom_overrides = {"PROG": "custom"}
    handle(args, custom_dir.resolve(), custom_overrides)


def add_prog_args(parser):
    subparsers = parser.add_subparsers(dest="prog", required=True)
    coremark_parser = subparsers.add_parser("coremark")
    coremark_parser.add_argument("--dir", type=str, default=None)
    coremark_parser.add_argument("--coremark-iterations", "--iter", type=int, default=None)
    embench_parser = subparsers.add_parser("embench")
    embench_parser.add_argument("--dir", type=str, default=None)
    embench_parser.add_argument("--bench", type=str, default="crc32")
    embench_parser.add_argument("--global-scale-factor", "--gsf", type=int, default=None)
    custom_parser = subparsers.add_parser("custom")
    custom_parser.add_argument("--dir", type=str, default=None, required=True)

    coremark_parser.set_defaults(func=handle_coremark)
    embench_parser.set_defaults(func=handle_embench)
    custom_parser.set_defaults(func=handle_custom)


def handle(args, make_dir, extra_overrides):
    sess = None
    if args.session is not None:
        session_dir = Path(args.session)
        assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
        sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    overrides = parse_override_args(args)
    if extra_overrides:
        overrides = {**extra_overrides, **overrides}
    found = lookup_simulator(args.simulator)
    assert found, f"Invoke func for found for simulator: {args.simulator}"
    parser_func, invoke_func = found
    runner_kwargs = parser_func(args)
    invoke_func(
        sess,
        make_dir,
        program=args.prog,
        overrides=overrides,
        label=args.label,
        dest_dir=args.dest,
        verbose=args.verbose,
        resume=args.resume,
        trace=args.trace,
        load=args.load,
        cleanup=args.cleanup,
        force=args.force,
        **runner_kwargs,
    )
    sess.save()


# def get_parser():
#     parser = argparse.ArgumentParser()
#     add_common_args(parser)
#     add_prog_args(parser)
#     return parser
#
#
# def main(argv):
#     parser = get_parser()
#     args = parser.parse_args(argv)
#     args.func(args)
#
#
# if __name__ == "__main__":
#     main(sys.argv[1:])
