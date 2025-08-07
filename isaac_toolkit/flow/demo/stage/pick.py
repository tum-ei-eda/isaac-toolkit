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
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.generate.ise.check_ise_potential_per_llvm_bb import (
    check_ise_potential_per_llvm_bb,
)
from isaac_toolkit.generate.ise.check_ise_potential import check_ise_potential
from isaac_toolkit.generate.ise.choose_bbs import choose_bbs

logger = get_logger()


def analyze_bottleneck_bbs(sess: Session, force: bool = False, progress: bool = False):
    logger.info("Analyzing bottleneck BBs...")
    config = sess.config
    flow_config = config.flow
    assert flow_config is not None
    demo_config = flow_config.demo
    assert demo_config is not None
    choose_config = demo_config.choose
    assert choose_config is not None
    check_ise_potential(
        sess,
        min_supported=choose_config.check_potential_min_supported,
        allow_mem=choose_config.allow_mem,
        allow_loads=choose_config.allow_loads,
        allow_stores=choose_config.allow_stores,
        allow_branches=choose_config.allow_branches,
        allow_compressed=choose_config.allow_compressed,
        allow_custom=choose_config.allow_custom,
        allow_fp=choose_config.allow_fp,
        allow_system=choose_config.allow_system,
        force=force,
    )
    check_ise_potential_per_llvm_bb(
        sess,
        min_supported=choose_config.check_potential_min_supported,
        allow_mem=choose_config.allow_mem,
        allow_loads=choose_config.allow_loads,
        allow_stores=choose_config.allow_stores,
        allow_branches=choose_config.allow_branches,
        allow_compressed=choose_config.allow_compressed,
        allow_custom=choose_config.allow_custom,
        allow_fp=choose_config.allow_fp,
        allow_system=choose_config.allow_system,
        force=force,
    )
    choose_bbs(
        sess,
        threshold=choose_config.bb_threshold,
        min_weight=choose_config.bb_min_weight,
        min_supported_weight=choose_config.bb_min_supported_weight,
        min_instrs=choose_config.bb_min_instrs,
        max_num=choose_config.bb_max_num,
        force=force,
    )


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    set_log_level(console_level=args.log, file_level=args.log)
    analyze_bottleneck_bbs(sess, force=args.force)
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
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
