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

# from isaac_toolkit.logging import get_logger, set_log_level
from isaac_toolkit.analysis.static.dwarf import analyze_dwarf

# from isaac_toolkit.analysis.static.llvm_bbs import analyze_llvm_bbs
from isaac_toolkit.analysis.static.mem_footprint import analyze_mem_footprint
from isaac_toolkit.analysis.static.linker_map import analyze_linker_map
from isaac_toolkit.analysis.static.histogram.disass_instr import (
    create_disass_instr_hist,
)
from isaac_toolkit.analysis.static.histogram.disass_opcode import (
    create_disass_opcode_hist,
)

# from isaac_toolkit.analysis.dynamic.trace.trunc_trace import trunc_trace
from isaac_toolkit.analysis.dynamic.trace.instr_operands import analyze_instr_operands
from isaac_toolkit.analysis.dynamic.histogram.opcode import create_opcode_hist
from isaac_toolkit.analysis.dynamic.histogram.pc   import create_pc_hist

# from isaac_toolkit.analysis.dynamic.histogram.opcode_per_llvm_bb import (
#     create_opcode_per_llvm_bb_hist,
# )
from isaac_toolkit.analysis.dynamic.histogram.instr import create_instr_hist
# from isaac_toolkit.analysis.dynamic.trace.basic_blocks import analyze_basic_blocks
from isaac_toolkit.analysis.dynamic.trace.trace_bbs import collect_trace_bbs
from isaac_toolkit.analysis.dynamic.trace.map_llvm_bbs_new import map_llvm_bbs_new
from isaac_toolkit.analysis.dynamic.trace.track_used_functions import (
    track_unused_functions,
)

# logger = get_logger()
import logging

logger = logging.getLogger()


def analyze_artifacts(sess: Session, force: bool = False, progress: bool = False):
    logger.info("Analyzing RVF Demo artifacts...")
    analyze_dwarf(sess, force=force)
    # analyze_llvm_bbs(sess, force=force)
    analyze_mem_footprint(sess, force=force)
    analyze_linker_map(sess, force=force)
    # trunc_trace(sess, start_func="mlonmcu_run", force=force)
    # trunc_trace(sess, end_func="stop_bench", force=force)
    # analyze_instr_operands(sess, force=force)  # TODO: separate df and detect if available
    create_opcode_hist(sess, force=force)
    # create_opcode_per_llvm_bb_hist(sess, force=force)
    create_instr_hist(sess, force=force)
    create_pc_hist(sess, force=force)
    create_disass_instr_hist(sess, force=force)
    create_disass_opcode_hist(sess, force=force)
    # analyze_basic_blocks(sess, force=force)
    collect_trace_bbs(sess, force=force)
    # map_llvm_bbs_new(sess, force=force)
    track_unused_functions(sess, force=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    # set_log_level(console_level=args.log, file_level=args.log)
    analyze_artifacts(sess, force=args.force)
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
