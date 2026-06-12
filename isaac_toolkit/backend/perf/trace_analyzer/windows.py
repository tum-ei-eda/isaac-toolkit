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
from typing import List, Optional, Union


import sys
import logging
import argparse
from math import ceil
from pathlib import Path
from dataclasses import dataclass

import yaml
import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts

logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


@dataclass
class Window:
    start_idx: int
    end_idx: int


def validate_trace_index(df):
    expected = pd.RangeIndex(len(df))

    if not df.index.equals(expected):
        raise ValueError(
            "Instruction trace DataFrame index must be contiguous "
            f"and start at 0. Got [{df.index.min()}..{df.index.max()}] "
            f"with {len(df)} rows."
        )


def gen_windows(
    df,
    num_windows: Optional[int] = None,
    window_size: Optional[int] = None,
    overlap: float = 0.0,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    length: Optional[int] = None,
):
    assert (num_windows is None) != (window_size is None), "Exactly one of num_windows or window_size must be specified"

    # TODO: check if iloc or loc shall be used
    assert 0.0 <= overlap < 1.0

    n_total = len(df)

    start_idx = 0 if start_idx is None else start_idx
    # print("n_total", n_total)
    # print("length", length)

    if end_idx is None:
        if length is not None:
            end_idx = min(start_idx + length, n_total)
        else:
            end_idx = n_total
    else:
        assert length is None

    assert start_idx < end_idx

    region_size = end_idx - start_idx
    # print("region_size", region_size)

    windows = []

    if window_size is not None:
        assert num_windows is None
        stride = max(1, int(round(window_size * (1.0 - overlap))))

        cur = start_idx

        while cur + window_size <= end_idx:
            windows.append(
                Window(
                    start_idx=cur,
                    end_idx=cur + window_size,
                )
            )

            cur += stride

    else:
        # num_windows specified
        assert num_windows is not None
        assert window_size is None

        if num_windows == 1:
            windows.append(
                Window(
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )
        else:
            window_size = ceil(region_size / (1.0 + (num_windows - 1) * (1.0 - overlap)))
            # print("window_size", window_size)

            stride = max(
                1,
                int(round(window_size * (1.0 - overlap))),
            )

            cur = start_idx

            for _ in range(num_windows):
                w_end = min(cur + window_size, end_idx)

                windows.append(
                    Window(
                        start_idx=cur,
                        end_idx=w_end,
                    )
                )

                cur += stride

    return windows


def generate_windows_yaml(
    sess: Session,
    output: Optional[Union[str, Path]] = None,
    force: bool = False,
    num_windows: Optional[int] = None,
    window_size: Optional[int] = None,
    overlap: float = 0.0,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    length: Optional[int] = None,  # if end_idx is not provided, how many rows to consider after start_idx
):
    artifacts = sess.artifacts

    instr_trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(instr_trace_artifacts) == 1
    instr_trace_artifact = instr_trace_artifacts[0]
    instr_trace_df = instr_trace_artifact.df

    validate_trace_index(instr_trace_df)

    windows = gen_windows(
        instr_trace_df,
        num_windows=num_windows,
        window_size=window_size,
        overlap=overlap,
        start_idx=start_idx,
        end_idx=end_idx,
        length=length,
    )
    # print("len(windows)", len(windows))
    ranges_data = []

    for i, window in enumerate(windows):
        start = window.start_idx
        end = window.end_idx
        name = f"window{i}"
        new = [name, start, end]
        ranges_data.append(new)

    yaml_data = {"ranges": ranges_data}

    if output is None:
        profile_dir = sess.directory / "output"
        profile_dir.mkdir(exist_ok=True)
        out_name = "windows.yml"
        output = profile_dir / out_name
    if Path(output).is_file():
        assert force, f"Output file '{output}' already exists. Use --force to override!"
    with open(output, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    generate_windows_yaml(
        sess,
        output=args.output,
        force=args.force,
        num_windows=args.num_windows,
        window_size=args.window_size,
        overlap=args.overlap,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        length=args.length,
    )
    sess.save()
    # TODO: logging


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--num-windows", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)

    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
