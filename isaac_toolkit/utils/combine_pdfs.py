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
import argparse
import subprocess
from typing import Union
from pathlib import Path

import yaml


def dot2pdf_helper(in_file: Union[str, Path], out_file: Union[str, Path]):
    with open(out_file, "wb") as f:
        dot_args = ["dot", "-Tpdf", in_file]
        print(">", " ".join(map(str, dot_args)))
        _ = subprocess.run(dot_args, check=True, stdout=f)
    print(f"Converted {in_file} -> {out_file}")


def combine_pdfs(index, output=None):
    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)

    pdf_files = [
        Path(candidate_data["artifacts"]["io_sub"].replace(".pkl", ".pdf"))
        for candidate_data in combined_index_data["candidates"]
    ]
    dot_files = [
        Path(candidate_data["artifacts"]["io_sub"].replace(".pkl", ".dot"))
        for candidate_data in combined_index_data["candidates"]
    ]
    for i, pdf_file in enumerate(pdf_files):
        if not pdf_file.is_file():
            dot_file = dot_files[i]
            assert dot_file.is_file()
            dot2pdf_helper(dot_file, pdf_file)
    assert len(pdf_files) > 0, "No files found!"
    assert output is not None
    out_file = Path(output)
    combine_args = ["pdfunite", *pdf_files, out_file]
    # print(">", " ".join(map(str, combine_args)))
    _ = subprocess.run(combine_args, capture_output=True, text=True, check=True)
    # print(f"Wrote combined PDF file: {out_file}")


def handle(args):
    combine_pdfs(args.index, output=args.output)


def get_parser():
    parser = argparse.ArgumentParser(description="Extract dot files from index and merge graphs into single pdf")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
