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

import yaml


def sort_index(index, output=None, inplace=False, ascending=False, by=None, by2=None):
    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
    candidates = combined_index_data["candidates"]
    print("candidates", candidates)

    candidates = sorted(
        candidates, key=lambda x: (x["metrics"].get(by, 0.0), x["metrics"].get(by2, 0.0)), reverse=not ascending
    )
    combined_index_data["candidates"] = candidates
    print("sorted_candidates", candidates)

    if inplace:
        assert output is None
        out_file = index
    else:
        assert output is not None
        out_file = output

    with open(out_file, "w") as f:
        yaml.dump(combined_index_data, f)


def handle(args):
    sort_index(args.index, output=args.output, inplace=args.inplace, ascending=args.ascending, by=args.by, by2=args.by2)


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--ascending", action="store_true", help="TODO")
    parser.add_argument("--by", required=True, help="Metric used for sorting")
    parser.add_argument("--by2", default=None, help="Metric used for sorting 2nd level")
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
