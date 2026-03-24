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


def annotate_global_artifacts(data, index, inplace=False, output=None):
    # print("data", data)
    mapping = dict([tuple(x.split("=", 1)) for x in data])
    # print("mapping", mapping)

    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
    # candidates = combined_index_data["candidates"]
    # print("candidates", candidates)
    if isinstance(combined_index_data["global"]["artifacts"], list):
        assert len(combined_index_data["global"]["artifacts"]) == 0
        combined_index_data["global"]["artifacts"] = {}
    combined_index_data["global"]["artifacts"].update(mapping)

    if inplace:
        assert output is None
        out_file = index
    else:
        assert output is not None
        out_file = output

    with open(out_file, "w") as f:
        yaml.dump(combined_index_data, f)


def handle(args):
    annotate_global_artifacts(args.data, args.index, inplace=args.inplace, output=args.output)


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--data", action="append", help="TODO")
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
