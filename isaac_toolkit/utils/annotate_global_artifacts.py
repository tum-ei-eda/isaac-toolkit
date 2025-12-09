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
