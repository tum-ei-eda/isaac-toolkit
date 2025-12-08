import sys
import argparse

import yaml
import pandas as pd


def names_helper(index, output=None):
    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
        # TODO: index and cdsl should use the same instruction names?
        names = [
            candidate["properties"].get("InstrName", f"CUSTOM{i}")
            for i, candidate in enumerate(combined_index_data["candidates"])
        ]
        print("names", names)
        # num_candidates = len(names)
        names_df = pd.DataFrame({"instr": names})
        names_df["instr_lower"] = names_df["instr"].apply(lambda x: x.lower())
        names_df["idx"] = names_df["instr"].apply(lambda x: names.index(x))

    if output is None:
        print(names_df)
    else:
        names_df.to_csv(output, index=False)


def handle(args):
    names_helper(args.index, output=args.output)


def get_parser():
    parser = argparse.ArgumentParser(description="Extract instr names from index YAML and write to CSV")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output CSV file")
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
