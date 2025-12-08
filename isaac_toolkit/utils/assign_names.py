import sys
import argparse

import pandas as pd
import yaml


def assign_names(index, output=None, inplace=False, prefix="CUSTOM", csv=None, pkl=None):

    with open(index, "r") as f:
        combined_index_data = yaml.safe_load(f)
    candidates = combined_index_data["candidates"]
    # num_candidates = len(candidates)
    # print("candidates", candidates)

    names_data = []

    for i, candidate in enumerate(candidates):
        # print("i", i)
        properties = candidate.get("properties", {})
        # print("properties", properties)
        new_name = f"{prefix}{i}"
        properties["InstrName"] = new_name
        candidate["properties"] = properties
        num_fused_instrs = properties["#Instrs"]
        new_data = {"instr": new_name, "instr_lower": new_name.lower(), "idx": i, "num_fused_instrs": num_fused_instrs}
        names_data.append(new_data)

    names_df = pd.DataFrame(names_data)
    # print(names_df)

    if csv is not None:
        names_df.to_csv(csv, index=False)

    if pkl is not None:
        names_df.to_pickle(pkl)

    if inplace:
        assert output is None
        out_file = index
    else:
        assert output is not None
        out_file = output

    with open(out_file, "w") as f:
        yaml.dump(combined_index_data, f)


def handle(args):
    assign_names(args.index, output=args.output, inplace=args.inplace, prefix=args.prefix, csv=args.csv, pkl=args.pkl)


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("index", help="Index yaml file")
    parser.add_argument("-o", "--output", default=None, help="Output yaml file")
    parser.add_argument("--inplace", action="store_true", help="TODO")
    parser.add_argument("--prefix", default="CUSTOM", help="TODO")
    parser.add_argument("--csv", default=None, help="TODO")
    parser.add_argument("--pkl", default=None, help="TODO")
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
