import sys
import argparse

import pandas as pd


def calc_agg_counts(count_files, output=None):

    dfs = []

    for count_file in count_files:
        df_ = pd.read_pickle(count_file)
        dfs.append(df_)

    df = pd.concat(dfs)
    # print("df", df)

    df = df.groupby("instr").sum().reset_index()
    # df = df.sort_values("util_score", ascending=False)
    df = df.sort_values("estimated_reduction_rel", ascending=False)
    # print("df", df)

    if output is None:
        print(df)
    else:
        df.to_csv(output, index=False)


def handle(args):
    calc_agg_counts(args.index, output=args.output)


def get_parser():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("count_files", nargs="+", help="Pickle files")
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
