import sys
import argparse
from pathlib import Path

import pandas as pd


def analyze_compare_perf(compare_perf_csv, uarchs_csv, output=None, print_df=False):

    compare_perf_csv = Path(compare_perf_csv)
    assert compare_perf_csv.is_file()
    compare_perf_df = pd.read_csv(compare_perf_csv)
    if "Unnamed: 0" in compare_perf_df.columns:
        compare_perf_df.drop(columns=["Unnamed: 0"], inplace=True)
    print(compare_perf_df, compare_perf_df.columns, len(compare_perf_df))

    uarchs_csv = Path(uarchs_csv)
    assert uarchs_csv.is_file()
    uarchs_df = pd.read_csv(uarchs_csv)
    # uarchs_df.rename(index={0: "idx"}, inplace=True)
    if "Unnamed: 0" in uarchs_df.columns:
        uarchs_df.drop(columns=["Unnamed: 0"], inplace=True)
    print(uarchs_df, uarchs_df.columns, len(uarchs_df))

    uarchs_bench_df = compare_perf_df[["uArch", "Run Cycles", "Run Cycles (rel.)", "Run CPI"]]
    uarchs_bench_df = uarchs_bench_df.iloc[1:]
    sorted_uarchs_bench_df = uarchs_bench_df.sort_values("Run CPI")
    print("uarchs_bench_df")
    print(uarchs_bench_df)
    print("sorted_uarchs_bench_df")
    print(sorted_uarchs_bench_df)
    sorted_uarchs = list(sorted_uarchs_bench_df["uArch"].values)
    sorted_cpis = list(sorted(set(list(sorted_uarchs_bench_df["Run CPI"].values))))
    # TODO: same rank if same score?
    # TODO: replace CPI with Score
    print("sorted_uarchs", sorted_uarchs)
    print("sorted_cpis", sorted_cpis)
    merged_df = pd.merge(uarchs_bench_df, uarchs_df, left_on="uArch", right_on="uarch", how="inner", suffixes=("", ""))
    merged_df.drop(columns=["uarch"], inplace=True)
    # merged_df["Rank"] = merged_df["uArch"].apply(lambda x: sorted_uarchs.index(x) + 1)
    merged_df["Rank"] = merged_df["Run CPI"].apply(lambda x: sorted_cpis.index(x) + 1)
    merged_df = merged_df.sort_values("Rank")
    # merged_df.drop(columns=["Run Cycles", "Run Cycles (rel.)", "Run CPI", "uarch_lower", "variant"], inplace=True)
    merged_df.drop(columns=["Run Cycles", "Run Cycles (rel.)", "uarch_lower", "variant"], inplace=True)
    # TODO: attach costs

    df = merged_df

    assert print_df or output is not None

    if print_df:
        with pd.option_context("display.max_columns", None, "display.width", 1000, "display.max_colwidth", 400, "display.min_rows", 100):
            print(df)

    if output is not None:
        # TODO: assert csv suffix?
        df.to_csv(output)


def handle(args):
    analyze_compare_perf(
        args.report,
        args.uarchs_csv,
        output=args.output,
        print_df=args.print_df,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Collect metrics from MLonMCU Report")
    parser.add_argument("report", help="Compare Perf CSV file")
    parser.add_argument("--uarchs-csv", default=None, required=True, help="uArchs CSV file")
    parser.add_argument("-o", "--output", default=None, help="Output file")
    parser.add_argument("--print-df", action="store_true", help="Print DataFrame")
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
