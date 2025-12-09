import sys
import argparse
from pathlib import Path

import pandas as pd


def analyze_compare(report, mem_report=None, output=None, print_df=False):
    COLS = ["Model", "Arch", "Run Instructions", "Run Instructions (rel.)"]
    MEM_COLS = ["Model", "Arch", "Total ROM", "Total RAM", "ROM code", "ROM code (rel.)"]
    COMMON_COLS = list(set(COLS) & set(MEM_COLS))

    report_file = Path(report)
    assert report_file.is_file()
    report_df = pd.read_csv(report_file)[COLS]
    # print(report_df, report_df.columns)

    if mem_report:
        mem_report_file = Path(mem_report)
        assert mem_report_file.is_file()
        mem_report_df = pd.read_csv(mem_report_file)[MEM_COLS]
        # print(mem_report_df, mem_report_df.columns)
        report_df = report_df.merge(mem_report_df, on=COMMON_COLS)

    # print(report_df, report_df.columns)

    df = report_df

    assert print_df or output is not None

    if print_df:
        with pd.option_context("display.max_columns", None):
            print(df)

    if output is not None:
        # TODO: assert csv suffix?
        df.to_csv(output)


def handle(args):
    analyze_compare(
        args.report,
        mem_report=args.mem_report,
        output=args.output,
        print_df=args.print_df,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Collect metrics from MLonMCU Report")
    parser.add_argument("report", help="Report CSV file")
    parser.add_argument("--mem-report", default=None, help="Memory report CSV file")
    # parser.add_argument("--mem", action="store_true", help="Compare mem instead of Runtime")
    # parser.add_argument("-i", "--isax", default="auto", help="ISAX name")
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
