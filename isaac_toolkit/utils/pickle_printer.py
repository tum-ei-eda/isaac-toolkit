import sys
import pickle

import pandas as pd


# TODO: argparse (--mem, ...)

def print_memory_footprint(df):
    print("================")
    print("Memory Footprint")
    print("================")
    mem = pd.concat([df.dtypes, df.memory_usage(deep=True)], axis=1).sort_values(1, ascending=False)
    mem.rename(columns={0: "dtype", 1: "mem"}, inplace=True)
    total = mem["mem"].sum()
    print(mem)
    print("TOTAL:", total)


def main():
    assert len(sys.argv) == 2, "Unexpected number of arguments"
    file = sys.argv[1]
    with open(file, "rb") as f:
        data = pickle.load(f)
    # PRINT = False
    PRINT = True
    if PRINT:
        print("Unpickled Data:")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None, "max_colwidth", 150):
            print(data)

    MEM = True
    if MEM:
       print_memory_footprint(data)



if __name__ == "__main__":
    main()
