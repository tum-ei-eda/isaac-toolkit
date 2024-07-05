import sys
import pickle

import pandas as pd


def main():
    assert len(sys.argv) == 2, "Unexpected number of arguments"
    file = sys.argv[1]
    with open(file, "rb") as f:
        data = pickle.load(f)
    print("Unpickled Data:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None, "max_colwidth", 150):
        print(data)

if __name__ == "__main__":
    main()
