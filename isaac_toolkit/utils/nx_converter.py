import sys
import pickle
from pathlib import Path

import networkx as nx


def main():
    assert len(sys.argv) == 3, "Unexpected number of arguments"
    in_file = Path(sys.argv[1])
    assert in_file.is_file()
    assert in_file.suffix == ".pkl"
    out_file = Path(sys.argv[2])
    fmt = out_file.suffix[1:]
    assert fmt in ["dot", "png", "pdf"]
    with open(in_file, "rb") as f:
        graph = pickle.load(f)
    graph = nx.nx_agraph.to_agraph(graph)
    if fmt == "dot":
        graph.write(out_file)
    else:
        # prog = "neato"
        prog = "dot"
        graph.draw(out_file, prog=prog)

if __name__ == "__main__":
    main()
