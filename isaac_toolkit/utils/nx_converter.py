#
# Copyright (c) 2025 TUM Department of Electrical and Computer Engineering.
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
