import sys
import argparse
from pathlib import Path

from neo4j import GraphDatabase
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, GraphArtifact, filter_artifacts


def algo(G):
    print("algo")
    print("G", G, dir(G))
    if G.number_of_nodes() == 0:
        return
    mapping = dict(zip(G.nodes.keys(), range(len(G.nodes))))
    G = nx.relabel_nodes(G, mapping)
    topo = list(reversed(list(nx.topological_sort(G))))
    print("topo", topo)
    invalid = [False] * len(topo)
    fanout = [0] * len(topo)
    processed = [False] * len(topo)
    fanout_org = [0] * len(topo)
    for node in G.nodes:
        print("node", node)
        print("node_data", G.nodes[node])
        od = G.out_degree(node)
        is_output = G.out_degree(node) == 0
        if is_output:
            od += 1
        print("od", od)
        fanout_org[topo.index(node)] = od
        is_input = G.nodes[node].get("label") == "Const"  # TODO: add to db
        load_instrs = ["LB", "LBU", "LH", "LHU", "LW"]
        is_load = G.nodes[node].get("label") in load_instrs  # TODO: add to db
        is_invalid = is_input or is_load
        if is_invalid:
            invalid[topo.index(node)] = True
    max_misos = []
    for node in topo:
        print("node", node)
        if processed[topo.index(node)]:
            continue
        fanout = fanout_org.copy()
        processed[topo.index(node)] = True
        if invalid[topo.index(node)]:
            continue
        max_miso = [False] * len(topo)
        max_miso[topo.index(node)] = True

        def generate_max_miso(node, count):
            ins = G.in_edges(node)
            print("ins", ins)
            for src, dest in ins:
                print("src", src)
                print("dest", dest)
                fanout[topo.index(src)] -= 1
                if not invalid[topo.index(src)] and fanout[topo.index(src)] == 0:
                    max_miso[topo.index(src)] = True
                    processed[topo.index(src)] = True
                    count = generate_max_miso(src, count + 1)
            return count

        size = generate_max_miso(node, 1)
        # size = generate_max_miso(node, G, max_miso, 1, invalid, fanout, processed)
        print("size", size)
        if size > 1:
            max_misos.append(max_miso)
    print("max_misos", max_misos, len(max_misos))
    from itertools import compress

    max_misos_ = [list(compress(topo, max_miso)) for max_miso in max_misos]
    print("max_misos_", max_misos_)
    # max_misos__ = [nx.subgraph_view(G, filter_node=lambda node: max_miso[topo.index(node)]) for max_miso in max_misos]
    # max_misos__ = [nx.subgraph_view(G, filter_node=lambda node: node in max_miso) for max_miso in max_misos_]
    # max_misos__ = [nx.subgraph_view(G, filter_node=lambda node: node in max_miso) for max_miso in max_misos_]
    max_misos__ = [G.subgraph(max_miso) for max_miso in max_misos_]
    print("max_misos__", max_misos__)
    for i, mig in enumerate(max_misos__):
        print("i,mig", i, mig)
        write_dot(mig, f"maxmiso{i}.dot")
        labeldict = {node: mig.nodes[node]["label"] for node in mig.nodes}
        print("labeldict", labeldict)
        nx.draw(mig, labels=labeldict, with_labels=True)
        plt.savefig(f"maxmiso{i}.png")
        plt.close()
    labeldict = {node: G.nodes[node]["label"] for node in G.nodes}
    print("labeldict", labeldict)
    nx.draw(G, labels=labeldict, with_labels=True)
    plt.savefig(f"full.png")
    plt.close()

    print("invalid", invalid)
    print("fanout", fanout)
    print("processed", processed)
    print("fanout_org", fanout_org)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    graphs = sess.graphs
    print("graphs", graphs)
    cdfg = filter_artifacts(graphs, lambda x: x.name == "memgraph_mir_cdfg")
    print("cdfg", cdfg)
    assert len(cdfg) == 1
    cdfg = cdfg[0]
    graph = cdfg.graph
    print("graph", graph)

    # attrs = {}  # TODO
    # artifact = GraphArtifact("memgraph_mir_cfdf", G, attrs=attrs)
    # print("artifact", artifact, dir(artifact), artifact.flags)
    # sess.artifacts.append(artifact)
    # sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
