import sys
import argparse
from pathlib import Path

from neo4j import GraphDatabase
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import matplotlib.pyplot as plt

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, GraphArtifact


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    override = args.force
    memgraph_config = sess.config.memgraph
    hostname = memgraph_config.hostname
    port = memgraph_config.port
    user = memgraph_config.user
    password = memgraph_config.password
    # TODO: database?

    driver = GraphDatabase.driver(f"bolt://{hostname}:{port}", auth=(user, password))

    query = """
    MATCH (n)-[r]->(c) RETURN *
    """

    results = driver.session().run(query)

    G = nx.MultiDiGraph()

    nodes = list(results.graph()._nodes.values())
    for node in nodes:
        if len(node._labels) > 0:
            label = list(node._labels)[0]
        else:
            label = "?!"
        name = node._properties.get("name", "?")
        G.add_node(node.id, xlabel=label, label=name, properties=node._properties)

    rels = list(results.graph()._relationships.values())
    for rel in rels:
        label = rel.type
        # G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, label=label, type=rel.type, properties=rel._properties)
        G.add_edge(
            rel.start_node.id, rel.end_node.id, key=rel.id, label=label, type=rel.type, properties=rel._properties
        )
    print("G", G, dir(G))
    attrs = {
        "kind": "cdfg",
        "by": "isaac_toolkit.frontend.memgraph.llvm_mir_cdfg",
        # "mod_name": ""
        # "func_name": ""
        # "bb_name": ""
    }
    artifact = GraphArtifact("memgraph_mir_cdfg", G, attrs=attrs)
    print("artifact", artifact, dir(artifact), artifact.flags)
    sess.add_artifact(artifact, overrride=override)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--force", "-f", action="store_true")
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
