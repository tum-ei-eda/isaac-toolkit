#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
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
import argparse
from pathlib import Path

from neo4j import GraphDatabase, Query
import networkx as nx
import matplotlib.pyplot as plt

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, GraphArtifact


def legalize_str(x):
    # print("legalize_str", x, type(x))
    legalized = x.replace("/", "_")
    # print("legalized", legalized)
    return legalized


def get_cfg_artifacts(driver, label: str = "default"):
    query = Query(
        f"""
    MATCH (n)-[r:CFG]->(c)
    WHERE n.session = "{label}"
    RETURN *
    """
    )

    session = driver.session()
    try:
        results = session.run(query)
        # print("results", results)

        G = nx.MultiDiGraph()

        nodes = list(results.graph()._nodes.values())
        # print("nodes", nodes, len(nodes))
        module_func_nodes = {}
        for node in nodes:
            props = node._properties
            module_name = props["module_name"]
            func_name = props["func_name"]
            if module_name not in module_func_nodes:
                module_func_nodes[module_name] = {}
            if func_name not in module_func_nodes[module_name]:
                module_func_nodes[module_name][func_name] = set()
            module_func_nodes[module_name][func_name].add(node.id)
            if len(node._labels) > 0:
                label = list(node._labels)[0]
            else:
                label = props.get("op_type", "unknown")
            name = node._properties.get("name", "?")
            G.add_node(node.id, xlabel=label, label=name, properties=node._properties)

        rels = list(results.graph()._relationships.values())
        for rel in rels:
            props = rel._properties
            label = rel.type
            # G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, label=label, type=rel.type, properties=rel._properties)
            G.add_edge(
                rel.start_node.id,
                rel.end_node.id,
                key=rel.id,
                label=label,
                type=rel.type,
                properties=props,
            )
        # print("G", G, dir(G))
        # print("mfn", module_func_nodes)
        ret = []
        for module_name, func_nodes in module_func_nodes.items():
            for func_name, nodes in func_nodes.items():
                G_ = G.subgraph(nodes).copy()
                attrs = {
                    "kind": "cfg",
                    "by": "isaac_toolkit.frontend.memgraph.llvm_mir_cdfg",
                    "module_name": module_name,
                    "func_name": func_name,
                }
                artifact = GraphArtifact(f"{legalize_str(module_name)}/{func_name}/llvm_cfg", G_, attrs=attrs)
                # print("artifact", artifact, dir(artifact), artifact.flags)
                ret.append(artifact)
    finally:
        session.close()

    # print("ret", ret)
    return ret


def get_dfg_artifacts(driver, label: str = "default"):
    query = Query(
        f"""
    MATCH (n)-[r:DFG]->(c)
    WHERE n.session = "{label}"
    RETURN *
    """
    )

    session = driver.session()
    try:
        results = session.run(query)
        # print("results", results)

        G = nx.MultiDiGraph()

        nodes = list(results.graph()._nodes.values())
        # print("nodes", nodes, len(nodes))
        module_func_bb_nodes = {}
        for node in nodes:
            props = node._properties
            module_name = props["module_name"]
            func_name = props["func_name"]
            bb_name = props["basic_block"]
            if module_name not in module_func_bb_nodes:
                module_func_bb_nodes[module_name] = {}
            if func_name not in module_func_bb_nodes[module_name]:
                module_func_bb_nodes[module_name][func_name] = {}
            if bb_name not in module_func_bb_nodes[module_name][func_name]:
                module_func_bb_nodes[module_name][func_name][bb_name] = set()
            module_func_bb_nodes[module_name][func_name][bb_name].add(node.id)
            if len(node._labels) > 0:
                label = list(node._labels)[0]
            else:
                label = props.get("op_type", "unknown")
            name = node._properties.get("name", "?")
            G.add_node(node.id, xlabel=label, label=name, properties=node._properties)

        rels = list(results.graph()._relationships.values())
        for rel in rels:
            props = rel._properties
            label = rel.type
            # G.add_edge(rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id, label=label, type=rel.type, properties=rel._properties)
            G.add_edge(
                rel.start_node.id,
                rel.end_node.id,
                key=rel.id,
                label=label,
                type=rel.type,
                properties=props,
            )
        # print("G", G, dir(G))
        # print("mfbn", module_func_bb_nodes)
        ret = []
        for module_name, func_bb_nodes in module_func_bb_nodes.items():
            for func_name, bb_nodes in func_bb_nodes.items():
                for bb_name, nodes in bb_nodes.items():
                    G_ = G.subgraph(nodes).copy()
                    attrs = {
                        "kind": "dfg",
                        "by": "isaac_toolkit.frontend.memgraph.llvm_mir_cdfg",
                        "module_name": module_name,
                        "func_name": func_name,
                        "bb_name": bb_name,
                    }
                    artifact = GraphArtifact(
                        f"{legalize_str(module_name)}/{func_name}/{bb_name}/llvm_dfg",
                        G_,
                        attrs=attrs,
                    )
                    # print("artifact", artifact, dir(artifact), artifact.flags)
                    ret.append(artifact)
    finally:
        session.close()

    # print("ret", ret)
    return ret


def load_cdfg(sess: Session, label: str = "default", force: bool = False):
    memgraph_config = sess.config.memgraph
    hostname = memgraph_config.hostname
    port = memgraph_config.port
    user = memgraph_config.user
    password = memgraph_config.password
    # TODO: database?

    driver = GraphDatabase.driver(f"bolt://{hostname}:{port}", auth=(user, password))
    try:
        cfgs = get_cfg_artifacts(driver, label=label)
        print("cfgs", cfgs)
        for cfg in cfgs:
            sess.add_artifact(cfg, override=force)
        dfgs = get_dfg_artifacts(driver, label=label)
        print("dfgs", dfgs)
        for dfg in dfgs:
            sess.add_artifact(dfg, override=force)
    finally:
        driver.close()


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    load_cdfg(sess, label=args.label, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--label", default="default", required=True)
    parser.add_argument("--force", "-f", action="store_true")
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
