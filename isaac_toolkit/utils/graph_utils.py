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
import logging
from pathlib import Path

import networkx as nx
from networkx.drawing.nx_agraph import write_dot


logger = logging.getLogger("graph_utils")


def graph_to_file(graph, dest, fmt="auto"):
    if not isinstance(dest, Path):
        dest = Path(dest)
    if fmt == "auto":
        fmt = dest.suffix[1:].upper()
    prog = "dot"
    # TODO: support pkl
    if fmt == "DOT":
        write_dot(graph, dest)
    elif fmt in ["PDF", "PNG"]:
        graph = nx.nx_agraph.to_agraph(graph)
        graph.draw(str(dest), prog=prog)
        graph.close()
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")


def memgraph_to_nx(results):
    graph = nx.MultiDiGraph()
    nodes = list(results.graph()._nodes.values())
    # print("nodes", nodes)
    for node in nodes:
        # print("node", node)
        if len(node._labels) > 0:
            label = list(node._labels)[0]
        else:
            label = "?!"
        name = node._properties.get("name", "?")
        graph.add_node(
            int(node.element_id), key=int(node.element_id), xlabel=label, label=name, properties=node._properties
        )

    rels = list(results.graph()._relationships.values())
    for rel in rels:
        label = rel.type
        # graph.add_edge(
        #     rel.start_node.element_id, rel.end_node.element_id, key=rel.element_id,
        #     label=label, type=rel.type, properties=rel._properties,
        # )
        graph.add_edge(
            int(rel.start_node.element_id),
            int(rel.end_node.element_id),
            key=int(rel.element_id),
            label=label,
            type=rel.type,
            properties=rel._properties,
        )
    return graph


# def calc_inputs(G, sub, ignore_const: bool = False):
def calc_inputs(G, sub):
    # print("calc_inputs", G, sub)
    # print("G.nodes", G.nodes)
    inputs = []
    constants = []
    sub_nodes = sub.nodes
    # print("sub_nodes", sub_nodes)
    for node in sub_nodes:
        # print("node", node, G.nodes[node].get("label"))
        ins = G.in_edges(node)
        # print("ins", ins)
        for in_ in ins:
            # print("in_", in_, G.nodes[in_[0]].get("label"))
            src = in_[0]
            # print("src", src, G.nodes[src].get("label"))
            # print("src in sub_nodes", src in sub_nodes)
            # print("src not in inputs", src not in inputs)
            # op_type = G.nodes[src]["properties"]["op_type"]
            if not (src in sub_nodes) and (src not in inputs):
                # print("IN")
                if G.nodes[src]["properties"]["op_type"] == "constant":
                    constants.append(src)
                else:
                    inputs.append(src)
    # print("ret", ret)
    return len(inputs), inputs, len(constants), constants


def calc_outputs(G, sub):
    # print("calc_outputs", sub)
    ret = 0
    sub_nodes = sub.nodes
    # print("sub_nodes", sub_nodes)
    outputs = []
    for node in sub_nodes:
        # print("node", node, G.nodes[node].get("label"))
        if G.nodes[node]["properties"]["op_type"] == "output":
            # print("A")
            # print("OUT2")
            ret += 1
            if node not in outputs:
                outputs.append(node)
        else:
            # print("B")
            outs = G.out_edges(node)
            # print("outs", outs)
            for out_ in outs:
                # print("out_", out_, G.nodes[out_[0]].get("label"))
                dst = out_[1]
                # print("dst", dst, G.nodes[dst].get("label"))
                if dst not in sub_nodes:
                    # print("OUT")
                    ret += 1
                    if node not in outputs:
                        outputs.append(node)
    # print("ret", ret)
    return ret, outputs


def get_instructions(sub):
    ret = []
    sub_nodes = sub.nodes
    for node in sub_nodes:
        # print("node", node)
        node_data = sub.nodes[node]
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        name = node_properties["name"]
        op_type = node_properties["op_type"]
        if op_type == "input":
            continue
        ret.append(name)
    return ret


def calc_weights(sub):
    weights = []
    freqs = []
    sub_nodes = sub.nodes
    for node in sub_nodes:
        # print("node", node)
        node_data = sub.nodes[node]
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        op_type = node_properties["op_type"]
        if op_type == "input":
            continue
        instr_freq = node_properties.get("instr_freq", None)
        if instr_freq is None:
            return None, None
        instr_rel_weight = node_properties.get("instr_rel_weight", None)
        if instr_rel_weight is None:
            return None, None
        freqs.append(instr_freq)
        weights.append(instr_rel_weight)
    # print("weights", weights)
    # print("freqs", freqs)
    # crossBB = True
    crossBB = False
    if not crossBB:
        assert len(set(weights)) == 1
    total_weight = sum(weights)
    if not crossBB:
        assert len(set(freqs)) == 1
        freq = freqs[0]
    else:
        freq = max(freqs)
    # print("total_weight", total_weight)
    # print("freq", freq)
    # input("?!")
    return total_weight, freq


def calc_weights_iso(graph, nodes):
    weights = []
    freqs = []
    assert len(nodes) == len(set(nodes))
    for node in nodes:
        # print("node", node)
        node_data = graph.nodes[node]
        # print("node_data", node_data)
        node_properties = node_data["properties"]
        op_type = node_properties["op_type"]
        if op_type == "input":
            continue
        instr_freq = node_properties.get("instr_freq", None)
        if instr_freq is None:
            return None, None
        instr_rel_weight = node_properties.get("instr_rel_weight", None)
        if instr_rel_weight is None:
            return None, None
        freqs.append(instr_freq)
        weights.append(instr_rel_weight)
    # print("weights", weights)
    # print("freqs", freqs)
    # crossBB = True
    crossBB = False
    if not crossBB:
        # print("weights", weights)
        assert len(set(weights)) == 1
    total_weight = sum(weights)
    if not crossBB:
        assert len(set(freqs)) == 1
        freq = freqs[0]
    else:
        freq = max(freqs)
    # print("total_weight", total_weight)
    # print("freq", freq)
    # input("?!")
    return total_weight, freq
