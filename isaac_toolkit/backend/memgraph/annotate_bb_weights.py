import sys
import argparse
from pathlib import Path

from neo4j import GraphDatabase, Query

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts


def anonotate_helper(
    driver,
    func_name: str,
    bb_name: str,
    num_instrs: int,
    freq: int,
    rel_weight: float,
    label: str = "default",
    check: bool = True,
    timeout: float = 120,
):
    bb_query = Query(
        f"""
    MATCH (a:BB)
    WHERE a.func_name = "{func_name}"
    AND a.name = "{bb_name}"
    AND a.session = "{label}"
    SET a.bb_freq = {freq}
    SET a.bb_instrs = {num_instrs}
    SET a.bb_rel_weight = {rel_weight}
    // RETURN *
    """,
        timeout=timeout,
    )

    results = driver.session().run(bb_query)
    # print("results", results)
    # print("results.df", results.df)
    # input("<>")

    instrs_query = Query(
        f"""
    MATCH (a:INSTR)
    WHERE a.func_name = "{func_name}"
    AND a.basic_block = "{bb_name}"
    AND a.session = "{label}"
    AND a.op_type != "input"
    AND a.op_type != "constant"
    SET a.instr_freq = {freq}
    SET a.instr_rel_weight = {rel_weight / num_instrs}
    RETURN COUNT(a)
    """,
        timeout=timeout,
    )

    results = driver.session().run(instrs_query)
    # print("results", results, dir(results))
    if check:
        pass
        # df = results.to_df()
        # count = df.iloc[0, 0]
        # print("count", count)
        # print("num_instrs", num_instrs)
        # assert count <= (num_instrs * 1.05 + 5)
    # input("<>")


def annotate_bb_weights(sess: Session, label: str = "default", force: bool = False):
    memgraph_config = sess.config.memgraph
    hostname = memgraph_config.hostname
    port = memgraph_config.port
    user = memgraph_config.user
    password = memgraph_config.password

    driver = GraphDatabase.driver(f"bolt://{hostname}:{port}", auth=(user, password))

    artifacts = sess.artifacts
    llvm_bbs_artifacts = filter_artifacts(
        artifacts, lambda x: x.flags & ArtifactFlag.TABLE and x.name == "llvm_bbs_new"
    )  # TODO: optional or different pass
    assert len(llvm_bbs_artifacts) == 1
    llvm_bbs_artifact = llvm_bbs_artifacts[0]
    llvm_bbs_df = llvm_bbs_artifact.df
    # print("llvm_bbs_df", llvm_bbs_df, llvm_bbs_df.columns)
    # input("?")
    for index, row in llvm_bbs_df.iterrows():
        # print("index", index)
        # print("row", row)
        func_name = row["func_name"]
        bb_name = row["bb_name"]
        num_instrs = row["num_instrs"]
        freq = row["freq"]
        rel_weight = row["rel_weight"]
        anonotate_helper(driver, func_name, bb_name, num_instrs, freq, rel_weight, label=label, check=True)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    annotate_bb_weights(sess, label=args.label, force=args.force)
    sess.save()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default="info", choices=["critical", "error", "warning", "info", "debug"]
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
