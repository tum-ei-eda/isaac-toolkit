import sys
import logging
import argparse
from pathlib import Path
# from collections import defaultdict

import pandas as pd

from isaac_toolkit.session import Session
from isaac_toolkit.session.artifact import ArtifactFlag, TableArtifact, filter_artifacts


logging.basicConfig(level=logging.DEBUG)  # TODO
logger = logging.getLogger(__name__)


def collect_instructions(disass_df):
    instrs = disass_df["instr"].value_counts().to_dict()
    instrs_data = []
    for instr_name, instr_count in instrs.items():
        instr_data = {"instr": instr_name, "count": instr_count}
        instrs_data.append(instr_data)
    instrs_df = pd.DataFrame(instrs_data)
    total_count = instrs_df["count"].sum()
    instrs_df["rel_count"] = instrs_df["count"] / total_count
    instrs_df.sort_values("count", ascending=False, inplace=True)

    return instrs_df


def create_disass_instr_hist(sess: Session, force: bool = False):
    artifacts = sess.artifacts
    disass_table_artifacts = filter_artifacts(artifacts, lambda x: x.name == "disass")
    assert len(disass_table_artifacts) == 1
    disass_df = disass_table_artifacts[0].df

    instrs_df = collect_instructions(disass_df)

    attrs = {
        "kind": "histogram",
        "by": __name__,
    }

    instrs_artifact = TableArtifact("disass_instrs_hist", instrs_df, attrs=attrs)
    sess.add_artifact(instrs_artifact, override=force)


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    create_disass_instr_hist(sess, force=args.force)
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
