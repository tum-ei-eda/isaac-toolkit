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
import time
import tempfile
import argparse
from pathlib import Path

from isaac_toolkit.session import Session
from isaac_toolkit.session.session import get_table_artifact_ext
from isaac_toolkit.session.artifact import ArtifactFlag, filter_artifacts
from isaac_toolkit.session.config import ArtifactsSettings
from isaac_toolkit.report.sess_mem_usage import get_artifact_size
from isaac_toolkit.report.sess_disk_usage import get_file_size


def handle(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    artifacts = sess.artifacts
    # print("artifacts", artifacts)
    trace_artifacts = filter_artifacts(artifacts, lambda x: x.flags & ArtifactFlag.INSTR_TRACE)
    assert len(trace_artifacts) == 1
    trace_artifact = trace_artifacts[0]
    # print("trace_artifact", trace_artifact)
    t0 = time.time()
    trace_df = trace_artifact.df
    t1 = time.time()
    del trace_df
    artifact_size = get_artifact_size(trace_artifact)
    print("artifact_size", round(artifact_size/1000, 2))
    # print("trace_df", trace_df)
    artifacts_settings_dict = {
        "instr_trace": {
            "fmt": args.fmt,
            "engine": args.engine,
            "compression_method": args.method,
            "compression_level": int(args.level) if args.level is not None else None,
        }
    }
    artifacts_settings = ArtifactsSettings.from_dict(artifacts_settings_dict)
    print("artifacts_settings", artifacts_settings)
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)
        ext = get_table_artifact_ext(artifacts_settings.instr_trace)
        fname = f"{trace_artifact.name}.{ext}"
        temp_file = Path(tmpdirname) / fname
        t2 = time.time()
        trace_artifact._save(temp_file, artifacts_settings=artifacts_settings)
        # input("!")
        t3 = time.time()
        compressed_size = get_file_size(temp_file)
        print("compressed_size", round(compressed_size/1000, 2))
        t4 = time.time()
        trace_artifact._load(temp_file)
        t5 = time.time()
    # ts = [t0, t1, t2, t3, t4, t5]
    # print("ts", ts)
    tds = {"t_cache": round(t1 - t0, 3), "t_save": round(t3 - t2, 3), "t_load": round(t5 - t4, 3)}
    print("tds", tds)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    parser.add_argument("--fmt", default="pickle")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--level", default=None)
    # TODO: allow overriding memgraph config?
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle(args)


if __name__ == "__main__":
    main(sys.argv[1:])
