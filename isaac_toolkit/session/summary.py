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
import argparse
from pathlib import Path

from . import Session


def get_summary(sess):
    config_text = sess.config.to_yaml()
    # inputs_text = "\n".join(f" - {artifact.summary()}" for artifact in sess.inputs)
    # outputs_text = "\n".join(f" - {artifact.summary()}" for artifact in sess.outputs)
    # temps_text = "\n".join(f" - {artifact.summary()}" for artifact in sess.temps)
    artifacts_text = "\n".join(f" - {artifact.summary()}" for artifact in sess.artifacts)
    return f"""Summary of ISAAC session {sess.directory}

Config:
```
---
{config_text}
```

Artifacts:
{artifacts_text}
"""


def handle_summary(args):
    assert args.session is not None
    session_dir = Path(args.session)
    assert session_dir.is_dir(), f"Session dir does not exist: {session_dir}"
    sess = Session.from_dir(session_dir)
    text = get_summary(sess)
    assert len(text) > 0
    print(text)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )  # TODO: move to defaults
    parser.add_argument("--session", "--sess", "-s", type=str, required=True)
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    handle_summary(args)


if __name__ == "__main__":
    main(sys.argv[1:])
