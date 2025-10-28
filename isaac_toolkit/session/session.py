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
from pathlib import Path

import yaml

from .config import IsaacConfig, DEFAULT_CONFIG
from .artifact import (
    FileArtifact,
    ElfArtifact,
    InstrTraceArtifact,
    SourceArtifact,
    TableArtifact,
    M2ISARArtifact,
    GraphArtifact,
    ArtifactFlag,
    PythonArtifact,
    filter_artifacts,
)
from isaac_toolkit.logging import get_logger, set_log_level, set_log_file

logger = get_logger()


# def create_dirs(base, dirnames):
#     assert isinstance(base, Path)
#     for dirname in dirnames:
#         (base / dirname).mkdir()


def load_artifacts(base):
    artifacts_yaml = base / "artifacts.yml"
    if not artifacts_yaml.is_file():
        return []
    with open(artifacts_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    artifacts = yaml_data.get("artifacts", None)
    assert artifacts is not None
    artifacts_ = []
    for artifact in artifacts:
        name = artifact.get("name", None)
        assert name is not None
        dest = artifact.get("dest", None)
        assert dest is not None
        flags = artifact.get("flags", None)
        assert flags is not None
        attrs = artifact.get("attrs", None)
        assert attrs is not None
        flags_ = ArtifactFlag(flags)
        # print("flags_", flags)
        # if flags_ & (ArtifactFlag.INPUT | ArtifactFlag.OUTPUT | ArtifactFlag.TEMP):
        #     path = artifact.get("path", None)
        #     assert path is not None
        if flags_ & ArtifactFlag.ELF:
            artifact_ = ElfArtifact.from_dict(artifact)
            # (name, dest, flags=flags, attrs=attrs)
        elif flags_ & ArtifactFlag.INSTR_TRACE:
            artifact_ = InstrTraceArtifact.from_dict(artifact)
        elif flags_ & (ArtifactFlag.SOURCE | ArtifactFlag.DISASS):
            artifact_ = SourceArtifact.from_dict(artifact)
        elif flags_ & ArtifactFlag.GRAPH:
            artifact_ = GraphArtifact.from_dict(artifact)
        elif flags_ & ArtifactFlag.TABLE:
            artifact_ = TableArtifact.from_dict(artifact)
        elif flags_ & ArtifactFlag.M2ISAR:
            artifact_ = M2ISARArtifact.from_dict(artifact)
        elif flags_ & ArtifactFlag.PYTHON:
            artifact_ = PythonArtifact.from_dict(artifact)
        else:
            logger.warning("Unhandled artifact type!")
            artifact_ = FileArtifact.from_dict(artifact)
            # raise RuntimeError("Unhandled case!")
        artifacts_.append(artifact_)
    # TODO: check for duplicates
    # print("artifacts", artifacts)
    # print("artifacts_", artifacts_)
    return artifacts_


class Session:

    def __init__(self, session_dir: Path, config: IsaacConfig):
        self.directory = session_dir.resolve()
        self.config = config
        self._artifacts = load_artifacts(self.directory)

    @property
    def artifacts(self):
        return self._artifacts

    def add_artifact(self, artifact, override=False):
        logger.debug("Adding artifact to session")
        artifact_names = [x.name for x in self._artifacts]
        if artifact.name in artifact_names:
            if override:
                logger.debug("Overriding artifact")
                idx = artifact_names.index(artifact.name)
                del self._artifacts[idx]
            else:
                raise RuntimeError(
                    f"Artifact with name {artifact.name} already exists. Use override=True or cleanup session."
                )
        self._artifacts.append(artifact)

    # @property
    # def inputs(self):
    #     return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.INPUT & x.flags)

    # @property
    # def outputs(self):
    #     return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.OUTPUT & x.flags)

    # @property
    # def temps(self):
    #     return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.TEMP & x.flags)

    @property
    def graphs(self):
        return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.GRAPH & x.flags)

    @property
    def tables(self):
        return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.TABLE & x.flags)

    # @property
    # def temp_dir(self):
    #     return self.directory / "temp"

    # @property
    # def inputs_dir(self):
    #     return self.directory / "inputs"

    # @property
    # def outputs_dir(self):
    #     return self.directory / "outputs"

    # @property
    # def graphs_dir(self):
    #     return self.directory / "graphs"

    # @property
    # def tables_dir(self):
    #     return self.directory / "tables"

    def validate(self):
        pass
        # assert self.temp_dir.is_dir()
        # assert self.inputs_dir.is_dir()
        # assert self.outputs_dir.is_dir()
        # self.config.validate()

    def save_artifacts(self):
        # print("save_artifacts")
        logger.debug("Saving artifacts...")
        artifacts_ = []
        for artifact in self.artifacts:
            dest_dir = None
            dest_file = artifact.name
            if isinstance(artifact, ElfArtifact):
                dest_dir = self.directory / "elf"
            elif isinstance(artifact, InstrTraceArtifact):
                dest_dir = self.directory / "instr_trace"
                dest_file = f"{dest_file}.pkl"
            elif isinstance(artifact, SourceArtifact):
                dest_dir = self.directory / "source"
            elif isinstance(artifact, GraphArtifact):
                # assert not artifact.is_input and not artifact.is_output
                dest_dir = self.directory / "graph"
                dest_file = f"{dest_file}.pkl"
            elif isinstance(artifact, TableArtifact):
                # assert not artifact.is_input and not artifact.is_output
                dest_dir = self.directory / "table"
                dest_file = f"{dest_file}.pkl"
            elif isinstance(artifact, M2ISARArtifact):
                # assert not artifact.is_input and not artifact.is_output
                dest_dir = self.directory / "model"
                dest_file = f"{dest_file}.m2isarmodel"
            elif isinstance(artifact, PythonArtifact):
                # assert not artifact.is_input and not artifact.is_output
                dest_dir = self.directory / "misc"
                dest_file = f"{dest_file}.pkl"
            if dest_dir is None:
                dest_dir = self.directory / "misc"
            assert dest_file is not None
            assert len(dest_file) > 0
            assert dest_file[0] != "/"
            dest = dest_dir / dest_file
            dest.parent.mkdir(parents=True, exist_ok=True)
            artifact.save(dest)
            metadata = artifact.to_dict()
            metadata["dest"] = str(dest)
            artifacts_.append(metadata)
        yaml_data = {"artifacts": artifacts_}
        artifacts_yaml = self.directory / "artifacts.yml"
        with open(artifacts_yaml, "w") as f:
            yaml.dump(yaml_data, f)

    def save(self):
        # print("save")
        self.config.to_yaml_file(self.directory / "config.yml")
        self.save_artifacts()

    @staticmethod
    def create(session_dir):
        if isinstance(session_dir, str):
            session_dir = Path(session_dir)
        assert isinstance(session_dir, Path)
        assert session_dir.parent.is_dir(), f"Parent directory does not exist: {session_dir.parent}"
        session_dir.mkdir()
        # create_dirs(session_dir, ["inputs", "outputs", "temp", "graphs", "tables"])
        config = IsaacConfig.from_dict(DEFAULT_CONFIG)
        sess = Session(session_dir, config)
        log_file = session_dir / "session.log"
        set_log_file(log_file)
        sess.save()
        return sess

    @staticmethod
    def from_dir(session_dir):
        if isinstance(session_dir, str):
            session_dir = Path(session_dir)
        assert isinstance(session_dir, Path)
        assert session_dir.is_dir(), f"Directory does not exist: {session_dir}"
        config_file = session_dir / "config.yml"
        assert config_file.is_file(), f"Missing config file: {config_file}"
        config = IsaacConfig.from_yaml_file(config_file)
        log_file = session_dir / "session.log"
        set_log_file(log_file)
        set_log_level(console_level=config.logging.console.level, file_level=config.logging.file.level)
        sess = Session(session_dir, config)
        sess.validate()
        sess.save()
        return sess
