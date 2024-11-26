import logging
from typing import Union, Optional
from enum import IntFlag, auto
from pathlib import Path

import yaml
import pandas as pd

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


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("session")
# logger = logging.getLogger(__name__)


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
            artifact_ = ElfArtifact(name, dest, flags=flags, attrs=attrs)
        elif flags_ & ArtifactFlag.INSTR_TRACE:
            import pandas as pd

            df = pd.read_pickle(dest)
            artifact_ = InstrTraceArtifact(name, df, flags=flags, attrs=attrs)
        elif flags_ & (ArtifactFlag.SOURCE | ArtifactFlag.DISASS):
            artifact_ = SourceArtifact(name, dest, flags=flags, attrs=attrs)
        elif flags_ & ArtifactFlag.GRAPH:
            # TODO: move to artifact.py
            import pickle

            with open(dest, "rb") as f:
                graph = pickle.load(f)
            artifact_ = GraphArtifact(name, graph, flags=flags, attrs=attrs)
        elif flags_ & ArtifactFlag.TABLE:
            import pandas as pd

            df = pd.read_pickle(dest)
            artifact_ = TableArtifact(name, df, flags=flags, attrs=attrs)
        elif flags_ & ArtifactFlag.M2ISAR:
            import pickle

            with open(dest, "rb") as f:
                model = pickle.load(f)
            artifact_ = M2ISARArtifact(name, model, flags=flags, attrs=attrs)
        elif flags_ & ArtifactFlag.PYTHON:
            import pickle

            with open(dest, "rb") as f:
                data = pickle.load(f)
            artifact_ = PythonArtifact(name, data, flags=flags, attrs=attrs)
        else:
            logger.warning("Unhandled artifact type!")
            artifact_ = FileArtifact(name, dest, flags=flags, attrs=attrs)
            raise RuntimeError(f"Unhandled case!")
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
        logger.info("Adding artifact to session")
        artifact_names = [x.name for x in self._artifacts]
        if artifact.name in artifact_names:
            if override:
                logger.info("Overriding artifact")
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
        logger.info("Saving artifacts...")
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
            artifacts_.append(
                {
                    "name": artifact.name,
                    "flags": int(artifact.flags),
                    "type": type(artifact).__name__,
                    "dest": str(dest),
                    "hash": artifact.hash,
                    "attrs": artifact.attrs,
                }
            )
        yaml_data = {"artifacts": artifacts_}
        artifacts_yaml = self.directory / "artifacts.yml"
        with open(artifacts_yaml, "w") as f:
            yaml.dump(yaml_data, f)

    def save(self):
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
        sess = Session(session_dir, config)
        sess.validate()
        sess.save()
        return sess
