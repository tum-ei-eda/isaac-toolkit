import logging
from typing import Union, Optional
from enum import IntFlag, auto
from pathlib import Path

import yaml
import pandas as pd

from .config import IsaacConfig, DEFAULT_CONFIG
from .artifact import FileArtifact, TableArtifact, GraphArtifact, ArtifactFlag, filter_artifacts


logger = logging.getLogger("session")


def create_dirs(base, dirnames):
    assert isinstance(base, Path)
    for dirname in dirnames:
        (base / dirname).mkdir()


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
        print("flags_", flags)
        if flags_ & (ArtifactFlag.INPUT | ArtifactFlag.OUTPUT | ArtifactFlag.TEMP):
            artifact_ = FileArtifact(name, dest, flags=flags, attrs=attrs)
            # path = artifact.get("path", None)
            # assert path is not None
        elif flags_ & ArtifactFlag.GRAPH:
            # TODO: move to artifact.py
            import pickle
            with open(dest, "rb") as f:
                df = pickle.load(f)
            artifact_ = GraphArtifact(name, graph, flags=flags, attrs=attrs)
        elif flags_ & ArtifactFlag.TABLE:
            import pandas as pd
            df = pd.read_pickle(dest)
            artifact_ = GraphArtifact(name, df, flags=flags, attrs=attrs)
        else:
            raise RuntimeError(f"Unhandled case!")
        artifacts_.append(artifact_)
    # TODO: check for duplicates
    return artifacts_



class Session:

    def __init__(self, session_dir: Path, config: IsaacConfig):
        self.directory = session_dir.resolve()
        self.config = config
        self.artifacts = load_artifacts(self.directory)

    @property
    def inputs(self):
        return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.INPUT & x.flags)

    @property
    def outputs(self):
        return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.OUTPUT & x.flags)

    @property
    def temps(self):
        return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.TEMP & x.flags)

    @property
    def graphs(self):
        return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.GRAPH & x.flags)

    @property
    def tables(self):
        return filter_artifacts(self.artifacts, lambda x: ArtifactFlag.TABLE & x.flags)

    @property
    def temp_dir(self):
        return self.directory / "temp"

    @property
    def inputs_dir(self):
        return self.directory / "inputs"

    @property
    def outputs_dir(self):
        return self.directory / "outputs"

    @property
    def graphs_dir(self):
        return self.directory / "graphs"

    @property
    def tables_dir(self):
        return self.directory / "tables"

    def validate(self):
        assert self.temp_dir.is_dir()
        assert self.inputs_dir.is_dir()
        assert self.outputs_dir.is_dir()
        # self.config.validate()

    def save_artifacts(self):
        logger.info("Saving artifacts...")
        artifacts_ = []
        for artifact in self.artifacts:
            print("a", artifact)
            dest_dir = None
            if isinstance(artifact, FileArtifact):
                if artifact.is_input:
                    dest_dir = self.inputs_dir
                elif artifact.is_output:
                    dest_dir = self.outputs_dir
                else:
                    dest_dir = self.temp_dir
                dest_file = artifact.name
            elif isinstance(artifact, GraphArtifact):
                assert not artifact.is_input and not artifact.is_output
                dest_dir = self.graphs_dir
                dest_file = f"{artifact.name}.pkl"
            elif isinstance(artifact, TableArtifact):
                assert not artifact.is_input and not artifact.is_output
                dest_dir = self.tables_dir
                dest_file = f"{artifact.name}.pkl"
            assert dest_dir is not None
            assert dest_file is not None
            dest = dest_dir / dest_file
            artifact.save(dest)
            artifacts_.append({"name": artifact.name, "flags": int(artifact.flags), "type": type(artifact).__name__, "dest": str(dest), "hash": artifact.hash, "attrs": artifact.attrs})
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
        create_dirs(session_dir, ["inputs", "outputs", "temp", "graphs", "tables"])
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
