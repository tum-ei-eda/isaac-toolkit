import shutil
import logging
from typing import Union, Optional, Dict, Any
from enum import IntFlag, auto
from pathlib import Path

import yaml
import pickle
import pandas as pd
import networkx as nx

from .config import IsaacConfig, DEFAULT_CONFIG

class ArtifactFlag(IntFlag):
    INPUT = auto()
    OUTPUT = auto()
    TEMP = auto()
    TABLE = auto()
    GRAPH = auto()
    ELF = auto()


def filter_artifacts(artifacts, func):
    return list(filter(func, artifacts))


class Artifact:

    def __init__(self, name: str, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        self.name = name
        self._flags = flags if flags is not None else ArtifactFlag(0)
        self.attrs = attrs if attrs is not None else {}

    @property
    def flags(self):
        return self._flags

    @property
    def hash(self):
        return "N/A"  # TODO

    def __repr__(self):
        return f"{type(self).__name__}({vars(self)})"

    def summary(self):
        return str(self)

    @property
    def is_input(self):
        return self.flags & ArtifactFlag.INPUT

    @property
    def is_output(self):
        return self.flags & ArtifactFlag.OUTPUT

    def save(self, dest: Path):
        raise NotImplementedError("Artifact.save() impl missing")


class FileArtifact(Artifact):

    def __init__(self, name: str, path: Union[str, Path], flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, flags=flags)
        self.path = Path(path)

    def save(self, dest: Path):
        if dest.resolve() == self.path.resolve():
            return
        shutil.copyfile(self.path, dest)


class PythonArtifact(Artifact):

    def __init__(self, name: str, data, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, flags=flags, attrs=attrs)
        self.data = data

    def save(self, dest):
        with open(dest, "wb") as f:
            pickle.dump(self.data, f)


class TableArtifact(PythonArtifact):

    def __init__(self, name: str, df: pd.DataFrame, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, data=df, flags=flags)

    @property
    def flags(self):
        return super().flags | ArtifactFlag.TABLE

    @property
    def df(self):
        return self.data

    def save(self, dest):
        self.df.to_pickle("./dummy.pkl")


class GraphArtifact(PythonArtifact):

    def __init__(self, name: str, graph: nx.Graph, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, data=graph, flags=flags)

    @property
    def flags(self):
        return super().flags | ArtifactFlag.GRAPH

    @property
    def graph(self):
        return self.data

    def summary(self):
        return f"{self.name}: {self.data}"
