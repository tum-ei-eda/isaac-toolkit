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
    # INPUT = auto()
    # OUTPUT = auto()
    # TEMP = auto()
    TABLE = auto()
    GRAPH = auto()
    ELF = auto()
    INSTR_TRACE = auto()
    SOURCE = auto()
    M2ISAR = auto()


def filter_artifacts(artifacts, func):
    return list(filter(func, artifacts))


# class ExportedArtifact()

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
        return f"{type(self).__name__}({self.name}, attrs={self.attrs})"

    def summary(self):
        return f"{self.name}: {self}"

    # @property
    # def is_input(self):
    #     return self.flags & ArtifactFlag.INPUT

    # @property
    # def is_output(self):
    #     return self.flags & ArtifactFlag.OUTPUT

    def save(self, dest: Path):
        raise NotImplementedError("Artifact.save() impl missing")


class FileArtifact(Artifact):

    def __init__(self, name: str, path: Union[str, Path], flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, flags=flags, attrs=attrs)
        self.path = Path(path)

    def save(self, dest: Path):
        if dest.resolve() == self.path.resolve():
            return
        shutil.copyfile(self.path, dest)


class ElfArtifact(FileArtifact):

    @property
    def flags(self):
        return super().flags | ArtifactFlag.ELF


class SourceArtifact(FileArtifact):

    @property
    def flags(self):
        return super().flags | ArtifactFlag.SOURCE


class PythonArtifact(Artifact):

    def __init__(self, name: str, data, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, flags=flags, attrs=attrs)
        self.data = data

    def save(self, dest):
        with open(dest, "wb") as f:
            pickle.dump(self.data, f)


class M2ISARArtifact(PythonArtifact):

    def __init__(self, name: str, model, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, data=model, flags=flags, attrs=attrs)

    @property
    def flags(self):
        return super().flags | ArtifactFlag.M2ISAR

    @property
    def model(self):
        return self.data


class TableArtifact(PythonArtifact):

    def __init__(self, name: str, df: pd.DataFrame, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, data=df, flags=flags, attrs=attrs)

    @property
    def flags(self):
        return super().flags | ArtifactFlag.TABLE

    @property
    def df(self):
        return self.data

    def save(self, dest):
        self.df.to_pickle(dest)

    def summary(self):
        return f"{self.name}: pd.DataFrame(shape={self.df.shape}, columns={list(self.df.columns)})"


class InstrTraceArtifact(TableArtifact):
    # TODO: csv instead of pickle?

    @property
    def flags(self):
        return super().flags | ArtifactFlag.INSTR_TRACE


class GraphArtifact(PythonArtifact):

    def __init__(self, name: str, graph: nx.Graph, flags: ArtifactFlag = None, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(name, data=graph, flags=flags, attrs=attrs)

    @property
    def flags(self):
        return super().flags | ArtifactFlag.GRAPH

    @property
    def graph(self):
        return self.data

    def summary(self):
        return f"{self.name}: {self.data}"
