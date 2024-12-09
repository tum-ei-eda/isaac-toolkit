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


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("artifact")


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
    PYTHON = auto()
    DISASS = auto()


def filter_artifacts(artifacts, func):
    return list(filter(func, artifacts))


# class ExportedArtifact()


class Artifact:

    def __init__(
        self,
        name: str,
        path: Optional[Path],
        flags: Optional[ArtifactFlag] = None,
        attrs: Optional[Dict[str, Any]] = None,
        autoload: bool = False,
    ):
        self.name = name
        self.path = Path(path) if path is not None else path
        self._flags = flags if flags is not None else ArtifactFlag(0)
        self.attrs = attrs if attrs is not None else {}
        self.autoload = autoload
        self.changed: bool = True
        self.imported: bool = False
        if autoload:
            self.load()

    @classmethod
    def from_dict(cls, data: dict):
        name = data.get("name", None)
        assert name is not None
        dest = data.get("dest", None)
        assert dest is not None
        flags = data.get("flags", None)
        assert flags is not None
        attrs = data.get("attrs", None)
        assert attrs is not None
        flags_ = ArtifactFlag(flags)
        ret = cls(name, path=dest, flags=flags, attrs=attrs)
        ret.changed = False
        return ret

    def to_dict(self):
        return {
            "name": self.name,
            "flags": int(self.flags),
            "type": type(self).__name__,
            "dest": str(self.path),
            "hash": self.hash,
            "attrs": self.attrs,
        }

    @property
    def exported(self):
        return self.path is not None and self.path.is_file()

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

    def update(self):
        self.changed = True

    # @property
    # def is_input(self):
    #     return self.flags & ArtifactFlag.INPUT

    # @property
    # def is_output(self):
    #     return self.flags & ArtifactFlag.OUTPUT

    def _save(self, dest: Path):
        raise NotImplementedError("Artifact._save() impl missing")

    def save(self, dest: Path):
        if not self.changed and self.exported:
            logger.debug("Artifact '%s' is already exported and unchanged", self.name)
            return
        logger.debug("Exporting artifact '%s' to '%s'", self.name, dest)
        self._save(dest)
        if self.path is None or not self.path.is_file():
            self.path = dest
        self.changed = False

    def _load(self, dest: Path):
        raise NotImplementedError("Artifact._load() impl missing")

    def load(self, source: Optional[Path] = None):
        if self.imported:
            logger.debug("Artifact '%s' is already imported", self.name)
            # TODO: check sha?
            return
        if source is None:
            assert self.path is not None
            source = self.path
        logger.debug("Loading artifact '%s' from '%s'", self.name, source)
        self._load(source)
        self.imported = True
        self.changed = False


class FileArtifact(Artifact):

    def __init__(
        self,
        name: str,
        path: Optional[Union[str, Path]] = None,
        flags: ArtifactFlag = None,
        attrs: Optional[Dict[str, Any]] = None,
        autoload: bool = False,
    ):
        super().__init__(name, path=path, flags=flags, attrs=attrs, autoload=autoload)
        # TODO: content/raw?

    def _save(self, dest: Path):
        if isinstance(dest, str):
            dest = Path(dest)
        if dest.resolve() == self.path.resolve():
            return
        shutil.copyfile(self.path, dest)


class ElfArtifact(FileArtifact):

    @property
    def flags(self):
        return super().flags | ArtifactFlag.ELF


class DisassArtifact(FileArtifact):

    @property
    def flags(self):
        return super().flags | ArtifactFlag.DISASS


class SourceArtifact(FileArtifact):

    @property
    def flags(self):
        return super().flags | ArtifactFlag.SOURCE


class PythonArtifact(Artifact):

    def __init__(
        self,
        name: str,
        data=None,
        path: Optional[Path] = None,
        flags: ArtifactFlag = None,
        attrs: Optional[Dict[str, Any]] = None,
        autoload: bool = False,
    ):
        super().__init__(name, path=path, flags=flags, attrs=attrs, autoload=autoload)
        self._data = data

    @property
    def data(self):
        if self._data is None:
            self.load()
        return self._data

    @property
    def flags(self):
        return super().flags | ArtifactFlag.PYTHON

    def _save(self, dest):
        with open(dest, "wb") as f:
            pickle.dump(self.data, f)

    def _load(self, source: Path):
        with open(source, "rb") as f:
            data = pickle.load(f)
        self._data = data


class M2ISARArtifact(PythonArtifact):

    def __init__(
        self,
        name: str,
        model=None,
        path: Optional[Path] = None,
        flags: ArtifactFlag = None,
        attrs: Optional[Dict[str, Any]] = None,
        autoload: bool = False,
    ):
        super().__init__(name, data=model, flags=flags, attrs=attrs, autoload=autoload)

    @property
    def flags(self):
        return super().flags | ArtifactFlag.M2ISAR

    @property
    def model(self):
        return self.data


class TableArtifact(PythonArtifact):

    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        path: Optional[Path] = None,
        flags: ArtifactFlag = None,
        attrs: Optional[Dict[str, Any]] = None,
        autoload: bool = False,
    ):
        super().__init__(
            name, data=df, path=path, flags=flags, attrs=attrs, autoload=autoload
        )

    @property
    def flags(self):
        return super().flags | ArtifactFlag.TABLE

    @property
    def df(self):
        return self.data

    def _save(self, dest):
        self.df.to_pickle(dest)

    def _load(self, source: Path):
        df = pd.read_pickle(source)
        self._data = df

    def summary(self):
        return f"{self.name}: pd.DataFrame(shape={self.df.shape}, columns={list(self.df.columns)})"


class InstrTraceArtifact(TableArtifact):
    # TODO: csv instead of pickle?

    @property
    def flags(self):
        return super().flags | ArtifactFlag.INSTR_TRACE


class GraphArtifact(PythonArtifact):

    def __init__(
        self,
        name: str,
        graph: Optional[nx.Graph] = None,
        path: Optional[Path] = None,
        flags: ArtifactFlag = None,
        attrs: Optional[Dict[str, Any]] = None,
        autoload: bool = False,
    ):
        super().__init__(
            name, data=graph, path=path, flags=flags, attrs=attrs, autoload=autoload
        )

    @property
    def flags(self):
        return super().flags | ArtifactFlag.GRAPH

    @property
    def graph(self):
        return self.data

    def summary(self):
        return f"{self.name}: {self.data}"
