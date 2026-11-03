"""Retargeting artifacts must unpack without a complete temporary tar copy."""
import io
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import zstandard

from isaac_toolkit.utils.compression import extract_zstd


def test_streaming_tar_zstd_extraction_with_nested_files_and_symlink(tmp_path):
    payload = b'compiler artifact\n' * 100
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as bundle:
        file = tarfile.TarInfo('output/llvm_install/bin/clang-19')
        file.size = len(payload)
        file.mode = 0o755
        bundle.addfile(file, io.BytesIO(payload))
        link = tarfile.TarInfo('output/llvm_install/bin/clang')
        link.type = tarfile.SYMTYPE
        link.linkname = 'clang-19'
        bundle.addfile(link)
    archive = tmp_path/'artifact.tar.zst'
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))
    output = tmp_path/'result'
    with patch.object(tempfile, 'TemporaryFile', side_effect=AssertionError('Must stream')):
        extract_zstd(archive, output)
    clang = output/'output/llvm_install/bin/clang'
    assert clang.is_symlink()
    assert clang.read_bytes() == payload
    assert clang.stat().st_mode & 0o111
    # A retry can overwrite a partial prior extraction.
    extract_zstd(archive, output)
    assert clang.read_bytes() == payload


def test_artifact_move_merges_retries_and_preserves_symlinks(tmp_path):
    from isaac_toolkit.retargeting.service.client import move_artifact_output
    for payload in (b'first compiler', b'retried compiler'):
        source = tmp_path/'output'
        binary = source/'llvm_install/bin/clang-19'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(payload)
        (binary.parent/'clang').symlink_to('clang-19')
        move_artifact_output(source, tmp_path)
        assert not source.exists()
        final = tmp_path/'llvm_install/bin/clang'
        assert final.is_symlink()
        assert final.read_bytes() == payload
