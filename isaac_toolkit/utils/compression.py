import sys
import tarfile
import tempfile
import shutil
from pathlib import Path


def extract_zstd(archive: Path, out_path: Path):
    """Extract a .zst archive (Zstandard-compressed tar)."""
    archive = Path(archive).expanduser()
    out_path = Path(out_path).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if sys.version_info >= (3, 14):
        from compression.zstd import ZstdFile

        with ZstdFile(archive, "rb") as f_in:
            with out_path.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        _legacy_decompress_zstd_stream(archive, out_path)


def _legacy_decompress_zstd_stream(archive: Path, out_path: Path) -> None:
    from zstandard import ZstdDecompressor

    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with archive.open("rb") as ifh:
            ZstdDecompressor().copy_stream(ifh, ofh)
        ofh.seek(0)
        with tarfile.open(fileobj=ofh) as z:
            z.extractall(out_path)
