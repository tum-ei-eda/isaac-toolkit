#
# Copyright (c) 2026 TUM Department of Electrical and Computer Engineering.
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
