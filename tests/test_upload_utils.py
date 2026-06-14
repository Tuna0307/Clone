from __future__ import annotations

import io
import tomllib
from pathlib import Path

from upload_utils import save_uploaded_files


class ChunkedUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._stream = io.BytesIO(content)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._stream.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def getbuffer(self) -> memoryview:
        raise AssertionError("save_uploaded_files should stream uploads instead of buffering them")


def test_save_uploaded_files_streams_upload_to_disk(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    upload = ChunkedUpload("incident.log", b"first line\nsecond line\n")
    upload_dir = tmp_path / "outputs" / "uploads" / "session-1"

    saved_paths = save_uploaded_files([upload], str(upload_dir))

    assert saved_paths == [str(upload_dir / "incident.log")]
    assert Path(saved_paths[0]).read_bytes() == b"first line\nsecond line\n"
    assert upload.read_sizes
    assert all(size > 0 for size in upload.read_sizes[:-1])


def test_streamlit_upload_limit_is_5gb():
    config_path = Path(".streamlit") / "config.toml"

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["server"]["maxUploadSize"] == 5120
