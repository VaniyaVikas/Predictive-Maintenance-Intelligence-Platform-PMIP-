"""Reproducible download of the UCI AI4I 2020 dataset (CC BY 4.0)."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

from .data import UCI_ZIP_URL


def download_ai4i(dest: Path, url: str = UCI_ZIP_URL, timeout: int = 60) -> Path:
    """Download and extract ``ai4i2020.csv`` to ``dest``. Returns the CSV path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pmip-downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("No CSV file found inside the downloaded archive.")
        dest.write_bytes(zf.read(names[0]))
    return dest
