"""Escrita atômica e download conferido por hash, usados por mais de um módulo.

Leve de propósito (só a biblioteca padrão): é importado tanto no processo da
LightGBM quanto no subprocesso do PyTorch.
"""

import hashlib
import urllib.request
from collections.abc import Callable
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_atomic(path: Path, write: Callable[[Path], object]) -> Path:
    """Grava num arquivo temporário e troca no fim: nunca fica meio arquivo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    write(tmp)
    tmp.replace(path)
    return path


def download_verified(url: str, path: Path, expected_sha256: str) -> Path:
    """Baixa `url` e só grava em `path` se o SHA-256 conferir."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    digest = sha256(data)
    if digest != expected_sha256:
        raise ValueError(
            f"hash do arquivo baixado não confere (esperado {expected_sha256}, "
            f"veio {digest})"
        )
    return write_atomic(path, lambda tmp: tmp.write_bytes(data))
