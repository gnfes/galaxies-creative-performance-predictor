import hashlib
import tomllib

import pytest

from creative_predictor import config
from creative_predictor.predict import bundle


def test_download_checks_the_hash(tmp_path):
    source = tmp_path / "publicado.joblib"
    source.write_bytes(b"modelo")
    target = tmp_path / "models" / "predictor.joblib"
    url = source.as_uri()

    with pytest.raises(ValueError, match="hash"):
        bundle.download(target, url, expected_sha256="0" * 64)
    assert not target.exists()

    good = hashlib.sha256(b"modelo").hexdigest()
    assert bundle.download(target, url, expected_sha256=good) == target
    assert target.read_bytes() == b"modelo"


def test_missing_model_points_to_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        bundle.load(tmp_path / "predictor.joblib")


def test_download_keeps_a_different_local_model(tmp_path):
    source = tmp_path / "publicado.joblib"
    source.write_bytes(b"modelo")
    good = hashlib.sha256(b"modelo").hexdigest()
    target = tmp_path / "predictor.joblib"
    target.write_bytes(b"retreinado")

    with pytest.raises(FileExistsError, match="--force"):
        bundle.download(target, source.as_uri(), expected_sha256=good)
    assert target.read_bytes() == b"retreinado"

    bundle.download(target, source.as_uri(), expected_sha256=good, force=True)
    assert target.read_bytes() == b"modelo"
    # Já é o publicado: não baixa de novo (a URL nem é consultada).
    assert bundle.download(target, "file:///nao/existe", expected_sha256=good) == target


def test_published_model_url_matches_the_project_version():
    pyproject = tomllib.loads((config.ROOT_DIR / "pyproject.toml").read_text())
    assert f"/v{pyproject['project']['version']}/" in bundle.MODEL_URL
