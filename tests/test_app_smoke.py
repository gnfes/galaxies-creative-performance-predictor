import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from creative_predictor import config

APP = str(config.ROOT_DIR / "src" / "creative_predictor" / "app" / "main.py")


def test_app_opens_without_the_local_dataset(monkeypatch, tmp_path):
    # A Galaxies pode ter só o modelo salvo: a tela abre e a aba de
    # demonstração avisa que precisa da base, sem quebrar as outras.
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("precisa da base" in i.value for i in app.info)
    assert app.selectbox[0].options == ["Outro canal (id do YouTube)"]
