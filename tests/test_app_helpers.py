import pandas as pd

from creative_predictor.app import helpers as h
from creative_predictor.predict import schemas


def prediction(creative_id="a", likes=120.0, low=60.0, high=300.0, flag=False):
    return schemas.Prediction(
        creative_id=creative_id,
        predicted_metric=likes,
        interval_low=low,
        interval_high=high,
        relative_to_channel=1.2,
        band="na_media",
        channel_median_likes=100.0,
        confidence_flag=flag,
        top_drivers=[
            schemas.Driver(
                feature="duration_s",
                label="duração do vídeo (s)",
                value="35",
                effect=0.12,
                direction="aumenta",
            ),
            schemas.Driver(
                feature="frames_pc",
                label="visual geral dos frames",
                value="",
                effect=-0.05,
                direction="reduz",
            ),
        ],
        comparables=[
            schemas.Comparable(
                video_id="v1",
                url="https://youtube.com/shorts/v1",
                title="Look do dia",
                channel="Canal A",
                likes=150,
                relative_to_channel=1.5,
                similarity=0.8,
            )
        ],
        blocks_used=["metadados", "sinais"],
        attributes={"tem rosto": "sim", "cortes de cena": 3.0},
    )


def test_channel_options_label_region_and_holdout_last():
    channels = pd.DataFrame(
        {
            "channel_id": ["UC2", "UC1", "UC3"],
            "name": ["Zeta", "Alfa", "Beta"],
            "sector": ["moda", "beleza", "automotivo"],
            "region": ["br", "global", "br"],
            "kind": ["criador", "marca", "marca"],
            "country": ["BR", "US", "BR"],
            "holdout": [False, False, True],
        }
    )
    out = h.channel_options(channels)
    assert out["label"].tolist() == [
        "Alfa (beleza, global)",
        "Zeta (moda, Brasil)",
        "Beta (automotivo, Brasil, fora do treino)",
    ]


def test_save_upload_names_by_content(tmp_path):
    first = h.save_upload(b"video", "Campanha.MOV", tmp_path)
    again = h.save_upload(b"video", "outro.mov", tmp_path)
    other = h.save_upload(b"outro video", "x.mp4", tmp_path)
    assert first == again and first.suffix == ".mov" and first.read_bytes() == b"video"
    assert other != first


def test_prediction_tables_are_readable():
    p = prediction()
    numbers = h.summary(p)
    assert numbers["intervalo"] == (60, 300)
    assert numbers["faixa"] == "na média do canal"
    assert numbers["blocos"] == ["metadados e texto", "sinais do vídeo e do áudio"]
    drivers = h.drivers_frame(p)
    assert drivers["fator"].tolist() == [
        "duração do vídeo (s) (35)",
        "visual geral dos frames",
    ]
    assert h.attributes_frame(p).shape == (2, 2)
    assert h.comparables_frame(p)["link"].iloc[0] == "https://youtube.com/shorts/v1"
    assert h.interval_frame(p)["mediana do canal"].iloc[0] == 100.0


def test_empty_prediction_parts_keep_their_columns():
    p = prediction().model_copy(update={"top_drivers": [], "comparables": []})
    assert list(h.drivers_frame(p).columns) == ["fator", "efeito", "sentido"]
    assert h.comparables_frame(p).empty


def test_ranking_and_demo_frames():
    ranking = schemas.Ranking(
        predictions=[prediction("b", 300, 200, 400, flag=True), prediction("a")]
    )
    table = h.ranking_frame(ranking)
    assert table["criativo"].tolist() == ["b", "a"]
    assert table["baixa confiança"].tolist() == ["sim", "não"]
    demo = h.demo_frame(ranking, {"a": 1000, "b": 250})
    assert demo.set_index("Short")["dentro do intervalo"].to_dict() == {
        "b": "sim",
        "a": "não",
    }


def test_results_tables_use_readable_names(tmp_path):
    pd.DataFrame(
        {
            "experimento": ["baseline_mediana_canal", "lgbm_extrator"],
            "n": [899, 899],
            "mae_log": [0.727, 0.672],
            "spearman_within_channel": [None, 0.39181],
            "pairwise_accuracy": [0.5, 0.648],
            "coverage": [None, None],
        }
    ).to_csv(tmp_path / "multimodal_results.csv", index=False)
    tables = h.results_tables(tmp_path)
    assert list(tables) == ["Ganho de cada bloco (teste com todos os blocos)"]
    table = tables["Ganho de cada bloco (teste com todos os blocos)"]
    assert list(table.columns) == [
        "modelo",
        "Shorts",
        "MAE log",
        "Spearman no canal",
        "Par a par",
    ]
    assert table["Spearman no canal"].tolist() == ["n/a", "0.392"]
    assert table["modelo"].iloc[1] == "+ rubrica do extrator (modelo final)"
    assert table["Par a par"].tolist() == ["0.500", "0.648"]


def test_numbers_in_brazilian_format():
    assert h.br(1234567) == "1.234.567"
    assert h.br(0.84, 2) == "0,84"
    assert h.br(3717.2) == "3.717"
