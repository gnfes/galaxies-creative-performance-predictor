"""Funções da tela de demonstração que não dependem do Streamlit.

Transformam as saídas da previsão (Pydantic) em tabelas, leem os resultados
de `reports/` e montam o "de-para" dos canais fora do treino. Ficam aqui, e
não em `app.main`, para poderem ser testadas sem o Streamlit instalado.
"""

import hashlib
from pathlib import Path

import pandas as pd

from creative_predictor import config
from creative_predictor.predict.schemas import Prediction, Ranking

UPLOADS_DIR = config.INTERIM_DIR / "app_uploads"
REPORTS_DIR = config.ROOT_DIR / "reports"

BAND_NAMES = {
    "abaixo": "abaixo do normal do canal",
    "na_media": "na média do canal",
    "acima": "acima da média do canal",
    "viral": "muito acima da média (viral)",
}
BLOCK_NAMES = {
    "metadados": "metadados e texto",
    "sinais": "sinais do vídeo e do áudio",
    "embeddings": "embeddings",
    "extrator": "rubrica do extrator",
}
METRIC_COLUMNS = {
    "n": "Shorts",
    "mae_log": "MAE log",
    "spearman_within_channel": "Spearman no canal",
    "pairwise_accuracy": "Par a par",
    "band_f1": "F1 das faixas",
    "coverage": "Cobertura",
}
EXPERIMENT_NAMES = {
    "baseline_mediana_canal": "Baseline: mediana do canal",
    "lgbm_metadados": "Metadados e texto",
    "lgbm_sinais": "+ sinais do vídeo e do áudio",
    "lgbm_embeddings": "+ embeddings",
    "lgbm_extrator": "+ rubrica do extrator (modelo final)",
    "lgbm_extrator_sem_rubrica_na_previsao": "Modelo final, sem a rubrica na previsão",
    "llm_sem_modelo_gemma4:12b": "LLM sem modelo (gemma4:12b)",
    "lgbm_todos_os_blocos": "Modelo treinado",
}


def br(value: float, decimals: int = 0) -> str:
    """Número no formato brasileiro: 1.234 e 0,84."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def channel_options(channels: pd.DataFrame) -> pd.DataFrame:
    """Canais da base para escolha, com um rótulo legível e ordenados."""
    out = channels[
        ["channel_id", "name", "sector", "region", "kind", "country", "holdout"]
    ].copy()
    where = out["region"].map({"br": "Brasil", "global": "global"})
    out["label"] = (
        out["name"]
        + " ("
        + out["sector"]
        + ", "
        + where
        + out["holdout"].map({True: ", fora do treino", False: ""})
        + ")"
    )
    return out.sort_values(["holdout", "name"]).reset_index(drop=True)


def save_upload(data: bytes, name: str, root: Path = UPLOADS_DIR) -> Path:
    """Grava o vídeo enviado pela tela, com nome pelo conteúdo (sem repetir)."""
    suffix = Path(name).suffix.lower() or ".mp4"
    path = root / f"{hashlib.sha256(data).hexdigest()[:16]}{suffix}"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return path


def summary(prediction: Prediction) -> dict:
    """Números principais de uma previsão, já arredondados para exibir."""
    return {
        "curtidas": round(prediction.predicted_metric),
        "intervalo": (
            round(prediction.interval_low),
            round(prediction.interval_high),
        ),
        "vezes_a_mediana": prediction.relative_to_channel,
        "mediana_do_canal": round(prediction.channel_median_likes),
        "faixa": BAND_NAMES[prediction.band],
        "blocos": [BLOCK_NAMES.get(b, b) for b in prediction.blocks_used],
    }


def interval_frame(prediction: Prediction) -> pd.DataFrame:
    """Previsão, intervalo e mediana do canal, para o gráfico de faixa."""
    return pd.DataFrame(
        {
            "criativo": [prediction.creative_id],
            "previsto": [prediction.predicted_metric],
            "mínimo": [prediction.interval_low],
            "máximo": [prediction.interval_high],
            "mediana do canal": [prediction.channel_median_likes],
        }
    )


def drivers_frame(prediction: Prediction) -> pd.DataFrame:
    """Fatores que mais pesaram (contribuição SHAP em log)."""
    rows = [
        {
            "fator": d.label + (f" ({d.value})" if d.value else ""),
            "efeito": d.effect,
            "sentido": d.direction,
        }
        for d in prediction.top_drivers
    ]
    return pd.DataFrame(rows, columns=["fator", "efeito", "sentido"])


def attributes_frame(prediction: Prediction) -> pd.DataFrame:
    """Atributos extraídos do vídeo, em duas colunas."""
    return pd.DataFrame(
        [{"atributo": k, "valor": str(v)} for k, v in prediction.attributes.items()],
        columns=["atributo", "valor"],
    )


def comparables_frame(prediction: Prediction) -> pd.DataFrame:
    """Shorts reais parecidos, com o desempenho observado."""
    rows = [
        {
            "título": c.title,
            "canal": c.channel,
            "curtidas": c.likes,
            "vezes a mediana do canal": c.relative_to_channel,
            "link": c.url,
        }
        for c in prediction.comparables
    ]
    return pd.DataFrame(
        rows,
        columns=["título", "canal", "curtidas", "vezes a mediana do canal", "link"],
    )


def ranking_frame(ranking: Ranking) -> pd.DataFrame:
    """Criativos na ordem do ranking, com os números de cada um."""
    rows = [
        {
            "posição": i,
            "criativo": p.creative_id,
            "curtidas previstas": round(p.predicted_metric),
            "intervalo de 80%": f"{br(p.interval_low)} a {br(p.interval_high)}",
            "vezes a mediana": br(p.relative_to_channel, 2),
            "baixa confiança": "sim" if p.confidence_flag else "não",
        }
        for i, p in enumerate(ranking.predictions, 1)
    ]
    return pd.DataFrame(rows)


def demo_frame(ranking: Ranking, real: dict[str, int]) -> pd.DataFrame:
    """Previsto × real dos Shorts de um canal de demonstração."""
    rows = []
    for p in ranking.predictions:
        likes = real[p.creative_id]
        rows.append(
            {
                "Short": p.creative_id,
                "previsto": round(p.predicted_metric),
                "mínimo": round(p.interval_low),
                "máximo": round(p.interval_high),
                "real": likes,
                "dentro do intervalo": "sim"
                if p.interval_low <= likes <= p.interval_high
                else "não",
            }
        )
    return pd.DataFrame(rows)


def _readable(table: pd.DataFrame) -> pd.DataFrame:
    table = table[[c for c in ["experimento", *METRIC_COLUMNS] if c in table]]
    table = table.dropna(axis=1, how="all").copy()
    table["experimento"] = table["experimento"].map(
        lambda e: EXPERIMENT_NAMES.get(e, e)
    )
    if "n" in table:
        table["n"] = table["n"].astype(int)
    metrics = [c for c in table if c not in ("experimento", "n")]
    # Texto já formatado: uma coluna só de texto (sem misturar com números).
    table[metrics] = table[metrics].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    return table.rename(columns={"experimento": "modelo", **METRIC_COLUMNS})


def results_tables(root: Path = REPORTS_DIR) -> dict[str, pd.DataFrame]:
    """Tabelas de resultados do projeto, com nomes legíveis (as que existirem)."""
    tables = {}
    blocks = root / "multimodal_results.csv"
    if blocks.is_file():
        tables["Ganho de cada bloco (teste com todos os blocos)"] = _readable(
            pd.read_csv(blocks)
        )
    llm = root / "llm_baseline_results.csv"
    if llm.is_file():
        tables["Modelo treinado × LLM sem modelo"] = _readable(pd.read_csv(llm))
    return tables
