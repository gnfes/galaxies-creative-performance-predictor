"""Ganho de cada bloco multimodal sobre o modelo de metadados e texto.

Blocos cumulativos:
1. metadados: histórico do canal, formato, horário, contagens do texto e
   embedding do título (o modelo de `model.train`);
2. sinais: medidas diretas do vídeo e do áudio (cor, rostos, texto na tela,
   movimento, cortes, energia, fala);
3. embeddings: frames, áudio e transcrição (EmbeddingGemma 2, reduzidos por
   PCA ajustado no treino);
4. extrator: rubrica respondida pelo `gemma-4-31b-it`.

Os modelos treinam com todos os Shorts (mídia ausente fica vazia, o que a
LightGBM trata nativamente) e são comparados no mesmo subconjunto do teste em
que todos os blocos existem.

Uso:
    uv run python -m creative_predictor.model.multimodal
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from creative_predictor import config
from creative_predictor.features import extract, media_embed
from creative_predictor.model import dataset as ds
from creative_predictor.model import train as tr
from creative_predictor.model.metrics import band_edges, evaluate

RESULTS_PATH = config.ROOT_DIR / "reports" / "multimodal_results.csv"
EXPERIMENT = "multimodal"
OBJECTIVE = "huber"
MEDIA_COMPONENTS = 16  # componentes principais por tipo de embedding

SIGNAL_FEATURES = [
    "brightness_mean",
    "contrast_mean",
    "saturation_mean",
    "colorfulness_mean",
    "face_frames_share",
    "face_max_count",
    "face_area_max",
    "face_in_hook",
    "text_frames_share",
    "text_area_mean",
    "text_words_mean",
    "text_in_hook",
    "motion_mean",
    "motion_std",
    "motion_hook",
    "cuts",
    "cuts_per_10s",
    "cuts_in_hook",
    "first_cut_s",
    "audio_db_mean",
    "audio_db_std",
    "silence_share",
    "speech_share",
    "speech_words",
    "speech_words_per_second",
    "speech_language",
]
# Campos de texto livre da rubrica (gancho, listas de elementos) ficam de fora.
RUBRIC_TEXT = {
    "vlm_hook_primeiros_3s",
    "vlm_elementos_video_top5",
    "vlm_elementos_audio_top5",
}
RUBRIC_CATEGORICAL = ["vlm_cor_dominante", "vlm_tom_narracao"]
SPEECH_LANGUAGES = ["pt", "en"]


def media_columns(table: pd.DataFrame) -> None:
    """Ajusta as colunas vindas da extração para o modelo (no lugar).

    O idioma da fala vira pt / en / outro, e as categorias da rubrica viram
    categóricas. Usado no treino e na previsão de um criativo novo.
    """
    if "speech_language" not in table:
        table["speech_language"] = None
    language = table["speech_language"]
    table["speech_language"] = pd.Categorical(
        language.where(language.isin(SPEECH_LANGUAGES) | language.isna(), "outro"),
        categories=[*SPEECH_LANGUAGES, "outro"],
    )
    for col in RUBRIC_CATEGORICAL:
        if col in table:
            table[col] = table[col].astype("category")


def blank(table: pd.DataFrame, columns: list[str]) -> None:
    """Esvazia colunas (no lugar), mantendo as categóricas como categóricas:
    a LightGBM recusa uma categórica do treino que chega como número."""
    for col in columns:
        if col in table and isinstance(table[col].dtype, pd.CategoricalDtype):
            table[col] = pd.Categorical(
                [None] * len(table), categories=table[col].cat.categories
            )
        else:
            table[col] = np.nan


def fit_media_pcas(table: pd.DataFrame, fit_rows: pd.Index) -> dict[str, PCA]:
    """PCA de cada tipo de embedding, ajustado só nos Shorts do treino."""
    pcas = {}
    for kind in media_embed.KINDS:
        fit_ids = table.loc[fit_rows, "video_id"]
        vectors = media_embed.load(kind, fit_ids)
        if len(vectors) < MEDIA_COMPONENTS * 4:
            continue
        pca = PCA(n_components=MEDIA_COMPONENTS, random_state=tr.SEED)
        pcas[kind] = pca.fit(np.vstack(list(vectors.values())))
    return pcas


def add_media_embeddings(
    table: pd.DataFrame, pcas: dict[str, PCA], vectors: dict[str, dict] | None = None
) -> list[str]:
    """Componentes principais dos embeddings de frames, áudio e transcrição.

    `vectors` (tipo → id → vetor) vem pronto na previsão; no treino, é lido
    do cache. Short sem embedding fica com as colunas vazias.
    """
    columns = []
    for kind, pca in pcas.items():
        found = (
            vectors[kind]
            if vectors is not None
            else media_embed.load(kind, table["video_id"])
        )
        names = [f"{kind}_pc{i}" for i in range(pca.n_components_)]
        table[names] = np.nan
        has = table["video_id"].isin(found)
        if has.any():
            table.loc[has, names] = pca.transform(
                np.vstack([found[v] for v in table.loc[has, "video_id"]])
            )
        columns += names
    return columns


def rubric_features(columns) -> list[str]:
    """Campos da rubrica usados como feature (fora os de texto livre)."""
    return [c for c in columns if c.startswith("vlm_") and c not in RUBRIC_TEXT]


def build() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, PCA]]:
    """Tabela de modelagem com todos os blocos, as features de cada bloco e os
    PCAs (título e cada tipo de embedding) ajustados no treino."""
    table = ds.build_table()
    rows = ds.modeling_rows(table)
    fit_rows = rows.index[rows["split"] == "train"]
    title_cols, title_pca = tr.add_text_embeddings(table, fit_rows)
    shorts = pd.read_parquet(config.PROCESSED_DIR / "shorts.parquet")
    media = extract.assemble(extract.sample_ids(shorts))
    media = media.drop(columns=["transcript"], errors="ignore")
    table = table.merge(media, on="video_id", how="left").set_index(table.index)
    media_columns(table)
    pcas = fit_media_pcas(table, fit_rows)
    embed_cols = add_media_embeddings(table, pcas)
    blocks = {
        "metadados": [f for block in tr.BLOCKS.values() for f in block] + title_cols,
        "sinais": [f for f in SIGNAL_FEATURES if f in table],
        "embeddings": embed_cols,
        "extrator": rubric_features(table.columns),
    }
    return table, blocks, {"title": title_pca, **pcas}


def run() -> pd.DataFrame:
    import mlflow

    table, blocks, _ = build()
    rows = ds.modeling_rows(table)
    split = tr.make_split(rows, ("br", "global"))
    edges = band_edges(split.train["target"])
    # Subconjunto do teste em que todos os blocos existem. Cada bloco é
    # checado por um campo comum a todos os setores (os campos da rubrica
    # específicos de um setor ficam vazios nos outros).
    markers = ["brightness_mean", "vlm_forca_hook_0a10"] + [
        f"{kind}_pc0" for kind in media_embed.KINDS if f"{kind}_pc0" in table
    ]
    complete = split.test.dropna(subset=[m for m in markers if m in table])

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)

    results, features = [], []
    coverage = {
        "treino_com_sinais": float(split.train["brightness_mean"].notna().mean()),
        "treino_com_extrator": float(
            split.train[blocks["extrator"][0]].notna().mean()
            if blocks["extrator"]
            else 0.0
        ),
    }
    baseline = evaluate(complete, np.zeros(len(complete)), edges)
    results.append(
        tr.log_run(
            "baseline_mediana_canal",
            {"blocos": "nenhum", "avaliacao": "teste_completo"},
            baseline,
        )
    )
    for name, block in blocks.items():
        if not block:
            continue
        features = features + block
        model = tr.fit_lgbm(split.train, split.valid, features, OBJECTIVE)
        params = {
            "blocos": "+".join(list(blocks)[: list(blocks).index(name) + 1]),
            "n_features": len(features),
            "rodadas": model.best_iteration,
            "avaliacao": "teste_completo",
            **coverage,
        }
        result = evaluate(complete, model.predict(complete[features]), edges)
        results.append(tr.log_run(f"lgbm_{name}", params, result, model))
    # O modelo completo quando a rubrica não é pedida na previsão (criativo
    # sigiloso ou extrator fora do ar): os campos dela ficam vazios.
    if blocks["extrator"]:
        without = complete[features].copy()
        blank(without, blocks["extrator"])
        result = evaluate(complete, model.predict(without), edges)
        results.append(
            tr.log_run(
                "lgbm_extrator_sem_rubrica_na_previsao",
                params | {"rubrica_na_previsao": False},
                result,
            )
        )
    return pd.DataFrame(results)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    results = run()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    columns = [
        "experimento",
        "n",
        "mae_log",
        "spearman_target",
        "spearman_within_channel",
        "pairwise_accuracy",
        "band_f1",
    ]
    with pd.option_context("display.width", 200):
        print(
            results[[c for c in columns if c in results]]
            .round(3)
            .to_string(index=False)
        )
    print(f"\nResultados em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
