"""Baselines e modelo de metadados e texto, com registro no MLflow.

Experimentos (todos avaliados no teste do Brasil, jun a ago/2026):
1. Baselines do brief: mediana do canal (y = 0), média do desempenho recente
   do canal e regressão linear das curtidas sobre os inscritos.
2. LightGBM por blocos de features: histórico do canal; + formato e horário;
   + contagens do texto; + embeddings do título (com `--text-embeddings`).
3. Treino só com canais do Brasil contra Brasil + globais.
4. Split por canal: canais do Brasil nunca vistos no treino (5 grupos).
5. Intervalo de 80% por regressão quantílica (P10 e P90).

Uso:
    uv run python -m creative_predictor.model.train [--text-embeddings]
"""

import argparse
import os
import warnings
from dataclasses import dataclass

import lightgbm as lgb

# Sem a dica que o MLflow imprime ao ser importado.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from creative_predictor import config
from creative_predictor.features.text import cache_path, embed_texts
from creative_predictor.model import dataset as ds
from creative_predictor.model.metrics import band_edges, conformal_margin, evaluate

EXPERIMENT = "metadados-e-texto"
RESULTS_PATH = config.ROOT_DIR / "reports" / "metadata_text_results.csv"
SEED = 42
TEXT_COMPONENTS = 16  # componentes principais do embedding do título
OBJECTIVES = ("huber", "regression_l1", "regression")
INTERVAL = (0.1, 0.9)

BLOCKS = {
    "historico": [
        "sector",
        "region",
        "kind",
        *ds.HISTORY_FEATURES,
        *ds.CONTROL_FEATURES,
    ],
    "formato_horario": [
        "category_id",
        "definition",
        *ds.FORMAT_FEATURES,
        *ds.TIME_FEATURES,
    ],
    "texto_contagens": ds.TEXT_STAT_FEATURES,
}

PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def make_split(rows: pd.DataFrame, train_regions: tuple[str, ...]) -> Split:
    """Treino e validação nas regiões pedidas; teste sempre só no Brasil."""
    in_regions = rows["region"].isin(train_regions)
    return Split(
        train=rows[(rows["split"] == "train") & in_regions],
        valid=rows[(rows["split"] == "valid") & in_regions],
        test=rows[(rows["split"] == "test") & (rows["region"] == "br")],
    )


def fit_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    objective: str,
    alpha: float | None = None,
    num_rounds: int | None = None,
) -> lgb.Booster:
    """LightGBM com parada antecipada na validação (ou rodadas fixas)."""
    params = PARAMS | {"objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    data = lgb.Dataset(train[features], train["target"])
    if num_rounds is not None:
        return lgb.train(params, data, num_boost_round=num_rounds)
    valid_data = lgb.Dataset(valid[features], valid["target"], reference=data)
    return lgb.train(
        params | {"metric": "quantile" if alpha is not None else "l1"},
        data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def add_text_embeddings(
    table: pd.DataFrame, fit_rows: pd.Index
) -> tuple[list[str], PCA]:
    """Acrescenta componentes principais do embedding do título à tabela.

    Só os Shorts da janela (treino, teste e demonstração) recebem embedding;
    os de histórico anterior não entram no modelo. O PCA é ajustado só nas
    linhas de treino (`fit_rows`).
    """
    window = table.index[table["in_window"]]
    vectors = embed_texts(
        table.loc[window, "video_id"], table.loc[window, "title"], cache_path("title")
    )
    pca = PCA(n_components=TEXT_COMPONENTS, random_state=SEED)
    pca.fit(vectors.loc[table.loc[fit_rows, "video_id"]].to_numpy())
    columns = [f"title_pc{i}" for i in range(TEXT_COMPONENTS)]
    table[columns] = np.nan
    table.loc[window, columns] = pca.transform(vectors.to_numpy())
    return columns, pca


def subscriber_baseline(split: Split, channels: pd.DataFrame) -> np.ndarray:
    """Regressão de log(curtidas) sobre log(inscritos), convertida para o alvo.

    Inscritos são os da data da coleta, não os da publicação (limitação da
    API); por isso só entram aqui, como baseline pedida pelo brief.
    """
    subscribers = channels.set_index("channel_id")["subscribers"]

    def x(frame):
        return np.log1p(frame["channel_id"].map(subscribers).astype(float)).to_frame()

    train_x = x(split.train)
    known = train_x.iloc[:, 0].notna().to_numpy()
    reg = LinearRegression().fit(
        train_x[known], np.log1p(split.train.loc[known, "likes"].astype(float))
    )
    test_x = x(split.test)
    # Canal que oculta os inscritos fica com a mediana do próprio canal (y = 0).
    pred = np.zeros(len(split.test))
    ok = test_x.iloc[:, 0].notna().to_numpy()
    if ok.any():
        log_likes = reg.predict(test_x[ok])
        median = split.test.loc[ok, "history_median_likes"].to_numpy()
        pred[ok] = log_likes - np.log1p(median)
    return pred


def channel_split_predictions(
    rows: pd.DataFrame,
    features: list[str],
    objective: str,
    num_rounds: int,
    alpha: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Previsões no teste para canais do Brasil nunca vistos no treino.

    Cada grupo de canais do Brasil sai inteiro do treino (todos os períodos) e
    é avaliado no período de teste; canais globais ficam sempre no treino.
    """
    br_channels = rows.loc[rows["region"] == "br", "channel_id"].unique()
    folds = GroupKFold(n_splits=5).split(br_channels, groups=br_channels)
    parts, preds = [], []
    for _, held_idx in folds:
        held = rows["channel_id"].isin(set(br_channels[held_idx]))
        train = rows[(rows["split"] != "test") & ~held]
        test = rows[(rows["split"] == "test") & held]
        model = fit_lgbm(
            train, None, features, objective, alpha=alpha, num_rounds=num_rounds
        )
        parts.append(test)
        preds.append(model.predict(test[features]))
    return pd.concat(parts), np.concatenate(preds)


def channel_split_interval(
    rows: pd.DataFrame,
    features: list[str],
    rounds: tuple[int, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Limites P10 e P90 para canais do Brasil nunca vistos no treino."""
    bounds = [
        channel_split_predictions(rows, features, "quantile", n, alpha=a)
        for a, n in zip(INTERVAL, rounds, strict=True)
    ]
    (held, low), (_, high) = bounds
    return held, np.minimum(low, high), np.maximum(low, high)


def log_run(
    name: str, params: dict, result: dict, model: lgb.Booster | None = None
) -> dict:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(
            {k: v for k, v in result.items() if v is not None and np.isfinite(v)}
        )
        if model is not None:
            importance = pd.Series(
                model.feature_importance("gain"), index=model.feature_name()
            ).sort_values(ascending=False)
            mlflow.log_text(importance.to_csv(header=["gain"]), "importance.csv")
    return {"experimento": name, **params, **result}


def run(text_embeddings: bool = False) -> pd.DataFrame:
    table = ds.build_table()
    channels = pd.read_parquet(config.PROCESSED_DIR / "channels.parquet")
    rows = ds.modeling_rows(table)
    blocks = dict(BLOCKS)
    if text_embeddings:
        fit_rows = rows.index[rows["split"] == "train"]
        blocks["texto_embeddings"], _ = add_text_embeddings(table, fit_rows)
        rows = ds.modeling_rows(table)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)

    split = make_split(rows, ("br", "global"))
    edges = band_edges(split.train["target"])
    results = []

    def evaluate_test(pred, frame=split.test, **interval):
        return evaluate(frame, pred, edges, **interval)

    # 1. Baselines
    base = {"modelo": "baseline", "treino": "br+global"}
    results.append(
        log_run(
            "baseline_mediana_canal",
            base,
            evaluate_test(np.zeros(len(split.test))),
        )
    )
    recent = split.test["recent_target_mean"].fillna(0).to_numpy()
    results.append(log_run("baseline_media_recente_canal", base, evaluate_test(recent)))
    results.append(
        log_run(
            "baseline_inscritos",
            base,
            evaluate_test(subscriber_baseline(split, channels)),
        )
    )

    # 2. Objetivo escolhido na validação, com todas as features de metadados.
    all_features = [f for block in blocks.values() for f in block]
    valid_mae = {}
    for objective in OBJECTIVES:
        model = fit_lgbm(split.train, split.valid, all_features, objective)
        valid_mae[objective] = model.best_score["valid_0"]["l1"]
    objective = min(valid_mae, key=valid_mae.get)
    print(f"Objetivo escolhido na validação: {objective} ({valid_mae})")

    # Blocos cumulativos.
    features: list[str] = []
    for block, block_features in blocks.items():
        features = features + block_features
        model = fit_lgbm(split.train, split.valid, features, objective)
        params = {
            "modelo": "lightgbm",
            "treino": "br+global",
            "blocos": "+".join(list(blocks)[: list(blocks).index(block) + 1]),
            "objetivo": objective,
            "rodadas": model.best_iteration,
            "n_features": len(features),
        }
        result = evaluate_test(model.predict(split.test[features]))
        results.append(log_run(f"lgbm_{block}", params, result, model))
    # O último bloco acumula todas as features.
    full_features, full_model = features, model

    # 3. Treino só com o Brasil.
    br = make_split(rows, ("br",))
    model = fit_lgbm(br.train, br.valid, full_features, objective)
    params = {
        "modelo": "lightgbm",
        "treino": "br",
        "blocos": "todos",
        "objetivo": objective,
        "rodadas": model.best_iteration,
        "n_features": len(full_features),
    }
    results.append(
        log_run(
            "lgbm_todos_treino_br",
            params,
            evaluate_test(model.predict(br.test[full_features])),
            model,
        )
    )

    # 4. Canais nunca vistos.
    test_rows, preds = channel_split_predictions(
        rows, full_features, objective, full_model.best_iteration
    )
    params = {
        "modelo": "lightgbm",
        "treino": "br+global",
        "blocos": "todos",
        "objetivo": objective,
        "avaliacao": "canais_nao_vistos",
    }
    results.append(
        log_run(
            "lgbm_todos_canais_nao_vistos", params, evaluate(test_rows, preds, edges)
        )
    )

    # 5. Intervalo de 80%.
    low_model, high_model = (
        fit_lgbm(split.train, split.valid, full_features, "quantile", alpha=a)
        for a in INTERVAL
    )
    low = low_model.predict(split.test[full_features])
    high = high_model.predict(split.test[full_features])
    params = {
        "modelo": "lightgbm",
        "treino": "br+global",
        "blocos": "todos",
        "objetivo": objective,
        "intervalo": f"P{int(INTERVAL[0] * 100)}-P{int(INTERVAL[1] * 100)}",
    }
    low, high = np.minimum(low, high), np.maximum(low, high)
    interval = evaluate_test(
        full_model.predict(split.test[full_features]), low=low, high=high
    )
    # As métricas pontuais são as do último bloco; aqui só as do intervalo.
    keys = ("n", "coverage", "interval_width")
    results.append(
        log_run("lgbm_todos_intervalo", params, {k: interval[k] for k in keys})
    )

    # Intervalo calibrado: a folga conformal vem da validação e é testada no
    # período de teste, que o ajuste nunca viu.
    v_low = low_model.predict(split.valid[full_features])
    v_high = high_model.predict(split.valid[full_features])
    margin = conformal_margin(
        split.valid["target"],
        np.minimum(v_low, v_high),
        np.maximum(v_low, v_high),
        coverage=INTERVAL[1] - INTERVAL[0],
    )
    calibrated = evaluate_test(
        full_model.predict(split.test[full_features]),
        low=low - margin,
        high=high + margin,
    )
    results.append(
        log_run(
            "lgbm_todos_intervalo_conformal",
            params | {"folga_conformal": round(margin, 4)},
            {k: calibrated[k] for k in keys},
        )
    )

    # Intervalo em canais nunca vistos, sem folga e com a folga calibrada nos
    # canais do treino.
    rounds = (low_model.best_iteration, high_model.best_iteration)
    held, u_low, u_high = channel_split_interval(rows, full_features, rounds)
    assert held.index.equals(test_rows.index)  # mesmos grupos, mesma ordem
    for name, m in [("sem_folga", 0.0), ("com_folga", margin)]:
        result = evaluate(test_rows, preds, edges, low=u_low - m, high=u_high + m)
        results.append(
            log_run(
                f"lgbm_todos_intervalo_canais_nao_vistos_{name}",
                params | {"avaliacao": "canais_nao_vistos", "folga": round(m, 4)},
                {k: result[k] for k in keys},
            )
        )

    # Mesmas métricas no subconjunto que vai para a extração multimodal, para
    # comparar depois com os modelos que usam vídeo.
    sample = split.test["media_sample"].to_numpy()
    results.append(
        log_run(
            "lgbm_todos_amostra_multimodal",
            params | {"avaliacao": "amostra_multimodal"},
            evaluate(
                split.test[sample],
                full_model.predict(split.test.loc[sample, full_features]),
                edges,
            ),
        )
    )
    return pd.DataFrame(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--text-embeddings",
        action="store_true",
        help="inclui embeddings do título (calcula com o EmbeddingGemma 2 se "
        "não estiverem no cache)",
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    results = run(args.text_embeddings)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    columns = [
        "experimento",
        "mae_log",
        "mape",
        "spearman_likes",
        "spearman_target",
        "spearman_within_channel",
        "pairwise_accuracy",
        "band_f1",
        "coverage",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            results[[c for c in columns if c in results]]
            .round(3)
            .to_string(index=False)
        )
    print(f"\nResultados em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
