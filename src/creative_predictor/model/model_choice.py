"""Por que LightGBM: comparação de modelos e desempenho mês a mês no teste.

Todos os modelos usam as mesmas features de metadados e texto (com o
embedding do título) e o mesmo split temporal. A escolha é feita pelo MAE na
validação; o teste só confirma. Compara:

- LightGBM com as perdas Huber, L1 e L2 (a escolhida vai para o modelo final);
- o gradient boosting do scikit-learn (`HistGradientBoostingRegressor`), que
  também aceita campos vazios;
- uma regressão linear (Ridge), com campos vazios preenchidos pela mediana e
  categorias em one-hot.

Grava `reports/model_choice_results.csv`. O mês a mês usa o modelo de
metadados e texto escolhido, no teste (junho a agosto de 2026).

Uso:
    uv run python -m creative_predictor.model.model_choice
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from creative_predictor import config
from creative_predictor.model import dataset as ds
from creative_predictor.model import train as tr
from creative_predictor.model.metrics import band_edges, evaluate

RESULTS_PATH = config.ROOT_DIR / "reports" / "model_choice_results.csv"
KEYS = ("n", "mae_log", "spearman_within_channel", "pairwise_accuracy", "band_f1")


def _sklearn_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Features num formato que o scikit-learn aceita: números como float e o
    resto (categorias e textos curtos) como texto."""
    out = frame[features].copy()
    for column in out:
        if pd.api.types.is_bool_dtype(out[column]) or pd.api.types.is_numeric_dtype(
            out[column]
        ):
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
        else:
            out[column] = out[column].astype(str).astype(object)
    return out


def _sklearn_models(x: pd.DataFrame) -> dict:
    categorical = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
    numeric = [c for c in x if c not in categorical]
    ridge = make_pipeline(
        ColumnTransformer(
            [
                (
                    "num",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    numeric,
                ),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        ),
        Ridge(alpha=10.0),
    )
    hgb = make_pipeline(
        ColumnTransformer(
            [
                ("num", "passthrough", numeric),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ]
        ),
        HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=500, early_stopping=True, random_state=0
        ),
    )
    return {"sklearn_hist_gradient_boosting": hgb, "ridge_linear": ridge}


def run() -> pd.DataFrame:
    table = ds.build_table()
    rows = ds.modeling_rows(table)
    blocks = dict(tr.BLOCKS)
    fit_rows = rows.index[rows["split"] == "train"]
    blocks["texto_embeddings"], _ = tr.add_text_embeddings(table, fit_rows)
    rows = ds.modeling_rows(table)
    split = tr.make_split(rows, ("br", "global"))
    edges = band_edges(split.train["target"])
    features = [f for block in blocks.values() for f in block]

    results, preds = [], {}
    for objective in tr.OBJECTIVES:
        model = tr.fit_lgbm(split.train, split.valid, features, objective)
        name = f"lightgbm_{objective}"
        preds[name] = model.predict(split.test[features])
        valid = evaluate(split.valid, model.predict(split.valid[features]), edges)
        results.append((name, valid, evaluate(split.test, preds[name], edges)))

    x_train = _sklearn_frame(split.train, features)
    for name, model in _sklearn_models(x_train).items():
        model.fit(x_train, split.train["target"])
        valid_pred = model.predict(_sklearn_frame(split.valid, features))
        preds[name] = model.predict(_sklearn_frame(split.test, features))
        valid = evaluate(split.valid, valid_pred, edges)
        results.append((name, valid, evaluate(split.test, preds[name], edges)))

    out = pd.DataFrame(
        [
            {"analise": "modelo", "modelo": name}
            | {f"validacao_{k}": valid[k] for k in KEYS}
            | {f"teste_{k}": test[k] for k in KEYS}
            for name, valid, test in results
        ]
    )
    chosen = out.loc[out["validacao_mae_log"].idxmin(), "modelo"]
    out["escolhido_na_validacao"] = out["modelo"] == chosen

    # Mês a mês no teste, com o modelo escolhido.
    month = split.test["published_at"].dt.strftime("%Y-%m")
    monthly = []
    for value in sorted(month.unique()):
        part = month == value
        metrics = evaluate(split.test[part], preds[chosen][part.to_numpy()], edges)
        baseline = evaluate(split.test[part], np.zeros(part.sum()), edges)
        monthly.append(
            {"analise": "mes_no_teste", "modelo": chosen, "mes": value}
            | {f"teste_{k}": metrics[k] for k in KEYS}
            | {"teste_mae_log_baseline_mediana": baseline["mae_log"]}
        )
    return pd.concat([out, pd.DataFrame(monthly)], ignore_index=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    results = run()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(results.round(3).to_string(index=False))
    print(f"\nResultados em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
