"""Modelo pronto para uso: treino final, salvamento e carregamento.

O pacote guarda tudo o que a previsão precisa além do código: os três
LightGBM (previsão e quantis P10 e P90 do intervalo), o PCA dos embeddings do
título, os limites das faixas e a largura de intervalo a partir da qual a
previsão é marcada como de baixa confiança.

A configuração (features, objetivo, rodadas) vem da validação temporal, como
nos experimentos; o modelo salvo é retreinado com todos os Shorts com alvo,
para usar também os meses mais recentes.

Uso:
    uv run python -m creative_predictor.predict.bundle
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from creative_predictor import config
from creative_predictor.model import dataset as ds
from creative_predictor.model import train as tr
from creative_predictor.model.metrics import band_edges

BUNDLE_PATH = config.DATA_DIR / "models" / "predictor.joblib"
OBJECTIVE = "huber"  # escolhido na validação (ver reports/metadata_text_results.csv)
WIDE_INTERVAL_QUANTILE = 0.8  # intervalo mais largo que 80% da validação = incerto


@dataclass
class ModelBundle:
    features: list[str]
    point: lgb.Booster
    low: lgb.Booster
    high: lgb.Booster
    title_pca: PCA
    band_edges: np.ndarray
    wide_interval: float
    trained_at: str
    n_train: int
    channels_in_training: set[str] = field(default_factory=set)

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, ...]:
        """Previsão, P10 e P90 do alvo em log (relativo ao canal)."""
        x = rows[self.features]
        low, high = self.low.predict(x), self.high.predict(x)
        return self.point.predict(x), np.minimum(low, high), np.maximum(low, high)

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Contribuição SHAP de cada feature (TreeSHAP da própria LightGBM)."""
        values = self.point.predict(rows[self.features], pred_contrib=True)
        return pd.DataFrame(values[:, :-1], columns=self.features, index=rows.index)


def train_bundle() -> ModelBundle:
    table = ds.build_table()
    rows = ds.modeling_rows(table)
    fit_rows = rows.index[rows["split"] == "train"]
    title_cols, pca = tr.add_text_embeddings(table, fit_rows)
    rows = ds.modeling_rows(table)
    features = [f for block in tr.BLOCKS.values() for f in block] + title_cols

    split = tr.make_split(rows, ("br", "global"))
    point = tr.fit_lgbm(split.train, split.valid, features, OBJECTIVE)
    low = tr.fit_lgbm(split.train, split.valid, features, "quantile", alpha=0.1)
    high = tr.fit_lgbm(split.train, split.valid, features, "quantile", alpha=0.9)
    lo, hi = low.predict(split.valid[features]), high.predict(split.valid[features])
    wide = float(np.quantile(np.abs(hi - lo), WIDE_INTERVAL_QUANTILE))

    def refit(model, objective, alpha=None):
        return tr.fit_lgbm(
            rows,
            None,
            features,
            objective,
            alpha=alpha,
            num_rounds=model.best_iteration,
        )

    return ModelBundle(
        features=features,
        point=refit(point, OBJECTIVE),
        low=refit(low, "quantile", 0.1),
        high=refit(high, "quantile", 0.9),
        title_pca=pca,
        band_edges=band_edges(split.train["target"]),
        wide_interval=wide,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_train=len(rows),
        channels_in_training=set(rows["channel_id"]),
    )


def save(bundle: ModelBundle, path: Path = BUNDLE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp)
    tmp.replace(path)
    return path


def load(path: Path = BUNDLE_PATH) -> ModelBundle:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} não existe: treine com "
            "`uv run python -m creative_predictor.predict.bundle`"
        )
    return joblib.load(path)


def main() -> None:
    # Pelo nome do módulo (e não por `__main__`), para o pickle guardar a
    # classe como `creative_predictor.predict.bundle.ModelBundle`.
    from creative_predictor.predict import bundle as module  # noqa: PLW0406

    bundle = module.train_bundle()
    path = module.save(bundle)
    print(
        f"Modelo com {len(bundle.features)} features, treinado em "
        f"{bundle.n_train} Shorts, salvo em {path}"
    )


if __name__ == "__main__":
    main()
