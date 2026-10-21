"""Modelo pronto para uso: treino final, salvamento e carregamento.

O pacote guarda tudo o que a previsão precisa além do código: os três
LightGBM (previsão e quantis P10 e P90 do intervalo), as folgas conformais que
calibram o intervalo para 80% (geral e dos Shorts de risco), os PCAs dos
embeddings (título, frames, áudio e transcrição), as features de cada bloco,
os limites das faixas e a largura de intervalo a partir da qual a previsão é
marcada como de baixa confiança.

O modelo usa todos os blocos (metadados e texto, sinais do vídeo e do áudio,
embeddings e rubrica do extrator). Na base de treino, a maioria dos Shorts
não tem vídeo extraído e muitos não têm rubrica; a LightGBM aprende com esses
campos vazios, e por isso o mesmo modelo prevê um criativo só com texto, com
vídeo sem rubrica ou com tudo.

A configuração (features, objetivo, rodadas) vem da validação temporal, como
nos experimentos; o modelo salvo é retreinado com todos os Shorts com alvo,
para usar também os meses mais recentes.

O modelo publicado fica anexado à release do GitHub; `--download` baixa
essa versão e confere o hash antes de gravar (o arquivo é um pickle, então só
o arquivo publicado deve ser carregado).

Uso:
    uv run python -m creative_predictor.predict.bundle --download  # modelo publicado
    uv run python -m creative_predictor.predict.bundle             # treina de novo
"""

import argparse
import urllib.error
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from creative_predictor import config
from creative_predictor.features.history import history_risk
from creative_predictor.files import download_verified, sha256, write_atomic
from creative_predictor.model import dataset as ds
from creative_predictor.model import multimodal as mm
from creative_predictor.model import train as tr
from creative_predictor.model.metrics import band_edges, conformal_margin

BUNDLE_PATH = config.DATA_DIR / "models" / "predictor.joblib"
OBJECTIVE = "huber"  # escolhido na validação (ver reports/metadata_text_results.csv)
WIDE_INTERVAL_QUANTILE = 0.8  # intervalo mais largo que 80% da validação = incerto
# Modelo publicado: anexado à release v1.0.0 do GitHub, com o hash conferido no
# download. A URL só muda quando o modelo muda, não a cada versão do código.
MODEL_URL = (
    "https://github.com/gnfes/galaxies-creative-performance-predictor/"
    "releases/download/v1.0.0/predictor.joblib"
)
MODEL_SHA256 = "7859296cfe4e0d5860090c2aab36bff09c8d18a8cdcf7757dcadf534695a07b1"


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
    # Folga conformal somada aos dois lados do intervalo (calibrada na
    # validação temporal para cobrir 80%).
    conformal_margin: float = 0.0
    # Features de cada bloco (metadados, sinais, embeddings, extrator) e o PCA
    # de cada tipo de embedding do vídeo.
    blocks: dict[str, list[str]] = field(default_factory=dict)
    media_pcas: dict[str, PCA] = field(default_factory=dict)
    # Folga dos Shorts de risco (histórico desatualizado ou canal em queda,
    # `features.history.history_risk`), calibrada só neles: o modelo erra
    # mais nesses casos, e a folga única deixaria o intervalo curto demais.
    risk_margin: float = 0.0

    def predict(
        self, rows: pd.DataFrame, risk: np.ndarray | None = None
    ) -> tuple[np.ndarray, ...]:
        """Previsão, P10 e P90 do alvo em log (relativo ao canal)."""
        x = rows[self.features]
        low, high = self.low.predict(x), self.high.predict(x)
        low, high = np.minimum(low, high), np.maximum(low, high)
        margin = np.full(len(rows), self.conformal_margin)
        if risk is not None:
            margin = np.where(
                risk, max(self.risk_margin, self.conformal_margin), margin
            )
        return self.point.predict(x), low - margin, high + margin

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Contribuição SHAP de cada feature (TreeSHAP da própria LightGBM)."""
        values = self.point.predict(rows[self.features], pred_contrib=True)
        return pd.DataFrame(values[:, :-1], columns=self.features, index=rows.index)


def train_bundle() -> ModelBundle:
    table, blocks, pcas = mm.build()
    rows = ds.modeling_rows(table)
    features = [f for block in blocks.values() for f in block]

    split = tr.make_split(rows, ("br", "global"))
    point = tr.fit_lgbm(split.train, split.valid, features, OBJECTIVE)
    low = tr.fit_lgbm(split.train, split.valid, features, "quantile", alpha=0.1)
    high = tr.fit_lgbm(split.train, split.valid, features, "quantile", alpha=0.9)
    lo, hi = low.predict(split.valid[features]), high.predict(split.valid[features])
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    margin = conformal_margin(split.valid["target"], lo, hi, coverage=0.8)
    risk = history_risk(table).reindex(split.valid.index).to_numpy(bool)
    # Sem Shorts de risco na validação (outra base), vale a folga geral.
    risk_margin = (
        conformal_margin(split.valid["target"][risk], lo[risk], hi[risk], coverage=0.8)
        if risk.any()
        else margin
    )
    wide = float(np.quantile(hi - lo + 2 * margin, WIDE_INTERVAL_QUANTILE))

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
        title_pca=pcas.pop("title"),
        band_edges=band_edges(split.train["target"]),
        wide_interval=wide,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_train=len(rows),
        channels_in_training=set(rows["channel_id"]),
        conformal_margin=margin,
        blocks=blocks,
        media_pcas=pcas,
        risk_margin=risk_margin,
    )


def save(bundle: ModelBundle, path: Path = BUNDLE_PATH) -> Path:
    return write_atomic(path, lambda tmp: joblib.dump(bundle, tmp))


def load(path: Path = BUNDLE_PATH) -> ModelBundle:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} não existe: baixe o modelo publicado com "
            "`uv run python -m creative_predictor.predict.bundle --download` "
            "ou treine de novo com `uv run python -m creative_predictor.predict.bundle`"
        )
    return joblib.load(path)


def download(
    path: Path = BUNDLE_PATH,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
    force: bool = False,
) -> Path:
    """Baixa o modelo publicado e só o grava se o hash conferir.

    Se já houver um modelo diferente do publicado (retreinado localmente), não
    o substitui sem `force`.
    """
    if path.is_file():
        if sha256(path.read_bytes()) == expected_sha256:
            return path
        if not force:
            raise FileExistsError(
                f"{path} já existe e não é o modelo publicado (talvez retreinado); "
                "use --force para substituí-lo"
            )
    return download_verified(url, path, expected_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--download",
        action="store_true",
        help="baixa o modelo publicado na release em vez de treinar",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="com --download, substitui um modelo local diferente do publicado",
    )
    args = parser.parse_args()
    if args.download:
        try:
            path = download(force=args.force)
        except urllib.error.URLError as error:
            raise SystemExit(f"Não foi possível baixar {MODEL_URL}: {error}") from None
        except (FileExistsError, ValueError) as error:
            raise SystemExit(str(error)) from None
        print(f"Modelo publicado em {path}")
        return
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
