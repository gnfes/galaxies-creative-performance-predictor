"""Validação dos sinais de baixa confiança ligados ao histórico do canal.

Dois candidatos, ambos calculáveis antes da publicação:
- histórico desatualizado: os Shorts que definem a mediana do canal são
  antigos (`history_age_days` acima de um limite);
- canal em queda: os 5 Shorts estabilizados mais recentes rendem bem abaixo
  da mediana e o ritmo de postagem caiu nos últimos 90 dias.

O limite do histórico desatualizado é escolhido na validação (o que mais
separa o erro absoluto, com um mínimo de Shorts marcados) e conferido no
teste. Também mede uma folga conformal própria para os Shorts marcados
(intervalo mais largo onde o modelo erra mais) e o efeito de usar a idade do
histórico como feature do modelo. Resultados em
`reports/confidence_results.csv`.

Uso:
    uv run python -m creative_predictor.model.confidence
"""

import argparse

import numpy as np
import pandas as pd

from creative_predictor import config
from creative_predictor.features.history import STALE_HISTORY_DAYS, channel_decline
from creative_predictor.model import dataset as ds
from creative_predictor.model import multimodal as mm
from creative_predictor.model import train as tr
from creative_predictor.model.metrics import band_edges, conformal_margin, evaluate

RESULTS_PATH = config.ROOT_DIR / "reports" / "confidence_results.csv"
STALE_CANDIDATES = (180, 270, 365, 540, 730)
MIN_FLAGGED = 50  # Shorts marcados na validação para um limite valer
COVERAGE = tr.INTERVAL[1] - tr.INTERVAL[0]


def group_errors(frame: pd.DataFrame, pred: np.ndarray, flagged: pd.Series) -> dict:
    """Erro médio (viés) e absoluto dos Shorts marcados e dos demais."""
    error = frame["target"].to_numpy() - pred
    flag = flagged.reindex(frame.index).fillna(False).to_numpy(bool)
    out = {"marcados": int(flag.sum()), "fracao_marcada": float(flag.mean())}
    for name, mask in (("marcados", flag), ("demais", ~flag)):
        out[f"vies_{name}"] = float(error[mask].mean()) if mask.any() else np.nan
        out[f"mae_{name}"] = float(np.abs(error[mask]).mean()) if mask.any() else np.nan
    return out


def choose_threshold(valid: pd.DataFrame, pred: np.ndarray) -> tuple[int, list]:
    """Limite da idade do histórico que mais separa o erro na validação."""
    rows, best, best_gap = [], None, -np.inf
    for days in STALE_CANDIDATES:
        flagged = valid["history_age_days"] > days
        stats = group_errors(valid, pred, flagged)
        rows.append({"limite_dias": days} | stats)
        gap = stats["mae_marcados"] - stats["mae_demais"]
        if stats["marcados"] >= MIN_FLAGGED and gap > best_gap:
            best, best_gap = days, gap
    return best, rows


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((y >= low) & (y <= high)))


def run() -> pd.DataFrame:
    table, blocks, _ = mm.build()
    table["decline"] = channel_decline(table)
    rows = ds.modeling_rows(table)
    split = tr.make_split(rows, ("br", "global"))
    edges = band_edges(split.train["target"])
    features = [f for block in blocks.values() for f in block]

    point = tr.fit_lgbm(split.train, split.valid, features, mm.OBJECTIVE)
    low_m, high_m = (
        tr.fit_lgbm(split.train, split.valid, features, "quantile", alpha=a)
        for a in tr.INTERVAL
    )

    def predict(frame):
        lo, hi = low_m.predict(frame[features]), high_m.predict(frame[features])
        return point.predict(frame[features]), np.minimum(lo, hi), np.maximum(lo, hi)

    v_pred, v_lo, v_hi = predict(split.valid)
    t_pred, t_lo, t_hi = predict(split.test)
    results = []

    # 1. Limite escolhido na validação e conferido no teste.
    threshold, scan = choose_threshold(split.valid, v_pred)
    if threshold != STALE_HISTORY_DAYS:
        print(
            f"Atenção: a validação escolheu {threshold} dias, e a previsão usa "
            f"{STALE_HISTORY_DAYS} (features.history.STALE_HISTORY_DAYS)."
        )
    results += [{"analise": "limite_validacao"} | r for r in scan]
    for name, valid_flag, test_flag in (
        (
            f"historico_desatualizado_{threshold}d",
            split.valid["history_age_days"] > threshold,
            split.test["history_age_days"] > threshold,
        ),
        ("canal_em_queda", split.valid["decline"], split.test["decline"]),
    ):
        results.append(
            {"analise": f"{name}_validacao"}
            | group_errors(split.valid, v_pred, valid_flag)
        )
        results.append(
            {"analise": f"{name}_teste"} | group_errors(split.test, t_pred, test_flag)
        )

    # 2. Folga conformal única × folga própria para os Shorts de risco
    # (histórico desatualizado, e também com a regra de queda).
    y_valid, y_test = split.valid["target"].to_numpy(), split.test["target"].to_numpy()
    margin = conformal_margin(y_valid, v_lo, v_hi, COVERAGE)
    v_stale = (split.valid["history_age_days"] > threshold).to_numpy()
    t_stale = (split.test["history_age_days"] > threshold).to_numpy()
    groups = {
        "desatualizado": (v_stale, t_stale),
        "desatualizado_ou_queda": (
            v_stale | split.valid["decline"].to_numpy(bool),
            t_stale | split.test["decline"].to_numpy(bool),
        ),
    }
    # Para cada grupo, a cobertura dele com a folga única e com a própria,
    # sobre os mesmos Shorts.
    for name, (v_mask, t_mask) in groups.items():
        own = (
            conformal_margin(y_valid[v_mask], v_lo[v_mask], v_hi[v_mask], COVERAGE)
            if v_mask.any()
            else margin
        )
        m = np.where(t_mask, max(own, margin), margin)
        lo, hi = t_lo - m, t_hi + m
        results.append(
            {
                "analise": f"intervalo_grupo_{name}",
                "folga_geral": round(margin, 4),
                "folga_grupo": round(own, 4),
                "marcados": int(t_mask.sum()),
                "cobertura_grupo_folga_unica": coverage(
                    y_test[t_mask], t_lo[t_mask] - margin, t_hi[t_mask] + margin
                ),
                "cobertura_grupo_folga_propria": coverage(
                    y_test[t_mask], lo[t_mask], hi[t_mask]
                ),
                "cobertura_teste_folga_unica": coverage(
                    y_test, t_lo - margin, t_hi + margin
                ),
                "cobertura_teste_folga_por_grupo": coverage(y_test, lo, hi),
                "largura_media_folga_unica": float(np.mean(t_hi - t_lo + 2 * margin)),
                "largura_media_folga_por_grupo": float(np.mean(hi - lo)),
            }
        )

    # 3. Idade do histórico como feature do modelo.
    with_age = [*features, "history_age_days"]
    model = tr.fit_lgbm(split.train, split.valid, with_age, mm.OBJECTIVE)
    for name, pred in (
        ("modelo_sem_idade", t_pred),
        ("modelo_com_idade", model.predict(split.test[with_age])),
    ):
        result = evaluate(split.test, pred, edges)
        results.append(
            {"analise": name}
            | {
                k: result[k]
                for k in (
                    "n",
                    "mae_log",
                    "mape",
                    "median_ape",
                    "spearman_likes",
                    "spearman_target",
                    "spearman_within_channel",
                    "pairwise_accuracy",
                    "band_f1",
                )
            }
            | group_errors(split.test, pred, split.test["history_age_days"] > threshold)
        )
    return pd.DataFrame(results)


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
