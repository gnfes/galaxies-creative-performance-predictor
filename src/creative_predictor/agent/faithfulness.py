"""Checagem automática das explicações escritas pelo LLM do agente.

O agente promete que o LLM não inventa nem altera números. Esta checagem
gera explicações para Shorts do teste, com a mesma instrução e os mesmos
dados do nó `explain`, e confere em cada uma:
- números: todo número citado existe nos dados que o LLM recebeu (com
  tolerância de arredondamento e de formato, como 1.234 ou 0,59);
- fatores: a explicação cita pelo menos um dos fatores que mais pesaram;
- confiança: a explicação afirma baixa confiança quando ela existe, e não a
  afirma quando não existe ("não há sinais de baixa confiança" conta como
  negar).

Resultado por explicação em `reports/explanation_faithfulness.csv`.

Uso:
    uv run python -m creative_predictor.agent.faithfulness [--n 100]
"""

import argparse
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from creative_predictor import config

RESULTS_PATH = config.ROOT_DIR / "reports" / "explanation_faithfulness.csv"
N_EXPLANATIONS = 100
SEED = 42
# Números que a instrução ou o formato justificam sem estar nos dados.
ALLOWED = {80.0, 5.0}
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_LOW_CONFIDENCE = re.compile(r"baixa confianca|confianca[^.]*\bbaixa\b|incerte|cautela")
_NEGATION = re.compile(r"\b(nao|sem|nenhum|nenhuma)\b")


def _plain(text: str) -> str:
    """Minúsculas e sem acentos, para comparar nomes de fatores."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def readings(token: str) -> set[float]:
    """Valores possíveis de um número escrito (1.234 ou 1,234; 0,59 ou 0.59)."""
    out = set()
    for thousands, decimal in ((".", ","), (",", ".")):
        try:
            out.add(float(token.replace(thousands, "").replace(decimal, ".")))
        except ValueError:
            continue
    return out


def numbers_in(text: str) -> list[str]:
    return _NUMBER.findall(text)


# Campos que são razões (vezes a mediana): só eles podem aparecer no texto
# como percentual ("41% abaixo" de 0,59×).
RATIO_KEYS = {"vezes_a_mediana_do_canal", "vezes_a_mediana_do_seu_canal"}


def data_numbers(data) -> set[float]:
    """Todos os números dos dados enviados ao LLM (inclusive dentro de textos)."""
    out = set()
    if isinstance(data, dict):
        for value in data.values():
            out |= data_numbers(value)
    elif isinstance(data, list | tuple):
        for value in data:
            out |= data_numbers(value)
    elif isinstance(data, bool) or data is None:
        pass
    elif isinstance(data, int | float):
        out.add(float(data))
    elif isinstance(data, str):
        for token in numbers_in(data):
            out |= readings(token)
    return out


def ratio_numbers(data) -> set[float]:
    """Valores dos campos de razão, em qualquer nível dos dados."""
    out = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if key in RATIO_KEYS and isinstance(value, int | float):
                out.add(float(value))
            else:
                out |= ratio_numbers(value)
    elif isinstance(data, list | tuple):
        for value in data:
            out |= ratio_numbers(value)
    return out


def supported(token: str, known: set[float], ratios: set[float] = frozenset()) -> bool:
    """Se o número citado bate com algum dado: o próprio valor (arredondado)
    ou, só para as razões, o percentual equivalente."""
    for value in readings(token):
        for ref in known | ALLOWED:
            if abs(value - ref) <= max(0.01 * abs(ref), 0.051):
                return True
            if abs(value - round(ref)) < 1e-9:
                return True
        for ref in ratios:
            if (
                abs(value - 100 * ref) <= 0.51
                or abs(value - 100 * abs(1 - ref)) <= 0.51
            ):
                return True
    return False


def affirms_low_confidence(explanation: str) -> bool:
    """Se alguma frase afirma (sem negar) baixa confiança ou incerteza."""
    for sentence in re.split(r"[.!?;]", _plain(explanation)):
        if _LOW_CONFIDENCE.search(sentence) and not _NEGATION.search(sentence):
            return True
    return False


def check(summary: dict, explanation: str) -> dict:
    """Resultado das três checagens para uma explicação."""
    known, ratios = data_numbers(summary), ratio_numbers(summary)
    tokens = numbers_in(explanation)
    unsupported = [t for t in tokens if not supported(t, known, ratios)]
    text = _plain(explanation)
    factors = [_plain(f["fator"]) for f in summary["fatores"]]
    cites_factor = any(f in text for f in factors) or any(
        word in text
        for f in factors
        for word in f.split()
        if len(word) > 5  # palavras de conteúdo do nome do fator
    )
    return {
        "numeros_citados": len(tokens),
        "numeros_sem_base": len(unsupported),
        "exemplos_sem_base": "; ".join(unsupported[:5]),
        "cita_fator": cites_factor,
        "baixa_confianca": summary["baixa_confianca"],
        "afirma_baixa_confianca": affirms_low_confidence(explanation),
    }


def run(n: int) -> pd.DataFrame:
    from creative_predictor.agent import graph
    from creative_predictor.demo import to_inputs
    from creative_predictor.model import dataset as ds
    from creative_predictor.predict import service

    llm = graph._llm()
    if llm is None:
        raise RuntimeError("Ollama indisponível: a checagem precisa do LLM do agente")
    shorts = pd.read_parquet(config.PROCESSED_DIR / "shorts.parquet")
    channels = pd.read_parquet(config.PROCESSED_DIR / "channels.parquet")
    rows = ds.modeling_rows(ds.build_table(shorts, channels))
    test = rows[(rows["split"] == "test") & (rows["region"] == "br")]
    picked = shorts[shorts["video_id"].isin(test["video_id"])].sample(
        n=min(n, len(test)), random_state=SEED
    )
    results = []
    for inp, row in zip(
        to_inputs(picked, channels, use_video=False), picked.itertuples(), strict=True
    ):
        prediction = service.predict_creative(inp)
        summary = graph._summary(prediction)
        prompt = graph.EXPLAIN_PROMPT.format(
            data=json.dumps(summary, ensure_ascii=False)
        )
        explanation = llm.invoke(prompt).content.strip()
        results.append(
            {"video_id": row.video_id, "explicacao": explanation}
            | check(summary, explanation)
        )
        print(f"{len(results)}/{len(picked)}", flush=True)
    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> dict:
    low = results[results["baixa_confianca"]]
    high = results[~results["baixa_confianca"]]
    return {
        "explicacoes": len(results),
        "sem_numero_inventado": float((results["numeros_sem_base"] == 0).mean()),
        "numeros_sem_base_total": int(results["numeros_sem_base"].sum()),
        "numeros_citados_total": int(results["numeros_citados"].sum()),
        "citam_fator": float(results["cita_fator"].mean()),
        "baixa_confianca_afirmada": float(low["afirma_baixa_confianca"].mean())
        if len(low)
        else np.nan,
        "alta_confianca_contradita": float(high["afirma_baixa_confianca"].mean())
        if len(high)
        else np.nan,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=N_EXPLANATIONS)
    args = parser.parse_args(argv)
    results = run(args.n)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    for key, value in summarize(results).items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"\nResultados em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
