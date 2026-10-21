"""LLM sem modelo × modelo treinado, nos mesmos Shorts do teste.

O LLM local (`gemma4:12b`, o mesmo do agente) recebe o que uma pessoa de
marketing teria em mãos antes de publicar e chuta as curtidas:
- os 8 frames do Short, o título, a duração e a transcrição;
- a mediana de curtidas do canal e os últimos Shorts estabilizados do canal,
  com título e curtidas.

O chute vira o mesmo alvo do modelo (`log((curtidas+1)/(mediana+1))`) e é
avaliado com as mesmas métricas, nos mesmos Shorts. A amostra tem até 5
Shorts por canal do teste (só canais com 3 ou mais), para a ordenação dentro
do canal ter pares. As respostas ficam em cache por vídeo.

Uso:
    uv run python -m creative_predictor.model.llm_baseline ask [--limit 200]
    uv run python -m creative_predictor.model.llm_baseline evaluate
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from creative_predictor import config
from creative_predictor.features import frames as fr
from creative_predictor.features.history import HISTORY_GAP_DAYS
from creative_predictor.features.transcribe import TRANSCRIPTS_DIR, WHISPER_MODEL

LLM = "gemma4:12b"
CACHE_DIR = config.INTERIM_DIR / "llm_baseline"
RESULTS_PATH = config.ROOT_DIR / "reports" / "llm_baseline_results.csv"
SAMPLE_LIMIT = 200
PER_CHANNEL = 5
MIN_PER_CHANNEL = 3
CONTEXT_POSTS = 10
N_FRAMES = 8
SEED = 42

PROMPT = """Você é especialista em YouTube Shorts de {sector}. Antes da \
publicação, estime quantas curtidas este Short terá cerca de 90 dias depois \
de publicado.

Canal: mediana de {median:.0f} curtidas nos Shorts anteriores.
Últimos Shorts do canal (título → curtidas):
{history}

Short a avaliar (frames anexos, em ordem; os 3 primeiros são dos 3 \
primeiros segundos):
Título: {title}
Duração: {duration} s
Transcrição: {transcript}

Responda só o JSON pedido."""


class Guess(BaseModel):
    curtidas_previstas: int = Field(ge=0, description="Curtidas em ~90 dias")
    justificativa: str = Field(description="Uma frase")


def cache_path(video_id: str, model: str = LLM) -> Path:
    return CACHE_DIR / model.replace(":", "_") / f"{video_id}.json"


def sample_path(model: str = LLM) -> Path:
    """Ids sorteados na primeira execução: a avaliação usa os mesmos Shorts."""
    return CACHE_DIR / model.replace(":", "_") / "sample.json"


def fixed_sample(rows: pd.DataFrame, limit: int = SAMPLE_LIMIT) -> pd.DataFrame:
    """A amostra salva, ou uma nova (salva para as próximas execuções)."""
    path = sample_path()
    if path.is_file():
        ids = json.loads(path.read_text())
        return rows[rows["video_id"].isin(ids)]
    picked = sample(rows, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(picked["video_id"].tolist()))
    return picked


def frames_of(video_id: str) -> list[Path] | None:
    paths = [fr.FRAMES_DIR / video_id / f"frame_{i}.jpg" for i in range(N_FRAMES)]
    return paths if all(p.is_file() for p in paths) else None


def transcript_of(video_id: str) -> str:
    path = TRANSCRIPTS_DIR / WHISPER_MODEL / f"{video_id}.json"
    if not path.is_file():
        return ""
    return json.loads(path.read_text())["text"]


def sample(rows: pd.DataFrame, limit: int = SAMPLE_LIMIT) -> pd.DataFrame:
    """Até `PER_CHANNEL` Shorts do teste por canal, só os com frames."""
    test = rows[(rows["split"] == "test") & (rows["region"] == "br")]
    test = test[test["video_id"].map(lambda v: frames_of(v) is not None)]
    rng = np.random.default_rng(SEED)
    picked = []
    for _, group in test.groupby("channel_id", sort=True):
        if len(group) < MIN_PER_CHANNEL:
            continue
        take = rng.choice(len(group), size=min(PER_CHANNEL, len(group)), replace=False)
        picked.append(group.iloc[np.sort(take)])
    out = pd.concat(picked)
    # Corta por canal inteiro, para não deixar canal com um Short só.
    channels = out["channel_id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(channels)
    kept, total = [], 0
    for channel in channels:
        n = (out["channel_id"] == channel).sum()
        if total + n > limit:
            break
        kept.append(channel)
        total += n
    return out[out["channel_id"].isin(kept)]


def history_text(table: pd.DataFrame, row) -> str:
    """Últimos Shorts estabilizados do canal antes do Short (título → curtidas)."""
    cutoff = row.published_at - pd.Timedelta(days=HISTORY_GAP_DAYS)
    past = table[
        (table["channel_id"] == row.channel_id)
        & (table["published_at"] <= cutoff)
        & table["likes"].notna()
    ].sort_values("published_at")
    lines = [
        f"- {title[:90]} → {int(likes)}"
        for title, likes in past[["title", "likes"]].tail(CONTEXT_POSTS).to_numpy()
    ]
    return "\n".join(lines) or "(sem Shorts estabilizados)"


def ask(row, table: pd.DataFrame, model: str = LLM) -> dict:
    """Chute do LLM para um Short (do cache, se já existe)."""
    path = cache_path(row.video_id, model)
    if path.is_file():
        return json.loads(path.read_text())
    import ollama

    prompt = PROMPT.format(
        sector=row.sector,
        median=row.history_median_likes,
        history=history_text(table, row),
        title=row.title,
        duration="?" if pd.isna(row.duration_s) else int(row.duration_s),
        transcript=(transcript_of(row.video_id) or "(sem fala)")[:1500],
    )
    started = time.time()
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(p) for p in frames_of(row.video_id)],
            }
        ],
        format=Guess.model_json_schema(),
        options={"temperature": 0},
        think=False,
    )
    guess = Guess.model_validate_json(response.message.content)
    record = guess.model_dump() | {"seconds": round(time.time() - started, 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(path)
    return record


def llm_target(guesses: pd.Series, median: pd.Series) -> pd.Series:
    """Chute em curtidas no mesmo alvo do modelo."""
    return np.log((guesses.astype(float) + 1) / (median.astype(float) + 1))


def run_ask(limit: int) -> None:
    from creative_predictor.model import dataset as ds

    table = ds.build_table()
    picked = fixed_sample(ds.modeling_rows(table), limit)
    print(f"{len(picked)} Shorts de {picked['channel_id'].nunique()} canais")
    failures = 0
    for n, row in enumerate(picked.itertuples(), 1):
        try:
            record = ask(row, table)
        except Exception as error:  # noqa: BLE001 (um Short não para a amostra)
            failures += 1
            print(f"{row.video_id}: falha ({str(error)[:120]})", flush=True)
            continue
        if n % 10 == 0:
            print(f"{n}/{len(picked)} ({record['seconds']} s)", flush=True)
    print(f"Pronto; {failures} falhas")


def run_evaluate() -> pd.DataFrame:
    """Métricas do LLM, do modelo completo e da baseline nos mesmos Shorts."""
    from creative_predictor.model import multimodal as mm
    from creative_predictor.model import train as tr
    from creative_predictor.model.dataset import modeling_rows
    from creative_predictor.model.metrics import band_edges, evaluate

    table, blocks, _ = mm.build()
    rows = modeling_rows(table)
    split = tr.make_split(rows, ("br", "global"))
    edges = band_edges(split.train["target"])
    picked = fixed_sample(rows)
    guesses = {
        v: json.loads(cache_path(v).read_text())["curtidas_previstas"]
        for v in picked["video_id"]
        if cache_path(v).is_file()
    }
    picked = picked[picked["video_id"].isin(guesses)]
    features = [f for block in blocks.values() for f in block]
    model = tr.fit_lgbm(split.train, split.valid, features, mm.OBJECTIVE)
    llm = llm_target(picked["video_id"].map(guesses), picked["history_median_likes"])
    results = [
        {"experimento": "baseline_mediana_canal"}
        | evaluate(picked, np.zeros(len(picked)), edges),
        {"experimento": f"llm_sem_modelo_{LLM}"}
        | evaluate(picked, llm.to_numpy(), edges),
        {"experimento": "lgbm_todos_os_blocos"}
        | evaluate(picked, model.predict(picked[features]), edges),
    ]
    return pd.DataFrame(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["ask", "evaluate"])
    parser.add_argument("--limit", type=int, default=SAMPLE_LIMIT)
    args = parser.parse_args(argv)
    if args.stage == "ask":
        run_ask(args.limit)
        return
    results = run_evaluate()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    columns = ["experimento", "n", "mae_log", "spearman_within_channel"]
    columns += ["pairwise_accuracy", "band_f1"]
    print(results[columns].round(3).to_string(index=False))
    print(f"\nResultados em {RESULTS_PATH}")


if __name__ == "__main__":
    main()
