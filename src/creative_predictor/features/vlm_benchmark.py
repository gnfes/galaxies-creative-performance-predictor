"""Benchmark dos candidatos a extrator da rubrica.

30 Shorts do período de treino (10 por setor; 7 do Brasil e 3 globais), com
o mesmo material para todos: vídeo para o Gemini, frames + transcrição para
os demais. Para cada candidato mede JSON válido, concordância com a
referência (`gemini-3.5-flash-lite` com o vídeo nativo), segundos e tokens
por vídeo. A escolha do extrator sai desses números.

Os números em `reports/vlm_benchmark*.csv` foram medidos com a primeira versão
da instrução, antes de ela pedir booleanos como true/false e avisar sobre letra
de música na transcrição. Como o cache é separado por versão da instrução,
rodar de novo pede tudo outra vez aos modelos (horas de API).

Uso:
    uv run python -m creative_predictor.features.vlm_benchmark --prepare
    uv run python -m creative_predictor.features.vlm_benchmark --models gemini-3.5-flash-lite,gemma-4-31b-it
"""

import argparse
import typing
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from creative_predictor import config
from creative_predictor.collect.media import download_many, media_dir
from creative_predictor.features import frames as fr
from creative_predictor.features.rubric import RUBRICS
from creative_predictor.features.transcribe import transcribe
from creative_predictor.features.vlm import CANDIDATES, REFERENCE, Creative

REPORTS_DIR = config.ROOT_DIR / "reports"
PER_SECTOR = {"br": 7, "global": 3}
SEED = 42
TRAIN_END = pd.Timestamp("2026-03-01", tz="UTC")


def select_videos(shorts: pd.DataFrame) -> pd.DataFrame:
    """10 Shorts por setor da amostra multimodal, no período de treino."""
    pool = shorts[
        shorts["media_sample"]
        & ~shorts["holdout"]
        & shorts["in_window"]
        & (shorts["published_at"] < TRAIN_END)
    ]
    picked = [
        group.sample(PER_SECTOR[region], random_state=SEED)
        for (_, region), group in pool.groupby(["sector", "region"])
    ]
    return pd.concat(picked).sort_values(["sector", "region", "video_id"])


def prepare(videos: pd.DataFrame) -> list[Creative]:
    """Baixa, extrai frames e áudio e transcreve; devolve os prontos."""
    statuses = download_many(videos["video_id"]).set_index("video_id")
    creatives = []
    for row in videos.itertuples():
        if statuses.loc[row.video_id, "status"] != "ok":
            continue
        video = media_dir(row.video_id) / "video.mp4"
        duration = fr.probe_duration(video)
        transcript = transcribe(
            row.video_id, fr.audio_for(row.video_id, video), duration
        )
        creatives.append(
            Creative(
                video_id=row.video_id,
                sector=row.sector,
                video=video,
                frames=fr.frames_for(row.video_id, video),
                transcript=transcript["text"],
                title=row.title,
            )
        )
    return creatives


def _field_kinds() -> dict[str, str]:
    """Tipo de comparação de cada campo: nota, booleano, categoria ou texto."""
    kinds = {}
    for schema in RUBRICS.values():
        for name, info in schema.model_fields.items():
            ann = info.annotation
            base = typing.get_args(ann)[0] if typing.get_origin(ann) else ann
            if name.endswith("_0a10"):
                kinds[name] = "nota"
            elif ann is bool:
                kinds[name] = "booleano"
            elif isinstance(base, type) and issubclass(base, Enum):
                kinds[name] = "categoria"
            else:
                kinds[name] = "texto"
    return kinds


def _value(rubric, field):
    value = getattr(rubric, field, None)
    return value.value if isinstance(value, Enum) else value


def compare(records: dict[str, dict[str, dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resumo por modelo e concordância por campo com a referência.

    `records[modelo][video_id]` é o resultado de `Extractor.extract`.
    """
    kinds = _field_kinds()
    reference = records.get(REFERENCE, {})
    summary, per_field = [], []
    for model, by_video in records.items():
        valid = [r for r in by_video.values() if r.get("rubric") is not None]
        row = {
            "modelo": model,
            "videos": len(by_video),
            "json_valido": len(valid) / len(by_video) if by_video else np.nan,
            "segundos_por_video": np.mean([r["seconds"] for r in by_video.values()]),
            # Média que ignora vídeos sem contagem de tokens (NaN se nenhum tem).
            "tokens_entrada": pd.Series(
                [r.get("input_tokens") for r in by_video.values()], dtype=float
            ).mean(),
            "tokens_saida": pd.Series(
                [r.get("output_tokens") for r in by_video.values()], dtype=float
            ).mean(),
        }
        if model != REFERENCE:
            agreements = {"booleano": [], "categoria": [], "nota_mae": []}
            for field, kind in kinds.items():
                if kind == "texto":
                    continue
                pairs = [
                    (_value(reference[v]["rubric"], field), _value(r["rubric"], field))
                    for v, r in by_video.items()
                    if r.get("rubric") is not None
                    and reference.get(v, {}).get("rubric") is not None
                    and hasattr(r["rubric"], field)
                ]
                if not pairs:
                    continue
                ref, cand = map(np.array, zip(*pairs, strict=True))
                entry = {"modelo": model, "campo": field, "tipo": kind, "n": len(pairs)}
                if kind == "nota":
                    entry["mae"] = float(np.mean(np.abs(ref - cand)))
                    entry["spearman"] = (
                        float(spearmanr(ref, cand).statistic)
                        if len(set(ref)) > 1 and len(set(cand)) > 1
                        else np.nan
                    )
                    agreements["nota_mae"].append(entry["mae"])
                else:
                    entry["concordancia"] = float(np.mean(ref == cand))
                    agreements[kind].append(entry["concordancia"])
                per_field.append(entry)
            row["concordancia_booleanos"] = np.mean(agreements["booleano"] or [np.nan])
            row["concordancia_categorias"] = np.mean(
                agreements["categoria"] or [np.nan]
            )
            row["mae_notas"] = np.mean(agreements["nota_mae"] or [np.nan])
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(per_field)


def run(models: list[str], creatives: list[Creative]) -> dict[str, dict[str, dict]]:
    records = {}
    for model in models:
        extractor = CANDIDATES[model]()
        records[model] = {}
        for creative in creatives:
            record = extractor.extract(creative)
            records[model][creative.video_id] = record
            flag = "ok" if record["rubric"] is not None else record.get("error")
            print(f"{model} {creative.video_id}: {flag}")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prepare", action="store_true", help="só baixa e prepara os vídeos"
    )
    parser.add_argument(
        "--models",
        default=",".join(CANDIDATES),
        help="candidatos separados por vírgula (a referência entra sempre)",
    )
    args = parser.parse_args(argv)
    shorts = pd.read_parquet(config.PROCESSED_DIR / "shorts.parquet")
    creatives = prepare(select_videos(shorts))
    print(f"{len(creatives)} vídeos prontos")
    if args.prepare:
        return
    models = [REFERENCE] + [m for m in args.models.split(",") if m != REFERENCE]
    summary, per_field = compare(run(models, creatives))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "vlm_benchmark.csv", index=False)
    per_field.to_csv(REPORTS_DIR / "vlm_benchmark_fields.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
