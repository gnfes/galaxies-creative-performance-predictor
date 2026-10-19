"""Demonstração "de-para" com Shorts reais de canais fora do treino.

Para cada canal de demonstração, pega os últimos Shorts publicados e os trata
como criativos inéditos: o agente extrai os atributos do vídeo (quando ele foi
baixado), prevê e ordena cada grupo como se ainda não tivessem sido
publicados, e o resultado é comparado com as curtidas reais. Os vídeos são
públicos, então a rubrica do extrator é pedida. O relatório sai em
`reports/demo.md`.

Uso:
    uv run python -m creative_predictor.demo [--per-channel 5] [--no-llm] [--no-video]
"""

import argparse
from itertools import combinations

import pandas as pd

from creative_predictor import config
from creative_predictor.collect.media import media_dir
from creative_predictor.predict.schemas import ChannelInfo, CreativeInput, Ranking

REPORT_PATH = config.ROOT_DIR / "reports" / "demo.md"


def demo_groups(per_channel: int) -> list[tuple[str, pd.DataFrame]]:
    """Últimos Shorts com curtidas públicas de cada canal de demonstração."""
    shorts = pd.read_parquet(config.PROCESSED_DIR / "shorts.parquet")
    channels = pd.read_parquet(config.PROCESSED_DIR / "channels.parquet")
    names = channels.set_index("channel_id")["name"]
    demo = shorts[shorts["holdout"] & shorts["in_window"] & shorts["likes"].notna()]
    groups = []
    for channel_id, group in demo.groupby("channel_id"):
        recent = group.sort_values("published_at").tail(per_channel)
        if len(recent) >= 2:
            groups.append((names[channel_id], recent))
    return groups


def to_inputs(
    group: pd.DataFrame, channels: pd.DataFrame, use_video: bool = True
) -> list[CreativeInput]:
    info = channels.set_index("channel_id")
    inputs = []
    for i, row in enumerate(group.itertuples(), 1):
        country = info.loc[row.channel_id, "country"]
        video = media_dir(row.video_id) / "video.mp4"
        has_video = use_video and video.is_file()
        inputs.append(
            CreativeInput(
                creative_id=f"{i}. {row.title[:60]}",
                caption=row.title,
                description=row.description,
                tags=list(row.tags) if row.tags is not None else [],
                category_id=row.category_id,
                planned_datetime=row.published_at,
                sector=row.sector,
                duration_s=None if pd.isna(row.duration_s) else float(row.duration_s),
                creative=video if has_video else None,
                use_vlm=has_video,
                channel=ChannelInfo(
                    channel_id=row.channel_id,
                    region=row.region,
                    kind=row.kind,
                    country=None if pd.isna(country) else country,
                ),
            )
        )
    return inputs


def pairwise_hits(ranking: Ranking, real: dict[str, int]) -> tuple[int, int]:
    """Pares em que a ordem prevista bate com a das curtidas reais."""
    order = [p.creative_id for p in ranking.predictions]
    hits = total = 0
    for a, b in combinations(order, 2):  # a está acima de b na previsão
        if real[a] == real[b]:
            continue
        total += 1
        hits += real[a] > real[b]
    return hits, total


def render(results: list[tuple[str, Ranking, dict[str, int]]]) -> str:
    lines = [
        "# Demonstração com canais fora do treino",
        "",
        (
            "Shorts reais de canais que o modelo nunca viu, previstos como se "
            "fossem inéditos e comparados com as curtidas reais (coletadas em "
            "13/10/2026)."
        ),
        "",
    ]
    all_hits = all_total = inside = count = 0
    for channel, ranking, real in results:
        hits, total = pairwise_hits(ranking, real)
        all_hits, all_total = all_hits + hits, all_total + total
        lines += [
            f"## {channel}",
            "",
            "| Criativo | Previsto | Intervalo 80% | Real | Baixa confiança | Blocos |",
            "|---|---:|---:|---:|---|---|",
        ]
        for p in ranking.predictions:
            likes = real[p.creative_id]
            ok = p.interval_low <= likes <= p.interval_high
            inside, count = inside + ok, count + 1
            lines.append(
                f"| {p.creative_id} | {p.predicted_metric:,.0f} | "
                f"{p.interval_low:,.0f}–{p.interval_high:,.0f} | {likes:,} | "
                f"{'sim' if p.confidence_flag else 'não'} | "
                f"{', '.join(p.blocks_used)} |"
            )
        lines += ["", f"Pares na ordem certa: {hits} de {total}.", ""]
        if ranking.explanation:
            lines += [f"> {ranking.explanation}", ""]
    # O resumo vai logo depois da introdução, antes do primeiro canal.
    lines.insert(
        4,
        f"**Resumo:** {all_hits} de {all_total} pares na ordem certa "
        f"({all_hits / max(all_total, 1):.0%}); {inside} de {count} curtidas reais "
        f"dentro do intervalo de 80%.\n",
    )
    return "\n".join(lines)


def run(per_channel: int, use_llm: bool, use_video: bool = True) -> str:
    from creative_predictor.agent.graph import analyze
    from creative_predictor.predict.service import rank_creatives

    channels = pd.read_parquet(config.PROCESSED_DIR / "channels.parquet")
    results = []
    for channel, group in demo_groups(per_channel):
        inputs = to_inputs(group, channels, use_video)
        ranking = analyze(inputs) if use_llm else rank_creatives(inputs)
        real = {
            i.creative_id: int(v) for i, v in zip(inputs, group["likes"], strict=True)
        }
        results.append((channel, ranking, real))
        print(f"{channel}: {pairwise_hits(ranking, real)}", flush=True)
    return render(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-channel", type=int, default=5)
    parser.add_argument(
        "--no-llm", action="store_true", help="sem a explicação do LLM (mais rápido)"
    )
    parser.add_argument(
        "--no-video", action="store_true", help="só metadados e texto, sem o vídeo"
    )
    args = parser.parse_args(argv)
    report = run(args.per_channel, use_llm=not args.no_llm, use_video=not args.no_video)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"\nRelatório em {REPORT_PATH}")


if __name__ == "__main__":
    main()
