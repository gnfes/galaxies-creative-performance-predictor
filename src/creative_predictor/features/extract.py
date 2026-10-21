"""Extração multimodal da amostra inteira, em etapas retomáveis.

- `local`: para cada Short já baixado, frames, áudio, transcrição e medidas
  visuais e de áudio (CPU).
- `embed`: embeddings dos frames, do áudio e da transcrição (EmbeddingGemma 2,
  no GPU do Mac quando houver).
- `vlm`: rubrica do extrator escolhido (`vlm.EXTRACTOR`) a partir dos frames,
  da transcrição e do título, com pedidos em paralelo e limite de tokens por
  minuto da API.
- `assemble`: junta tudo em `data/processed/multimodal.parquet`, um Short por
  linha.

Cada etapa só faz o que falta (tudo fica em cache por vídeo), registra falhas
em `data/interim/extract_errors.jsonl` e segue. Com `--follow`, as etapas
`local` e `vlm` acompanham o download, que roda em paralelo, até não sobrar
nada. A ordem é embaralhada de forma fixa: se o tempo acabar antes do fim, o
que estiver pronto é uma amostra representativa de todos os canais.

Uso:
    uv run python -m creative_predictor.features.extract local --follow [--shard 0/2]
    uv run python -m creative_predictor.features.extract vlm --follow
    uv run python -m creative_predictor.features.extract embed --follow
    uv run python -m creative_predictor.features.extract assemble
"""

import argparse
import json
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from creative_predictor import config
from creative_predictor.collect.media import media_dir, read_status, shuffled_sample
from creative_predictor.features import frames as fr
from creative_predictor.features import media_embed
from creative_predictor.features.transcribe import TRANSCRIPTS_DIR, WHISPER_MODEL
from creative_predictor.features.transcribe import transcribe as run_transcribe
from creative_predictor.features.visual import VISUAL_DIR, VisualExtractor
from creative_predictor.features.vlm import (
    CANDIDATES,
    EXTRACTOR,
    Creative,
    cache_file,
    read_cached,
)

ERRORS_PATH = config.INTERIM_DIR / "extract_errors.jsonl"
MULTIMODAL_PATH = config.PROCESSED_DIR / "multimodal.parquet"
N_FRAMES = 8
# Cota gratuita do Gemma: 16 mil tokens por minuto. Fica uma folga, e cada
# pedido (8 frames + transcrição + instrução + resposta) conta ~3,2 mil.
TOKENS_PER_MINUTE = 15_000
TOKENS_PER_REQUEST = 3_200
VLM_WORKERS = 24  # a API demora minutos por pedido; o limite real é o de tokens
# Tentativas por vídeo em cada execução: um vídeo que sempre falha (arquivo
# corrompido, resposta inválida) não gasta CPU nem cota a cada volta.
MAX_ATTEMPTS = 2


def sample_ids(shorts: pd.DataFrame) -> pd.DataFrame:
    """Shorts da amostra multimodal, na mesma ordem embaralhada do download."""
    return shuffled_sample(shorts)


def log_error(stage: str, video_id: str, error: BaseException | str) -> None:
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "stage": stage,
                    "video_id": video_id,
                    "error": str(error)[:500],
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def downloaded(video_id: str) -> Path | None:
    status = read_status(video_id)
    if status and status["status"] == "ok":
        video = media_dir(video_id) / "video.mp4"
        return video if video.is_file() else None
    return None


def local_done(video_id: str) -> bool:
    return (VISUAL_DIR / f"{video_id}.json").is_file() and (
        TRANSCRIPTS_DIR / WHISPER_MODEL / f"{video_id}.json"
    ).is_file()


def run_local(
    video_id: str, visual: VisualExtractor, video: Path | None = None
) -> tuple[list[Path], Path | None, str]:
    """Frames, áudio, transcrição e medidas visuais de um Short.

    Devolve os frames, o áudio e o texto transcrito. Sem `video`, usa o
    arquivo baixado da base.
    """
    video = video or downloaded(video_id)
    duration = fr.probe_duration(video)
    frames = fr.frames_for(video_id, video)
    audio = fr.audio_for(video_id, video)
    transcript = run_transcribe(video_id, audio, duration)
    visual(video_id, video, frames, audio, duration)
    return frames, audio, transcript["text"]


def creative_for(row) -> Creative | None:
    """Material do extrator para um Short, se a etapa local já terminou."""
    frames_dir = fr.FRAMES_DIR / row.video_id
    frames = [frames_dir / f"frame_{i}.jpg" for i in range(N_FRAMES)]
    transcript = TRANSCRIPTS_DIR / WHISPER_MODEL / f"{row.video_id}.json"
    if not (transcript.is_file() and all(p.is_file() for p in frames)):
        return None
    try:
        text = json.loads(transcript.read_text())["text"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None  # ainda sendo escrita por outro processo
    return Creative(
        video_id=row.video_id,
        sector=row.sector,
        video=media_dir(row.video_id) / "video.mp4",
        frames=frames,
        transcript=text,
        title=row.title,
    )


class TokenRateLimiter:
    """Segura cada pedido até caber no limite de tokens da última janela."""

    def __init__(
        self,
        tokens_per_window: int,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.limit, self.window = tokens_per_window, window
        self._clock, self._sleep = clock, sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= self.window:
                    self._events.popleft()
                used = sum(t for _, t in self._events)
                if not self._events or used + tokens <= self.limit:
                    self._events.append((now, tokens))
                    return
                wait = self.window - (now - self._events[0][0])
            self._sleep(max(wait, 0.05))


def _follow(
    pending: Callable[[], list], process: Callable[[list], int], idle_minutes: float
) -> None:
    """Processa o que estiver pendente até nada novo aparecer por um tempo."""
    idle_since = None
    while True:
        todo = pending()
        done = process(todo) if todo else 0
        if done:
            idle_since = None
            continue
        idle_since = idle_since or time.monotonic()
        if time.monotonic() - idle_since > idle_minutes * 60:
            return
        time.sleep(60)


def shard(sample: pd.DataFrame, spec: str | None) -> pd.DataFrame:
    """Parte `i/n` da amostra (`0/2`, `1/2`...), para dividir a etapa local
    entre processos; sem `spec`, a amostra inteira."""
    if not spec:
        return sample
    index, total = (int(x) for x in spec.split("/"))
    if not 0 <= index < total:
        raise ValueError(f"shard inválido: {spec}")
    return sample.iloc[index::total]


def local_stage(sample: pd.DataFrame, follow: bool, idle_minutes: float) -> None:
    visual = VisualExtractor()
    attempts: Counter[str] = Counter()

    def pending():
        return [
            v
            for v in sample["video_id"]
            if attempts[v] < MAX_ATTEMPTS and not local_done(v) and downloaded(v)
        ]

    def process(ids):
        done, started = 0, time.time()
        for n, video_id in enumerate(ids, 1):
            attempts[video_id] += 1
            try:
                run_local(video_id, visual)
                done += 1
            except Exception as error:  # noqa: BLE001 (um vídeo ruim não para a base)
                log_error("local", video_id, error)
            if n % 50 == 0:
                rate = (time.time() - started) / n
                print(f"local: {n}/{len(ids)} ({rate:.1f} s/vídeo)", flush=True)
        return done

    if follow:
        _follow(pending, process, idle_minutes)
    else:
        process(pending())


def vlm_stage(
    sample: pd.DataFrame, follow: bool, idle_minutes: float, workers: int
) -> None:
    extractor = CANDIDATES[EXTRACTOR]()
    limiter = TokenRateLimiter(TOKENS_PER_MINUTE)
    # A vez no limite é pedida a cada envio, inclusive nas novas tentativas.
    extractor.before_request = lambda: limiter.acquire(TOKENS_PER_REQUEST)
    attempts: Counter[str] = Counter()

    def pending():
        out = []
        for row in sample.itertuples():
            if attempts[row.video_id] >= MAX_ATTEMPTS:
                continue
            if cache_file(EXTRACTOR, row.video_id, row.sector).is_file():
                continue
            creative = creative_for(row)
            if creative is not None:
                out.append(creative)
        return out

    def process(creatives):
        done, failed, started = 0, 0, time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extractor.extract, c): c for c in creatives}
            for future in as_completed(futures):
                creative = futures[future]
                attempts[creative.video_id] += 1
                try:
                    record = future.result()
                except Exception as error:  # noqa: BLE001 (um vídeo não para a base)
                    record = {"rubric": None, "error": repr(error)}
                if record.get("rubric") is None:
                    failed += 1
                    log_error("vlm", creative.video_id, record.get("error", ""))
                else:
                    done += 1
                total = done + failed
                if total % 25 == 0:
                    per_min = 60 * total / (time.time() - started)
                    print(
                        f"vlm: {total}/{len(creatives)} ({failed} falhas, "
                        f"{per_min:.1f} vídeos/min)",
                        flush=True,
                    )
        return done

    if follow:
        _follow(pending, process, idle_minutes)
    else:
        process(pending())


def embed_stage(sample: pd.DataFrame, follow: bool, idle_minutes: float) -> None:
    """Embeddings de frames, áudio e transcrição dos Shorts já processados."""
    embedder = media_embed.MediaEmbedder()
    attempts: Counter[str] = Counter()

    def pending():
        out = []
        for row in sample.itertuples():
            if attempts[row.video_id] >= MAX_ATTEMPTS or media_embed.done(row.video_id):
                continue
            creative = creative_for(row)
            if creative is not None:
                out.append(creative)
        return out

    def process(creatives):
        done, started = 0, time.time()
        for n, creative in enumerate(creatives, 1):
            attempts[creative.video_id] += 1
            audio = fr.AUDIO_DIR / f"{creative.video_id}.wav"
            try:
                embedder(
                    creative.video_id,
                    creative.frames,
                    audio if audio.is_file() else None,
                    creative.transcript,
                )
                done += 1
            except Exception as error:  # noqa: BLE001 (um vídeo não para a base)
                log_error("embed", creative.video_id, error)
            if n % 50 == 0:
                rate = (time.time() - started) / n
                print(f"embed: {n}/{len(creatives)} ({rate:.1f} s/vídeo)", flush=True)
        return done

    if follow:
        _follow(pending, process, idle_minutes)
    else:
        process(pending())


def _flatten(rubric: dict) -> dict:
    """Rubrica em colunas numéricas: booleanos 0/1, notas 0–10 e categorias.

    As listas de elementos viram texto, para embeddings depois.
    """
    out = {}
    for key, value in rubric.items():
        if isinstance(value, int):  # inclui booleanos, que viram 0/1
            out[f"vlm_{key}"] = float(value)
        elif isinstance(value, Enum):
            out[f"vlm_{key}"] = value.value
        elif isinstance(value, list):
            out[f"vlm_{key}"] = "; ".join(value)
        else:
            out[f"vlm_{key}"] = value
    return out


def media_record(video_id: str, sector: str) -> dict:
    """Tudo o que já foi extraído de um Short, numa linha plana.

    Só lê o cache: nenhum cálculo e nenhum pedido à API. É a mesma linha que
    a base de treino usa e que a previsão monta para um criativo novo.
    """
    out = {"video_id": video_id}
    transcript = TRANSCRIPTS_DIR / WHISPER_MODEL / f"{video_id}.json"
    if transcript.is_file():
        t = json.loads(transcript.read_text())
        out |= {
            "transcript": t["text"],
            "speech_language": t["language"],
            "speech_share": t["speech_share"],
            "speech_words": t["n_words"],
            "speech_words_per_second": t["words_per_second"],
        }
    visual = VISUAL_DIR / f"{video_id}.json"
    if visual.is_file():
        out |= json.loads(visual.read_text())
    rubric = read_cached(EXTRACTOR, video_id, sector)
    if rubric is not None:
        out |= _flatten(dict(rubric))
    return out


def assemble(sample: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por Short da amostra com o que já foi extraído."""
    return pd.DataFrame(
        [media_record(row.video_id, row.sector) for row in sample.itertuples()]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["local", "vlm", "embed", "assemble"])
    parser.add_argument(
        "--follow", action="store_true", help="acompanha o download até acabar"
    )
    parser.add_argument(
        "--idle-minutes",
        type=float,
        default=60,
        help="com --follow, para depois desse tempo sem nada novo",
    )
    parser.add_argument("--workers", type=int, default=VLM_WORKERS)
    parser.add_argument(
        "--shard", help="etapa local: processa só a parte i/n (ex.: 0/2)"
    )
    args = parser.parse_args(argv)

    shorts = pd.read_parquet(config.PROCESSED_DIR / "shorts.parquet")
    sample = sample_ids(shorts)
    if args.stage == "local":
        local_stage(shard(sample, args.shard), args.follow, args.idle_minutes)
    elif args.stage == "vlm":
        vlm_stage(sample, args.follow, args.idle_minutes, args.workers)
    elif args.stage == "embed":
        embed_stage(sample, args.follow, args.idle_minutes)
    else:
        table = assemble(sample)
        MULTIMODAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(MULTIMODAL_PATH, index=False)
        filled = table.notna().mean().round(3)
        print(f"{len(table)} Shorts; cobertura por bloco:")
        for col in ("transcript", "brightness_mean", "vlm_forca_hook_0a10"):
            if col in filled:
                print(f"  {col}: {filled[col]:.1%}")
        print(f"Salvo em {MULTIMODAL_PATH}")


if __name__ == "__main__":
    main()
