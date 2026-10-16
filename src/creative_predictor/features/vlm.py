"""Extratores que respondem a rubrica sobre um Short.

Dois formatos de entrada:
- vídeo nativo (Gemini): o arquivo inteiro vai para a API;
- frames + transcrição (Gemma pela API ou modelos locais no Ollama): os 8
  frames e a transcrição do Whisper, que custam bem menos tokens que o vídeo.

Cada resposta bruta fica em `data/raw/vlm/<modelo>/<instrução>/<video_id>.json`,
com tempo e tokens, e nunca é pedida de novo (falhas não vão para o cache). A
pasta `<instrução>` muda quando o texto da instrução muda. A resposta vira a
rubrica do setor validada com Pydantic; resposta inválida fica registrada como
erro.
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from creative_predictor import config
from creative_predictor.features.rubric import Rubric, instructions, rubric_for

VLM_DIR = config.RAW_DIR / "vlm"


@dataclass
class Creative:
    """Um criativo pronto para o extrator."""

    video_id: str
    sector: str
    video: Path
    frames: list[Path]
    transcript: str
    title: str = ""


def _is_rate_limit(error: BaseException) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "503" in text


_retry = retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=10, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)


def prompt_id(sector: str) -> str:
    """Identificador curto da instrução: mudar o texto muda a pasta do cache,
    para a base inteira ser respondida com a mesma versão."""
    return hashlib.sha256(instructions(sector).encode()).hexdigest()[:8]


def cache_file(model: str, video_id: str, sector: str, root: Path = VLM_DIR) -> Path:
    return root / model.replace("/", "__") / prompt_id(sector) / f"{video_id}.json"


def _read_json(path: Path) -> dict | None:
    """Conteúdo de um arquivo do cache; None se não existe ou está corrompido."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_cached(
    model: str, video_id: str, sector: str, root: Path = VLM_DIR
) -> Rubric | None:
    """Rubrica já extraída para um Short, sem chamar o modelo."""
    record = _read_json(cache_file(model, video_id, sector, root))
    if record is None:
        return None
    try:
        return parse_rubric(record["text"], sector)
    except (ValueError, ValidationError, json.JSONDecodeError):
        return None


def parse_rubric(text: str, sector: str) -> Rubric:
    """Primeiro objeto JSON do texto, validado contra a rubrica do setor.

    Lê só o primeiro objeto completo: texto depois dele (inclusive com chaves)
    é ignorado.
    """
    start = (text or "").find("{")
    if start < 0:
        raise ValueError("resposta sem JSON")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    return rubric_for(sector).model_validate(data)


def _context(creative: Creative) -> str:
    transcript = creative.transcript or "(sem fala)"
    return f"Título publicado: {creative.title}\nTranscrição do áudio: {transcript}"


class Extractor:
    name: str

    def call(self, creative: Creative) -> tuple[str, dict]:
        """Texto da resposta e uso de tokens."""
        raise NotImplementedError

    # Chamado antes de cada pedido ao provedor, inclusive nas novas tentativas
    # (ex.: esperar a vez no limite de tokens por minuto).
    before_request: Callable[[], None] | None = None

    def _wait_turn(self) -> None:
        if self.before_request is not None:
            self.before_request()

    def cache_path(self, creative: Creative, root: Path = VLM_DIR) -> Path:
        return cache_file(self.name, creative.video_id, creative.sector, root)

    def extract(self, creative: Creative, root: Path = VLM_DIR) -> dict:
        """Resposta do modelo (do cache, se já existe) com rubrica validada."""
        path = self.cache_path(creative, root)
        record = _read_json(path)
        if record is None:
            started = time.time()
            try:
                text, usage = self.call(creative)
                record = {"text": text, **usage}
            # Qualquer falha do provedor (cota, rede, recusa) vira erro
            # registrado, para o benchmark contar e seguir para o próximo.
            except Exception as error:  # noqa: BLE001
                record = {"text": None, "error": str(error)[:500]}
            record["seconds"] = round(time.time() - started, 2)
        try:
            record["rubric"] = parse_rubric(record["text"], creative.sector)
        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            record["rubric"] = None
            record.setdefault("error", f"rubrica inválida: {str(error)[:300]}")
        # Só respostas válidas vão para o cache: falha do provedor ou rubrica
        # inválida é pedida de novo na próxima execução.
        if record["rubric"] is not None and not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            cached = {k: v for k, v in record.items() if k != "rubric"}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cached, ensure_ascii=False))
            tmp.replace(path)
        return record


class GeminiVideo(Extractor):
    """Gemini com o vídeo nativo (referência de qualidade)."""

    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        from google import genai

        self.name = model
        self.client = genai.Client(api_key=config.require_env("GEMINI_API_KEY"))

    @_retry
    def call(self, creative: Creative) -> tuple[str, dict]:
        from google.genai import types

        self._wait_turn()

        uploaded = self.client.files.upload(file=str(creative.video))
        try:
            while uploaded.state and uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = self.client.files.get(name=uploaded.name)
            response = self.client.models.generate_content(
                model=self.name,
                contents=[
                    uploaded,
                    _context(creative),
                    instructions(creative.sector),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    # JSON Schema padrão (o `response_schema` da API não aceita
                    # o `additionalProperties` da rubrica).
                    response_json_schema=rubric_for(
                        creative.sector
                    ).model_json_schema(),
                ),
            )
        finally:
            self.client.files.delete(name=uploaded.name)
        return response.text, _usage(response)


class GeminiFrames(Extractor):
    """Modelo da API (Gemma) com os frames e a transcrição."""

    def __init__(self, model: str = "gemma-4-31b-it"):
        from google import genai

        self.name = model
        self.client = genai.Client(api_key=config.require_env("GEMINI_API_KEY"))

    @_retry
    def call(self, creative: Creative) -> tuple[str, dict]:
        from google.genai import types

        self._wait_turn()

        images = [
            types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")
            for path in creative.frames
        ]
        response = self.client.models.generate_content(
            model=self.name,
            contents=[
                *images,
                (
                    "Os frames acima são de um Short, em ordem; os 3 primeiros "
                    "são dos 3 primeiros segundos."
                ),
                _context(creative),
                instructions(creative.sector),
            ],
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text, _usage(response)


class OllamaFrames(Extractor):
    """Modelo local no Ollama com os frames e a transcrição."""

    def __init__(self, model: str = "gemma4:12b"):
        self.name = model

    def call(self, creative: Creative) -> tuple[str, dict]:
        import ollama

        response = ollama.chat(
            model=self.name,
            messages=[
                {
                    "role": "user",
                    "content": "Os frames anexos são de um Short, em ordem; os 3 "
                    "primeiros são dos 3 primeiros segundos.\n"
                    f"{_context(creative)}\n{instructions(creative.sector)}",
                    "images": [str(p) for p in creative.frames],
                }
            ],
            format=rubric_for(creative.sector).model_json_schema(),
            options={"temperature": 0},
            think=False,
        )
        return response.message.content, {
            "input_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
        }


def _usage(response) -> dict:
    usage = response.usage_metadata
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
    }


CANDIDATES = {
    "gemini-3.5-flash-lite": lambda: GeminiVideo("gemini-3.5-flash-lite"),
    "gemma-4-31b-it": lambda: GeminiFrames("gemma-4-31b-it"),
    "gemma-4-26b-a4b-it": lambda: GeminiFrames("gemma-4-26b-a4b-it"),
    "gemma4:12b": lambda: OllamaFrames("gemma4:12b"),
    "qwen3.5:9b": lambda: OllamaFrames("qwen3.5:9b"),
}
REFERENCE = "gemini-3.5-flash-lite"
# Escolhido no benchmark de 28 Shorts: o mais próximo da referência e com pesos
# abertos, para a Galaxies rodar o mesmo modelo localmente com criativos inéditos.
EXTRACTOR = "gemma-4-31b-it"
LOCAL = {"gemma4:12b", "qwen3.5:9b"}
