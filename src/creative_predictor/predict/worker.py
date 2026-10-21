"""Processo separado para os modelos em PyTorch usados na previsão.

No macOS, a LightGBM (com a `libomp` do sistema) e o PyTorch (com a sua
própria `libomp`) não convivem no mesmo processo: carregar o EmbeddingGemma
num processo que já importou a LightGBM derruba o interpretador. Por isso os
embeddings, e a extração do vídeo do criativo (`predict.media`, que também
usa PyTorch no OCR), rodam num subprocesso independente
(`python -m creative_predictor.predict.worker`), que nunca importa a
LightGBM. Ele é iniciado na primeira chamada, mantém os modelos carregados e
conversa por linhas JSON na entrada e na saída padrão.

Este módulo não pode importar nada que carregue a LightGBM.
"""

import atexit
import json
import subprocess
import sys
import threading

import numpy as np

_process: subprocess.Popen | None = None
_lock = threading.Lock()


def _start() -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-m", "creative_predictor.predict.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    atexit.register(process.terminate)
    return process


def _ask(request: dict) -> dict:
    global _process
    with _lock:
        if _process is None or _process.poll() is not None:
            _process = _start()
        _process.stdin.write(json.dumps(request) + "\n")
        _process.stdin.flush()
        line = _process.stdout.readline()
    if not line:
        raise RuntimeError("o subprocesso de PyTorch terminou sem responder")
    reply = json.loads(line)
    if "error" in reply:
        raise RuntimeError(f"subprocesso de PyTorch: {reply['error']}")
    return reply


def embed(texts: list[str]) -> np.ndarray:
    """Embeddings do EmbeddingGemma 2 (mesma configuração do treino)."""
    reply = _ask({"texts": list(texts)})
    return np.asarray(reply["vectors"], dtype=np.float32)


def media(video: str, sector: str, title: str, use_vlm: bool) -> dict:
    """Atributos do vídeo de um criativo (ver `predict.media.CreativeMedia`)."""
    return _ask(
        {
            "media": {
                "video": str(video),
                "sector": sector,
                "title": title,
                "use_vlm": use_vlm,
            }
        }
    )


def _serve(replies) -> None:
    """Laço do subprocesso: uma linha JSON por pedido, uma por resposta.

    Os modelos só carregam no primeiro pedido que precisa deles.
    """
    models: dict = {}

    def handle(request: dict) -> dict:
        if "media" in request:
            if "media" not in models:
                from creative_predictor.predict.media import CreativeMedia

                models["media"] = CreativeMedia()
                # O modelo completo também codifica texto (mesmos vetores):
                # uma cópia só do EmbeddingGemma na memória.
                models.pop("text", None)
            return models["media"](**request["media"])
        if "media" in models:
            encode = models["media"].encode_texts
        else:
            if "text" not in models:
                from creative_predictor.features.text import load_encoder

                models["text"] = load_encoder()
            encode = models["text"]
        return {"vectors": encode(request["texts"]).tolist()}

    for line in sys.stdin:
        try:
            reply = handle(json.loads(line))
        except Exception as error:  # noqa: BLE001 (o erro volta para quem pediu)
            reply = {"error": str(error)[:500]}
        replies.write(json.dumps(reply) + "\n")
        replies.flush()


if __name__ == "__main__":
    # Só as respostas usam a saída padrão; qualquer impressão das bibliotecas
    # (barras de progresso, avisos) vai para a saída de erro.
    _replies, sys.stdout = sys.stdout, sys.stderr
    _serve(_replies)
