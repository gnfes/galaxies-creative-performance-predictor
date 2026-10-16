"""Processo separado para os modelos em PyTorch usados na previsão.

No macOS, a LightGBM (com a `libomp` do sistema) e o PyTorch (com a sua
própria `libomp`) não convivem no mesmo processo: carregar o EmbeddingGemma
num processo que já importou a LightGBM derruba o interpretador. Por isso os
embeddings são calculados num subprocesso independente
(`python -m creative_predictor.predict.worker`), que nunca importa a
LightGBM. Ele é iniciado na primeira chamada, mantém o modelo carregado e
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


def embed(texts: list[str]) -> np.ndarray:
    """Embeddings do EmbeddingGemma 2 (mesma configuração do treino)."""
    global _process
    with _lock:
        if _process is None or _process.poll() is not None:
            _process = _start()
        _process.stdin.write(json.dumps({"texts": list(texts)}) + "\n")
        _process.stdin.flush()
        line = _process.stdout.readline()
    if not line:
        raise RuntimeError("o processo de embeddings terminou sem responder")
    reply = json.loads(line)
    if "error" in reply:
        raise RuntimeError(f"processo de embeddings: {reply['error']}")
    return np.asarray(reply["vectors"], dtype=np.float32)


def _serve(replies) -> None:
    """Laço do subprocesso: uma linha JSON por pedido, uma por resposta."""
    from creative_predictor.features.text import load_encoder

    encoder = load_encoder()
    for line in sys.stdin:
        try:
            vectors = encoder(json.loads(line)["texts"])
            reply = {"vectors": vectors.tolist()}
        except Exception as error:  # noqa: BLE001 (o erro volta para quem pediu)
            reply = {"error": str(error)[:500]}
        replies.write(json.dumps(reply) + "\n")
        replies.flush()


if __name__ == "__main__":
    # Só as respostas usam a saída padrão; qualquer impressão das bibliotecas
    # (barras de progresso, avisos) vai para a saída de erro.
    _replies, sys.stdout = sys.stdout, sys.stderr
    _serve(_replies)
