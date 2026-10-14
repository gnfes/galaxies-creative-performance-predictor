"""Features do texto publicado: contagens simples e embeddings do título.

Os embeddings vêm do EmbeddingGemma 2 (`google/embeddinggemma-2`), carregado
só com o codificador de texto, com o prefixo de classificação recomendado pelo
modelo e cortados em 256 dimensões (Matryoshka). Ficam em cache em
`data/interim/embeddings/`, um arquivo por modelo e campo, para cada título
ser calculado uma vez só.
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from creative_predictor import config

EMBEDDING_MODEL = "google/embeddinggemma-2"
EMBEDDING_DIM = 256
EMBEDDING_PROMPT = "Classification"
EMBEDDINGS_DIR = config.INTERIM_DIR / "embeddings"

_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
_HASHTAG = re.compile(r"#\w+")
_URL = re.compile(r"https?://\S+")


def _upper_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    return sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0


def text_stats(shorts: pd.DataFrame) -> pd.DataFrame:
    """Contagens do título, da descrição e das tags de cada Short."""
    title = shorts["title"].fillna("")
    description = shorts["description"].fillna("")
    out = pd.DataFrame(index=shorts.index)
    out["title_chars"] = title.str.len()
    out["title_words"] = title.str.split().str.len()
    out["title_hashtags"] = title.str.count(_HASHTAG.pattern)
    out["title_emojis"] = title.map(lambda s: len(_EMOJI.findall(s)))
    out["title_question"] = title.str.contains("?", regex=False).astype(int)
    out["title_exclamation"] = title.str.contains("!", regex=False).astype(int)
    out["title_digits"] = title.str.count(r"\d")
    out["title_upper_ratio"] = title.map(_upper_ratio)
    out["description_chars"] = description.str.len()
    out["description_links"] = description.str.count(_URL.pattern)
    out["description_hashtags"] = description.str.count(_HASHTAG.pattern)
    out["n_tags"] = shorts["tags"].map(lambda tags: 0 if tags is None else len(tags))
    return out


Encoder = Callable[[list[str]], np.ndarray]


def load_encoder(model_name: str = EMBEDDING_MODEL) -> Encoder:
    """Codificador de texto do EmbeddingGemma 2 (baixa o modelo na 1ª vez)."""
    import torch
    from sentence_transformers import SentenceTransformer

    # float32: o modelo é pequeno, e float16 estoura a faixa de ativações dele
    # (devolve NaN em silêncio).
    model = SentenceTransformer(
        model_name,
        model_kwargs={"torch_dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
    )

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            prompt_name=EMBEDDING_PROMPT,
            truncate_dim=EMBEDDING_DIM,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        ).astype(np.float32)

    return encode


def cache_path(field: str, model_name: str = EMBEDDING_MODEL) -> Path:
    """Arquivo de cache; muda junto com modelo, prompt e dimensão."""
    model = model_name.replace("/", "__")
    name = f"{field}__{model}__{EMBEDDING_PROMPT}__{EMBEDDING_DIM}d.parquet"
    return EMBEDDINGS_DIR / name


def embed_texts(
    ids: pd.Series,
    texts: pd.Series,
    path: Path,
    encoder: Encoder | None = None,
) -> pd.DataFrame:
    """Embeddings por id, calculando só os que ainda não estão no cache.

    `encoder` recebe uma lista de textos e devolve uma matriz; se for omitido,
    o EmbeddingGemma 2 só é carregado quando há texto novo para calcular.
    Devolve um DataFrame indexado pelo id, uma coluna por dimensão.
    """
    cached = pd.read_parquet(path) if path.is_file() else pd.DataFrame()
    todo = ~ids.isin(cached.index)
    if todo.any():
        encode = encoder or load_encoder()
        new_ids = ids[todo].drop_duplicates()
        vectors = encode(texts.loc[new_ids.index].fillna("").tolist())
        new = pd.DataFrame(
            vectors,
            index=pd.Index(new_ids.to_numpy(), name="id"),
            columns=[f"e{i}" for i in range(vectors.shape[1])],
        )
        cached = pd.concat([cached, new])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        cached.to_parquet(tmp)
        tmp.replace(path)
    return cached.loc[ids.to_numpy()]
