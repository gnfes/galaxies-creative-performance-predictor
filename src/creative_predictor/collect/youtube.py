"""Cliente fino da YouTube Data API v3 com cache em disco das respostas brutas.

Cada chamada `list` é salva como JSON em `data/raw/youtube/<recurso>/`, com a
data da consulta. Rodar de novo reaproveita o cache e não gasta cota; para
forçar uma consulta nova, use `refresh=True`.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from creative_predictor import config

CACHE_DIR = config.RAW_DIR / "youtube"

# Custo em unidades de cota de uma chamada `list` (cota padrão: 10.000/dia).
QUOTA_COST = {"channels": 1, "playlistItems": 1, "videos": 1, "search": 100}


class YouTubeClient:
    def __init__(self, service: Any = None, cache_dir: Path = CACHE_DIR):
        self._service = service
        self.cache_dir = cache_dir
        self.quota_used = 0

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = build(
                "youtube",
                "v3",
                developerKey=config.require_env("YOUTUBE_API_KEY"),
                cache_discovery=False,
            )
        return self._service

    def list(self, resource: str, *, refresh: bool = False, **params: Any) -> dict:
        """Chama `<resource>.list(**params)`, usando o cache quando houver."""
        path = self._cache_path(resource, params)
        if path.is_file() and not refresh:
            return json.loads(path.read_text())["response"]

        response = getattr(self.service, resource)().list(**params).execute()
        self.quota_used += QUOTA_COST.get(resource, 1)

        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "resource": resource,
            "params": params,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "response": response,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        return response

    def _cache_path(self, resource: str, params: dict) -> Path:
        key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / resource / f"{digest}.json"


def is_not_found(error: HttpError) -> bool:
    """Playlist ou canal inexistente (ex.: canal sem nenhum Short)."""
    return error.resp.status == 404
