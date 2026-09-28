"""Caminhos do projeto e leitura de variáveis de ambiente (.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT_DIR / "configs"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

load_dotenv(ROOT_DIR / ".env")


def require_env(name: str) -> str:
    """Lê uma variável de ambiente obrigatória, com erro claro se faltar."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Variável {name} ausente — defina no .env (ver .env.example)."
        )
    return value
