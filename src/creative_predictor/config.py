"""Caminhos do projeto e leitura de variáveis de ambiente (.env).

A raiz do projeto (onde ficam .env, configs/ e data/) vem de
CREATIVE_PREDICTOR_ROOT, se definida. Sem ela, é a raiz do repositório
quando o pacote roda a partir do código-fonte, ou o diretório atual quando
o pacote foi instalado como dependência.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def _default_root(module_file: Path = Path(__file__)) -> Path:
    repo_root = module_file.resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        return repo_root
    return Path.cwd()


ROOT_DIR = Path(os.getenv("CREATIVE_PREDICTOR_ROOT") or _default_root()).resolve()
CONFIGS_DIR = ROOT_DIR / "configs"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Registro dos experimentos na raiz do projeto, de onde quer que o código rode
# (um notebook em notebooks/ não cria um registro separado).
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "mlartifacts"

load_dotenv(ROOT_DIR / ".env")


def require_env(name: str) -> str:
    """Lê uma variável de ambiente obrigatória, com erro claro se faltar."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Variável {name} ausente — defina no .env (ver .env.example)."
        )
    return value
