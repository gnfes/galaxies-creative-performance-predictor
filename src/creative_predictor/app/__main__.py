"""Abre a tela de demonstração: `python -m creative_predictor.app`.

Argumentos extras vão para o `streamlit run` (ex.: `--server.port 8502`).
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).with_name("main.py")
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(script), *sys.argv[1:]]
    )


if __name__ == "__main__":
    sys.exit(main())
