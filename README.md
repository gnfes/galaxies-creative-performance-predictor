# galaxies-creative-performance-predictor
Agente preditivo de performance de criativos em mídias sociais para moda, beleza e automotivo — projeto Inteli Academy x Galaxies

## Requisitos

- [uv](https://docs.astral.sh/uv/) e Python 3.12 (`uv sync` instala as dependências).
- macOS: a LightGBM precisa da biblioteca OpenMP do sistema:
  ```bash
  brew install libomp
  ```
- `ffmpeg` no PATH, para extrair frames e áudio dos vídeos.
- Chaves no `.env` (modelo em `.env.example`).
