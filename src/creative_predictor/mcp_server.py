"""Servidor MCP: expõe a previsão de criativos como ferramentas para agentes.

Ferramentas:
- `predict_creative`: previsão de um criativo (número, intervalo, confiança,
  fatores, comparáveis), sem LLM;
- `rank_creatives`: vários criativos ordenados pelo desempenho previsto;
- `analyze_creatives`: o agente completo, com explicação em linguagem natural
  escrita pelo LLM local a partir das saídas do modelo.

Entradas e saídas seguem os modelos Pydantic de `predict.schemas`.

Uso:
    uv run python -m creative_predictor.mcp_server            # stdio
    uv run python -m creative_predictor.mcp_server --http 8000
"""

import argparse

from fastmcp import FastMCP

from creative_predictor.predict import service
from creative_predictor.predict.schemas import CreativeInput, Prediction, Ranking

mcp = FastMCP(
    "creative-predictor",
    instructions=(
        "Prevê curtidas de YouTube Shorts de moda, beleza e automotivo antes da "
        "publicação, em relação ao histórico do canal, com intervalo de 80%, "
        "sinal de baixa confiança, fatores que mais pesaram e Shorts "
        "comparáveis. Com o caminho do vídeo em `creative`, os atributos do "
        "vídeo entram na previsão; `use_vlm` autoriza enviar frames e "
        "transcrição à API do Google para a rubrica (só criativos públicos). "
        "Os números vêm de um modelo treinado; use analyze_creatives quando "
        "quiser também a explicação em texto."
    ),
)


@mcp.tool
def predict_creative(creative: CreativeInput) -> Prediction:
    """Prevê o desempenho de um criativo antes da publicação."""
    return service.predict_creative(creative)


@mcp.tool
def rank_creatives(creatives: list[CreativeInput]) -> Ranking:
    """Ordena criativos pelo desempenho previsto em relação ao canal."""
    return service.rank_creatives(creatives)


@mcp.tool
def analyze_creatives(creatives: list[CreativeInput]) -> Ranking:
    """Previsão, comparáveis, explicação em texto e ranking (agente completo)."""
    from creative_predictor.agent.graph import analyze

    return analyze(creatives)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--http", type=int, metavar="PORTA", help="serve por HTTP")
    args = parser.parse_args(argv)
    if args.http:
        mcp.run(transport="http", port=args.http)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
