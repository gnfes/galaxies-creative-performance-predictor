import re

from creative_predictor import config
from creative_predictor.agent import graph


def test_agent_diagram_in_docs_matches_the_graph():
    text = (config.ROOT_DIR / "docs" / "agent.md").read_text()
    block = re.search(r"<!-- grafo:inicio -->\n```mermaid\n(.*?)```", text, re.DOTALL)
    assert block, "diagrama do grafo não encontrado em docs/agent.md"
    assert block.group(1).strip() == graph.mermaid().strip(), (
        "docs/agent.md desatualizado: gere de novo com "
        "`python -m creative_predictor.agent.graph --mermaid`"
    )
