import re

import pytest

from creative_predictor import config
from creative_predictor.agent import graph


@pytest.mark.parametrize("name", ["agent.md", "agent.en.md"])
def test_agent_diagram_in_docs_matches_the_graph(name):
    text = (config.ROOT_DIR / "docs" / name).read_text()
    block = re.search(r"<!-- grafo:inicio -->\n```mermaid\n(.*?)```", text, re.DOTALL)
    assert block, f"diagrama do grafo não encontrado em docs/{name}"
    assert block.group(1).strip() == graph.mermaid().strip(), (
        f"docs/{name} desatualizado: gere de novo com "
        "`python -m creative_predictor.agent.graph --mermaid`"
    )
