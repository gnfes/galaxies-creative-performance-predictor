"""Agente em LangGraph que avalia criativos antes da publicação.

Fluxo fixo, um nó por ferramenta:

    extrair → prever → comparáveis → explicar → ordenar (se houver mais de um)

O nó de extração roda, para cada criativo com vídeo, os mesmos passos da base
de treino (frames, transcrição, sinais, embeddings e, se autorizada, a
rubrica do extrator); a previsão usa esses atributos.

O número sempre vem do modelo treinado (`predict.service`). O LLM local só
escreve a explicação em linguagem natural, a partir dos fatores, do
intervalo, do sinal de confiança e dos comparáveis que as ferramentas
devolveram; a instrução proíbe inventar ou alterar números. Sem o LLM
disponível, a explicação cai para um texto padrão e o resto funciona igual.

O LLM é configurável por `CREATIVE_PREDICTOR_LLM` (padrão `gemma4:12b`, no
Ollama).

    from creative_predictor.agent.graph import analyze
    result = analyze([CreativeInput(...), CreativeInput(...)])

O diagrama do grafo (Mermaid, gerado do próprio código) sai com:

    uv run python -m creative_predictor.agent.graph --mermaid
"""

import json
import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from creative_predictor.predict import service
from creative_predictor.predict.schemas import CreativeInput, Prediction, Ranking

DEFAULT_LLM = "gemma4:12b"

EXPLAIN_PROMPT = """Você explica, em português e em até 5 frases, a previsão de \
desempenho de um YouTube Short antes da publicação, para um time de marketing.
Use SOMENTE os dados abaixo. Não invente nem altere números; cite os valores \
como estão. Diga o que puxa a previsão para cima e para baixo, e mencione a \
incerteza quando houver baixa confiança. Não use listas.

Dados (JSON):
{data}"""

RANK_PROMPT = """Você compara criativos de YouTube Shorts antes da publicação, \
em português e em até 5 frases, para um time de marketing. Use SOMENTE os dados \
abaixo, sem inventar ou alterar números. Diga qual tende a render mais e por \
quê, e se a diferença é clara ou está dentro da incerteza dos intervalos.

Dados (JSON, do maior para o menor desempenho previsto):
{data}"""


BAND_LABELS = {
    "abaixo": "abaixo do normal do canal",
    "na_media": "na média do canal",
    "acima": "acima da média do canal",
    "viral": "muito acima da média (viral)",
}
# Fatores cujo valor faz sentido para quem lê; nos demais (escalas internas
# do modelo) o LLM recebe só o nome e o sentido do efeito.
READABLE_VALUES = {
    "duration_s",
    "local_hour",
    "title_chars",
    "title_hashtags",
    "title_emojis",
    "n_tags",
    "days_since_prev",
    "posts_prev_30d",
    "face_frames_share",
    "face_in_hook",
    "text_frames_share",
    "text_in_hook",
    "cuts",
    "cuts_per_10s",
    "first_cut_s",
    "speech_words_per_second",
    "speech_language",
}
WEEKDAYS = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def _driver_value(driver) -> str | None:
    if driver.feature == "local_weekday" and driver.value:
        return WEEKDAYS[int(float(driver.value))]
    readable = driver.feature in READABLE_VALUES or driver.feature.startswith("vlm_")
    return driver.value if readable else None


class AgentState(TypedDict, total=False):
    inputs: list[CreativeInput]
    media: list[dict | None]
    predictions: list[Prediction]
    ranking: Ranking | None


_llm_cache: dict[str, object] = {}


def _llm():
    """LLM local do Ollama, ou None se não estiver disponível.

    O modelo que respondeu fica guardado; uma falha não fica, para a próxima
    chamada tentar de novo (ex.: Ollama iniciado depois).
    """
    name = os.getenv("CREATIVE_PREDICTOR_LLM", DEFAULT_LLM)
    if name in _llm_cache:
        return _llm_cache[name]
    try:
        from langchain_ollama import ChatOllama

        model = ChatOllama(model=name, temperature=0.2, reasoning=False)
        model.invoke("ok")  # falha cedo se o Ollama ou o modelo não existir
        _llm_cache[name] = model
        return model
    except Exception:  # noqa: BLE001 (sem LLM, a explicação usa o texto padrão)
        return None


def _summary(prediction: Prediction) -> dict:
    """O que o LLM pode usar: números já calculados, sem nada a inventar."""
    return {
        "criativo": prediction.creative_id,
        "curtidas_previstas": prediction.predicted_metric,
        "intervalo_80": [prediction.interval_low, prediction.interval_high],
        "vezes_a_mediana_do_canal": prediction.relative_to_channel,
        "faixa": BAND_LABELS[prediction.band],
        "baixa_confianca": prediction.confidence_flag,
        "motivos_da_baixa_confianca": prediction.confidence_reasons,
        "fatores": [
            {"fator": d.label, "valor": _driver_value(d), "efeito": d.direction}
            for d in prediction.top_drivers
        ],
        "atributos_do_video": prediction.attributes,
        "comparaveis": [
            {"titulo": c.title, "vezes_a_mediana_do_seu_canal": c.relative_to_channel}
            for c in prediction.comparables[:3]
        ],
    }


def extract(state: AgentState) -> AgentState:
    """Atributos do vídeo de cada criativo (None quando não há arquivo).

    Uma URL é baixada aqui, uma vez; os nós seguintes recebem o criativo com o
    caminho local (ou sem vídeo, se o download falhou, com o aviso em `notes`).
    """
    inputs, media = [], []
    for item in state["inputs"]:
        item, notes = service.local_creative(item)
        found = service.creative_media(item)
        if notes:
            found = {"record": {}, "vectors": {}, "notes": notes}
        inputs.append(item)
        media.append(found)
    return {"inputs": inputs, "media": media}


def predict(state: AgentState) -> AgentState:
    return {
        "predictions": [
            service.predict_creative(item, with_comparables=False, media=media)
            for item, media in zip(state["inputs"], state["media"], strict=True)
        ]
    }


def comparables(state: AgentState) -> AgentState:
    predictions = [
        p.model_copy(update={"comparables": service.comparables(item)})
        for p, item in zip(state["predictions"], state["inputs"], strict=True)
    ]
    return {"predictions": predictions}


def explain(state: AgentState) -> AgentState:
    llm = _llm()
    if llm is None:
        return {}
    predictions = []
    for p in state["predictions"]:
        prompt = EXPLAIN_PROMPT.format(data=json.dumps(_summary(p), ensure_ascii=False))
        text = llm.invoke(prompt).content.strip()
        predictions.append(p.model_copy(update={"explanation": text or p.explanation}))
    return {"predictions": predictions}


def rank(state: AgentState) -> AgentState:
    ordered = sorted(
        state["predictions"], key=lambda p: p.relative_to_channel, reverse=True
    )
    best = ordered[0]
    text = (
        f"Melhor previsão: {best.creative_id} "
        f"({best.relative_to_channel:.2f}× a mediana do canal)."
    )
    llm = _llm()
    if llm is not None:
        data = json.dumps([_summary(p) for p in ordered], ensure_ascii=False)
        text = llm.invoke(RANK_PROMPT.format(data=data)).content.strip() or text
    return {"ranking": Ranking(predictions=ordered, explanation=text)}


def _route(state: AgentState) -> str:
    return "rank" if len(state["predictions"]) > 1 else END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract", extract)
    graph.add_node("predict", predict)
    graph.add_node("comparables", comparables)
    graph.add_node("explain", explain)
    graph.add_node("rank", rank)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "predict")
    graph.add_edge("predict", "comparables")
    graph.add_edge("comparables", "explain")
    graph.add_conditional_edges("explain", _route, ["rank", END])
    graph.add_edge("rank", END)
    return graph.compile()


def analyze(inputs: list[CreativeInput]) -> Ranking:
    """Avalia um ou mais criativos; com mais de um, devolve também o ranking."""
    state = build_graph().invoke({"inputs": list(inputs), "ranking": None})
    if state.get("ranking") is not None:
        return state["ranking"]
    return Ranking(predictions=state["predictions"])


def mermaid() -> str:
    """Diagrama Mermaid do grafo compilado (o mesmo de `docs/agent.md`)."""
    return build_graph().get_graph().draw_mermaid()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agente de previsão de criativos")
    parser.add_argument("--mermaid", action="store_true", help="imprime o grafo")
    if parser.parse_args().mermaid:
        print(mermaid())
