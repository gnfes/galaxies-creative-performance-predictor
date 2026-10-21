"""Contrato da ferramenta: entrada e saída validadas com Pydantic.

Segue o contrato mínimo do brief: o criativo (vídeo e legenda planejada), a
data prevista de publicação, o canal (id ou histórico) e o setor; na saída,
a previsão de curtidas com intervalo, o sinal de baixa confiança, o
desempenho relativo ao canal, os fatores que mais pesaram, a explicação e os
criativos comparáveis.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Sector = Literal["moda", "beleza", "automotivo"]
Band = Literal["abaixo", "na_media", "acima", "viral"]


class PastPost(BaseModel):
    """Um Short já publicado pelo canal (histórico)."""

    published_at: datetime
    likes: int | None = Field(None, description="Curtidas; nulo se ocultas")
    duration_s: float | None = None
    title: str = ""


class ChannelInfo(BaseModel):
    """Canal do criativo: id do YouTube ou o histórico informado à mão."""

    channel_id: str | None = Field(
        None, description="Id do canal (UC...); o histórico vem da base ou da API"
    )
    history: list[PastPost] | None = Field(
        None, description="Shorts anteriores, quando o canal não está na base"
    )
    region: Literal["br", "global"] = "br"
    kind: Literal["marca", "criador"] = "marca"
    country: str | None = Field("BR", description="País do canal, para o fuso")
    created_at: datetime | None = None


class CreativeInput(BaseModel):
    """Um criativo a avaliar antes da publicação."""

    creative: Path | str | None = Field(
        None,
        description="Vídeo do criativo (opcional): caminho na máquina que roda a "
        "previsão ou URL http(s), inclusive links do YouTube",
    )
    caption: str = Field(..., description="Título/legenda planejada")
    description: str = ""
    tags: list[str] = Field(default_factory=list, description="Tags planejadas")
    category_id: str | None = Field(
        None, description="Categoria do YouTube; sem ela, vale a mais comum do canal"
    )
    planned_datetime: datetime = Field(
        ..., description="Data e hora previstas (com fuso)"
    )
    channel: ChannelInfo
    sector: Sector
    duration_s: float | None = Field(
        None, description="Duração em segundos; lida do vídeo quando houver"
    )
    creative_id: str = Field("criativo", description="Nome para identificar no ranking")
    use_vlm: bool = Field(
        False,
        description=(
            "Pede a rubrica do extrator (gemma-4-31b-it) pela API gratuita do "
            "Google, que pode usar os dados para treino: só para criativos "
            "públicos ou liberados. Sem ela, o vídeo entra pelos sinais e "
            "embeddings, calculados localmente."
        ),
    )


class Driver(BaseModel):
    """Fator que mais pesou na previsão (contribuição SHAP no alvo em log)."""

    feature: str
    label: str
    value: str
    effect: float = Field(..., description="Contribuição em log; > 0 empurra para cima")
    direction: Literal["aumenta", "reduz"]


class Comparable(BaseModel):
    """Short real e parecido, com o desempenho observado."""

    video_id: str
    url: str
    title: str
    channel: str
    likes: int | None
    relative_to_channel: float = Field(..., description="Curtidas ÷ mediana do canal")
    similarity: float


class Prediction(BaseModel):
    creative_id: str
    predicted_metric: float = Field(..., description="Curtidas previstas (~90 dias)")
    interval_low: float
    interval_high: float
    relative_to_channel: float = Field(
        ...,
        description="Desempenho em relação ao canal: (curtidas + 1) ÷ (mediana + 1); "
        "o +1 suaviza canais com poucas curtidas",
    )
    band: Band
    channel_median_likes: float
    confidence_flag: bool = Field(..., description="True = baixa confiança")
    confidence_reasons: list[str] = []
    top_drivers: list[Driver] = []
    explanation: str = ""
    comparables: list[Comparable] = []
    blocks_used: list[str] = Field(
        default_factory=list,
        description="Blocos de features presentes: metadados, sinais, "
        "embeddings, extrator",
    )
    attributes: dict[str, str | float | None] = Field(
        default_factory=dict,
        description="Atributos extraídos do vídeo (sinais e rubrica), legíveis",
    )
    notes: list[str] = Field(
        default_factory=list, description="Avisos sobre a extração do vídeo"
    )


class Ranking(BaseModel):
    """Criativos ordenados do maior para o menor desempenho previsto."""

    predictions: list[Prediction]
    explanation: str = ""
