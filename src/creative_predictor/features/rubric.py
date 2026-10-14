"""Rubrica que o VLM responde sobre cada Short (só JSON, validado com Pydantic).

Campos do brief agrupados por tema, mais as adaptações por setor e um campo
que diz se o vídeo trata mesmo do setor do canal (canais mistos publicam
receitas, memes, futebol...). Notas `_0a10` vão de 0 a 10.

O mesmo extrator responde a base inteira e os criativos novos: misturar
extratores faria o modelo aprender "quem respondeu" em vez do conteúdo.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Score = Annotated[int, Field(ge=0, le=10)]
Short = Annotated[str, Field(max_length=120)]


class Narration(StrEnum):
    AUSENTE = "ausente"
    CALMO = "calmo"
    ANIMADO = "animado"
    INSTRUCIONAL = "instrucional"


class Color(StrEnum):
    PRETO = "preto"
    BRANCO = "branco"
    CINZA = "cinza"
    VERMELHO = "vermelho"
    LARANJA = "laranja"
    AMARELO = "amarelo"
    VERDE = "verde"
    AZUL = "azul"
    ROXO = "roxo"
    ROSA = "rosa"
    MARROM = "marrom"
    BEGE = "bege"


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Gancho
    hook_primeiros_3s: Short = Field(
        description="O que acontece nos 3 primeiros segundos, em até 15 palavras"
    )
    forca_hook_0a10: Score = Field(
        description="Quanto os 3 primeiros segundos prendem a atenção"
    )
    # Pessoas e produto
    tem_rosto: bool = Field(description="Aparece rosto humano visível")
    close_up_produto: bool = Field(description="Há close no produto")
    before_after_ou_transformacao: bool = Field(
        description="Mostra antes e depois ou uma transformação"
    )
    try_on_ou_demonstracao: bool = Field(
        description="Alguém usa, veste ou demonstra o produto"
    )
    # Texto e edição
    texto_na_tela: bool = Field(description="Há texto sobreposto ao vídeo")
    densidade_texto_0a10: Score = Field(description="Quantidade de texto na tela")
    ritmo_cortes_0a10: Score = Field(description="0 = plano único, 10 = cortes rápidos")
    # Áudio
    energia_musica_0a10: Score = Field(
        description="Energia da música de fundo (0 = sem música)"
    )
    tom_narracao: Narration = Field(description="Tom da fala, se houver")
    # Estética
    brilho_saturacao_0a10: Score = Field(description="Imagem clara e saturada")
    cor_dominante: Color = Field(description="Cor que predomina na imagem")
    estetica_geral_0a10: Score = Field(description="Qualidade visual e de produção")
    # Mensagem e síntese
    clareza_beneficio_produto_0a10: Score = Field(
        description="Quão claro fica o benefício do produto (0 se não há produto)"
    )
    elementos_video_top5: list[Short] = Field(
        max_length=5, description="Até 5 elementos visuais marcantes, curtos"
    )
    elementos_audio_top5: list[Short] = Field(
        max_length=5, description="Até 5 elementos sonoros marcantes, curtos"
    )
    potencial_viral_0a10: Score = Field(
        description="Potencial de engajamento do vídeo, na sua avaliação"
    )
    assunto_no_setor: bool = Field(
        description="O vídeo trata do setor do canal (e não de outro assunto)"
    )


class BeautyRubric(Rubric):
    swatch: bool = Field(description="Mostra swatch (amostra do produto na pele)")
    aplicacao_na_pele: bool = Field(description="Mostra a aplicação no rosto ou pele")


class FashionRubric(Rubric):
    corpo_inteiro: bool = Field(
        description="Mostra o look de corpo inteiro (e não só detalhe da peça)"
    )
    styling: bool = Field(description="Mostra combinação de peças ou dicas de estilo")


class AutoRubric(Rubric):
    close_do_carro: bool = Field(description="Close em detalhes do carro")
    som_do_motor: bool = Field(description="O som do motor aparece em destaque")
    pov_direcao: bool = Field(description="Câmera do ponto de vista de quem dirige")
    interior: bool = Field(description="Mostra o interior do carro")


RUBRICS: dict[str, type[Rubric]] = {
    "beleza": BeautyRubric,
    "moda": FashionRubric,
    "automotivo": AutoRubric,
}


def rubric_for(sector: str) -> type[Rubric]:
    return RUBRICS[sector]


def instructions(sector: str) -> str:
    """Instrução comum a todos os extratores, com a descrição de cada campo."""
    schema = rubric_for(sector)
    fields = "\n".join(
        f"- {name}: {info.description}" for name, info in schema.model_fields.items()
    )
    return (
        f"Você analisa um YouTube Short de um canal do setor de {sector}. "
        "Responda somente com um objeto JSON com exatamente estes campos, "
        "em português, sem texto fora do JSON. Notas terminadas em _0a10 são "
        "inteiros de 0 a 10. tom_narracao é um de: "
        f"{', '.join(n.value for n in Narration)}. cor_dominante é uma de: "
        f"{', '.join(c.value for c in Color)}.\n{fields}"
    )
