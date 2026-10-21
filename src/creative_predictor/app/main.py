"""Tela de demonstração (Streamlit) da previsão de curtidas de Shorts.

Uma interface leve sobre a mesma função Python e o mesmo agente que o
servidor MCP expõe: prever um criativo, comparar criativos do mesmo canal,
ver previsto × real em canais fora do treino e consultar os resultados.
Roda localmente, porque precisa da base, dos modelos e do `ffmpeg`.

Uso:
    uv run --extra app python -m creative_predictor.app
"""

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from creative_predictor import config
from creative_predictor.app import helpers as h
from creative_predictor.app import style
from creative_predictor.predict.schemas import ChannelInfo, CreativeInput, Prediction

OTHER_CHANNEL = "Outro canal (id do YouTube)"
# Paleta da Galaxies (azul de destaque, cinzas e o violeta do ícone).
BAND_COLOR = "#C7DBFF"  # faixa do intervalo
PREDICTION_COLOR = style.ACCENT  # previsão
MEDIAN_COLOR = style.MUTED  # mediana do canal
REAL_COLOR = "#8B5CF6"  # curtidas reais
INCREASE_COLOR = style.ACCENT  # fator que aumenta a previsão
DECREASE_COLOR = "#F97066"  # fator que reduz
SECTORS = ["moda", "beleza", "automotivo"]

st.set_page_config(page_title="Previsão de curtidas de Shorts", layout="wide")
st.markdown(style.CSS, unsafe_allow_html=True)


def section(label: str, title: str, text: str | None = None) -> None:
    """Abertura de uma aba: rótulo pequeno, título e, se houver, um texto."""
    st.markdown(style.eyebrow(label), unsafe_allow_html=True)
    st.subheader(title)
    if text:
        st.caption(text)


@st.cache_data
def load_channels() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "channels.parquet"
    if not path.is_file():
        return pd.DataFrame(
            columns=["channel_id", "label", "sector", "region", "kind", "holdout"]
        )
    return h.channel_options(pd.read_parquet(path))


def has_base() -> bool:
    return (config.PROCESSED_DIR / "shorts.parquet").is_file()


@st.cache_data
def history_end() -> str:
    """Data do Short mais recente da janela da base (fim do histórico)."""
    shorts = pd.read_parquet(
        config.PROCESSED_DIR / "shorts.parquet", columns=["published_at", "in_window"]
    )
    return shorts.loc[shorts["in_window"], "published_at"].max().strftime("%d/%m/%Y")


@st.cache_data
def load_demo_groups(per_channel: int) -> dict[str, pd.DataFrame]:
    from creative_predictor.demo import demo_groups

    return dict(demo_groups(per_channel))


@st.cache_data
def load_results() -> dict[str, pd.DataFrame]:
    return h.results_tables()


def run_safely(task, *args):
    """Roda a previsão e mostra um aviso curto em vez do erro completo."""
    try:
        return task(*args)
    except Exception as error:  # noqa: BLE001 (o erro vira aviso na tela)
        st.error(f"Não foi possível prever: {error}")
        return None


def channel_picker(key: str) -> tuple[ChannelInfo, str] | None:
    """Escolha do canal: da base ou por id do YouTube. Devolve (canal, setor)."""
    options = load_channels()
    labels = [*options["label"], OTHER_CHANNEL]
    choice = st.selectbox("Canal", labels, key=f"{key}_channel")
    if choice == OTHER_CHANNEL:
        channel_id = st.text_input("Id do canal (UC...)", key=f"{key}_channel_id")
        sector = st.selectbox("Setor", SECTORS, key=f"{key}_sector")
        kind = st.radio(
            "Tipo", ["marca", "criador"], horizontal=True, key=f"{key}_kind"
        )
        if not channel_id:
            st.info("Informe o id do canal; o histórico vem da API do YouTube.")
            return None
        return ChannelInfo(channel_id=channel_id, kind=kind), sector
    row = options[options["label"] == choice].iloc[0]
    st.caption(
        f"O histórico dos canais da base vai até {history_end()}, o fim da janela "
        "coletada: para datas muito depois disso, o modelo vê uma pausa no canal. "
        "Canais fora da base, informados pelo id, usam o histórico atual, buscado "
        "na API do YouTube."
    )
    country = None if pd.isna(row["country"]) else row["country"]
    channel = ChannelInfo(
        channel_id=row["channel_id"],
        region=row["region"],
        kind=row["kind"],
        country=country,
    )
    return channel, row["sector"]


def creative_fields(key: str, number: int | None = None) -> dict:
    """Campos de um criativo (legenda, data, vídeo)."""
    suffix = f" {number}" if number else ""
    caption = st.text_input(f"Legenda{suffix}", key=f"{key}_caption")
    col1, col2 = st.columns(2)
    day = col1.date_input(
        "Data planejada",
        dt.datetime.now(dt.UTC).date() + dt.timedelta(days=1),
        key=f"{key}_day",
    )
    hour = col2.time_input(
        "Horário (local do canal)", dt.time(18, 0), key=f"{key}_hour"
    )
    video = st.file_uploader(
        f"Vídeo{suffix} (opcional)", type=["mp4", "mov", "webm"], key=f"{key}_video"
    )
    duration = None
    if video is None:
        duration = st.number_input(
            "Duração (s), sem vídeo",
            min_value=1,
            max_value=180,
            value=30,
            key=f"{key}_duration",
        )
    return {
        "caption": caption,
        "planned_datetime": dt.datetime.combine(day, hour),
        "video": video,
        "duration_s": duration,
    }


def build_input(
    fields: dict, channel: ChannelInfo, sector: str, use_vlm: bool, name: str
):
    creative = None
    if fields["video"] is not None:
        creative = h.save_upload(fields["video"].getvalue(), fields["video"].name)
    return CreativeInput(
        creative=creative,
        caption=fields["caption"],
        planned_datetime=fields["planned_datetime"],
        sector=sector,
        channel=channel,
        duration_s=None if creative else fields["duration_s"],
        use_vlm=use_vlm,
        creative_id=name,
    )


def run_options(key: str) -> tuple[bool, bool]:
    use_vlm = st.checkbox(
        "Pedir a rubrica do extrator",
        key=f"{key}_vlm",
        help="Envia frames e transcrição à API gratuita do Google, que pode usar "
        "os dados para treino. Só para criativos públicos. Leva de 1 a 10 minutos.",
    )
    with_llm = st.checkbox(
        "Explicação pelo LLM local",
        value=True,
        key=f"{key}_llm",
        help="Usa o gemma4:12b no Ollama. Sem ele, a explicação é um texto padrão.",
    )
    return use_vlm, with_llm


def plain(chart):
    """Gráfico sem fundo nem moldura, para ficar limpo dentro do cartão branco."""
    return chart.configure(background="transparent").configure_view(strokeWidth=0)


def interval_chart(prediction: Prediction) -> alt.LayerChart:
    data = h.interval_frame(prediction)
    base = alt.Chart(data).encode(y=alt.Y("criativo:N", title=None, axis=None))
    band = base.mark_rule(size=14, opacity=0.6, color=BAND_COLOR).encode(
        x=alt.X("mínimo:Q", title="curtidas"), x2="máximo:Q"
    )
    point = base.mark_point(filled=True, size=160, color=PREDICTION_COLOR).encode(
        x="previsto:Q", tooltip=["previsto", "mínimo", "máximo"]
    )
    median = base.mark_tick(thickness=2, size=30, color=MEDIAN_COLOR).encode(
        x="mediana do canal:Q", tooltip=["mediana do canal"]
    )
    return (band + point + median).properties(height=90)


def drivers_chart(prediction: Prediction) -> alt.Chart:
    data = h.drivers_frame(prediction)
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("efeito:Q", title="contribuição (log)"),
            y=alt.Y(
                "fator:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=360, labelOverlap=False),
            ),
            color=alt.Color(
                "sentido:N",
                scale=alt.Scale(
                    domain=["aumenta", "reduz"], range=[INCREASE_COLOR, DECREASE_COLOR]
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["fator", "efeito", "sentido"],
        )
        .properties(height=alt.Step(34))
    )


def show_prediction(prediction: Prediction) -> None:
    numbers = h.summary(prediction)
    col1, col2, col3 = st.columns(3)
    col1.metric("Curtidas previstas", h.br(numbers["curtidas"]))
    low, high = numbers["intervalo"]
    col2.metric("Intervalo de 80%", f"{h.br(low)} a {h.br(high)}")
    col3.metric(
        "Vezes a mediana",
        f"{h.br(numbers['vezes_a_mediana'], 2)}×",
        help=f"Mediana do canal: {h.br(numbers['mediana_do_canal'])} curtidas",
    )
    st.write(
        f"Curtidas cerca de 90 dias depois da publicação. Faixa: "
        f"**{numbers['faixa']}**. Blocos usados: {', '.join(numbers['blocos'])}."
    )
    if prediction.confidence_flag:
        st.warning("Baixa confiança: " + "; ".join(prediction.confidence_reasons) + ".")
    st.altair_chart(plain(interval_chart(prediction)), width="stretch")
    st.caption(
        "Ponto azul: previsão. Faixa azul-clara: intervalo de 80%. Traço cinza: "
        "mediana do canal."
    )
    if prediction.explanation:
        st.markdown("**Explicação**")
        st.write(prediction.explanation)
    st.markdown("**Fatores que mais pesaram**")
    st.altair_chart(plain(drivers_chart(prediction)), width="stretch")
    attributes = h.attributes_frame(prediction)
    if len(attributes):
        with st.expander("Atributos extraídos do vídeo"):
            st.dataframe(attributes, hide_index=True, width="stretch")
    comparables = h.comparables_frame(prediction)
    if len(comparables):
        with st.expander("Shorts reais parecidos"):
            st.dataframe(
                comparables,
                hide_index=True,
                width="stretch",
                column_config={"link": st.column_config.LinkColumn("link")},
            )
    for note in prediction.notes:
        st.caption(note)


def signature(*parts) -> str:
    """Identifica as entradas de um resultado (para não mostrar um antigo)."""
    return repr(
        [
            (p.name, p.size) if hasattr(p, "getvalue") else p
            for part in parts
            for p in (part.values() if isinstance(part, dict) else [part])
        ]
    )


def predict_tab() -> None:
    form, output = st.columns([5, 7], gap="large")
    with form, st.container(key="card_one_form"):
        section("Criativo", "Prever um criativo")
        picked = channel_picker("one")
        fields = creative_fields("one")
        use_vlm, with_llm = run_options("one")
        clicked = st.button(
            "Prever", type="primary", disabled=picked is None, width="stretch"
        )
    key = signature(
        fields, picked and (picked[0].channel_id, picked[1]), use_vlm, with_llm
    )
    with output, st.container(key="card_one_result"):
        predict_result(clicked, picked, fields, use_vlm, with_llm, key)


def predict_result(clicked, picked, fields, use_vlm, with_llm, key) -> None:
    section("Resultado", "Previsão")
    if clicked and picked:
        if not fields["caption"]:
            st.error("Escreva a legenda do criativo.")
            return
        channel, sector = picked
        inp = build_input(fields, channel, sector, use_vlm, "criativo")
        with st.spinner("Extraindo o vídeo e consultando o modelo..."):
            if with_llm:
                from creative_predictor.agent.graph import analyze

                ranking = run_safely(analyze, [inp])
                result = ranking.predictions[0] if ranking else None
            else:
                from creative_predictor.predict.service import predict_creative

                result = run_safely(predict_creative, inp)
        st.session_state["one"] = (key, result)
    saved = st.session_state.get("one")
    if saved and saved[0] == key and saved[1] is not None:
        show_prediction(saved[1])
    else:
        st.caption(
            "Escolha o canal, escreva a legenda e clique em Prever. O resultado "
            "aparece aqui: curtidas previstas, intervalo, fatores e explicação."
        )


def compare_tab() -> None:
    with st.container(key="card_many_channel"):
        section(
            "Ranking",
            "Comparar criativos do mesmo canal",
            "Versões de um mesmo Short, ordenadas pelas curtidas previstas em "
            "relação ao canal.",
        )
        picked = channel_picker("many")
        count = st.number_input("Quantos criativos", 2, 4, 2, key="many_count")
    all_fields = []
    for i, col in enumerate(st.columns(count), 1):
        with col, st.container(key=f"card_many_{i}"):
            st.markdown(style.eyebrow(f"Criativo {i}"), unsafe_allow_html=True)
            all_fields.append(creative_fields(f"many_{i}", i))
    use_vlm, with_llm = run_options("many")
    key = signature(
        *all_fields, picked and (picked[0].channel_id, picked[1]), use_vlm, with_llm
    )
    if st.button("Comparar", type="primary", disabled=picked is None) and picked:
        if not all(f["caption"] for f in all_fields):
            st.error("Escreva a legenda de todos os criativos.")
            return
        channel, sector = picked
        inputs = [
            build_input(f, channel, sector, use_vlm, f"criativo {i}")
            for i, f in enumerate(all_fields, 1)
        ]
        with st.spinner("Extraindo os vídeos, consultando o modelo e ordenando..."):
            if with_llm:
                from creative_predictor.agent.graph import analyze

                result = run_safely(analyze, inputs)
            else:
                from creative_predictor.predict.service import rank_creatives

                result = run_safely(rank_creatives, inputs)
        st.session_state["many"] = (key, result)
    saved = st.session_state.get("many")
    if saved and saved[0] == key and saved[1] is not None:
        ranking = saved[1]
        with st.container(key="card_many_result"):
            section("Resultado", "Ranking")
            st.dataframe(h.ranking_frame(ranking), hide_index=True, width="stretch")
            if ranking.explanation:
                st.write(ranking.explanation)
            for p in ranking.predictions:
                with st.expander(p.creative_id):
                    show_prediction(p)


def demo_tab() -> None:
    controls, output = st.columns([4, 8], gap="large")
    with controls, st.container(key="card_demo_form"):
        section(
            "Canais nunca vistos",
            "Previsto × real",
            "Os Shorts mais recentes de um canal que o modelo nunca viu, previstos "
            "como se ainda não tivessem sido publicados e comparados com as "
            "curtidas reais.",
        )
        if not has_base():
            st.info("Esta aba precisa da base coletada (`data/processed/`).")
            return
        per_channel = st.slider("Shorts por canal", 2, 8, 5)
        groups = load_demo_groups(per_channel)
        name = st.selectbox("Canal de demonstração", sorted(groups))
        use_video = st.checkbox("Usar o vídeo quando ele foi baixado", value=True)
        clicked = st.button("Prever os Shorts", type="primary", width="stretch")
    with output, st.container(key="card_demo_result"):
        demo_result(clicked, groups, name, per_channel, use_video)


def demo_result(clicked, groups, name, per_channel, use_video) -> None:
    from creative_predictor.demo import pairwise_hits, to_inputs

    section("Resultado", name)
    if clicked:
        from creative_predictor.predict.service import rank_creatives

        channels = pd.read_parquet(config.PROCESSED_DIR / "channels.parquet")
        group = groups[name]
        inputs = to_inputs(group, channels, use_video)
        with st.spinner("Prevendo os Shorts do canal..."):
            ranking = run_safely(rank_creatives, inputs)
        real = {
            i.creative_id: int(v) for i, v in zip(inputs, group["likes"], strict=True)
        }
        st.session_state["demo"] = ((name, per_channel, use_video), ranking, real)
    saved = st.session_state.get("demo")
    if not (saved and saved[0] == (name, per_channel, use_video) and saved[1]):
        st.caption("Clique em Prever os Shorts para ver o previsto e o real.")
        return
    _, ranking, real = saved
    hits, total = pairwise_hits(ranking, real)
    table = h.demo_frame(ranking, real)
    inside = (table["dentro do intervalo"] == "sim").sum()
    col1, col2 = st.columns(2)
    col1.metric("Pares na ordem certa", f"{hits} de {total}")
    col2.metric("Curtidas reais no intervalo", f"{inside} de {len(table)}")
    base = alt.Chart(table).encode(
        y=alt.Y(
            "Short:N",
            title=None,
            sort=None,
            axis=alt.Axis(labelLimit=320, labelOverlap=False),
        )
    )
    chart = (
        base.mark_rule(size=10, opacity=0.6, color=BAND_COLOR).encode(
            x=alt.X("mínimo:Q", title="curtidas"), x2="máximo:Q"
        )
        + base.mark_point(filled=True, size=120, color=PREDICTION_COLOR).encode(
            x="previsto:Q", tooltip=["previsto", "mínimo", "máximo"]
        )
        + base.mark_point(
            shape="diamond", filled=True, size=140, color=REAL_COLOR
        ).encode(x="real:Q", tooltip=["real"])
    ).properties(height=alt.Step(32))
    st.altair_chart(plain(chart), width="stretch")
    st.caption(
        "Ponto azul: previsão. Faixa azul-clara: intervalo de 80%. Losango "
        "violeta: curtidas reais."
    )
    st.dataframe(table, hide_index=True, width="stretch")


def results_tab() -> None:
    section(
        "Experimentos",
        "Resultados do projeto",
        "Teste temporal com Shorts de canais do Brasil de junho a agosto de 2026, "
        "nunca vistos no treino. Detalhes no README e no notebook de resultados.",
    )
    for i, (title, table) in enumerate(load_results().items()):
        with st.container(key=f"card_results_{i}"):
            st.markdown(f"**{title}**")
            st.dataframe(table, hide_index=True, width="stretch")


st.markdown(
    style.hero(
        "Shorts",
        "Saiba quantas curtidas o criativo terá",
        "antes",
        "de publicar.",
        "Moda, beleza e automotivo. O número vem de um modelo treinado com "
        "Shorts reais; o LLM só explica.",
    ),
    unsafe_allow_html=True,
)
tabs = st.tabs(
    ["Prever um criativo", "Comparar criativos", "Canais fora do treino", "Resultados"]
)
with tabs[0]:
    predict_tab()
with tabs[1]:
    compare_tab()
with tabs[2]:
    demo_tab()
with tabs[3]:
    results_tab()
st.markdown(
    style.footer("Projeto Inteli Academy × Galaxies. Demonstração local."),
    unsafe_allow_html=True,
)
