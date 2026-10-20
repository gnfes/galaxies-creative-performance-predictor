"""Visual da tela de demonstração, inspirado na identidade da Galaxies.

O tema (cores, fonte DM Sans, cantos) fica em `.streamlit/config.toml`; aqui
ficam só o CSS que o tema não cobre (cartões brancos com sombra suave, abas
em pílula, métricas em cartão) e os blocos de HTML do cabeçalho. Os cartões
são `st.container(key="card_...")`, que o Streamlit marca com a classe
`st-key-card_...`.
"""

import html

ACCENT = "#0065FF"
INK = "#111827"
MUTED = "#6B7280"
LINE = "#E5E7EB"
# Sombra em camadas, como a dos cartões do site da Galaxies.
SHADOW = (
    "0 0.6px 1px -1px rgba(0,0,0,.10), 0 2.3px 4px -2px rgba(0,0,0,.09), "
    "0 10px 18px -3px rgba(0,0,0,.05)"
)

CSS = f"""
<style>
.block-container {{ padding-top: 2.5rem; max-width: 1240px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}

/* Cartões brancos */
[class*="st-key-card"] {{
  background: #FFFFFF;
  border: 1px solid {LINE};
  border-radius: 16px;
  box-shadow: {SHADOW};
  padding: 1.25rem 1.4rem;
}}
[class*="st-key-card"] [data-testid="stVegaLiteChart"] svg {{
  background: transparent !important;
}}

/* Abas em pílula */
[data-testid="stTabs"] [role="tablist"] {{
  gap: 4px; background: #EBECF0; padding: 4px; border-radius: 12px;
  width: fit-content; max-width: 100%; margin: 0 auto 1.5rem auto;
  overflow-x: auto; scrollbar-width: none;
}}
[data-testid="stTabs"] [role="tablist"]::after {{ display: none; }}
[data-testid="stTabs"] [role="tab"] {{
  border-radius: 9px; padding: 6px 16px; height: auto; color: #374151;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  background: #FFFFFF; color: {INK}; box-shadow: {SHADOW};
}}
/* Sublinhado da aba ativa: a pílula branca já marca a seleção. */
[data-testid="stTabs"] .react-aria-SelectionIndicator {{ display: none; }}

/* Métricas em cartão */
[data-testid="stMetric"] {{
  background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px;
  padding: 0.8rem 1rem; box-shadow: {SHADOW};
}}
[data-testid="stMetricValue"] {{ color: {INK}; letter-spacing: -0.02em; }}

/* Botão principal com a sombra do site */
[data-testid="stBaseButton-primary"] {{
  box-shadow: 0 0.6px 1px -1px rgba(0,0,0,.22), 0 1.8px 3.3px -1.9px rgba(0,0,0,.21),
    0 4.8px 8.6px -2.8px rgba(0,0,0,.18), 0 15px 27px -3.8px rgba(0,0,0,.08);
}}

/* Cabeçalho */
.gx-hero {{ text-align: center; margin: 0.5rem auto 1.75rem auto; max-width: 760px; }}
.gx-chip {{
  display: inline-flex; align-items: center; gap: 8px; background: #EBECF0;
  border-radius: 999px; padding: 3px 12px 3px 3px; font-size: 13px; color: #374151;
}}
.gx-chip b {{
  background: {ACCENT}; color: #FFFFFF; border-radius: 999px; padding: 2px 9px;
  font-size: 11px; font-weight: 600; letter-spacing: .04em;
}}
.gx-title {{
  font-size: 2.9rem; font-weight: 600; line-height: 1.15; letter-spacing: -0.02em;
  color: {INK}; margin: 0.9rem 0 0.6rem 0;
}}
.gx-title span {{ color: {ACCENT}; }}
.gx-sub {{ color: #374151; font-size: 1.02rem; line-height: 1.5; }}
.gx-eyebrow {{
  display: inline-block; background: #FFFFFF; border: 1px solid {LINE};
  border-radius: 8px; padding: 2px 10px; font-size: 11px; font-weight: 600;
  letter-spacing: .06em; color: #374151; text-transform: uppercase;
}}
.gx-eyebrow::before {{ content: "◆ "; color: {ACCENT}; }}
.gx-footer {{ text-align: center; color: {MUTED}; font-size: 12px; margin-top: 2.5rem; }}
@media (max-width: 640px) {{
  .gx-title {{ font-size: 1.9rem; }}
  .gx-sub {{ font-size: 0.95rem; }}
  [class*="st-key-card"] {{ padding: 1rem; }}
}}
</style>
"""


def hero(chip: str, before: str, accent: str, after: str, subtitle: str) -> str:
    """Cabeçalho centralizado: selo, título com uma palavra em azul e subtítulo."""
    e = html.escape
    return (
        '<div class="gx-hero">'
        f'<span class="gx-chip"><b>{e(chip)}</b>Inteli Academy × Galaxies</span>'
        f'<div class="gx-title">{e(before)} <span>{e(accent)}</span> {e(after)}</div>'
        f'<div class="gx-sub">{e(subtitle)}</div>'
        "</div>"
    )


def eyebrow(text: str) -> str:
    """Rótulo pequeno em caixa alta acima do título de uma seção."""
    return f'<span class="gx-eyebrow">{html.escape(text)}</span>'


def footer(text: str) -> str:
    return f'<div class="gx-footer">{html.escape(text)}</div>'
