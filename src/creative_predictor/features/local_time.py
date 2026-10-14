"""Hora e dia da semana de publicação no fuso do canal.

O público de um canal reage no horário local dele, não em UTC. Canais do
Brasil usam o horário de São Paulo; canais globais, o fuso do país declarado
na API (nos EUA, o fuso do leste, onde está a maior parte do público). Sem
país declarado, canais do Brasil continuam em São Paulo e os globais ficam
em UTC.
"""

import pandas as pd

BR_TIMEZONE = "America/Sao_Paulo"
TIMEZONES = {
    "BR": BR_TIMEZONE,
    "US": "America/New_York",
    "GB": "Europe/London",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "IT": "Europe/Rome",
    "ES": "Europe/Madrid",
    "SE": "Europe/Stockholm",
}


def channel_timezone(country: str | None, region: str) -> str:
    if country in TIMEZONES:
        return TIMEZONES[country]
    return BR_TIMEZONE if region == "br" else "UTC"


def local_time(published_at: pd.Series, timezone: pd.Series) -> pd.DataFrame:
    """`local_hour` (0–23) e `local_weekday` (0 = segunda) por Short.

    `published_at` precisa ter fuso (UTC na base); `timezone` traz o fuso de
    cada linha, com o mesmo índice.
    """
    out = pd.DataFrame(
        index=published_at.index, columns=["local_hour", "local_weekday"]
    )
    for tz, index in timezone.groupby(timezone).groups.items():
        local = published_at.loc[index].dt.tz_convert(tz)
        out.loc[index, "local_hour"] = local.dt.hour
        out.loc[index, "local_weekday"] = local.dt.dayofweek
    return out.astype("int64")
