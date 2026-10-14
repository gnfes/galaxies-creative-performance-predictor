import pandas as pd
import pytest

from creative_predictor.features import local_time


@pytest.mark.parametrize(
    ("country", "region", "expected"),
    [
        ("BR", "br", "America/Sao_Paulo"),
        (None, "br", "America/Sao_Paulo"),
        ("US", "global", "America/New_York"),
        (None, "global", "UTC"),
        ("JP", "global", "UTC"),
    ],
)
def test_channel_timezone(country, region, expected):
    assert local_time.channel_timezone(country, region) == expected


def test_local_time_converts_each_row_to_its_timezone():
    published = pd.Series(
        pd.to_datetime(["2026-03-02T02:30:00Z", "2026-03-02T02:30:00Z"], utc=True),
        index=[10, 20],
    )
    tz = pd.Series(["America/Sao_Paulo", "UTC"], index=[10, 20])
    out = local_time.local_time(published, tz)
    # 02:30 UTC de segunda = 23:30 de domingo em São Paulo.
    assert out.loc[10].tolist() == [23, 6]
    assert out.loc[20].tolist() == [2, 0]
    assert out.index.tolist() == [10, 20]
