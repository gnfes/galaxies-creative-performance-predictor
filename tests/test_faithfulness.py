from creative_predictor.agent import faithfulness as ft

SUMMARY = {
    "curtidas_previstas": 143.2,
    "intervalo_80": [85.0, 570.7],
    "vezes_a_mediana_do_canal": 0.59,
    "baixa_confianca": True,
    "fatores": [
        {"fator": "duração do vídeo (s)", "valor": "35", "efeito": "aumenta"},
        {"fator": "desempenho recente do canal", "valor": None, "efeito": "reduz"},
    ],
    "comparaveis": [{"titulo": "Top 3 carros de 2026", "vezes": 1.4}],
}


def test_reads_numbers_in_both_formats():
    assert ft.readings("1.234") >= {1234.0}
    assert 0.59 in ft.readings("0,59")


def test_numbers_must_come_from_the_data():
    ok = ft.check(
        SUMMARY,
        "A previsão é de 143 curtidas, entre 85 e 571 (0,59 vezes a mediana, ou "
        "41% abaixo), puxada pela duração de 35 s; há incerteza.",
    )
    assert ok["numeros_sem_base"] == 0
    assert ok["cita_fator"] and ok["afirma_baixa_confianca"]
    invented = ft.check(SUMMARY, "Deve chegar a 900 curtidas.")
    assert invented["numeros_sem_base"] == 1
    assert invented["exemplos_sem_base"] == "900"


def test_denying_low_confidence_is_not_affirming_it():
    assert not ft.affirms_low_confidence("Não há indícios de baixa confiança.")
    assert not ft.affirms_low_confidence("O vídeo não apresenta baixa confiança.")
    assert ft.affirms_low_confidence("Existe baixa confiança nesta estimativa.")
    assert ft.affirms_low_confidence("A confiança nesta estimativa é baixa.")
    assert ft.affirms_low_confidence("Há incerteza, pois o canal posta pouco.")


def test_percent_readings_only_for_ratio_fields():
    summary = SUMMARY | {"atributos_do_video": {"brilho": 0.5}}
    # 0,59× pode virar "59%" ou "41% abaixo"; o brilho 0,5 não vira "50%".
    assert ft.check(summary, "Fica em 59% da mediana.")["numeros_sem_base"] == 0
    assert ft.check(summary, "Alcance de 50% do público.")["numeros_sem_base"] == 1
