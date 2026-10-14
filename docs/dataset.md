# Ficha da base: YouTube Shorts de moda, beleza e automotivo

Base usada para treinar e avaliar o modelo que prevê, antes da publicação, quantas curtidas um
YouTube Short vai receber em relação ao histórico do próprio canal. A análise exploratória completa
está em [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb), que lê as tabelas de `data/processed/`
(geradas pela coleta abaixo). Para executar o notebook de ponta a ponta no ambiente do projeto:

```bash
uv run python -m ipykernel install --prefix .venv --name creative-predictor
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=creative-predictor notebooks/01_eda.ipynb
```

O primeiro comando registra o kernel do ambiente do projeto dentro de `.venv`, para o Jupyter não
usar outro Python instalado na máquina.

## Origem

- **Fonte**: YouTube Data API v3 (rota pública e oficial, cota gratuita de 10.000 unidades/dia).
- **Canais**: lista curada à mão em [`configs/channel_candidates.yaml`](../configs/channel_candidates.yaml),
  validada pela API e exportada para [`configs/channels.yaml`](../configs/channels.yaml). Cada canal
  foi conferido como canal oficial da marca ou do criador (título, descrição e volume de inscritos).
- **Shorts de cada canal**: playlist de Shorts do canal (`UUSH` + id do canal sem o prefixo `UC`),
  com detalhes de cada vídeo em `videos.list` e estatísticas do canal em `channels.list`.
- **Data da coleta**: 13/10/2026. Curtidas, views, comentários e inscritos são os desse dia.
- **Reprodução**:
  ```bash
  uv run python -m creative_predictor.collect.channels validate
  uv run python -m creative_predictor.collect.channels export
  uv run python -m creative_predictor.collect.shorts
  ```
  As respostas brutas da API ficam em cache em `data/raw/` (fora do git). Uma coleta em outra data
  traz estatísticas daquele dia, então os números mudam.

## Período

- **Janela de publicação**: 15/10/2024 a 28/08/2026. O início coincide com a data em que os Shorts
  passaram a aceitar até 3 minutos, então toda a base segue a mesma regra de formato e ficam de fora
  vídeos verticais antigos reclassificados como Shorts. O fim deixa pelo menos 30 dias entre a
  publicação e a coleta (o Short mais novo tinha 46 dias), para as métricas estarem estabilizadas.
- **Histórico**: além da janela, os 50 Shorts anteriores de cada canal, usados só para calcular o
  histórico de curtidas do canal.

## Volume

| | Canais | Shorts na janela | Amostra multimodal |
|---|---:|---:|---:|
| Automotivo, Brasil | 19 | 6.405 | 2.176 |
| Automotivo, global | 9 | 3.418 | 1.029 |
| Beleza, Brasil | 20 | 6.771 | 2.119 |
| Beleza, global | 11 | 3.872 | 1.229 |
| Moda, Brasil | 21 | 7.555 | 2.178 |
| Moda, global | 7 | 2.166 | 823 |
| **Treino e teste** | **87** | **30.187** | **9.554** |
| Canais de demonstração | 13 | 215 | 215 |

- 60 dos 87 canais são do Brasil (47 criadores e 13 marcas); os 27 globais são 5 criadores e 22
  marcas, de EUA, Reino Unido, Alemanha, França, Itália, Espanha e Suécia (5 sem país declarado
  na API).
- 29.916 Shorts da janela têm alvo (106 têm curtidas ocultas e 165 não têm histórico suficiente).
- **Amostra multimodal**: até 120 Shorts por canal (com curtidas visíveis e até 3 minutos),
  espalhados no tempo ao longo da janela, que seguem para download, extração de frames, transcrição e atributos por VLM. Espalhar no tempo
  garante que todos os canais cubram o mesmo período no split temporal.
- **Canais de demonstração**: 13 canais fora do treino, com poucos Shorts na janela (Garnier, Quem
  Disse Berenice, Camila Gaio, H&M, Toyota do Brasil, Jeep, Chevrolet, Nissan, Reserva, Hering,
  Riachuelo, Alini Bosko, Garagem do Bellote). Servem para a demonstração com métrica real conhecida
  e para medir o modelo em canais que ele nunca viu.

## Setores e critérios de seleção

- Moda, beleza e automotivo, os setores com mais clientes da Galaxies. Marcas e criadores de
  conteúdo, com o Brasil como núcleo (os clientes da Galaxies são brasileiros) e canais globais como
  dados extras de treino.
- Canal entra se tem pelo menos 50 Shorts na janela, conteúdo próprio do setor (sem compilações,
  reacts ou postagem automática), e se não fica de fora por uma das regras automáticas:
  - **mídia paga**: mais de 30% dos 50 Shorts recentes com curtidas abaixo de 0,3% das views
    (Shorts impulsionados como anúncio, cujas views medem a verba e não o conteúdo);
  - **curtidas ocultas**: curtidas escondidas em mais da metade dos 50 Shorts recentes (sem alvo).
- Canais com mediana de curtidas abaixo de 10 ficam na base, marcados com `low_engagement`, porque
  são o perfil de marca que a Galaxies atende; o alvo deles é mais ruidoso.

## Tabelas e esquema

`data/processed/shorts.parquet` (um Short por linha):

| Coluna | Descrição |
|---|---|
| `video_id`, `channel_id` | Identificadores do Short e do canal |
| `published_at` | Data e hora de publicação (UTC) |
| `title`, `description`, `tags` | Texto publicado |
| `category_id`, `default_language`, `default_audio_language` | Categoria e idiomas declarados pelo canal |
| `duration_s`, `definition`, `has_captions` | Formato do vídeo |
| `made_for_kids`, `licensed_content`, `has_paid_placement`, `topic_categories` | Metadados de status e conteúdo |
| `views`, `likes`, `comments` | Métricas na data da coleta (nulas quando ocultas) |
| `thumbnail_url`, `url` | Referências à mídia |
| `sector`, `region`, `kind`, `holdout` | Setor, região, tipo de canal e canal de demonstração |
| `in_window`, `media_sample` | Short da janela; Short da amostra multimodal |
| `collected_on`, `age_days` | Data da coleta e idade do Short nesse dia |

`data/processed/channels.parquet` (um canal por linha): nome, handle, setor, região, tipo,
`low_engagement`, `holdout`, país, data de criação, inscritos, views e vídeos na data da coleta.

Correspondência com o esquema mínimo para retreino: `post_id` = `video_id`; `caption` = `title`
(+ `description`); `media_ref` = `url`; `metric_target` = `likes`; `channel_stats_at_collection` =
colunas de `channels.parquet`; `sector` = `sector`.

## Alvo

`y = log((curtidas + 1) / (mediana + 1))`, em que `mediana` é a mediana das curtidas dos 20 Shorts
do mesmo canal publicados pelo menos 30 dias antes do Short (mínimo de 10). Implementação em
[`creative_predictor.features.target`](../src/creative_predictor/features/target.py).

- **Intervalo de 30 dias**: as curtidas da base são as da data da coleta. Sem o intervalo, um post
  da véspera entraria no histórico com curtidas que acumulou depois da publicação do Short previsto.
  Com 30 dias, o histórico tem só métricas estabilizadas, que são também as únicas disponíveis ao
  prever um criativo novo.
- O alvo relativo isola o efeito do conteúdo do tamanho do canal, que é o que interessa para
  escolher entre criativos do mesmo cliente. A previsão absoluta volta por
  `curtidas = (mediana + 1) · exp(y) − 1`.
- **Curtidas, e não views**: em 31/03/2025 o YouTube passou a contar como view cada vez que um Short
  começa a tocar ou repete. A razão curtidas/view da base cai de ~0,06 para ~0,035 nesse mês, então
  views de antes e de depois não são comparáveis. Views de canais de marca também incluem mídia paga.
- Distribuição: mediana ≈ 0 e desvio-padrão ≈ 1,1. Quantis P25 = −0,57 (0,57× a mediana do canal),
  P75 = 0,51 (1,66×) e P95 = 1,92 (6,8×), que são a proposta para as faixas abaixo / na média /
  acima / viral.

## O que a base não representa

- **Outras plataformas e formatos**: só YouTube Shorts; nada de TikTok, Reels ou vídeos longos.
- **Anúncios pagos**: a base é de posts orgânicos. Os canais com sinal claro de impulsionamento
  ficaram de fora, mas um Short impulsionado pontualmente pode continuar na base.
- **Desempenho em outras métricas**: o alvo são as curtidas; alcance, retenção, cliques e vendas
  não estão disponíveis pela API pública.
- **Todas as marcas do setor**: grandes marcas brasileiras sem canal oficial ativo de Shorts (GAC,
  GWM, Renner, C&A, Shein Brasil) não estão na base; criadores de provador cobrem parte do varejo
  de moda.
- **Público**: não há dado demográfico de quem curtiu.
- **Evolução no tempo de cada post**: cada Short tem uma única leitura de métricas, na data da
  coleta.

## Limitações

- A playlist de Shorts (`UUSH`) não é documentada pela API. Ela se mostrou confiável na base
  (ordem cronológica e só 88 dos 30.187 Shorts da janela, 0,3%, acima de 3 minutos), mas pode
  mudar sem aviso.
- Inscritos, views e curtidas são os da data da coleta, não os da data da publicação. Por isso o
  alvo usa o histórico de curtidas dos posts anteriores, e os inscritos entram só como referência
  da baseline.
- Vídeos mais antigos tiveram mais tempo para acumular curtidas: o alvo médio passa de −0,05 (30 a
  60 dias) para +0,07 (480 a 800 dias). A idade na coleta é controlada na modelagem.
- O idioma do áudio declarado pelo canal não é confiável (código `sa`, sânscrito, em 1.272 Shorts;
  19% sem idioma). O idioma vem da transcrição.
- Fuso horário: canais do Brasil usam o horário de São Paulo; canais globais usam o fuso do país
  declarado (o fuso do leste nos EUA) e UTC quando o país não é informado.
- Curtidas ocultas pontuais (106 Shorts) ficam sem alvo e fora do histórico.

## Licença e uso

Os dados vêm da YouTube Data API v3 e seguem os Termos de Serviço dos serviços de API do YouTube. Os
vídeos, títulos e descrições pertencem aos canais que os publicaram. O repositório não redistribui
mídia, respostas brutas da API nem as tabelas coletadas: guarda só a lista de canais, o código que
refaz a coleta e resultados agregados.
