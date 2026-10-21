# Model card: previsão de curtidas de YouTube Shorts

Português · [English](model.en.md)

Ficha do modelo no formato de *model cards* (Mitchell et al., 2019, e o modelo do Hugging Face). A base está descrita em [`dataset.md`](dataset.md), e o sistema em volta do modelo, em [`architecture.md`](architecture.md).

## Detalhes

- **O que é**: um LightGBM que prevê, antes da publicação, as curtidas de um YouTube Short de moda, beleza ou automotivo em relação ao histórico do próprio canal, com intervalo de 80% e sinal de baixa confiança.
- **Alvo**: `log((curtidas + 1) / (mediana + 1))`, em que a mediana é a das curtidas dos 20 Shorts do canal publicados pelo menos 30 dias antes. As curtidas absolutas voltam pela mediana do canal.
- **Versão**: 19/10/2026, treinado em 29.916 Shorts de 87 canais (`data/models/predictor.joblib`).
- **Arquitetura**:
  - três LightGBM: previsão com perda Huber e quantis P10 e P90 para o intervalo;
  - folga conformal (CQR) calibrada na validação;
  - faixas pelos quantis P25, P75 e P95 do alvo no treino (0,56×, 1,70× e 7,4× a mediana do canal).
  - PCA de 16 componentes por tipo de embedding.
- **Features**: 148, em quatro blocos:
  - **metadados e texto** (50): histórico do canal, formato, horário local, contagens do texto e embedding do título;
  - **sinais do vídeo e do áudio** (26): cor, rostos, texto na tela, movimento, cortes, volume e fala;
  - **embeddings** (48): frames, áudio e transcrição, com o EmbeddingGemma 2;
  - **rubrica do extrator** (24): respondida pelo `gemma-4-31b-it` a partir de 8 frames e da transcrição.
- **Licença do código**: nenhuma declarada (todos os direitos reservados). Os modelos de terceiros usados na extração (EmbeddingGemma 2, Gemma 4, Whisper, YuNet, EasyOCR) seguem as licenças deles.

## Uso pretendido

- **Para quê**: comparar criativos de um mesmo canal antes de publicar e estimar a faixa de curtidas esperada, com a margem de erro à vista.
- **Para quem**: a persona sintética da Galaxies, como ferramenta MCP, e times de marketing, pela função Python.
- **Melhor caso**: canal com histórico recente e estável de Shorts (pelo menos 10 Shorts com 30+ dias, postados nos últimos meses).

## Fora do escopo

- Prever views, comentários ou a evolução das curtidas nos primeiros dias.
- Anúncios pagos: a base tem só conteúdo orgânico, e canais com impulsionamento evidente foram excluídos.
- Outras plataformas (TikTok, Reels) ou formatos longos.
- Comparar canais diferentes pelo número absoluto previsto: o modelo é relativo ao canal.
- Decidir sozinho: a previsão é um insumo para a decisão, com a incerteza explícita.

## Fatores avaliados

- **Setor**: moda, beleza e automotivo.
- **Tipo de canal**: marca ou criador.
- **Porte**: quartis da mediana de curtidas do canal.
- **Canal visto ou não no treino**: split por canal.
- **Presença de vídeo e de rubrica na previsão.**
- **Histórico desatualizado**: canais cujos Shorts de referência são antigos.

## Métricas

- **Erro**: MAE no alvo em log e MAPE nas curtidas.
- **Ordenação**:
  - Spearman global;
  - Spearman dentro do canal;
  - acurácia par a par entre Shorts do mesmo canal (a mais próxima da decisão da Galaxies).
- **Faixas**: F1 macro nas quatro faixas (abaixo, na média, acima, viral).
- **Intervalo**: cobertura do intervalo de 80%.

## Dados de avaliação

- **Teste temporal**: Shorts de canais do Brasil publicados de junho a agosto de 2026 (2.821 Shorts, 58 canais), nunca vistos no treino nem na validação.
- **Canais nunca vistos**: os canais do Brasil divididos em 5 grupos, cada grupo fora do treino por vez, avaliados no mesmo período.
- **Demonstração**: os 5 Shorts mais recentes de 13 canais fora do treino (`reports/demo.md`).

## Dados de treino

- **Base**: Shorts publicados de 15/10/2024 a 28/08/2026, coletados pela YouTube Data API em 13/10/2026.
- **Treino**: até fevereiro de 2026; **validação**: março a maio de 2026.
- **O modelo salvo** é retreinado com todos os Shorts com alvo.
- **Parte multimodal**: até 120 Shorts por canal, espalhados no tempo. 9.662 têm sinais e embeddings e 9.648 têm rubrica, cerca de 32% das linhas de treino. Nos demais, esses campos ficam vazios.

## Resultados

**Ganho de cada bloco**, nos 899 Shorts do teste com todos os blocos:

| Modelo | MAE log | Spearman no canal | Par a par |
|---|---:|---:|---:|
| Baseline: mediana do canal | 0,727 | n/a | 0,500 |
| Metadados e texto | 0,683 | 0,315 | 0,622 |
| + sinais | 0,686 | 0,350 | 0,629 |
| + embeddings | 0,673 | 0,371 | 0,644 |
| + rubrica (modelo final) | 0,672 | 0,392 | 0,648 |
| Modelo final, sem a rubrica na previsão | 0,680 | 0,358 | 0,646 |

Contra o modelo de metadados e texto, com bootstrap sobre os canais (intervalo de 90%), o modelo final ganha +0,025 no par a par (+0,005 a +0,048) e +0,080 no Spearman dentro do canal (+0,045 a +0,116). No teste inteiro (2.821 Shorts, só um terço com vídeo): MAE 0,653, MAPE 0,93 (erro percentual mediano de 48%), Spearman global 0,879 (0,862 só com a mediana do canal, porque mede sobretudo o porte do canal), Spearman no canal 0,375, par a par 0,620 e F1 nas faixas 0,267. Em canais nunca vistos (modelo de metadados e texto): MAE 0,689, par a par 0,593.

**Modelo × LLM sem modelo**, nos mesmos 198 Shorts do teste. O LLM recebeu frames, título, duração, transcrição, a mediana e os últimos Shorts do canal:

| Modelo | MAE log | Erro % mediano | Spearman no canal | Par a par |
|---|---:|---:|---:|---:|
| Baseline: mediana do canal | 0,776 | 57% | n/a | 0,500 |
| LLM `gemma4:12b` sem modelo | 1,127 | 78% | 0,014 | 0,504 |
| Modelo treinado | 0,701 | 50% | 0,422 | 0,668 |

O MAPE do LLM não é informativo (alguns chutes saem absurdamente altos); por isso a tabela traz a mediana do erro percentual.

**Intervalo de 80%**:
- cobertura de 79,7% no teste (75,8% sem calibração) e de 78,2% em canais nunca vistos (modelo de metadados e texto);
- nos Shorts de risco (12% do teste), 78,0% com a folga própria, contra 71,7% com a folga única;
- por setor: automotivo 76,9%, beleza 82,1%, moda 80,1%.

**Baixa confiança**:
- marca 29,8% do teste, que erram mais (MAE 0,886 contra 0,554);
- por motivo: intervalo largo 0,94; histórico desatualizado 1,23 (2,4% do teste); canal em queda 0,78 (10% do teste).

**Explicações do agente**: em 100 explicações conferidas automaticamente, nenhum dos 516 números citados estava fora dos dados que o LLM recebeu, e todas as de baixa confiança dizem isso.

**Demonstração** (13 canais fora do treino, 5 Shorts recentes de cada):
- 77 de 114 pares na ordem certa (68%);
- 27 de 61 curtidas reais dentro do intervalo de 80% (44%), por causa do histórico desatualizado de vários desses canais.

**Por grupo**:
- moda tem par a par 0,657 e beleza 0,592;
- criadores 0,644 e marcas 0,559.

**Faixas**: a previsão tende ao centro. Nos 2.821 Shorts do teste, nenhum foi previsto como viral (68 eram) e só 216 como abaixo do normal (802 eram).

## Considerações éticas

- **Dados**: só vídeos públicos e métricas agregadas. Nenhum dado pessoal de quem assiste. O repositório não redistribui mídia nem respostas da API.
- **Sigilo de criativos**: a rubrica envia frames e transcrição à API gratuita do Google, cujos termos permitem usar os dados para treino. Por isso ela só é pedida com `use_vlm=True`; para criativos sigilosos, a previsão usa só o que roda localmente. O `gemma-4-31b-it` tem pesos abertos e poderia rodar na própria máquina, mas o código atual só o chama pela API.
- **Viés de amostra**: canais grandes e ativos estão mais representados; canais pequenos e marcas que postam pouco erram mais.
- **Risco de uso**: uma previsão baixa pode desestimular conteúdo que o público receberia bem. O intervalo e o sinal de baixa confiança existem para que a decisão considere a incerteza.

## Ressalvas e recomendações

- O valor absoluto de curtidas é mal previsível; o uso recomendado é ordenar criativos do mesmo canal.
- **Histórico desatualizado**: quando os Shorts de referência do canal são antigos, o erro dobra (para cima ou para baixo; nos canais de demonstração, quase sempre para cima). Esses casos são marcados como de baixa confiança e ganham um intervalo mais largo.
- **Campanhas de marca impulsionadas** geram os maiores erros, porque a mídia paga não aparece nos dados.
- **Tempo de resposta**: com vídeo, a extração local leva ~30 s por criativo; com a rubrica pela API gratuita, mais 1 a 10 minutos.
