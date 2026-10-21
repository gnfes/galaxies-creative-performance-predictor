# Roteiro da demonstração

[English](demo-guide.en.md)

Roteiro de 6 a 8 minutos para mostrar a tela de demonstração. Ele usa a base coletada localmente (`data/processed/`); sem ela, funcionam a previsão e a comparação com canais informados pelo id ("Outro canal") e a aba de resultados.

## Antes de começar

- Abra o Ollama com o `gemma4:12b` (a explicação em texto depende dele).
- Abra a tela:
  ```bash
  uv run --extra app python -m creative_predictor.app
  ```
- Faça uma previsão de aquecimento: a primeira carrega o EmbeddingGemma 2 e leva cerca de 20 s; as seguintes levam poucos segundos sem o LLM e cerca de 1 minuto com ele.
- Deixe a rubrica do extrator desmarcada: ela envia frames à API gratuita do Google e leva de 1 a 10 minutos.

## 1. Abertura (30 s)

A pergunta é quantas curtidas um Short vai ter antes de ser publicado, em relação ao que o canal costuma ter. O número vem de um modelo treinado com 29.916 Shorts reais de 87 canais; o LLM só escreve a explicação a partir dos números prontos.

## 2. Prever um criativo (2 min)

- Canal: **Rita Saraiva** (moda). Legenda: "Look do dia: jeans, camisa branca e tênis". Data: 29/08/2026, logo depois do fim da base (28/08/2026). Desmarque a explicação pelo LLM para ir mais rápido.
- O que mostrar:
  - curtidas previstas, o intervalo de 80% e quantas vezes a mediana do canal isso representa;
  - o gráfico do intervalo, com a mediana do canal como referência;
  - os fatores que mais pesaram (contribuições SHAP), como o tema do título, a duração em relação ao habitual do canal e os dias desde o Short anterior.
- Mude a duração ou a legenda e preveja de novo: o resultado anterior some e o novo mostra como o número reage.
- Se perguntarem por datas de hoje: o histórico dos canais da base termina em 28/08/2026, e para uma data bem depois o modelo enxerga uma pausa no canal. Em uso real, sem a base local, o histórico de cada canal vem atualizado da API do YouTube; na tela, isso vale para canais fora da base, informados pelo id em "Outro canal".

## 3. Comparar criativos (2 min)

- Canal: **Karen Bachini** (beleza), data 29/08/2026, com 2 criativos: "Make rápida para o dia a dia com 3 produtos" com 45 s e "Testei a base mais viral do momento: vale a pena?" com 120 s. Deixe a explicação pelo LLM marcada. O segundo sai na frente (cerca de 2.500 curtidas previstas contra 1.900), puxado pela duração, o sinal simples mais forte da base.
- O que mostrar:
  - o ranking, em curtidas previstas e vezes a mediana do canal;
  - a explicação do agente, que cita os números e os fatores recebidos (numa checagem automática de 100 explicações, nenhum dos 521 números citados era inventado);
  - os intervalos sobrepostos: a explicação diz quando a diferença entre os criativos está dentro da incerteza. Com a mesma duração, as duas legendas ficam quase empatadas, e a explicação diz que a diferença não é clara.

## 4. Canais fora do treino (1 a 2 min)

Os Shorts mais recentes de canais que o modelo nunca viu, previstos como se ainda não tivessem sido publicados e comparados com as curtidas reais.

- **Nissan Brasil**: 8 de 9 pares na ordem certa e 4 de 5 curtidas reais dentro do intervalo.
- **H&M**: 7 de 10 pares e 5 de 5 dentro do intervalo.
- **Alini Bosko**, como contraponto: 9 de 10 pares, mas 0 de 5 dentro do intervalo. A ordem está certa e o nível, não: os Shorts de referência do canal são antigos (histórico desatualizado), e o modelo marca esse caso como de baixa confiança.
- No total dos 13 canais: 77 de 114 pares (68%) e 27 de 61 curtidas reais dentro do intervalo.

## 5. Resultados (1 min)

- Ganho de cada bloco: o modelo final (metadados, sinais do vídeo e do áudio, embeddings e rubrica) chega a 0,648 de acerto par a par, contra 0,622 só com metadados e texto; no Spearman dentro do canal, +0,080 (intervalo de 90% de +0,045 a +0,116).
- Modelo treinado × LLM sem modelo: pedir o número direto ao `gemma4:12b` dá 0,504 par a par (acaso); o modelo dá 0,668 nos mesmos Shorts.
- Intervalo de 80% cobre 79,7% do teste; os Shorts marcados com baixa confiança erram mais (MAE 0,886 contra 0,554).

## Perguntas prováveis

- **Por que curtidas e não views?** Views de marca incluem mídia paga e mudaram de definição em março de 2025; curtidas são mais estáveis (detalhes na [ficha da base](dataset.md)).
- **Funciona para um canal novo?** Sim, com o histórico da API do YouTube; a previsão sai marcada como de baixa confiança quando o canal tem poucos Shorts ou está fora da base de treino.
- **O criativo sigiloso vai para alguma API?** Não, a não ser que a rubrica seja pedida. Sem ela, frames, áudio e transcrição são processados localmente.
- **Quanto o modelo acerta?** Ordena dois Shorts do mesmo canal certo em cerca de 2 de cada 3 pares; o número absoluto tem erro grande, e o intervalo mostra isso.
