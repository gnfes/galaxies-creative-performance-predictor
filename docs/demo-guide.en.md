# Demo script

[Português](demo-guide.md)

A 6 to 8 minute script to present the demo screen. It uses the dataset collected locally (`data/processed/`); without it, prediction and comparison work with channels given by id ("Outro canal"), and so does the results tab.

## Before starting

- Open Ollama with `gemma4:12b` (the text explanation depends on it).
- Open the screen:
  ```bash
  uv run --extra app python -m creative_predictor.app
  ```
- Run a warm-up prediction: the first one loads EmbeddingGemma 2 and takes about 20 s; the next ones take a few seconds without the LLM and about 1 minute with it.
- Leave the extractor rubric unticked: it sends frames to Google's free API and takes 1 to 10 minutes.

## 1. Opening (30 s)

The question is how many likes a Short will get before it is published, relative to what the channel usually gets. The number comes from a model trained on 29,916 real Shorts from 87 channels; the LLM only writes the explanation from the finished numbers.

## 2. Predict a creative (2 min)

- Channel: **Rita Saraiva** (fashion). Caption: "Look do dia: jeans, camisa branca e tênis". Date: 08/29/2026, right after the end of the dataset (08/28/2026). Untick the LLM explanation to make it faster.
- What to show:
  - predicted likes, the 80% interval and how many times the channel median that represents;
  - the interval chart, with the channel median as a reference;
  - the top drivers (SHAP contributions), such as the title topic, the duration relative to the channel's usual and the days since the previous Short.
- Change the duration or the caption and predict again: the previous result disappears and the new one shows how the number reacts.
- If asked about today's dates: the history of the dataset channels ends on 08/28/2026, and for a much later date the model sees a pause in the channel. In real use, without the local dataset, each channel's history comes up to date from the YouTube API; on the screen, this applies to channels outside the dataset, given by id under "Outro canal".

## 3. Compare creatives (2 min)

- Channel: **Karen Bachini** (beauty), date 08/29/2026, with 2 creatives: "Make rápida para o dia a dia com 3 produtos" at 45 s and "Testei a base mais viral do momento: vale a pena?" at 120 s. Keep the LLM explanation ticked. The second one comes out ahead (about 2,500 predicted likes against 1,900), driven by duration, the strongest simple signal in the dataset.
- What to show:
  - the ranking, in predicted likes and times the channel median;
  - the agent's explanation, which cites the numbers and drivers it received (in an automatic check of 100 explanations, none of the 516 numbers cited was made up);
  - the overlapping intervals: the explanation says when the difference between the creatives is within the uncertainty. With the same duration, the two captions come out almost tied, and the explanation says the difference is not clear.

## 4. Channels outside training (1 to 2 min)

The latest Shorts from channels the model never saw, predicted as if they had not been published yet and compared with the actual likes.

- **Nissan Brasil**: 8 of 9 pairs in the right order and 4 of 5 actual likes inside the interval.
- **H&M**: 7 of 10 pairs and 5 of 5 inside the interval.
- **Alini Bosko**, as a counterpoint: 9 of 10 pairs, but 0 of 5 inside the interval. The order is right and the level is not: the channel's reference Shorts are old (stale history), and the model flags this case as low confidence.
- Across the 13 channels: 77 of 114 pairs (68%) and 27 of 61 actual likes inside the interval.

## 5. Results (1 min)

- Gain of each block: the final model (metadata, video and audio signals, embeddings and rubric) reaches 0.648 pairwise accuracy, against 0.622 with metadata and text only; in within-channel Spearman, +0.080 (90% interval from +0.045 to +0.116).
- Trained model vs. LLM without a model: asking `gemma4:12b` for the number directly gives 0.504 pairwise (chance); the model gives 0.668 on the same Shorts.
- The 80% interval covers 79.7% of the test; Shorts flagged as low confidence have larger errors (MAE 0.886 against 0.554).

## Likely questions

- **Why likes and not views?** Brand views include paid media and changed definition in March 2025; likes are more stable (details in the [dataset card](dataset.en.md)).
- **Does it work for a new channel?** Yes, with the history from the YouTube API; the prediction is flagged as low confidence when the channel has few Shorts or is outside the training data.
- **Does a confidential creative go to any API?** No, unless the rubric is requested. Without it, frames, audio and transcript are processed locally.
- **How accurate is the model?** It orders two Shorts from the same channel correctly in about 2 of every 3 pairs; the absolute number has a large error, and the interval shows it.
