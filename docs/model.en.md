# Model card: YouTube Shorts like prediction

[Português](model.md) · English

Model card following the *model cards* format (Mitchell et al., 2019, and the Hugging Face template). The dataset is described in [`dataset.en.md`](dataset.en.md), and the system around the model in [`architecture.en.md`](architecture.en.md).

## Details

- **What it is**: a LightGBM that predicts, before publication, the likes of a fashion, beauty or automotive YouTube Short relative to its own channel's history, with an 80% interval and a low-confidence flag.
- **Target**: `log((likes + 1) / (median + 1))`, where the median is that of the likes of the channel's 20 Shorts published at least 30 days earlier. Absolute likes come back through the channel median.
- **Version**: 10/19/2026, trained on 29,916 Shorts from 87 channels (`data/models/predictor.joblib`).
- **Architecture**:
  - three LightGBMs: the prediction with Huber loss and the P10 and P90 quantiles for the interval;
  - a conformal margin (CQR) calibrated on validation;
  - bands from the P25, P75 and P95 quantiles of the training target (0.56×, 1.70× and 7.4× the channel median).
  - a 16-component PCA per embedding type.
- **Features**: 148, in four blocks:
  - **metadata and text** (50): channel history, format, local time, text counts and title embedding;
  - **video and audio signals** (26): color, faces, on-screen text, motion, cuts, volume and speech;
  - **embeddings** (48): frames, audio and transcript, with EmbeddingGemma 2;
  - **extractor rubric** (24): answered by `gemma-4-31b-it` from 8 frames and the transcript.
- **Code license**: none declared (all rights reserved). The third-party models used in extraction (EmbeddingGemma 2, Gemma 4, Whisper, YuNet, EasyOCR) follow their own licenses.

## Intended use

- **What for**: comparing creatives of the same channel before publishing and estimating the expected range of likes, with the margin of error in view.
- **For whom**: Galaxies' synthetic persona, as an MCP tool, and marketing teams, through the Python function.
- **Best case**: a channel with a recent and steady Shorts history (at least 10 Shorts older than 30 days, posted in the last few months).

## Out of scope

- Predicting views, comments, or how likes evolve in the first days.
- Paid ads: the dataset only has organic content, and channels with obvious boosting were excluded.
- Other platforms (TikTok, Reels) or long formats.
- Comparing different channels by the predicted absolute number: the model is relative to the channel.
- Deciding on its own: the prediction is one input to the decision, with the uncertainty made explicit.

## Factors evaluated

- **Sector**: fashion, beauty and automotive.
- **Channel type**: brand or creator.
- **Size**: quartiles of the channel's median likes.
- **Channel seen or not in training**: channel split.
- **Video and rubric present at prediction time.**
- **Stale history**: channels whose reference Shorts are old.

## Metrics

- **Error**: MAE on the log target and MAPE on likes.
- **Ranking**:
  - global Spearman;
  - within-channel Spearman;
  - pairwise accuracy between Shorts of the same channel (the closest to Galaxies' decision).
- **Bands**: macro F1 over the four bands (below, average, above, viral).
- **Interval**: coverage of the 80% interval.

## Evaluation data

- **Temporal test set**: Shorts from Brazilian channels published from June to August 2026 (2,821 Shorts, 58 channels), never seen in training or validation.
- **Unseen channels**: Brazilian channels split into 5 groups, each group left out of training in turn, evaluated on the same period.
- **Demo**: the 5 latest Shorts of 13 channels outside training (`reports/demo.md`).

## Training data

- **Dataset**: Shorts published from 10/15/2024 to 08/28/2026, collected through the YouTube Data API on 10/13/2026.
- **Training**: up to February 2026; **validation**: March to May 2026.
- **The saved model** is refit on every Short with a target.
- **Multimodal part**: up to 120 Shorts per channel, spread over time. 9,662 have signals and embeddings and 9,648 have the rubric, about 32% of the training rows. In the rest, these fields are empty.

## Results

**Gain per block**, on the 899 test Shorts with every block:

| Model | MAE log | Within-channel Spearman | Pairwise |
|---|---:|---:|---:|
| Baseline: channel median | 0.727 | n/a | 0.500 |
| Metadata and text | 0.683 | 0.315 | 0.622 |
| + signals | 0.686 | 0.350 | 0.629 |
| + embeddings | 0.673 | 0.371 | 0.644 |
| + rubric (final model) | 0.672 | 0.392 | 0.648 |
| Final model, without the rubric at prediction time | 0.680 | 0.358 | 0.646 |

Against the metadata and text model, with a bootstrap over channels (90% interval), the final model gains +0.025 in pairwise accuracy (+0.005 to +0.048) and +0.080 in within-channel Spearman (+0.045 to +0.116). On the full test set (2,821 Shorts, only a third with video): MAE 0.653, MAPE 0.93 (median percentage error of 48%), global Spearman 0.879 (0.862 with the channel median alone, since it mostly measures channel size), within-channel Spearman 0.375, pairwise 0.620 and band F1 0.267. On unseen channels (metadata and text model): MAE 0.689, pairwise 0.593.

**Model × LLM without a model**, on the same 198 test Shorts. The LLM received frames, title, duration, transcript, the median and the channel's latest Shorts:

| Model | MAE log | Median % error | Within-channel Spearman | Pairwise |
|---|---:|---:|---:|---:|
| Baseline: channel median | 0.776 | 57% | n/a | 0.500 |
| LLM `gemma4:12b` without a model | 1.127 | 78% | 0.014 | 0.504 |
| Trained model | 0.701 | 50% | 0.422 | 0.668 |

The LLM's MAPE is not informative (some guesses come out absurdly high); that is why the table shows the median percentage error.

**80% interval**:
- 79.7% coverage on the test set (75.8% without calibration) and 78.2% on unseen channels (metadata and text model);
- on risky Shorts (12% of the test set), 78.0% with their own margin, against 71.7% with the single margin;
- by sector: automotive 76.9%, beauty 82.1%, fashion 80.1%.

**Low confidence**:
- flags 29.8% of the test set, which err more (MAE 0.886 against 0.554);
- by reason: wide interval 0.94; stale history 1.23 (2.4% of the test set); declining channel 0.78 (10% of the test set).

**Agent explanations**: in 100 automatically checked explanations, none of the 516 cited numbers was missing from the data the LLM received, and every low-confidence one says so.

**Demo** (13 channels outside training, 5 recent Shorts each):
- 77 of 114 pairs in the right order (68%);
- 27 of 61 real like counts inside the 80% interval (44%), because of the stale history of several of these channels.

**By group**:
- fashion has pairwise 0.657 and beauty 0.592;
- creators 0.644 and brands 0.559.

**Bands**: the prediction leans toward the center. Of the 2,821 test Shorts, none was predicted as viral (68 were) and only 216 as below normal (802 were).

## Ethical considerations

- **Data**: public videos and aggregated metrics only. No personal data about viewers. The repository does not redistribute media or API responses.
- **Confidentiality of creatives**: the rubric sends frames and transcript to Google's free API, whose terms allow the data to be used for training. That is why it is only requested with `use_vlm=True`; for confidential creatives, the prediction uses only what runs locally. `gemma-4-31b-it` has open weights and could run on one's own machine, but the current code only calls it through the API.
- **Sample bias**: large and active channels are better represented; small channels and brands that post little err more.
- **Risk of use**: a low prediction may discourage content the audience would like. The interval and the low-confidence flag exist so the decision takes the uncertainty into account.

## Caveats and recommendations

- The absolute number of likes is hard to predict; the recommended use is ranking creatives of the same channel.
- **Stale history**: when a channel's reference Shorts are old, the error doubles (up or down; on the demo channels, almost always up). These cases are flagged as low confidence and get a wider interval.
- **Boosted brand campaigns** produce the largest errors, because paid media does not show up in the data.
- **Response time**: with video, local extraction takes ~30 s per creative; with the rubric through the free API, another 1 to 10 minutes.
