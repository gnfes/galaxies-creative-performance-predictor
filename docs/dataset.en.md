# Dataset card: fashion, beauty and automotive YouTube Shorts

[Português](dataset.md) · English

Dataset used to train and evaluate the model that predicts, before publication, how many likes a
YouTube Short will get relative to its own channel's history. The full exploratory analysis is in
[`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) (in Portuguese), which reads the tables in
`data/processed/` (produced by the collection below). To run the notebook end to end in the
project environment:

```bash
uv run python -m ipykernel install --prefix .venv --name creative-predictor
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=creative-predictor notebooks/01_eda.ipynb
```

The first command registers the project environment's kernel inside `.venv`, so Jupyter does not
use another Python installed on the machine.

## Source

- **Source**: YouTube Data API v3 (public and official route, free quota of 10,000 units/day).
- **Channels**: hand-curated list in [`configs/channel_candidates.yaml`](../configs/channel_candidates.yaml),
  validated through the API and exported to [`configs/channels.yaml`](../configs/channels.yaml). Each
  channel was checked as the official channel of the brand or creator (title, description and
  subscriber count).
- **Shorts of each channel**: the channel's Shorts playlist (`UUSH` + channel id without the `UC`
  prefix), with each video's details from `videos.list` and channel statistics from `channels.list`.
- **Collection date**: 10/13/2026. Likes, views, comments and subscribers are those of that day.
- **Reproduction**:
  ```bash
  uv run python -m creative_predictor.collect.channels validate
  uv run python -m creative_predictor.collect.channels export
  uv run python -m creative_predictor.collect.shorts
  ```
  Raw API responses are cached in `data/raw/` (outside git). A collection on another date brings
  that day's statistics, so the numbers change.

## Period

- **Publication window**: 10/15/2024 to 08/28/2026. The start matches the date when Shorts started
  accepting up to 3 minutes, so the whole dataset follows the same format rule and old vertical
  videos reclassified as Shorts are left out. The end leaves at least 30 days between publication
  and collection (the newest Short was 46 days old), so metrics have settled.
- **History**: besides the window, the 50 previous Shorts of each channel, used only to compute the
  channel's like history.

## Volume

| | Channels | Shorts in the window | Multimodal sample |
|---|---:|---:|---:|
| Automotive, Brazil | 19 | 6,405 | 2,176 |
| Automotive, global | 9 | 3,418 | 1,029 |
| Beauty, Brazil | 20 | 6,771 | 2,119 |
| Beauty, global | 11 | 3,872 | 1,229 |
| Fashion, Brazil | 21 | 7,555 | 2,178 |
| Fashion, global | 7 | 2,166 | 823 |
| **Training and test** | **87** | **30,187** | **9,554** |
| Demo channels | 13 | 215 | 215 |

- 60 of the 87 channels are from Brazil (47 creators and 13 brands); the 27 global ones are 5
  creators and 22 brands, from the US, UK, Germany, France, Italy, Spain and Sweden (5 with no
  country declared in the API).
- 29,916 Shorts in the window have a target (106 have hidden likes and 165 lack enough history).
- **Multimodal sample**: up to 120 Shorts per channel (with visible likes and up to 3 minutes long),
  spread over the window, which go on to download, frame extraction, transcription and VLM
  attributes. Spreading them over time ensures every channel covers the same period in the
  temporal split.
- **Extraction (cut on 10/19/2026)**: of the 9,769 sampled Shorts (9,554 for training and test and
  215 for the demo), 9,665 were downloaded and processed (frames, transcript, video and audio
  signals, embeddings), and 9,640 have the `gemma-4-31b-it` rubric. Another 92 went offline or
  became private after the collection, and 12 stayed blocked by YouTube. The sample covers about
  32% of the training rows; in the rest, the video fields are empty.
- **Demo channels**: 13 channels outside training, with few Shorts in the window (Garnier, Quem
  Disse Berenice, Camila Gaio, H&M, Toyota do Brasil, Jeep, Chevrolet, Nissan, Reserva, Hering,
  Riachuelo, Alini Bosko, Garagem do Bellote). They serve the demo with a known real metric and
  measure the model on channels it has never seen.

## Sectors and selection criteria

- Fashion, beauty and automotive, the sectors with the most Galaxies clients. Brands and content
  creators, with Brazil at the core (Galaxies' clients are Brazilian) and global channels as extra
  training data.
- A channel is included if it has at least 50 Shorts in the window and its own sector content (no
  compilations, reacts or automated posting), and if it is not excluded by one of the automatic
  rules:
  - **paid media**: more than 30% of the 50 latest Shorts with likes below 0.3% of views (Shorts
    boosted as ads, whose views measure the budget rather than the content);
  - **hidden likes**: likes hidden on more than half of the 50 latest Shorts (no target).
- Channels with median likes below 10 stay in the dataset, flagged with `low_engagement`, because
  they are the brand profile Galaxies serves; their target is noisier.

## Tables and schema

`data/processed/shorts.parquet` (one Short per row):

| Column | Description |
|---|---|
| `video_id`, `channel_id` | Short and channel identifiers |
| `published_at` | Publication date and time (UTC) |
| `title`, `description`, `tags` | Published text |
| `category_id`, `default_language`, `default_audio_language` | Category and languages declared by the channel |
| `duration_s`, `definition`, `has_captions` | Video format |
| `made_for_kids`, `licensed_content`, `has_paid_placement`, `topic_categories` | Status and content metadata |
| `views`, `likes`, `comments` | Metrics on the collection date (null when hidden) |
| `thumbnail_url`, `url` | Media references |
| `sector`, `region`, `kind`, `holdout` | Sector, region, channel type and demo channel |
| `in_window`, `media_sample` | Short in the window; Short in the multimodal sample |
| `collected_on`, `age_days` | Collection date and the Short's age on that day |

`data/processed/channels.parquet` (one channel per row): name, handle, sector, region, type,
`low_engagement`, `holdout`, country, creation date, subscribers, views and videos on the
collection date.

`data/processed/multimodal.parquet` (one sampled Short per row, built by
`features.extract assemble`):
- transcript and speech measures: language, share with speech, words and words per second;
- video and audio signals: brightness, contrast, saturation, colorfulness, faces (YuNet),
  on-screen text (EasyOCR), motion, scene cuts (ffmpeg), and audio volume and silence;
- the extractor rubric fields, prefixed with `vlm_`.

The embeddings (EmbeddingGemma 2) live in `data/interim/embeddings/`, one file per Short and per
type (frames, audio and transcript).

Mapping to the minimum schema for retraining: `post_id` = `video_id`; `caption` = `title`
(+ `description`); `media_ref` = `url`; `metric_target` = `likes`; `channel_stats_at_collection` =
columns of `channels.parquet`; `sector` = `sector`.

## Target

`y = log((likes + 1) / (median + 1))`, where `median` is the median likes of the channel's 20
Shorts published at least 30 days before the Short (minimum of 10). Implemented in
[`creative_predictor.features.target`](../src/creative_predictor/features/target.py).

- **30-day gap**: the dataset's likes are those of the collection date. Without the gap, a post
  from the day before would enter the history with likes it gathered after the predicted Short was
  published. With 30 days, the history has only settled metrics, which are also the only ones
  available when predicting a new creative.
- The relative target separates the effect of the content from the size of the channel, which is
  what matters when choosing between creatives of the same client. The absolute prediction comes
  back through `likes = (median + 1) · exp(y) − 1`.
- **Likes, not views**: on 03/31/2025 YouTube started counting a view every time a Short starts
  playing or replays. The dataset's likes-per-view ratio drops from ~0.06 to ~0.035 that month, so
  views before and after are not comparable. Brand channel views also include paid media.
- Distribution: median ≈ 0 and standard deviation ≈ 1.1. Quantiles P25 = −0.57 (0.57× the channel
  median), P75 = 0.51 (1.66×) and P95 = 1.92 (6.8×), which define the below / average / above /
  viral bands.

## What the dataset does not represent

- **Other platforms and formats**: YouTube Shorts only; no TikTok, Reels or long videos.
- **Paid ads**: the dataset has organic posts. Channels with a clear boosting signal were left out,
  but a Short boosted once may still be in the dataset.
- **Performance on other metrics**: the target is likes; reach, retention, clicks and sales are not
  available through the public API.
- **Every brand in each sector**: large Brazilian brands without an active official Shorts channel
  (GAC, GWM, Renner, C&A, Shein Brasil) are not in the dataset; try-on creators cover part of
  fashion retail.
- **Audience**: there is no demographic data on who liked.
- **Each post over time**: each Short has a single metrics reading, on the collection date.

## Limitations

- The Shorts playlist (`UUSH`) is not documented by the API. It proved reliable in the dataset
  (chronological order, and only 88 of the 30,187 Shorts in the window, 0.3%, longer than 3
  minutes), but it may change without notice.
- Subscribers, views and likes are those of the collection date, not of the publication date. That
  is why the target uses the like history of previous posts, and subscribers only enter as the
  baseline reference.
- Older videos had more time to gather likes: the mean target goes from −0.05 (30 to 60 days) to
  +0.07 (480 to 800 days). Age at collection is controlled in modeling.
- The audio language declared by the channel is not reliable (code `sa`, Sanskrit, on 1,272 Shorts;
  19% with no language). The language comes from the transcript.
- Time zone: Brazilian channels use São Paulo time; global channels use the declared country's
  time zone (US Eastern for the US) and UTC when the country is not given.
- Occasional hidden likes (106 Shorts) leave those Shorts without a target and out of the history.

## License and use

The data comes from the YouTube Data API v3 and follows the YouTube API Services Terms of Service.
Videos, titles and descriptions belong to the channels that published them. The repository does not
redistribute media, raw API responses or the collected tables: it only keeps the channel list, the
code that redoes the collection, and aggregated results.
