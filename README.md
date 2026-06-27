# VK Advertising — Predicting User Gender

Machine-learning solution for the **VK / All Cups** task: predict a social-media user's **gender**
(`target ∈ {0, 1}`) from their advertising-request logs, so holiday gift-set promotions on
VKontakte, Odnoklassniki and Zen reach the right audience.

The full pipeline lives in a single self-contained, idempotent notebook:
**[`vk_gender_prediction.ipynb`](vk_gender_prediction.ipynb)**.

## Approach

Users have only ~1.14 requests each (median 1), so the signal is essentially **per-request**. Each
request is turned into features and aggregated to the user:

| Group | Features | Why |
|---|---|---|
| Referer embedding | `component0..9` (mean + std) | Encodes the *type* of site the ad ran on — strongest signal |
| User-agent | browser, os, browser/os major version | Device & software generation correlate with demographics |
| Referer structure | `domain` (target-encoded), `has_path` | Which host; homepage vs deep link |
| Geo | `geo_id` / country / region (target-encoded) | Regional gender patterns |
| Time | hour, day-of-week | Activity rhythm differs by audience |
| Activity | n_requests, n_domains, n_geos | Engagement intensity |

High-cardinality categoricals use **out-of-fold smoothed target encoding** (α = 20) to avoid leakage.
The model is a **LightGBM** gradient-boosted tree trained with **5-fold stratified cross-validation**;
test predictions average the five fold models (0.5 threshold → 0/1 label).

## Data

The competition CSVs are **not** included (large; `train.csv` exceeds GitHub's 100 MB limit, and the
dataset is not redistributed here). Place these files in the repo root before running:

```
train.csv  test.csv  train_labels.csv  test_users.csv  referer_vectors.csv  [geo_info.csv]
```

`geo_info.csv` is optional — the notebook detects and uses it automatically if present.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook vk_gender_prediction.ipynb   # then: Restart Kernel → Run All
```

Running top-to-bottom produces:

| Output | Description |
|---|---|
| `submission.csv` | `user_id;target` for every user in `test_users.csv` (format = `train_labels.csv`) |
| `model.pkl` | 5-fold LightGBM ensemble + metadata |
| `report.pdf` | project report (problem, method, results, feature importance) |

The pipeline is deterministic (fixed seeds) and runs in a few minutes on a laptop CPU.
