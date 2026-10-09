# CrisisLens

**Multimodal triage of disaster social media posts.** CrisisLens reads a post's text and image together, flags the ones responders need to see, sorts them by type of need, and maps them, so humans look at the right posts first.

> Status: Week 1 of 6 (data + baselines). Results below are filled in as experiments finish.

## Why

During hurricanes, floods, and wildfires, thousands of posts appear every hour. A few show trapped people, collapsed bridges, or rescue requests; most are noise. Text alone or images alone each miss urgent posts that the combination catches.

## Results

Test split of CrisisMMD v2.0 (agreed labels). Macro-F1 is the headline metric because the classes are imbalanced.

| Model | Informative | Humanitarian | Damage | Urgent recall |
| --- | --- | --- | --- | --- |
| TF-IDF + LR (text) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| BERTweet (text) | | | | |
| CLIP (image) | | | | |
| Gated fusion (text + image) | | | | |

## Dataset

[CrisisMMD v2.0](https://crisisnlp.qcri.org/crisismmd.html): human-labeled tweet + image pairs from seven 2017 disasters, with three tasks (informative, humanitarian category, damage severity). Not included in this repo; download it yourself and follow the dataset's [terms of use](https://crisisnlp.qcri.org/terms-of-use.html).

> Alam, Ofli, Imran. *CrisisMMD: Multimodal Twitter Datasets from Natural Disasters.* ICWSM 2018.
> Ofli, Alam, Imran. *Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response.* ISCRAM 2020.

## Setup

### 1. Get the data (once)

From the [dataset page](https://crisisnlp.qcri.org/crisismmd.html), download:

- `CrisisMMD_v2.0.tar.gz` (images + raw annotations, ~1.8 GB)
- `crisismmd_datasplit_agreed_label.zip` (the official train/dev/test splits)

Unpack both into one folder:

```
crisismmd/
├── CrisisMMD_v2.0/data_image/...
└── crisismmd_datasplit_agreed_label/task_*.tsv
```

### 2a. Run on Kaggle (recommended: free GPU from Week 2)

1. Kaggle → **Datasets → New Dataset** → upload the `crisismmd/` folder, name it `crisismmd`, keep it **private**.
2. New notebook → **Add Input** → your `crisismmd` dataset (it appears at `/kaggle/input/crisismmd`).
3. Settings → turn **Internet on**. In **Add-ons → Secrets**, add `WANDB_API_KEY`.
4. In the notebook:

```python
!git clone https://github.com/kshitideshpande/crisislens.git
%cd crisislens
%pip install -q -e .
!python scripts/check_data.py --data-dir /kaggle/input/crisismmd
```

### 2b. Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,track]"
python scripts/check_data.py --data-dir data/crisismmd
```

## Week 1: data + baselines

```bash
# 1. Sanity-check the download: row counts, labels, and that image files exist
python scripts/check_data.py --data-dir /kaggle/input/crisismmd

# 2. Explore: open notebooks/01_eda.ipynb, set DATA_DIR, run all, fill in "Takeaways"

# 3. Baseline: TF-IDF + logistic regression on all three tasks
python -m crisislens.train_baseline --config configs/baseline_tfidf.yaml
python -m crisislens.train_baseline --config configs/baseline_tfidf.yaml --wandb   # also log to W&B

# 4. Tests
pytest -q
```

Outputs land in `outputs/baseline_tfidf_lr/<task>/`: `results.json` (all metrics, the C sweep, top n-grams per class) and `confusion_test.png`.

> The damage task is labeled on images, so a text-only model will be weak there. That's expected, and it's the first piece of evidence for why the image matters.

## Project layout

```
configs/                one YAML per experiment
notebooks/01_eda.ipynb  data exploration
scripts/check_data.py   dataset sanity checks
src/crisislens/
  data.py               CrisisMMD loading + tweet normalization
  metrics.py            macro-F1, urgent-class recall, confusion matrix
  models/baselines.py   TF-IDF + logistic regression
  train_baseline.py     dev-tuned training + test evaluation
tests/                  runs on a tiny fake dataset (no download needed)
```

## Responsible use

CrisisLens is a triage aid that ranks what humans look at first. It never decides who gets help. The data is English-language tweets from 2017, so performance on today's platforms is unknown.
