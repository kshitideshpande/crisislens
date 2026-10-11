---
title: CrisisLens
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8501
pinned: false
short_description: Multimodal triage of disaster posts (text + image)
---

# CrisisLens demo

Replay a simulated hurricane and watch a text + image model flag urgent posts, sort them by type of need,
explain its decisions, and map them, or try it on your own post and photo.

- **Model:** BERTweet + CLIP ViT-B/32 with gated fusion, int8 ONNX on CPU.
- **Training data:** CrisisMMD v2.0 (research use; not shown here). The replay is a fictional storm with posts written
  for this demo and public-domain photos (credits in the About tab).
- **Code, results, and method:** [github.com/kshitideshpande/crisislens](https://github.com/kshitideshpande/crisislens)

CrisisLens is a triage aid. It ranks what people look at first; it never decides who gets help.

### Space settings
| Kind | Name | Value |
| --- | --- | --- |
| Variable | `MODEL_REPO` | your private model repo, e.g. `kshitid/crisislens-model` |
| Secret | `HF_TOKEN` | a **read** token that can access that repo |
