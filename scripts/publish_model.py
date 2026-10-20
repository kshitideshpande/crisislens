"""Upload the exported model to a PRIVATE Hugging Face model repo.

    export HF_TOKEN=hf_...            # a token with write access
    python scripts/publish_model.py --repo <your-username>/crisislens-model

The repo is private because the model was trained on CrisisMMD, whose terms
restrict the data to research use (docs/DATA.md). The demo Space downloads it
with a read token stored as a Space secret, so the public never gets the files.

Uploads only what inference needs: model.int8.onnx, meta.json, tokenizer and
preprocessor files, export_report.json, and a model card. No data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ALLOW = ["*model.int8.onnx", "meta.json", "export_report.json", "*.json", "*.txt", "*.codes", "*.model", "README.md"]

CARD = """---
license: other
tags: [crisis-informatics, multimodal, onnx]
---
# CrisisLens gated fusion model (private)

Text + image classifier for disaster social media posts (BERTweet + CLIP ViT-B/32,
gated fusion, int8 ONNX). Trained on CrisisMMD v2.0 for research on humanitarian
computing; kept private to respect the dataset's terms of use.

Tasks: {tasks}

Test macro-F1 for this checkpoint (seed {seed}): {scores}

Code: https://github.com/kshitideshpande/crisislens
"""


def main() -> None:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. kshitid/crisislens-model")
    ap.add_argument("--exported", default="exported")
    ap.add_argument("--run", default="outputs/fusion_gated_md40/seed_42", help="for the scores in the model card")
    args = ap.parse_args()

    exported = Path(args.exported)
    meta = json.loads((exported / "meta.json").read_text())
    metrics_path = Path(args.run) / "metrics.json"
    scores, seed = "see README", "?"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        scores = ", ".join(f"{t} {v['macro_f1']:.3f}" for t, v in m["test"].items())
        seed = m["seed"]
    (exported / "README.md").write_text(CARD.format(tasks=", ".join(meta["tasks"]), seed=seed, scores=scores))

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(args.repo)
    if not info.private:
        raise SystemExit(f"{args.repo} exists and is PUBLIC. Make it private on huggingface.co first.")
    api.upload_folder(folder_path=str(exported), repo_id=args.repo, repo_type="model", allow_patterns=ALLOW,
                      ignore_patterns=["*model.onnx"], commit_message="Upload CrisisLens model")
    print(f"Uploaded to https://huggingface.co/{args.repo} (private)")


if __name__ == "__main__":
    main()
