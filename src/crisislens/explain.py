"""Explain individual predictions: which words and which image regions mattered.

    python -m crisislens.explain --run outputs/fusion_gated/seed_42 --n 12

Words:  Integrated Gradients (Captum). Walk the word embeddings from a blank
        baseline to the real tweet and add up how much each token pushed the
        predicted class's probability. Scores are summed per word.
Image:  occlusion. Cover one patch of the image at a time and measure how much
        the predicted class's probability drops. Big drop = important region.

The CLI writes PNG panels for a mix of correct and wrong test predictions to
<run>/explanations/. They contain real CrisisMMD tweets and images, so they stay
in outputs/ (gitignored) and must not be published (see docs/DATA.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from crisislens.inference import occlude_image
from crisislens.torch_data import label_maps


def merge_subwords(tokens: list[str], scores: np.ndarray, special: set[str]) -> list[tuple[str, float]]:
    """Join subword pieces back into words, summing their scores.

    BERTweet marks a piece that continues into the next one with a trailing '@@';
    RoBERTa/GPT-style vocabularies mark the start of a new word with 'Ġ'.
    """
    gpt_style = any(t.startswith("Ġ") for t in tokens)
    words: list[list] = []
    continues = False
    for tok, s in zip(tokens, scores):
        if tok in special:
            continues = False
            continue
        piece = tok.removeprefix("Ġ").removesuffix("@@")
        new_word = not words or (not continues and (not gpt_style or tok.startswith("Ġ")))
        if new_word:
            words.append([piece, float(s)])
        else:
            words[-1][0] += piece
            words[-1][1] += float(s)
        continues = tok.endswith("@@")
    return [(w, s) for w, s in words]


class Explainer:
    def __init__(self, model, tokenizer, processor, max_length: int = 128, device=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.device = device or next(model.parameters()).device
        self.tasks = model.tasks
        self.maps = label_maps(self.tasks)

    # -- inputs -------------------------------------------------------------
    def _inputs(self, text: str, image: Image.Image | None) -> dict:
        batch: dict = {"text_present": torch.tensor([bool(text.strip())]),
                       "image_present": torch.tensor([image is not None])}
        if self.tokenizer is not None:
            enc = self.tokenizer([text or "."], truncation=True, max_length=self.max_length, return_tensors="pt")
            batch["input_ids"], batch["attention_mask"] = enc["input_ids"], enc["attention_mask"]
        if self.processor is not None:
            img = image if image is not None else Image.new("RGB", (224, 224))
            batch["pixel_values"] = self.processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
        else:
            batch["pixel_values"] = torch.zeros(1, 3, 224, 224)
        return batch

    @torch.no_grad()
    def predict(self, text: str, image: Image.Image | None) -> dict:
        out = self.model(self._inputs(text, image))
        res = {"image_reliance": float(out["image_reliance"][0]) if "image_reliance" in out else None}
        for t in self.tasks:
            p = torch.softmax(out[t][0].float(), -1).cpu().numpy()
            res[t] = {"label": self.maps[t][int(p.argmax())], "confidence": float(p.max()),
                      "probs": dict(zip(self.maps[t], p.round(4).tolist()))}
        return res

    # -- words: integrated gradients ---------------------------------------
    def word_scores(self, text: str, image: Image.Image | None, task: str, target: int, steps: int = 32):
        from captum.attr import LayerIntegratedGradients

        if self.tokenizer is None or not text.strip():
            return []
        batch = self._inputs(text, image)
        ids = batch["input_ids"].to(self.device)
        special_ids = set(self.tokenizer.all_special_ids)
        baseline = torch.tensor([[i if i in special_ids else self.tokenizer.pad_token_id for i in ids[0].tolist()]],
                                device=self.device)

        def forward(input_ids, attention_mask, pixel_values, text_present, image_present):
            b = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values,
                 "text_present": text_present, "image_present": image_present}
            return torch.softmax(self.model(b)[task].float(), -1)[:, target]

        lig = LayerIntegratedGradients(forward, self.model.text_encoder.embeddings)
        extra = tuple(batch[k].to(self.device) for k in ("attention_mask", "pixel_values", "text_present", "image_present"))
        attr = lig.attribute(ids, baselines=baseline, additional_forward_args=extra, n_steps=steps,
                             internal_batch_size=8)
        scores = attr.sum(-1)[0].detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        special = set(self.tokenizer.all_special_tokens) - {self.tokenizer.unk_token}  # keep unknown words visible
        return merge_subwords(tokens, scores, special)

    # -- image: occlusion -----------------------------------------------------
    def image_heatmap(self, text: str, image: Image.Image | None, task: str, target: int, grid: int = 7):
        if image is None or self.processor is None:
            return None
        batch = self._inputs(text, image)

        @torch.no_grad()
        def prob(pixels: np.ndarray) -> np.ndarray:
            n = len(pixels)
            b = {k: (v.repeat(n, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v) for k, v in batch.items()}
            b["pixel_values"] = torch.from_numpy(pixels).float()
            return torch.softmax(self.model(b)[task].float(), -1)[:, target].cpu().numpy()

        return occlude_image(prob, batch["pixel_values"][0].numpy(), grid=grid)

    def explain(self, text: str, image: Image.Image | None, task: str = "humanitarian") -> dict:
        pred = self.predict(text, image)
        target = self.maps[task].index(pred[task]["label"])
        return {"prediction": pred, "task": task, "target": pred[task]["label"],
                "words": self.word_scores(text, image, task, target),
                "heatmap": self.image_heatmap(text, image, task, target)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(explanation: dict, text: str, image: Image.Image | None, path: Path, true_label: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax_img.axis("off")
    if image is not None:
        ax_img.imshow(image.convert("RGB"))
        if explanation["heatmap"] is not None:
            h = np.clip(explanation["heatmap"], 0, None)
            ax_img.imshow(np.array(Image.fromarray((255 * h / (h.max() or 1)).astype(np.uint8)).resize(image.size)),
                          cmap="inferno", alpha=0.45)
    else:
        ax_img.text(0.5, 0.5, "no image", ha="center", va="center", color="gray")

    ax_txt.axis("off")
    pred = explanation["prediction"]
    head = f"Predicted: {explanation['target']} ({pred[explanation['task']]['confidence']:.0%})"
    if true_label:
        head += f"\nTrue: {true_label}"
    if pred.get("image_reliance") is not None:
        head += f"\nImage reliance: {pred['image_reliance']:.0%}"
    ax_txt.text(0, 1, head, va="top", fontsize=11, fontweight="bold", transform=ax_txt.transAxes)

    words = explanation["words"] or [(w, 0.0) for w in text.split()]
    top = max((abs(s) for _, s in words), default=1) or 1
    x, y = 0.0, 0.72
    for word, s in words:
        width = 0.012 * (len(word) + 1)
        if x + width > 1:
            x, y = 0.0, y - 0.09
        color = (0.85, 0.2, 0.2, min(1, abs(s) / top)) if s > 0 else (0.2, 0.4, 0.85, min(1, abs(s) / top))
        ax_txt.text(x, y, word, fontsize=10, transform=ax_txt.transAxes,
                    bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="none"))
        x += width
    ax_txt.text(0, 0.02, "red = pushed toward the prediction, blue = pushed away", fontsize=8, color="gray",
                transform=ax_txt.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    import pandas as pd

    from crisislens.data import build_multitask_table, find_image_root
    from crisislens.train import load_trained_model, pick_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--task", default="humanitarian")
    args = ap.parse_args()

    run_dir = Path(args.run)
    device = pick_device()
    model, tok, proc, cfg = load_trained_model(run_dir / "model.pt", device=device)
    explainer = Explainer(model, tok, proc, cfg["data"]["max_length"], device)

    preds = pd.read_csv(run_dir / "predictions_test.csv")
    preds = preds[preds[f"y_{args.task}"] != -100]
    logits = preds[[c for c in preds if c.startswith(f"logit_{args.task}_")]].to_numpy()
    preds = preds.assign(pred=logits.argmax(1), correct=logits.argmax(1) == preds[f"y_{args.task}"])
    picks = pd.concat([preds[preds.correct].sample(min(len(preds), args.n * 2 // 3), random_state=0),
                       preds[~preds.correct].sample(min((~preds.correct).sum(), args.n // 3), random_state=0)])

    table = build_multitask_table(cfg["data_dir"], tuple(cfg["tasks"])).set_index("row_id")
    image_root = find_image_root(cfg["data_dir"])
    names = label_maps([args.task])[args.task]
    out_dir = run_dir / "explanations"
    out_dir.mkdir(exist_ok=True)
    summary = []
    for i, row in enumerate(picks.itertuples()):
        post = table.loc[row.row_id]
        try:
            image = Image.open(image_root / post["image"]).convert("RGB")
        except OSError:
            image = None
        exp = explainer.explain(post["text"], image, args.task)
        tag = "ok" if row.correct else "wrong"
        render(exp, post["text"], image, out_dir / f"{i:02d}_{tag}.png", true_label=names[int(getattr(row, f"y_{args.task}"))])
        summary.append({"file": f"{i:02d}_{tag}.png", "row_id": row.row_id, "predicted": exp["target"],
                        "true": names[int(getattr(row, f"y_{args.task}"))],
                        "top_words": sorted(exp["words"], key=lambda w: -w[1])[:5]})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(summary)} panels to {out_dir}/")


if __name__ == "__main__":
    main()
