"""Train one neural model (text, image, concat, or gated) for one seed.

    python -m crisislens.train --config configs/fusion_gated.yaml --seed 42 --keep-model
    python -m crisislens.train --config configs/text_bertweet.yaml --seed 43 --wandb
    python -m crisislens.train --config configs/fusion_gated.yaml --max-train-batches 5 --epochs 1   # smoke test

Writes to <output_dir>/seed_<seed>/:
    metrics.json            dev + test metrics per task, training history, timing
    predictions_dev.csv     per-row logits (used by calibration and late fusion)
    predictions_test.csv
    model.pt                best checkpoint (only kept with --keep-model)

The checkpoint is picked by mean dev macro-F1 across tasks; test is evaluated
once, with that checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from crisislens.data import TASKS, build_multitask_table, find_image_root, leakage_report
from crisislens.metrics import evaluate, summary_line
from crisislens.models.encoders import load_image_encoder, load_text_encoder, set_trainable
from crisislens.models.fusion import FusionClassifier
from crisislens.torch_data import IGNORE, Collator, CrisisDataset, class_weights, label_maps, make_views


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def build_model(cfg: dict, encoders: tuple | None = None):
    """Return (model, tokenizer, image_processor).

    `encoders` = (text_model, tokenizer, image_model, processor) lets tests
    inject tiny random models instead of downloading BERTweet and CLIP.
    """
    m = cfg["model"]
    mode = m["mode"]
    uses_text, uses_image = mode in ("text", "concat", "gated"), mode in ("image", "concat", "gated")
    if encoders is not None:
        text, tokenizer, image, processor = encoders
    else:
        text, tokenizer = load_text_encoder(m["text_encoder"]) if uses_text else (None, None)
        image, processor = load_image_encoder(m["image_encoder"]) if uses_image else (None, None)

    set_trainable(text, not m.get("freeze_text", False))
    set_trainable(image, not m.get("freeze_image", False))
    maps = label_maps(cfg["tasks"])
    model = FusionClassifier(
        mode=mode,
        task_classes={t: len(v) for t, v in maps.items()},
        text_encoder=text if uses_text else None,
        image_encoder=image if uses_image else None,
        hidden_dim=m.get("hidden_dim", 512),
        dropout=m.get("dropout", 0.2),
        modality_dropout=m.get("modality_dropout", 0.0),
    )
    return model, (tokenizer if uses_text else None), (processor if uses_image else None)


def load_trained_model(checkpoint: str | Path, encoders: tuple | None = None, device: torch.device | None = None):
    """Rebuild a model from model.pt. Returns (model, tokenizer, image_processor, cfg)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model, tokenizer, processor = build_model(cfg, encoders)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or torch.device("cpu")).eval()
    return model, tokenizer, processor, cfg


def warmup_linear(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup to the base learning rate, then linear decay to zero."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------
# Prediction and metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run the model over a loader; return logits per task, labels, row ids, image reliance."""
    model.eval()
    logits = {t: [] for t in model.tasks}
    labels, row_ids, reliance = [], [], []
    for batch in loader:
        out = model(batch)
        for t in model.tasks:
            logits[t].append(out[t].float().cpu())
        labels.append(batch["labels"])
        row_ids += batch["row_id"]
        if "image_reliance" in out:
            reliance.append(out["image_reliance"].float().cpu())
    return {
        "logits": {t: torch.cat(v).numpy() for t, v in logits.items()},
        "labels": torch.cat(labels).numpy(),
        "row_ids": row_ids,
        "image_reliance": torch.cat(reliance).numpy() if reliance else None,
    }


def score(pred: dict, tasks: list[str]) -> dict[str, dict]:
    """Metrics per task on the rows that have a label for that task."""
    maps = label_maps(tasks)
    out = {}
    for j, task in enumerate(tasks):
        y = pred["labels"][:, j]
        keep = y != IGNORE
        if not keep.any():
            continue
        names = maps[task]
        y_true = [names[i] for i in y[keep]]
        y_pred = [names[i] for i in pred["logits"][task][keep].argmax(axis=1)]
        out[task] = evaluate(y_true, y_pred, names, TASKS[task].urgent_labels)
    return out


def predictions_frame(pred: dict, view_df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    """One row per post: ids, labels (-100 = none), and raw logits for every class."""
    df = pd.DataFrame({"row_id": pred["row_ids"]})
    df = df.merge(view_df[["row_id", "event_name"]], on="row_id", how="left")
    for j, task in enumerate(tasks):
        df[f"y_{task}"] = pred["labels"][:, j]
        for k in range(pred["logits"][task].shape[1]):
            df[f"logit_{task}_{k}"] = pred["logits"][task][:, k]
    if pred["image_reliance"] is not None:
        df["image_reliance"] = pred["image_reliance"]
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(cfg: dict, seed: int = 42, encoders: tuple | None = None, use_wandb: bool = False,
        keep_model: bool = False, device: torch.device | None = None) -> dict:
    set_seed(seed)
    device = device or pick_device()
    tasks = cfg["tasks"]
    tcfg, dcfg = cfg["train"], cfg.get("data", {})
    out_dir = Path(cfg["output_dir"]) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Data
    table = build_multitask_table(cfg["data_dir"], tuple(tasks))
    views = make_views(table, tasks, cfg.get("holdout_event"))
    leaks = leakage_report(table, tuple(tasks))
    print(f"device={device}  train={len(views['train'].df)}  dev={len(views['dev'].df)}  test={len(views['test'].df)}")
    print(f"leakage guard removed train labels: {leaks}")

    model, tokenizer, processor = build_model(cfg, encoders)
    model.to(device)
    image_root = find_image_root(cfg["data_dir"])
    if processor is not None and image_root is None:
        raise FileNotFoundError("Images not found: unpack CrisisMMD_v2.0.tar.gz into data_dir")

    collate = Collator(tokenizer, dcfg.get("max_length", 128))
    workers = dcfg.get("num_workers", 0)
    loader_kw = dict(collate_fn=collate, num_workers=workers, persistent_workers=workers > 0,
                     pin_memory=device.type == "cuda")
    datasets = {s: CrisisDataset(v, image_root, processor) for s, v in views.items()}
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=tcfg["batch_size"], shuffle=True, generator=gen, **loader_kw)
    eval_loaders = {s: DataLoader(datasets[s], batch_size=tcfg["batch_size"] * 2, **loader_kw) for s in ("dev", "test")}

    # Loss: one cross-entropy per task, class-weighted, ignoring rows without that label.
    maps = label_maps(tasks)
    losses = {}
    for j, t in enumerate(tasks):
        w = class_weights(views["train"].labels[:, j], len(maps[t])).to(device) if tcfg.get("class_weighted", True) else None
        losses[t] = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)

    # Optimizer and schedule
    optim = torch.optim.AdamW(
        model.param_groups(tcfg["lr_text"], tcfg["lr_image"], tcfg["lr_head"], tcfg["weight_decay"])
    )
    accum = tcfg.get("grad_accum", 1)
    max_batches = tcfg.get("max_train_batches")
    batches_per_epoch = min(len(train_loader), max_batches or len(train_loader))
    total_steps = max(1, math.ceil(batches_per_epoch / accum) * tcfg["epochs"])
    sched = warmup_linear(optim, int(tcfg.get("warmup_ratio", 0.1) * total_steps), total_steps)

    use_amp = tcfg.get("amp", "auto") in ("auto", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else contextlib.nullcontext

    run_wb = None
    if use_wandb:
        import wandb

        run_wb = wandb.init(project=cfg.get("wandb_project", "crisislens"), name=f"{cfg['name']}-s{seed}",
                            group=cfg["name"], config={**cfg, "seed": seed}, reinit=True)

    history, best, best_state, bad_epochs = [], -1.0, None, 0
    t_start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        running, seen = 0.0, 0
        for b, batch in enumerate(train_loader):
            if max_batches and b >= max_batches:
                break
            labels = batch["labels"].to(device)
            with autocast():
                out = model(batch)
                # Sum task losses; a task with no labels in this batch contributes 0.
                loss = sum(
                    losses[t](out[t].float(), labels[:, j]) for j, t in enumerate(tasks) if (labels[:, j] != IGNORE).any()
                )
            if not torch.is_tensor(loss):  # no labels at all in this batch
                continue
            loss = loss / accum
            (scaler.scale(loss) if scaler else loss).backward()
            if (b + 1) % accum == 0 or b + 1 == batches_per_epoch:
                if scaler:
                    scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("max_grad_norm", 1.0))
                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            running += loss.item() * accum
            seen += 1
            if run_wb and b % 20 == 0:
                run_wb.log({"train/loss": loss.item() * accum, "lr": sched.get_last_lr()[-1]})

        dev = score(predict(model, eval_loaders["dev"], device), tasks)
        dev_mean = float(np.mean([m["macro_f1"] for m in dev.values()]))
        row = {"epoch": epoch, "train_loss": running / max(1, seen), "dev_mean_macro_f1": dev_mean,
               **{f"dev_{t}_macro_f1": m["macro_f1"] for t, m in dev.items()}}
        history.append(row)
        print(f"epoch {epoch}  loss={row['train_loss']:.4f}  dev mean macroF1={dev_mean:.4f}  "
              + "  ".join(f"{t}={m['macro_f1']:.3f}" for t, m in dev.items()))
        if run_wb:
            run_wb.log(row)

        if dev_mean > best:
            best, bad_epochs = dev_mean, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg.get("patience", 2):
                print(f"early stop: no dev improvement for {bad_epochs} epochs")
                break
    train_seconds = time.time() - t_start

    # Final evaluation with the best checkpoint.
    model.load_state_dict(best_state)
    results = {"name": cfg["name"], "mode": cfg["model"]["mode"], "seed": seed, "device": str(device),
               "holdout_event": cfg.get("holdout_event"), "best_dev_mean_macro_f1": best,
               "history": history, "train_seconds": round(train_seconds, 1), "leakage_removed": leaks,
               "n_params": sum(p.numel() for p in model.parameters()),
               "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    for split in ("dev", "test"):
        pred = predict(model, eval_loaders[split], device)
        results[split] = score(pred, tasks)
        predictions_frame(pred, views[split].df, tasks).to_csv(out_dir / f"predictions_{split}.csv", index=False)
        for t, m in results[split].items():
            print(summary_line(f"{cfg['name']} {t}", split, m))
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    if keep_model:
        torch.save({"state_dict": best_state, "config": cfg, "label_maps": maps, "seed": seed}, out_dir / "model.pt")
    if run_wb:
        for t, m in results["test"].items():
            run_wb.summary[f"test/{t}/macro_f1"] = m["macro_f1"]
        run_wb.finish()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--max-train-batches", type=int)
    ap.add_argument("--keep-model", action="store_true", help="Save the best checkpoint to model.pt")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "data_dir": args.data_dir, "output_dir": args.output_dir, "train.epochs": args.epochs,
        "train.batch_size": args.batch_size, "data.num_workers": args.num_workers,
        "train.max_train_batches": args.max_train_batches,
    })
    run(cfg, seed=args.seed, use_wandb=args.wandb, keep_model=args.keep_model)


if __name__ == "__main__":
    main()
