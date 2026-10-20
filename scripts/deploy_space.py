"""Assemble the demo Space folder and (optionally) upload it to Hugging Face.

    python scripts/deploy_space.py --dry-run                       # just build build/space/ to inspect
    HF_TOKEN=hf_write... HF_READ_TOKEN=hf_read... \
      python scripts/deploy_space.py --space <user>/crisislens --model-repo <user>/crisislens-model

The Space gets only what the app needs: the app, the torch-free inference
module, the simulated replay with its public-domain photos and credits, and
the results table. Never the model weights (those load from the private model
repo at runtime) and never any CrisisMMD data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "space"


def assemble(out: Path = BUILD, scenario: Path = ROOT / "demo" / "scenario") -> Path:
    if not (scenario / "replay.json").exists():
        raise SystemExit("demo/scenario/replay.json missing: run scripts/build_scenario.py build first")
    replay = json.loads((scenario / "replay.json").read_text())
    if not replay.get("scenario", {}).get("simulated"):
        raise SystemExit("replay.json is not marked simulated; refusing to publish it")

    if out.exists():
        shutil.rmtree(out)
    (out / "crisislens").mkdir(parents=True)
    for name in ("streamlit_app.py", "Dockerfile", "requirements.txt", "README.md"):
        shutil.copy(ROOT / "app" / name, out / name)
    for name in ("inference.py", "text.py"):
        shutil.copy(ROOT / "src" / "crisislens" / name, out / "crisislens" / name)
    (out / "crisislens" / "__init__.py").write_text("")

    (out / "scenario").mkdir()
    shutil.copy(scenario / "replay.json", out / "scenario" / "replay.json")
    shutil.copy(scenario / "sources.csv", out / "scenario" / "sources.csv")
    used = {p["image"] for p in replay["posts"] if p.get("image")}
    for rel in sorted(used):
        (out / "scenario" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(scenario / rel, out / "scenario" / rel)

    results = ROOT / "outputs" / "results.md"
    if results.exists():
        shutil.copy(results, out / "results.md")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"assembled {out} ({size:.1f} MB, {len(used)} photos)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", help="e.g. kshitid/crisislens")
    ap.add_argument("--model-repo", help="private model repo the Space loads, e.g. kshitid/crisislens-model")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folder = assemble()
    if args.dry_run:
        return
    if not args.space or not args.model_repo:
        raise SystemExit("--space and --model-repo are required to upload")

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
    read_token = os.environ.get("HF_READ_TOKEN")
    if read_token:
        api.add_space_secret(args.space, "HF_TOKEN", read_token)
    else:
        print("[note] HF_READ_TOKEN not set: add a secret named HF_TOKEN (read token) in the Space settings")
    api.upload_folder(folder_path=str(folder), repo_id=args.space, repo_type="space",
                      commit_message="Deploy CrisisLens demo", delete_patterns=["*"])
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
