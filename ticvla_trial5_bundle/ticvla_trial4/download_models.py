#!/usr/bin/env python3
"""Download ONLY the two model artifacts needed for TIC-VLA DynaNav evaluation.

The full TIC-VLA Hugging Face dataset is ~538 GB and is NOT needed for this
benchmark. Do not download its entire snapshot. Pin both source revisions and
validate the published SHA-256 of the released action checkpoint.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from huggingface_hub import snapshot_download, hf_hub_download

BASE_REPO = "OpenGVLab/InternVL3-1B"
BASE_REVISION = "4415a3b810e636d11dfa86b0e9ba40bb00535aa8"
CHECKPOINT_REPO = "handsomeYun/TIC-VLA"
CHECKPOINT_REVISION = "f27fa2f85aa205098080c53bb1272e987e2c1a25"
CHECKPOINT_NAME = "TIC-VLA-model.ckpt"
CHECKPOINT_SHA256 = "376263f89fad0f42c267d85655019232edc91d36e214e23424804dd4cd42e036"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(operation, label: str):
    for attempt in range(1, 4):
        try:
            return operation()
        except Exception as exc:
            if attempt >= 3:
                raise RuntimeError(f"Could not download {label} after 3 attempts: {exc}") from exc
            print(f"{label} attempt {attempt} failed: {exc}", flush=True)
            print(f"Retrying in {30 * attempt}s; already downloaded files will be reused.", flush=True)
            time.sleep(30 * attempt)


def main():
    models = Path(os.environ.get("TICVLA_HOST_MODELS", str(Path.home() / "models"))).expanduser()
    base_dir = models / "InternVL3-1B"
    ckpt_dir = models / "TIC-VLA"
    base_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    # The first run of a Hugging Face snapshot can hit 429s on shared GPU VM IPs.
    # Single-worker download and optional HF_TOKEN mitigate this.
    # Always reconcile the pinned snapshot: a prior interrupted download could
    # have produced config.json and weights but missed tokenizer/custom-code files.
    print(f"Ensuring complete {BASE_REPO}@{BASE_REVISION} in {base_dir}", flush=True)
    retry(lambda: snapshot_download(
        repo_id=BASE_REPO,
        repo_type="model",
        revision=BASE_REVISION,
        local_dir=str(base_dir),
        token=token,
        max_workers=1,
    ), "InternVL3-1B")
    # File-integrity check works even if a previous attempt was interrupted.
    for f in ("config.json", "model.safetensors"):
        if not (base_dir / f).is_file() or (base_dir / f).stat().st_size == 0:
            raise RuntimeError(f"Missing/empty base model artifact: {base_dir / f}")
    ckpt = ckpt_dir / CHECKPOINT_NAME
    if not ckpt.is_file() or sha256_file(ckpt) != CHECKPOINT_SHA256:
        if ckpt.is_file():
            print("Checkpoint SHA mismatch. Downloading verified checkpoint again.", flush=True)
            ckpt.unlink()
        retry(lambda: hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            repo_type="dataset",
            filename=CHECKPOINT_NAME,
            revision=CHECKPOINT_REVISION,
            local_dir=str(ckpt_dir),
            token=token,
        ), "TIC-VLA checkpoint")
    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA mismatch: expected {CHECKPOINT_SHA256}, got {ckpt_sha}")
    print("TIC-VLA checkpoint SHA-256: PASS", flush=True)
    manifest = {
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "base_dir": str(base_dir),
        "base_model_size_bytes": (base_dir / "model.safetensors").stat().st_size,
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_file": str(ckpt),
        "checkpoint_bytes": ckpt.stat().st_size,
        "checkpoint_sha256": ckpt_sha,
    }
    (models / "trial4_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Two required models are present and validated.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MODEL DOWNLOAD FAILED: {exc}", file=sys.stderr)
        print("If Hugging Face returned 429, export a read-only HF_TOKEN and rerun bootstrap.", file=sys.stderr)
        sys.exit(1)
