"""
Download GLB files from TexVerse for verified garment models.
Uses metadata.json to resolve actual file paths per model.
Prefers highest available resolution (2K > 4K > 8K > 1K).

Usage:
    conda activate texverse
    python texverse/download_garments.py
"""
import json
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
GLB_DIR = HERE / "glbs"

REPO_MAP = {
    "glbs_1k": "YiboZhang2001/TexVerse-1K",
    "glbs_2k": "YiboZhang2001/TexVerse",
    "glbs_4k": "YiboZhang2001/TexVerse",
    "glbs_8k": "YiboZhang2001/TexVerse",
}

RESOLUTION_PREFERENCE = ["glbs_2k", "glbs_4k", "glbs_8k", "glbs_1k"]


def ensure_metadata():
    meta_path = HERE / "metadata.json"
    if not meta_path.exists():
        print("Downloading metadata.json...")
        hf_hub_download("YiboZhang2001/TexVerse", "metadata.json",
                        repo_type="dataset", local_dir=str(HERE))
    return meta_path


def pick_best_path(glb_paths):
    by_res = {}
    for p in glb_paths:
        for res_key in RESOLUTION_PREFERENCE:
            if res_key in p:
                by_res[res_key] = p
                break
    for res_key in RESOLUTION_PREFERENCE:
        if res_key in by_res:
            return by_res[res_key], res_key
    return glb_paths[0], "unknown"


def main():
    manifest = HERE / "verified_garments.jsonl"
    if not manifest.exists():
        print(f"ERROR: {manifest} not found.")
        sys.exit(1)

    meta_path = ensure_metadata()
    print("Loading metadata.json...")
    with open(meta_path) as f:
        metadata = json.load(f)

    with open(manifest) as f:
        records = [json.loads(line) for line in f]

    print(f"Verified garments: {len(records)}")
    GLB_DIR.mkdir(exist_ok=True)
    cache_dir = GLB_DIR / "_hf_cache"

    downloaded = 0
    skipped = 0
    failed = []
    res_counts = {}

    for i, rec in enumerate(records):
        uid = rec["model_id"]
        local_path = GLB_DIR / f"{uid}.glb"

        if local_path.exists():
            skipped += 1
            continue

        if uid not in metadata:
            failed.append((uid, "not in metadata"))
            continue

        glb_paths = metadata[uid].get("glb_paths", [])
        if not glb_paths:
            failed.append((uid, "no glb_paths"))
            continue

        repo_path, res_key = pick_best_path(glb_paths)
        repo_id = REPO_MAP.get(res_key, "YiboZhang2001/TexVerse")
        res_counts[res_key] = res_counts.get(res_key, 0) + 1

        print(f"[{i+1}/{len(records)}] {uid} ({res_key})...", end=" ", flush=True)

        try:
            tmp = hf_hub_download(
                repo_id, repo_path,
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
            shutil.copy2(tmp, local_path)
            downloaded += 1
            size_mb = local_path.stat().st_size / 1024 / 1024
            print(f"OK ({size_mb:.1f} MB)")
        except Exception as e:
            failed.append((uid, str(e)[:80]))
            print(f"FAILED")

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Failed: {len(failed)}")
    print(f"Resolution breakdown: {res_counts}")
    if failed:
        print("Failed:")
        for uid, reason in failed:
            print(f"  {uid}: {reason}")

    total = len(list(GLB_DIR.glob("*.glb")))
    print(f"Total GLBs on disk: {total}")


if __name__ == "__main__":
    main()
