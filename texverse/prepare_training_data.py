"""
Prepare TexGaussian training data from verified garments:
  1. Extract text descriptions from caption.json → garment_captions.csv
  2. Create train/test split → garment_train_list.txt, garment_test_list.txt

Usage:
    conda activate texverse
    python texverse/prepare_training_data.py [--test_ratio 0.1]
"""
import json
import argparse
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = HERE / "verified_garments.jsonl"
    with open(manifest) as f:
        records = [json.loads(line) for line in f]
    uids = [r["model_id"] for r in records]
    print(f"Verified garments: {len(uids)}")

    # --- captions CSV ---
    caption_path = HERE / "caption.json"
    if not caption_path.exists():
        from huggingface_hub import hf_hub_download
        print("Downloading caption.json...")
        hf_hub_download("YiboZhang2001/TexVerse", "caption.json",
                        repo_type="dataset", local_dir=str(HERE))

    with open(caption_path) as f:
        captions = json.load(f)

    csv_path = HERE / "garment_captions.csv"
    found = 0
    with open(csv_path, "w") as f:
        for uid in uids:
            text = captions.get(uid, "")
            if not text:
                text = "a garment"
            # truncate to ~200 chars for CLIP (77 tokens ≈ 200-300 chars)
            text = text.split(".")[0].strip()
            # escape commas for CSV
            text = text.replace('"', '""')
            f.write(f'{uid},"{text}"\n')
            found += 1
    print(f"Captions written: {csv_path} ({found} entries)")

    # --- train/test split ---
    random.seed(args.seed)
    shuffled = list(uids)
    random.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * args.test_ratio))
    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]

    train_path = HERE / "garment_train_list.txt"
    test_path = HERE / "garment_test_list.txt"

    with open(train_path, "w") as f:
        for uid in train_ids:
            f.write(uid + "\n")

    with open(test_path, "w") as f:
        for uid in test_ids:
            f.write(uid + "\n")

    print(f"Train: {len(train_ids)} → {train_path}")
    print(f"Test:  {len(test_ids)} → {test_path}")


if __name__ == "__main__":
    main()
