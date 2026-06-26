"""
Filter TexVerse annotations to produce a manifest of garment models with PBR materials.
Outputs JSONL manifest + summary counts. Does NOT download any assets.

Usage:
    conda activate texverse
    python texverse/build_garment_manifest.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- keyword lists ---

# High-confidence garment words (unambiguous — never mean something else)
STRONG_GARMENT_KW = [
    "shirt", "t-shirt", "tshirt", "blouse", "jacket", "coat", "hoodie",
    "sweater", "jumper", "cardigan", "tank top", "tunic", "blazer", "parka",
    "windbreaker", "overcoat", "raincoat", "anorak",
    "trousers", "jeans", "skirt", "leggings",
    "dress", "gown", "jumpsuit", "romper", "overalls", "dungarees", "kimono",
    "garment",
]

# Ambiguous keywords that need garment context in the annotation
# "top" excluded — too ambiguous (spatial "top of", "on top", kettlebell handles, etc.)
# Real garment tops are caught by shirt/blouse/jacket/hoodie/etc.
WEAK_GARMENT_KW = [
    "vest", "polo", "cape", "poncho",
    "pants", "shorts",
    "robe", "suit", "uniform", "outfit", "costume",
    "clothing", "apparel", "attire",
]

ALL_GARMENT_KW = STRONG_GARMENT_KW + WEAK_GARMENT_KW


GARMENT_CONTEXT_RE = re.compile(
    r'\b(wear|worn|wearing|cloth|fabric|sleeve|collar|button|zip|zipper|'
    r'knit|knitted|cotton|silk|linen|wool|polyester|neckline|hem|hemline|'
    r'cuff|pocket|fashion|casual|formal|garment|apparel|wardrobe|'
    r'sew|stitch|tailored|fitted|loose|baggy|pleated|flared|'
    r'waistband|lapel|seam|bodice|torso)\b', re.I
)

# non-garment object patterns
NON_GARMENT_OBJECT_RE = re.compile(
    r'\b(robot|car|vehicle|furniture|building|architect|weapon|sword|gun|'
    r'rifle|pistol|tank\b(?!\s*top)|ship|boat|airplane|aircraft|'
    r'house|castle|tower|bridge|machine|engine|motor|drone|'
    r'barrel|bottle|can|jar|box|crate|chest|'
    r'table|chair|desk|shelf|cabinet|lamp|'
    r'plant|tree|flower|rock|stone|terrain|landscape|'
    r'food|cake|pizza|burger|fruit|'
    r'animal|dog|cat|horse|bird|fish|dragon|monster|'
    r'phone|computer|laptop|keyboard|screen|monitor|'
    r'guitar|piano|drum|instrument|'
    r'ball|trophy|medal|coin|ring\b(?!\s)|crown|'
    r'door|window|gate|fence|wall|floor|ceiling|'
    r'pipe|wire|cable|hose|chain|'
    r'skateboard|bicycle|motorcycle|'
    r'spaceship|spacecraft|satellite|'
    r'card|ticket|poster|book|'
    r'mask\b(?!ed)|shield|armor\b(?!ed)|helmet)\b', re.I
)

# style patterns — not useful as realistic PBR training data
STYLISED_RE = re.compile(
    r'\b(low[- ]?poly|lowpoly|stylized|stylised|cartoon|cartoonish|'
    r'chibi|pixel|pixelated|voxel|blocky|minecraft|lego|'
    r'hand[- ]?painted|cel[- ]?shad|toon|anime|manga)\b', re.I
)

# character/figure patterns — model depicts a character, not a standalone garment
CHARACTER_RE = re.compile(
    r'\b(character|figurine|figure|person|people|creature|warrior|soldier|'
    r'knight|zombie|man\b|woman\b|boy\b|girl\b|child|human|'
    r'hero|villain|wizard|witch|elf|dwarf|orc|goblin|'
    r'pirate|ninja|samurai|gladiator|'
    r'player|athlete|dancer|'
    r'bust\b|mannequin|doll|puppet|'
    r'avatar|humanoid|android|cyborg)\b', re.I
)

# scene/environment patterns
SCENE_RE = re.compile(
    r'\b(scene|room|building|diorama|display|layout|interior|exterior|'
    r'environment|level|stage|arena|'
    r'bedroom|bathroom|kitchen|living\s*room|office|'
    r'garden|park|forest|city|street|alley)\b', re.I
)

TOPS = {"shirt", "t-shirt", "tshirt", "blouse", "jacket", "coat",
        "hoodie", "sweater", "jumper", "cardigan", "vest", "tank top",
        "polo", "tunic", "cape", "poncho", "blazer", "parka", "windbreaker",
        "overcoat", "raincoat", "anorak"}
BOTTOMS = {"trousers", "pants", "jeans", "shorts", "skirt", "leggings"}
ONEPIECE = {"dress", "gown", "robe", "suit", "jumpsuit", "romper", "overalls",
            "dungarees", "kimono", "uniform", "outfit", "costume"}


def load_ids(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def match_keywords(text, keywords):
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'(?:s|es|ed|ing)?\b'
        if re.search(pattern, text_lower):
            matched.append(kw)
    return matched


def classify_garment(keywords):
    cats = set()
    for kw in keywords:
        if kw in TOPS:
            cats.add("top")
        elif kw in BOTTOMS:
            cats.add("bottom")
        elif kw in ONEPIECE:
            cats.add("one-piece")
    if not cats:
        cats.add("generic")
    return sorted(cats)


def infer_model_type(text):
    """Classify whether the 3D model is a standalone garment vs character/scene."""
    first_sentence = text.split(".")[0].lower() if text else ""

    if SCENE_RE.search(first_sentence):
        return "scene"
    if CHARACTER_RE.search(first_sentence):
        return "character"

    garment_lead = re.match(
        r'^a\s+(?:3d\s+(?:model\s+of\s+)?)?'
        r'(?:(?:modern|classic|casual|formal|elegant|stylish|vintage|retro|'
        r'military|medieval|futuristic|simple|plain|colorful|dark|light|'
        r'white|black|red|blue|green|brown|gray|grey|pink|purple|yellow|'
        r'orange|gold|silver|leather|denim|cotton|silk|wool|knit|knitted|'
        r'quilted|striped|plaid|checkered|patterned|textured|long|short|'
        r'sleeveless|hooded|zipped|buttoned|fitted|loose|baggy|slim|'
        r'men\'?s|women\'?s|male|female|unisex)\s+)*'
        r'(' + '|'.join(re.escape(kw) for kw in ALL_GARMENT_KW) + r')\b',
        first_sentence
    )
    if garment_lead:
        return "standalone"

    if CHARACTER_RE.search(text):
        return "character"
    if SCENE_RE.search(text):
        return "scene"

    return "unclear"


def snippet(text, max_len=300):
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


REPO_ID = "YiboZhang2001/TexVerse"
METADATA_FILES = ["TexVerse_pbr_id_list.txt", "caption.json"]


def ensure_metadata():
    missing = [f for f in METADATA_FILES if not (HERE / f).exists()]
    if not missing:
        return
    from huggingface_hub import hf_hub_download
    for f in missing:
        print(f"Downloading {f} from {REPO_ID}...")
        hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=str(HERE))
        print(f"  Done.")


def main():
    ensure_metadata()

    print("Loading PBR ID list...")
    pbr_ids = load_ids(HERE / "TexVerse_pbr_id_list.txt")
    print(f"  PBR models: {len(pbr_ids):,}")

    print("Loading captions (this may take a moment — 452 MB)...")
    with open(HERE / "caption.json") as f:
        captions = json.load(f)
    print(f"  Captions loaded: {len(captions):,}")

    pbr_with_caption = pbr_ids & set(captions.keys())
    print(f"  PBR models with captions: {len(pbr_with_caption):,}")

    # --- filter ---
    garment_manifest = []
    stats = {"excluded_non_garment_obj": 0, "excluded_weak_no_context": 0,
             "excluded_stylised": 0}

    for model_id in sorted(pbr_with_caption):
        text = captions[model_id]

        strong_matches = match_keywords(text, STRONG_GARMENT_KW)
        weak_matches = match_keywords(text, WEAK_GARMENT_KW)
        garment_matches = strong_matches + weak_matches

        if not garment_matches:
            continue

        if STYLISED_RE.search(text):
            stats["excluded_stylised"] += 1
            continue

        # weak-only matches require garment context in the text
        if not strong_matches:
            if not GARMENT_CONTEXT_RE.search(text):
                stats["excluded_weak_no_context"] += 1
                continue

        # if the first sentence describes a non-garment object, skip
        first_sentence = text.split(".")[0]
        if NON_GARMENT_OBJECT_RE.search(first_sentence):
            first_strong = match_keywords(first_sentence, STRONG_GARMENT_KW)
            if not first_strong:
                stats["excluded_non_garment_obj"] += 1
                continue

        model_type = infer_model_type(text)

        garment_manifest.append({
            "model_id": model_id,
            "pbr": True,
            "matched_keywords": garment_matches,
            "category": classify_garment(garment_matches),
            "model_type": model_type,
            "annotation_snippet": snippet(text),
        })

    # --- output ---
    manifest_path = HERE / "garment_manifest.jsonl"

    with open(manifest_path, "w") as f:
        for rec in garment_manifest:
            f.write(json.dumps(rec) + "\n")

    # --- summary ---
    print(f"\n{'='*60}")
    print(f"GARMENT MANIFEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total PBR models:              {len(pbr_ids):>8,}")
    print(f"PBR models with captions:      {len(pbr_with_caption):>8,}")
    print(f"Garment matches (PBR):         {len(garment_manifest):>8,}")
    print(f"Excluded (stylised/low-poly):  {stats['excluded_stylised']:>8,}")
    print(f"Excluded (weak, no context):   {stats['excluded_weak_no_context']:>8,}")
    print(f"Excluded (non-garment obj):    {stats['excluded_non_garment_obj']:>8,}")

    # model_type breakdown
    type_counts = {}
    for rec in garment_manifest:
        t = rec["model_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nModel type breakdown:")
    for t in ["standalone", "character", "scene", "unclear"]:
        if t in type_counts:
            print(f"  {t:>12}: {type_counts[t]:>6,}")

    # category breakdown
    cat_counts = {}
    for rec in garment_manifest:
        for cat in rec["category"]:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"\nCategory breakdown (garments):")
    for cat in ["top", "bottom", "one-piece", "generic"]:
        if cat in cat_counts:
            print(f"  {cat:>12}: {cat_counts[cat]:>6,}")

    # standalone-only category breakdown
    standalone_cats = {}
    for rec in garment_manifest:
        if rec["model_type"] == "standalone":
            for cat in rec["category"]:
                standalone_cats[cat] = standalone_cats.get(cat, 0) + 1
    print(f"\nCategory breakdown (standalone garments only):")
    for cat in ["top", "bottom", "one-piece", "generic"]:
        if cat in standalone_cats:
            print(f"  {cat:>12}: {standalone_cats[cat]:>6,}")

    # top matched keywords
    kw_counts = {}
    for rec in garment_manifest:
        for kw in rec["matched_keywords"]:
            kw_counts[kw] = kw_counts.get(kw, 0) + 1
    print(f"\nTop 20 matched keywords:")
    for kw, count in sorted(kw_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {kw:>20}: {count:>6,}")

    print(f"\nManifest written to: {manifest_path}")

    print(f"\n--- Sample STANDALONE garment records ---")
    standalone = [r for r in garment_manifest if r["model_type"] == "standalone"]
    for rec in standalone[:8]:
        print(f"  {rec['model_id']}  [{', '.join(rec['category'])}]  kw={rec['matched_keywords']}")
        print(f"    {rec['annotation_snippet'][:150]}")

    print(f"\n--- Sample CHARACTER garment records ---")
    characters = [r for r in garment_manifest if r["model_type"] == "character"]
    for rec in characters[:5]:
        print(f"  {rec['model_id']}  [{', '.join(rec['category'])}]  kw={rec['matched_keywords']}")
        print(f"    {rec['annotation_snippet'][:150]}")

    print(f"\n--- Sample UNCLEAR garment records ---")
    unclear = [r for r in garment_manifest if r["model_type"] == "unclear"]
    for rec in unclear[:5]:
        print(f"  {rec['model_id']}  [{', '.join(rec['category'])}]  kw={rec['matched_keywords']}")
        print(f"    {rec['annotation_snippet'][:150]}")


if __name__ == "__main__":
    main()
