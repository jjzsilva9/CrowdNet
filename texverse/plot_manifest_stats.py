"""
Generate distribution plots for the TexVerse garment manifest.

Usage:
    conda activate texverse
    python texverse/plot_manifest_stats.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

COLORS = {
    "standalone": "#2ecc71",
    "character": "#3498db",
    "unclear": "#f39c12",
    "scene": "#e74c3c",
}
CAT_COLORS = {
    "top": "#3498db",
    "bottom": "#e74c3c",
    "one-piece": "#2ecc71",
    "generic": "#95a5a6",
}


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def fig1_model_type_pie(records):
    counts = Counter(r["model_type"] for r in records)
    labels = ["standalone", "character", "unclear", "scene"]
    sizes = [counts.get(l, 0) for l in labels]
    colors = [COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct=lambda p: f"{p:.1f}%\n({int(p*sum(sizes)/100):,})",
        colors=colors, startangle=90, pctdistance=0.75,
        textprops={"fontsize": 11}
    )
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    ax.set_title("TexVerse Garment Manifest — Model Type Distribution\n"
                 f"(n = {len(records):,} PBR garment matches)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "model_type_pie.png", dpi=150)
    plt.close(fig)


def fig2_category_by_type(records):
    types = ["standalone", "character", "unclear", "scene"]
    cats = ["top", "bottom", "one-piece", "generic"]

    data = {t: Counter() for t in types}
    for r in records:
        t = r["model_type"]
        for c in r["category"]:
            data[t][c] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(types))
    width = 0.2
    for i, cat in enumerate(cats):
        vals = [data[t].get(cat, 0) for t in types]
        bars = ax.bar([xi + i * width for xi in x], vals, width,
                      label=cat, color=CAT_COLORS[cat])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                        f"{v:,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(types, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Garment Category × Model Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    fig.savefig(OUT / "category_by_type.png", dpi=150)
    plt.close(fig)


def fig3_keyword_bars(records):
    kw_counts = Counter()
    for r in records:
        for kw in r["matched_keywords"]:
            kw_counts[kw] += 1

    top_kw = kw_counts.most_common(20)
    labels = [kw for kw, _ in top_kw]
    values = [v for _, v in top_kw]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], values[::-1], color="#3498db", edgecolor="white")
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Top 20 Matched Keywords (garment manifest)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    fig.savefig(OUT / "keyword_bars.png", dpi=150)
    plt.close(fig)


def fig4_standalone_breakdown(records):
    standalone = [r for r in records if r["model_type"] == "standalone"]
    kw_counts = Counter()
    for r in standalone:
        for kw in r["matched_keywords"]:
            kw_counts[kw] += 1

    top_kw = kw_counts.most_common(15)
    labels = [kw for kw, _ in top_kw]
    values = [v for _, v in top_kw]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.5]})

    # pie: category
    cat_counts = Counter()
    for r in standalone:
        for c in r["category"]:
            cat_counts[c] += 1
    pie_labels = [c for c in ["top", "bottom", "one-piece", "generic"] if c in cat_counts]
    pie_vals = [cat_counts[c] for c in pie_labels]
    pie_colors = [CAT_COLORS[c] for c in pie_labels]
    ax1.pie(pie_vals, labels=pie_labels,
            autopct=lambda p: f"{int(p*sum(pie_vals)/100)}", colors=pie_colors,
            textprops={"fontsize": 11})
    ax1.set_title(f"Standalone garments\ncategory split (n={len(standalone)})", fontsize=12, fontweight="bold")

    # bar: keywords
    bars = ax2.barh(labels[::-1], values[::-1], color="#2ecc71", edgecolor="white")
    for bar, v in zip(bars, values[::-1]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(v), va="center", fontsize=9)
    ax2.set_xlabel("Count", fontsize=11)
    ax2.set_title("Standalone garments — keywords", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "standalone_breakdown.png", dpi=150)
    plt.close(fig)


def fig5_funnel(pbr_total, captioned, garment, standalone):
    stages = [
        "PBR models\nin TexVerse",
        "PBR with\ncaptions",
        "Garment match\n(non-stylised)",
        "Standalone\ngarments",
    ]
    values = [pbr_total, captioned, garment, standalone]
    colors = ["#95a5a6", "#3498db", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1],
                   edgecolor="white", height=0.6)
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Filtering Funnel — TexVerse → Garment Subset", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    fig.savefig(OUT / "filtering_funnel.png", dpi=150)
    plt.close(fig)


def main():
    records = load_manifest(HERE / "garment_manifest.jsonl")

    print(f"Loaded {len(records):,} garment records")

    fig1_model_type_pie(records)
    fig2_category_by_type(records)
    fig3_keyword_bars(records)
    fig4_standalone_breakdown(records)
    fig5_funnel(158_518, 158_134, len(records), sum(1 for r in records if r["model_type"] == "standalone"))

    print(f"Plots saved to {OUT}/")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
