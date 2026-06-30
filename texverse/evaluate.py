"""
Evaluate finetuned TexGaussian on garment test set.
Computes FID, KID, CLIP score, LPIPS, PSNR, SSIM, and garment-specific metrics.

Usage:
    conda activate texgaussian
    python texverse/evaluate.py \
        --pred_dir texverse/workspace/<experiment>/eval_pred_images \
        --gt_dir texverse/renders \
        --testlist texverse/garment_test_list.txt \
        --captions texverse/garment_captions.csv \
        [--num_eval_views 20]
"""
import argparse
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from collections import defaultdict

HERE = Path(__file__).resolve().parent


def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def load_image_tensor(path, size=512):
    arr = load_image(path, size)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# --- Metrics ---

def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(pred, gt, window_size=11):
    """Simplified SSIM on grayscale."""
    from scipy.ndimage import uniform_filter
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred_gray = 0.2989 * pred[..., 0] + 0.587 * pred[..., 1] + 0.114 * pred[..., 2]
    gt_gray = 0.2989 * gt[..., 0] + 0.587 * gt[..., 1] + 0.114 * gt[..., 2]

    mu_x = uniform_filter(pred_gray, window_size)
    mu_y = uniform_filter(gt_gray, window_size)
    sigma_x2 = uniform_filter(pred_gray ** 2, window_size) - mu_x ** 2
    sigma_y2 = uniform_filter(gt_gray ** 2, window_size) - mu_y ** 2
    sigma_xy = uniform_filter(pred_gray * gt_gray, window_size) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean())


class LPIPSMetric:
    def __init__(self, device="cuda"):
        import lpips
        self.fn = lpips.LPIPS(net="alex").to(device)
        self.device = device

    def __call__(self, pred_tensor, gt_tensor):
        pred = pred_tensor.to(self.device) * 2 - 1
        gt = gt_tensor.to(self.device) * 2 - 1
        with torch.no_grad():
            return self.fn(pred, gt).item()


class CLIPScorer:
    def __init__(self, device="cuda"):
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        except ImportError:
            import clip
            self.model, self.preprocess = clip.load("ViT-L/14", device=device)
            self.tokenizer = clip.tokenize
        self.model = self.model.to(device).eval()
        self.device = device

    def image_features(self, img_tensor):
        img = F.interpolate(img_tensor, (224, 224), mode="bilinear", align_corners=False)
        img = img.to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def text_features(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def clip_score(self, img_tensor, text):
        img_feat = self.image_features(img_tensor)
        txt_feat = self.text_features(text)
        return (img_feat @ txt_feat.T).item()

    def image_similarity(self, img1_tensor, img2_tensor):
        f1 = self.image_features(img1_tensor)
        f2 = self.image_features(img2_tensor)
        return (f1 @ f2.T).item()


def compute_fid_kid(pred_images, gt_images):
    """Compute FID and KID using torchmetrics if available, else skip."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        print("  torchmetrics not available, skipping FID/KID")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(feature=2048, normalize=True, subset_size=min(50, len(pred_images))).to(device)

    for img in gt_images:
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        fid.update(t, real=True)
        kid.update(t, real=True)

    for img in pred_images:
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        fid.update(t, real=False)
        kid.update(t, real=False)

    fid_val = fid.compute().item()
    kid_mean, kid_std = kid.compute()

    return fid_val, (kid_mean.item(), kid_std.item())


# --- Garment-specific metrics ---

def compute_mr_plausibility(mr_dir, uids, num_views=20):
    """Check that metallic ≈ 0 and roughness is reasonable for cloth."""
    metallics = []
    roughnesses = []

    for uid in uids:
        for vid in range(num_views):
            mr_path = Path(mr_dir) / uid / f"{vid}_mr.png"
            if not mr_path.exists():
                continue
            img = load_image(mr_path)
            mask = np.any(img > 0.01, axis=-1)
            if mask.sum() == 0:
                continue
            roughnesses.append(img[mask, 1].mean())
            metallics.append(img[mask, 2].mean())

    return {
        "mean_metallic": float(np.mean(metallics)) if metallics else None,
        "std_metallic": float(np.std(metallics)) if metallics else None,
        "mean_roughness": float(np.mean(roughnesses)) if roughnesses else None,
        "std_roughness": float(np.std(roughnesses)) if roughnesses else None,
    }


def compute_multiview_clip_consistency(clip_scorer, pred_dir, uids, num_views=8):
    """Pairwise CLIP similarity across evenly-spaced views."""
    consistencies = []

    for uid in uids:
        view_indices = np.linspace(0, 63, num_views, dtype=int)
        feats = []
        for vid in view_indices:
            img_path = Path(pred_dir) / uid / f"{vid}.png"
            if not img_path.exists():
                continue
            img_t = load_image_tensor(img_path)
            feats.append(clip_scorer.image_features(img_t))

        if len(feats) < 2:
            continue

        feats = torch.cat(feats, dim=0)
        sim_matrix = feats @ feats.T
        mask = ~torch.eye(len(feats), dtype=bool, device=sim_matrix.device)
        pairwise = sim_matrix[mask]
        consistencies.append(pairwise.mean().item())

    return {
        "mean_consistency": float(np.mean(consistencies)) if consistencies else None,
        "std_consistency": float(np.std(consistencies)) if consistencies else None,
    }


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted renders")
    parser.add_argument("--gt_dir", required=True, help="Directory with GT renders (from render_views.py)")
    parser.add_argument("--testlist", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--num_eval_views", type=int, default=20)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    with open(args.testlist) as f:
        test_uids = [line.strip() for line in f if line.strip()]
    print(f"Test models: {len(test_uids)}")

    captions = {}
    with open(args.captions) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                captions[row[0]] = row[1]

    view_indices = np.linspace(0, 63, args.num_eval_views, dtype=int)

    # collect all images
    pred_images = []
    gt_images = []
    per_model = defaultdict(lambda: {"psnr": [], "ssim": []})

    print("Loading images...")
    for uid in test_uids:
        for vid in view_indices:
            pred_path = Path(args.pred_dir) / uid / f"{vid}.png"
            gt_path = Path(args.gt_dir) / uid / f"{vid}.png"
            if not pred_path.exists() or not gt_path.exists():
                continue
            pred = load_image(pred_path)
            gt = load_image(gt_path)
            pred_images.append(pred)
            gt_images.append(gt)
            per_model[uid]["psnr"].append(compute_psnr(pred, gt))
            per_model[uid]["ssim"].append(compute_ssim(pred, gt))

    print(f"Loaded {len(pred_images)} image pairs across {len(per_model)} models")

    results = {}

    # PSNR / SSIM
    all_psnr = [v for m in per_model.values() for v in m["psnr"]]
    all_ssim = [v for m in per_model.values() for v in m["ssim"]]
    results["psnr"] = {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr))}
    results["ssim"] = {"mean": float(np.mean(all_ssim)), "std": float(np.std(all_ssim))}
    print(f"PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f}")
    print(f"SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")

    # FID / KID
    print("Computing FID/KID...")
    fid_val, kid_val = compute_fid_kid(pred_images, gt_images)
    results["fid"] = fid_val
    results["kid"] = {"mean": kid_val[0], "std": kid_val[1]} if kid_val else None
    if fid_val is not None:
        print(f"FID: {fid_val:.2f}")
    if kid_val is not None:
        print(f"KID: {kid_val[0]:.4f} ± {kid_val[1]:.4f}")

    # LPIPS
    print("Computing LPIPS...")
    try:
        lpips_metric = LPIPSMetric()
        lpips_vals = []
        for uid in test_uids:
            for vid in view_indices:
                pred_path = Path(args.pred_dir) / uid / f"{vid}.png"
                gt_path = Path(args.gt_dir) / uid / f"{vid}.png"
                if not pred_path.exists() or not gt_path.exists():
                    continue
                lpips_vals.append(lpips_metric(load_image_tensor(pred_path), load_image_tensor(gt_path)))
        results["lpips"] = {"mean": float(np.mean(lpips_vals)), "std": float(np.std(lpips_vals))}
        print(f"LPIPS: {results['lpips']['mean']:.4f} ± {results['lpips']['std']:.4f}")
    except ImportError:
        print("  lpips not available, skipping")
        results["lpips"] = None

    # CLIP score
    print("Computing CLIP scores...")
    try:
        clip_scorer = CLIPScorer()
        clip_scores = []
        for uid in test_uids:
            text = captions.get(uid, "a garment")
            for vid in view_indices[:8]:
                pred_path = Path(args.pred_dir) / uid / f"{vid}.png"
                if not pred_path.exists():
                    continue
                score = clip_scorer.clip_score(load_image_tensor(pred_path), text)
                clip_scores.append(score)
        results["clip_score"] = {"mean": float(np.mean(clip_scores)), "std": float(np.std(clip_scores))}
        print(f"CLIP Score: {results['clip_score']['mean']:.4f} ± {results['clip_score']['std']:.4f}")

        # multi-view consistency
        print("Computing multi-view CLIP consistency...")
        consistency = compute_multiview_clip_consistency(clip_scorer, args.pred_dir, test_uids)
        results["multiview_clip_consistency"] = consistency
        if consistency["mean_consistency"]:
            print(f"Multi-view CLIP consistency: {consistency['mean_consistency']:.4f} ± {consistency['std_consistency']:.4f}")
    except (ImportError, Exception) as e:
        print(f"  CLIP scoring failed: {e}")
        results["clip_score"] = None
        results["multiview_clip_consistency"] = None

    # material plausibility
    print("Computing material plausibility...")
    mr_plaus = compute_mr_plausibility(args.gt_dir, test_uids, args.num_eval_views)
    results["material_plausibility_gt"] = mr_plaus
    print(f"GT metallic: {mr_plaus['mean_metallic']:.3f} ± {mr_plaus['std_metallic']:.3f}")
    print(f"GT roughness: {mr_plaus['mean_roughness']:.3f} ± {mr_plaus['std_roughness']:.3f}")

    # save
    output_path = args.output or str(HERE / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # summary table
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"{'Metric':<30} {'Value':>15}")
    print(f"{'-'*50}")
    for key in ["psnr", "ssim", "fid", "lpips", "clip_score"]:
        val = results.get(key)
        if val is None:
            print(f"{key:<30} {'N/A':>15}")
        elif isinstance(val, dict):
            print(f"{key:<30} {val['mean']:>10.4f} ± {val.get('std', 0):>.4f}")
        else:
            print(f"{key:<30} {val:>15.4f}")
    if results.get("kid") and isinstance(results["kid"], dict):
        print(f"{'kid':<30} {results['kid']['mean']:>10.4f} ± {results['kid']['std']:>.4f}")
    if results.get("multiview_clip_consistency") and results["multiview_clip_consistency"]["mean_consistency"]:
        c = results["multiview_clip_consistency"]
        print(f"{'mv_clip_consistency':<30} {c['mean_consistency']:>10.4f} ± {c['std_consistency']:>.4f}")


if __name__ == "__main__":
    main()
