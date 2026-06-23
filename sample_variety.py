"""
Sample 20 random garments from the latent space and render a combined grid image.
"""
import os, sys, time, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils_drape import load_udf, reconstruct

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
os.makedirs("output", exist_ok=True)

# Get garment type and number from CLI arguments
if len(sys.argv) < 3:
    print("Usage: python sample_variety.py <top|bottom> <num_garments>")
    sys.exit(1)

garment_type = sys.argv[1].lower()
try:
    num_garments = int(sys.argv[2])
except ValueError:
    print("Error: num_garments must be an integer")
    sys.exit(1)

if garment_type not in ["top", "bottom"]:
    print("Error: garment_type must be 'top' or 'bottom'")
    sys.exit(1)

if num_garments <= 0:
    print("Error: num_garments must be positive")
    sys.exit(1)

# Load appropriate decoder and latent codes
codes_file = f"{garment_type}_codes.pt"
udf_file = f"{garment_type}_udf.pt"
coords_encoder, latent_codes, decoder = load_udf(
    "checkpoints", codes_file, udf_file, device
)
n_latents = latent_codes.shape[0]
print(f"loaded {n_latents} {garment_type} latents, dim {latent_codes.shape[1]}")

# Warm-up
reconstruct(coords_encoder, decoder, latent_codes[[0]],
            udf_max_dist=0.1, resolution=64, differentiable=False)
if device.type == "cuda":
    torch.cuda.synchronize()

indices = np.random.choice(n_latents, size=num_garments, replace=False)
print(f"sampling {num_garments} indices: {indices.tolist()}")

resolution = 128
cols = int(np.ceil(np.sqrt(num_garments)))
rows = int(np.ceil(num_garments / cols))
fig, axes = plt.subplots(rows, cols, figsize=(20, 16),
                         subplot_kw={"projection": "3d"})

for i, idx in enumerate(indices):
    print(f"[{i+1:>2}/{num_garments}] idx={idx}", end="", flush=True)
    t0 = time.time()
    mesh, v, t = reconstruct(
        coords_encoder, decoder, latent_codes[[idx]],
        udf_max_dist=0.1, resolution=resolution, differentiable=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  verts={len(mesh.vertices):>5}  tris={len(mesh.faces):>5}  {dt:.2f}s")

    mesh.export(f"output/variety_{garment_type}_{i:02d}_idx{idx}.obj")

    ax = axes[i // cols][i % cols]
    verts = mesh.vertices
    faces = mesh.faces

    poly = Poly3DCollection(verts[faces], alpha=0.7, edgecolor="k", linewidth=0.1)
    poly.set_facecolor([0.6, 0.6, 0.9, 0.7])
    ax.add_collection3d(poly)

    # Set axis limits from mesh bounds
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_title(f"idx {idx}", fontsize=10)
    ax.axis("off")
    ax.view_init(elev=10, azim=135)

fig.suptitle(f"{num_garments} random {garment_type} garments (res={resolution})", fontsize=16)
fig.tight_layout()
out_path = f"output/variety_{garment_type}_grid.png"
fig.savefig(out_path, dpi=150)
print(f"\nwrote {out_path}")
