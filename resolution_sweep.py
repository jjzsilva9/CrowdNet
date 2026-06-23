"""
Resolution sweep: measure generation cost vs MeshUDF grid resolution.
Outputs CSV + a two-panel figure (time & tri count vs resolution).
"""
import os, time, csv, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils_drape import load_udf, reconstruct

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
os.makedirs("output", exist_ok=True)

coords_encoder, latent_codes_top, decoder_top = load_udf(
    "checkpoints", "top_codes.pt", "top_udf.pt", device
)
print(f"loaded {latent_codes_top.shape[0]} top latents, dim {latent_codes_top.shape[1]}")

top_idx = 208
resolutions = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]

# Warm-up run (first CUDA call includes context-init overhead)
print("warm-up run (res=64)...")
reconstruct(
    coords_encoder, decoder_top, latent_codes_top[[top_idx]],
    udf_max_dist=0.1, resolution=64, differentiable=False,
    use_fast_grid_filler=False,
)
if device.type == "cuda":
    torch.cuda.synchronize()
print("warm-up done\n")

rows = []
print(f"{'res':>6}  {'verts':>8}  {'tris':>8}  {'time_s':>8}")
print("-" * 38)

for res in resolutions:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    mesh, v, t = reconstruct(
        coords_encoder, decoder_top, latent_codes_top[[top_idx]],
        udf_max_dist=0.1, resolution=res, differentiable=False,
        use_fast_grid_filler=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    nv, nf = len(mesh.vertices), len(mesh.faces)
    rows.append({"resolution": res, "verts": nv, "tris": nf, "time_s": round(dt, 3)})
    print(f"{res:>6}  {nv:>8}  {nf:>8}  {dt:>8.3f}")

    mesh.export(f"output/top_res{res}.obj")

csv_path = "output/resolution_sweep.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["resolution", "verts", "tris", "time_s"])
    w.writeheader()
    w.writerows(rows)

print(f"\nwrote {csv_path}")

# --- Plot ---
res_list = [r["resolution"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(res_list, [r["time_s"] for r in rows], "o-")
ax1.set_xlabel("Grid resolution N")
ax1.set_ylabel("Generation time (s)")
ax1.set_title("Generation time vs resolution")

ax2.plot(res_list, [r["tris"] for r in rows], "s-")
ax2.set_xlabel("Grid resolution N")
ax2.set_ylabel("Triangle count")
ax2.set_title("Triangle count vs resolution")

fig.tight_layout()
fig_path = "output/resolution_sweep.png"
fig.savefig(fig_path, dpi=150)
print(f"wrote {fig_path}")
