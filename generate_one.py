"""
Minimal DrapeNet generation smoke test (no SMPL needed).
Place at the repo ROOT and run from there:  python generate_one.py
Decodes one garment's UDF latent -> MeshUDF mesh -> .obj, and logs tris + time.
"""
import os, time, torch
from utils_drape import load_udf, reconstruct

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
os.makedirs("output", exist_ok=True)

# Load the TOP decoder + its latent code bank (both ship in ./checkpoints)
coords_encoder, latent_codes_top, decoder_top = load_udf(
    "checkpoints", "top_codes.pt", "top_udf.pt", device
)
print(f"loaded {latent_codes_top.shape[0]} top latents, dim {latent_codes_top.shape[1]}")

top_idx = 208      # same index drape.py uses; valid range is 0..N-1 above
res = 1024          # MeshUDF grid resolution N -> the geometry-LOD knob (N^3 grid)

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
mesh, v, t = reconstruct(
    coords_encoder, decoder_top, latent_codes_top[[top_idx]],
    udf_max_dist=0.1, resolution=res, differentiable=False,
)
if device.type == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t0

mesh.export("output/top_smoke.obj")
print(f"res={res}  verts={len(mesh.vertices)}  tris={len(mesh.faces)}  time={dt:.2f}s")
print("wrote output/top_smoke.obj")