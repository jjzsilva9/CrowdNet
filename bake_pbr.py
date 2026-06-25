"""
Bake TexGaussian PBR textures onto UV maps for a DrapeNet garment mesh.

Full pipeline: gaussian generation → UV unwrap → multi-view texture fitting
→ export single OBJ + MTL with baked albedo + metallic-roughness texture maps.

Usage:
  conda activate texgaussian
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  python bake_texgaussian.py --mesh output/top_smoke.obj --prompt "a red cotton t-shirt"
"""
import sys
import os
import argparse
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TEXGAUSSIAN_ROOT = os.path.join(REPO_ROOT, "external", "TexGaussian")
sys.path.insert(0, TEXGAUSSIAN_ROOT)

from core.options import Options


def make_options(args):
    opt = Options()
    opt.use_material = True
    opt.use_text = True
    opt.use_checkpoint = False
    opt.gaussian_loss = False
    opt.use_local_pretrained_ckpt = False
    opt.lambda_lpips = 0.0
    opt.input_feature = "ND"
    opt.input_depth = 8
    opt.full_depth = 4
    opt.output_size = 512
    opt.fovy = 30
    opt.znear = 0.5
    opt.zfar = 2.5
    opt.out_channels = 7
    opt.force_cuda_rast = True
    opt.save_image = False
    opt.text_prompt = args.prompt
    opt.texture_cam_radius = 4.5
    opt.texture_name = args.name
    opt.output_dir = args.out_dir
    opt.mesh_path = args.mesh
    opt.ckpt_path = args.ckpt
    opt.pointcloud_dir = ""
    return opt


def export_pbr_mesh(converter, save_dir):
    """Export a single OBJ + MTL with albedo and MR texture maps."""
    import torch
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    albedo_np = torch.sigmoid(converter.albedo).detach().cpu().clamp(0, 1).numpy()
    albedo_img = (albedo_np * 255).astype(np.uint8)
    Image.fromarray(albedo_img).save(os.path.join(save_dir, "albedo.png"))

    mr_np = torch.sigmoid(converter.mr_albedo).detach().cpu().clamp(0, 1).numpy()
    mr_img = (mr_np * 255).astype(np.uint8)
    Image.fromarray(mr_img).save(os.path.join(save_dir, "metallic_roughness.png"))

    verts = converter.mesh.vertices.astype(np.float32)
    faces = converter.f.cpu().numpy()
    uvs = converter.vt.cpu().numpy()
    uv_faces = converter.ft.cpu().numpy()

    mtl_name = "material.mtl"
    mat_name = "texgaussian_pbr"

    with open(os.path.join(save_dir, mtl_name), "w") as f:
        f.write(f"newmtl {mat_name}\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.5 0.5 0.5\n")
        f.write("Ns 100.0\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write("map_Kd albedo.png\n")
        f.write("map_Pm metallic_roughness.png\n")

    obj_path = os.path.join(save_dir, "mesh.obj")
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"usemtl {mat_name}\n\n")

        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        norms = converter.mesh.vertex_normals
        for n in norms:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        for fi, uv_fi in zip(faces, uv_faces):
            i0, i1, i2 = fi + 1
            u0, u1, u2 = uv_fi + 1
            f.write(f"f {i0}/{u0}/{i0} {i1}/{u1}/{i1} {i2}/{u2}/{i2}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bake TexGaussian PBR textures onto UV maps"
    )
    parser.add_argument("--mesh", required=True, help="Path to garment OBJ")
    parser.add_argument("--prompt", default="a red cotton t-shirt")
    parser.add_argument("--ckpt", default=os.path.join(
        TEXGAUSSIAN_ROOT, "checkpoints", "PBR_model.safetensors"))
    parser.add_argument("--out_dir", default="output/texgaussian_baked")
    parser.add_argument("--name", default="garment",
                        help="Subfolder name for this bake")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Optimization iterations per fixed view")
    parser.add_argument("--texture_res", type=int, default=1024,
                        help="UV texture map resolution")
    parser.add_argument("--render_res", type=int, default=512,
                        help="Render resolution for fitting")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Mesh: {args.mesh}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Texture resolution: {args.texture_res}")
    print(f"Fitting iterations: {args.iters}")

    opt = make_options(args)

    import texture
    from texture import Converter
    texture.opt = opt

    print("\n[1/4] Loading model...")
    t0 = time.time()
    converter = Converter(opt).cuda()
    print(f"  Model loaded ({time.time()-t0:.1f}s)")

    print("\n[2/4] Loading checkpoint...")
    t1 = time.time()
    converter.load_ckpt(args.ckpt)
    print(f"  Checkpoint loaded ({time.time()-t1:.1f}s)")

    print("\n[3/4] Loading mesh and building octree...")
    t2 = time.time()
    converter.load_mesh(args.mesh)
    print(f"  Mesh loaded ({time.time()-t2:.1f}s)")

    print(f"\n[4/4] Fitting UV textures ({args.iters} iters × 6 fixed views "
          f"+ {args.iters*2} random views, ×2 for material)...")
    t3 = time.time()
    converter.fit_mesh_uv(
        iters=args.iters,
        resolution=args.render_res,
        texture_resolution=args.texture_res,
    )
    print(f"  Texture fitting done ({time.time()-t3:.1f}s)")

    save_dir = os.path.join(args.out_dir, args.name)
    export_pbr_mesh(converter, save_dir)

    print(f"\nDone! Output in {save_dir}/")
    print(f"  mesh.obj               — garment with UVs + normals")
    print(f"  material.mtl           — PBR material (albedo + MR maps)")
    print(f"  albedo.png             — baked albedo texture")
    print(f"  metallic_roughness.png — baked metallic-roughness texture")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
