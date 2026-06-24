"""
Zero-shot test: TexGaussian PBR splats on a DrapeNet garment mesh.
Renders with deferred PBR shading: albedo + metallic-roughness gaussians
are rasterized, then combined with screen-space normals and Cook-Torrance
lighting to produce a final lit image.

Usage:
  conda activate texgaussian
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  python test_texgaussian.py --mesh output/top_smoke.obj --prompt "a red cotton t-shirt"
"""
import sys
import os
import argparse
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TEXGAUSSIAN_ROOT = os.path.join(REPO_ROOT, "external", "TexGaussian")
sys.path.insert(0, TEXGAUSSIAN_ROOT)

from ocnn.octree import Octree, Points
import ocnn
from core.regression_models import TexGaussian
from core.options import Options
from external.clip import tokenize
from safetensors.torch import load_file
from kiui.cam import orbit_camera
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from PIL import Image


def make_options():
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
    return opt


def load_mesh_to_octree(mesh_path, opt, num_samples=200000):
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(mesh.vertices, axis=1)
    mesh.vertices /= np.max(distances)

    pts, idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[idx]

    points_gt = Points(
        points=torch.from_numpy(pts).float(),
        normals=torch.from_numpy(normals).float(),
    )
    points_gt.clip(min=-1, max=1)

    points_gt_cuda = points_gt.cuda(non_blocking=True)
    ot = Octree(depth=opt.input_depth, full_depth=opt.full_depth)
    ot.build_octree(points_gt_cuda)
    octree_in = ocnn.octree.merge_octrees([ot])
    octree_in.construct_all_neigh()

    xyzb = octree_in.xyzb(depth=octree_in.depth, nempty=True)
    x, y, z, b = xyzb
    xyz = torch.stack([x, y, z], dim=1)
    octree_in.position = 2 * xyz / (2 ** octree_in.depth) - 1

    input_data = octree_in.get_input_feature(feature=opt.input_feature, nempty=True)
    return octree_in, input_data


def rasterize_pass(gaussians, colors, raster_settings):
    """Single rasterizer pass with custom per-gaussian colors."""
    device = gaussians.device
    means3D = gaussians[:, 0:3].contiguous().float()
    opacity = gaussians[:, 3:4].contiguous().float()
    scales = gaussians[:, 4:7].contiguous().float()
    rotations = gaussians[:, 7:11].contiguous().float()

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
        shs=None,
        colors_precomp=colors.contiguous().float(),
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    return image, depth, alpha


def depth_to_normals(depth, alpha, fovy, size):
    """Estimate screen-space normals from depth buffer via finite differences."""
    tan_half = math.tan(0.5 * math.radians(fovy))
    pixel_size = 2.0 * tan_half / size

    d = depth.squeeze(0)  # [H, W]
    mask = alpha.squeeze(0) > 0.5  # [H, W]

    dz_dx = torch.zeros_like(d)
    dz_dy = torch.zeros_like(d)
    dz_dx[:, 1:-1] = (d[:, 2:] - d[:, :-2]) / 2.0
    dz_dy[1:-1, :] = (d[2:, :] - d[:-2, :]) / 2.0

    nx = -dz_dx / pixel_size
    ny = -dz_dy / pixel_size
    nz = torch.ones_like(d)

    normals = torch.stack([nx, ny, nz], dim=0)  # [3, H, W]
    normals = F.normalize(normals, dim=0)
    normals[:, ~mask] = 0
    return normals


def pbr_shade(albedo, mr, normals, alpha, cam_pos_world, view_matrix):
    """
    Deferred PBR shading (simplified Cook-Torrance).
    albedo: [3, H, W], mr: [3, H, W] (R=unused, G=roughness, B=metallic)
    normals: [3, H, W] in view space, alpha: [1, H, W]
    """
    mask = (alpha.squeeze(0) > 0.5).float()  # [H, W]

    roughness = mr[1:2].clamp(0.05, 1.0)  # [1, H, W]
    metallic = mr[2:3].clamp(0.0, 1.0)     # [1, H, W]

    # Lights in view space (camera at origin looking down -Z)
    key_dir = F.normalize(torch.tensor([0.5, 0.7, 0.8], device=albedo.device), dim=0)
    fill_dir = F.normalize(torch.tensor([-0.4, 0.3, 0.6], device=albedo.device), dim=0)
    view_dir = torch.tensor([0.0, 0.0, 1.0], device=albedo.device)

    key_color = torch.tensor([1.0, 0.95, 0.9], device=albedo.device).view(3, 1, 1)
    fill_color = torch.tensor([0.3, 0.35, 0.5], device=albedo.device).view(3, 1, 1)
    ambient = torch.tensor([0.15, 0.15, 0.18], device=albedo.device).view(3, 1, 1)

    def shade_light(L, light_color):
        H = F.normalize(L + view_dir, dim=0)
        NdotL = (normals * L.view(3, 1, 1)).sum(0, keepdim=True).clamp(0)  # [1, H, W]
        NdotH = (normals * H.view(3, 1, 1)).sum(0, keepdim=True).clamp(0)
        NdotV = normals[2:3].clamp(0)  # view is [0,0,1]

        # Fresnel (Schlick): F0 = lerp(0.04, albedo, metallic)
        f0 = 0.04 * (1 - metallic) + albedo * metallic
        fresnel = f0 + (1 - f0) * (1 - NdotV).clamp(0).pow(5)

        # GGX distribution (simplified)
        a2 = (roughness * roughness).clamp(min=1e-4)
        denom = NdotH * NdotH * (a2 - 1) + 1
        D = a2 / (math.pi * denom * denom + 1e-7)

        specular = fresnel * D * 0.25  # simplified, no G term
        diffuse = albedo * (1 - metallic) / math.pi

        return (diffuse + specular) * NdotL * light_color

    result = ambient * albedo
    result = result + shade_light(key_dir, key_color)
    result = result + shade_light(fill_dir, fill_color)

    result = result.clamp(0, 1)
    result = result * mask + (1 - mask)  # white background
    return result


def render_pbr_turntable(gaussians, mr_gaussians, octree_in, opt, n_views=8):
    device = torch.device("cuda")
    size = opt.output_size

    tan_half_fov = math.tan(0.5 * math.radians(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    batch_id = octree_in.batch_id(opt.input_depth, nempty=True)
    bg = torch.ones(3, dtype=torch.float32, device=device)
    radius = 4.5

    # Select gaussians for this batch
    mask = batch_id == 0
    g_batch = gaussians[mask]
    mr_batch = mr_gaussians[mask]
    albedo_colors = g_batch[:, 11:14].contiguous()
    mr_colors = mr_batch[:, 11:14].contiguous()

    pbr_images = []
    albedo_images = []
    mr_images = []

    for i in range(n_views):
        hor = i * (360 / n_views)
        pose = orbit_camera(0, hor, radius)

        cam_poses = torch.from_numpy(pose).unsqueeze(0).to(device)
        cam_poses[:, :3, 1:3] *= -1
        cam_view = torch.inverse(cam_poses).transpose(1, 2).squeeze(0)  # [4, 4]
        cam_view_proj = (cam_view @ proj_matrix)  # [4, 4]
        cam_pos = -cam_poses[0, :3, 3]

        settings = GaussianRasterizationSettings(
            image_height=size, image_width=size,
            tanfovx=tan_half_fov, tanfovy=tan_half_fov,
            bg=bg, scale_modifier=1,
            viewmatrix=cam_view, projmatrix=cam_view_proj,
            sh_degree=0, campos=cam_pos,
            prefiltered=False, debug=False,
        )

        albedo_img, depth, alpha = rasterize_pass(g_batch, albedo_colors, settings)
        mr_img, _, _ = rasterize_pass(g_batch, mr_colors, settings)

        normals = depth_to_normals(depth, alpha, opt.fovy, size)

        pbr = pbr_shade(albedo_img, mr_img, normals, alpha, cam_pos, cam_view)

        def to_np(t):
            return (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        pbr_images.append(to_np(pbr))
        albedo_images.append(to_np(albedo_img))
        mr_images.append(to_np(mr_img))

    return pbr_images, albedo_images, mr_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to garment OBJ")
    parser.add_argument("--prompt", default="a red cotton t-shirt")
    parser.add_argument("--ckpt", default=os.path.join(TEXGAUSSIAN_ROOT, "checkpoints", "PBR_model.safetensors"))
    parser.add_argument("--out_dir", default="output/texgaussian")
    parser.add_argument("--n_views", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Mesh: {args.mesh}")
    print(f"Prompt: '{args.prompt}'")

    opt = make_options()

    print("Loading model...")
    t0 = time.time()
    model = TexGaussian(opt, device).to(device)

    ckpt = load_file(args.ckpt, device="cpu")
    state_dict = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in ckpt.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    print(f"  Loaded {loaded} params, skipped {skipped} ({time.time()-t0:.1f}s)")

    print("Loading mesh and building octree...")
    t1 = time.time()
    octree_in, input_data = load_mesh_to_octree(args.mesh, opt)
    n_pts = octree_in.position.shape[0]
    print(f"  Octree points: {n_pts} ({time.time()-t1:.1f}s)")

    print("Encoding text prompt...")
    token = tokenize(args.prompt).to(device)
    text_embedding = model.text_encoder.encode(token).float()

    print("Running forward_gaussians...")
    t2 = time.time()
    with torch.no_grad():
        _, gaussians, mr_gaussians = model.forward_gaussians(
            input_data, octree_in,
            condition=text_embedding,
            data=None, ema=True,
        )
    print(f"  Gaussians: {gaussians.shape[0]} splats ({time.time()-t2:.2f}s)")

    print(f"Rendering {args.n_views} PBR turntable views...")
    t3 = time.time()
    with torch.no_grad():
        pbr_imgs, albedo_imgs, mr_imgs = render_pbr_turntable(
            gaussians, mr_gaussians, octree_in, opt, args.n_views,
        )
    print(f"  Render time: {time.time()-t3:.2f}s")

    for i, (pbr, alb, mr) in enumerate(zip(pbr_imgs, albedo_imgs, mr_imgs)):
        Image.fromarray(pbr).save(os.path.join(args.out_dir, f"pbr_{i:02d}.png"))
        Image.fromarray(alb).save(os.path.join(args.out_dir, f"albedo_{i:02d}.png"))
        Image.fromarray(mr).save(os.path.join(args.out_dir, f"mr_{i:02d}.png"))

    print(f"\nSaved to {args.out_dir}/")
    print(f"  pbr_*.png    — final lit render (Cook-Torrance, key+fill+ambient)")
    print(f"  albedo_*.png — raw albedo channel")
    print(f"  mr_*.png     — raw metallic-roughness (G=roughness, B=metallic)")


if __name__ == "__main__":
    main()
