"""
Render TexGaussian PBR gaussian splats using Relightable 3D Gaussian's
differentiable PBR renderer (GGX specular + environment lighting).

Usage:
  conda activate texgaussian
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  python render_texgaussian.py --mesh output/top_smoke.obj --prompt "a red cotton t-shirt"
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
R3DG_ROOT = os.path.join(REPO_ROOT, "external", "Relightable3DGaussian")
sys.path.insert(0, TEXGAUSSIAN_ROOT)

from ocnn.octree import Octree, Points
import ocnn
from core.regression_models import TexGaussian
from core.options import Options
from external.clip import tokenize
from safetensors.torch import load_file
from kiui.cam import orbit_camera
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings as OrigSettings,
    GaussianRasterizer as OrigGaussianRasterizer,
)
from PIL import Image

# Import R3DG: its internal imports expect `utils`, `scene`, `arguments` etc.
# as top-level packages. CrowdNet's own utils.py shadows R3DG's utils/ package,
# so we temporarily prioritise R3DG_ROOT on sys.path while importing.
_cwd_entries = [p for p in sys.path if os.path.isfile(os.path.join(p, "utils.py"))]
for p in _cwd_entries:
    sys.path.remove(p)
sys.path.insert(0, R3DG_ROOT)

from gaussian_renderer.neilf import GGX_specular
from scene.envmap import EnvLight
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb

# Restore sys.path
sys.path.remove(R3DG_ROOT)
for p in reversed(_cwd_entries):
    sys.path.insert(0, p)


def make_texgaussian_options():
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


def load_mesh_and_octree(mesh_path, opt, num_samples=200000):
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
    return mesh, octree_in, input_data


def assign_mesh_normals_to_gaussians(mesh, gaussian_positions):
    """Assign nearest mesh vertex normal to each gaussian."""
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)
    pos_np = gaussian_positions.cpu().numpy()
    _, indices = tree.query(pos_np)
    normals = mesh.vertex_normals[indices]
    return torch.from_numpy(normals.astype(np.float32)).cuda()


def splat(means3D, colors, opacity, scales, rotations, raster_settings):
    """Rasterize gaussians using the original diff-gaussian-rasterization."""
    device = means3D.device
    rasterizer = OrigGaussianRasterizer(raster_settings=raster_settings)
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
    return image, alpha


def render_pbr_view(gaussians_data, pose, env_light, opt, size,
                    sample_num=24, scale_modifier=1.0):
    """
    Render a single PBR view.
    - R3DG's GGX_specular + envmap for per-gaussian PBR color computation
    - Original diff-gaussian-rasterization for splatting (proven to work)
    """
    device = torch.device("cuda")
    means3D = gaussians_data["xyz"]
    opacity = gaussians_data["opacity"]
    scales = gaussians_data["scales"]
    rotations = gaussians_data["rotations"]
    base_color = gaussians_data["base_color"]
    roughness = gaussians_data["roughness"]
    normals = gaussians_data["normals"]

    # Camera setup (same convention as test_texgaussian.py)
    tan_half = math.tan(0.5 * math.radians(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half
    proj_matrix[1, 1] = 1 / tan_half
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    cam_poses = torch.from_numpy(pose).unsqueeze(0).to(device)
    cam_poses[:, :3, 1:3] *= -1
    cam_view = torch.inverse(cam_poses).transpose(1, 2).squeeze(0)
    cam_view_proj = cam_view @ proj_matrix
    cam_pos = -cam_poses[0, :3, 3]

    bg = torch.ones(3, dtype=torch.float32, device=device)

    raster_settings = OrigSettings(
        image_height=size, image_width=size,
        tanfovx=tan_half, tanfovy=tan_half,
        bg=bg, scale_modifier=scale_modifier,
        viewmatrix=cam_view, projmatrix=cam_view_proj,
        sh_degree=0, campos=cam_pos,
        prefiltered=False, debug=False,
    )

    # Per-gaussian PBR color via R3DG's rendering equation
    viewdirs = F.normalize(cam_pos - means3D, dim=-1)

    incident_dirs, incident_areas = fibonacci_sphere_sampling(
        normals, sample_num, random_rotate=False)

    incident_lights = env_light.direct_light(incident_dirs)

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i
    pbr_color = ((f_d + f_s) * transport).mean(dim=-2)

    # Splat each channel
    pbr_img, _ = splat(means3D, pbr_color, opacity, scales, rotations, raster_settings)
    albedo_img, _ = splat(means3D, base_color, opacity, scales, rotations, raster_settings)

    normal_vis = normals * 0.5 + 0.5
    normals_img, _ = splat(means3D, normal_vis, opacity, scales, rotations, raster_settings)

    # Linear PBR → sRGB
    pbr_srgb = rgb_to_srgb(pbr_img)

    return {
        "pbr": pbr_srgb,
        "albedo": albedo_img,
        "normals": normals_img,
    }


def to_np(t):
    return (t.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to garment OBJ")
    parser.add_argument("--prompt", default="a red cotton t-shirt")
    parser.add_argument("--ckpt", default=os.path.join(
        TEXGAUSSIAN_ROOT, "checkpoints", "PBR_model.safetensors"))
    parser.add_argument("--envmap", default=os.path.join(
        R3DG_ROOT, "env_map", "ocean_from_horn.jpg"),
        help="Environment map for lighting (HDR/LDR image)")
    parser.add_argument("--envmap_scale", type=float, default=1.0)
    parser.add_argument("--out_dir", default="output/texgaussian_pbr")
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--elevation", type=float, default=0)
    parser.add_argument("--radius", type=float, default=4.5)
    parser.add_argument("--sample_num", type=int, default=24,
                        help="Incident light samples per gaussian")
    parser.add_argument("--scale_modifier", type=float, default=1.0,
                        help="Multiply gaussian scales to fill holes (try 1.5-2.0)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Mesh: {args.mesh}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Envmap: {args.envmap}")

    opt = make_texgaussian_options()
    opt.output_size = args.size

    # Load TexGaussian model
    print("Loading TexGaussian model...")
    t0 = time.time()
    model = TexGaussian(opt, device).to(device)
    ckpt = load_file(args.ckpt, device="cpu")
    state_dict = model.state_dict()
    loaded = 0
    for k, v in ckpt.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
            loaded += 1
    print(f"  {loaded} params loaded ({time.time()-t0:.1f}s)")

    # Load mesh + octree
    print("Loading mesh...")
    t1 = time.time()
    mesh, octree_in, input_data = load_mesh_and_octree(args.mesh, opt)
    print(f"  {octree_in.position.shape[0]} octree points ({time.time()-t1:.1f}s)")

    # Generate gaussians
    print("Generating gaussians...")
    t2 = time.time()
    token = tokenize(args.prompt).to(device)
    text_embedding = model.text_encoder.encode(token).float()

    with torch.no_grad():
        _, gaussians, mr_gaussians = model.forward_gaussians(
            input_data, octree_in,
            condition=text_embedding, data=None, ema=True,
        )

    batch_id = octree_in.batch_id(opt.input_depth, nempty=True)
    mask = batch_id == 0
    g = gaussians[mask]
    mr_g = mr_gaussians[mask]
    print(f"  {g.shape[0]} splats ({time.time()-t2:.2f}s)")

    # Extract per-gaussian PBR properties
    xyz = g[:, 0:3].contiguous()
    opacity = g[:, 3:4].contiguous()
    scales = g[:, 4:7].contiguous()
    rotations = g[:, 7:11].contiguous()
    base_color = g[:, 11:14].contiguous()

    # MR channels: G=roughness, B=metallic (we use roughness for now)
    mr_colors = mr_g[:, 11:14].contiguous()
    roughness = mr_colors[:, 1:2].clamp(0.09, 0.99)

    # Assign normals from mesh vertices
    print("Computing per-gaussian normals from mesh...")
    normals = assign_mesh_normals_to_gaussians(mesh, xyz)
    normals = F.normalize(normals, dim=-1)

    gaussians_data = {
        "xyz": xyz, "opacity": opacity, "scales": scales,
        "rotations": rotations, "base_color": base_color,
        "roughness": roughness, "normals": normals,
    }

    # Load environment map
    print(f"Loading environment map...")
    env_light = EnvLight(args.envmap, scale=args.envmap_scale)

    # Render turntable
    print(f"Rendering {args.n_views} PBR views...")
    t3 = time.time()

    for i in range(args.n_views):
        hor = i * (360.0 / args.n_views)
        pose = orbit_camera(args.elevation, hor, args.radius)

        with torch.no_grad():
            result = render_pbr_view(
                gaussians_data, pose, env_light, opt, args.size,
                args.sample_num, args.scale_modifier)

        Image.fromarray(to_np(result["pbr"])).save(
            os.path.join(args.out_dir, f"pbr_{i:02d}.png"))
        Image.fromarray(to_np(result["albedo"])).save(
            os.path.join(args.out_dir, f"albedo_{i:02d}.png"))
        Image.fromarray(to_np(result["normals"])).save(
            os.path.join(args.out_dir, f"normals_{i:02d}.png"))

    print(f"  {time.time()-t3:.2f}s")
    print(f"\nSaved to {args.out_dir}/")
    print(f"  pbr_*.png     — PBR render (R3DG GGX + envmap, sRGB)")
    print(f"  albedo_*.png  — raw base color")
    print(f"  normals_*.png — per-gaussian normals (debug)")


if __name__ == "__main__":
    main()
