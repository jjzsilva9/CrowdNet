"""
Apply DressCode PBR textures to a DrapeNet mesh.
  1. Load bare OBJ (no UVs)
  2. UV-unwrap with xatlas
  3. Export textured OBJ + MTL referencing diffuse/normal/roughness maps

Usage:
  conda activate dresscode
  python apply_texture.py output/top_smoke.obj output/textures/deep_grey_fabric

The second arg is the texture prefix — it expects {prefix}_diffuse.png, _normal.png, _roughness.png.
Outputs: {mesh_stem}_textured.obj / .mtl in the same directory as the mesh.
"""
import sys
import os
import time
import numpy as np
import trimesh
import xatlas


def main():
    if len(sys.argv) < 3:
        print("Usage: python apply_texture.py <mesh.obj> <texture_prefix>")
        print("  e.g. python apply_texture.py output/top_smoke.obj output/textures/deep_grey_fabric")
        sys.exit(1)

    mesh_path = sys.argv[1]
    tex_prefix = sys.argv[2]

    diffuse_path = f"{tex_prefix}_diffuse.png"
    normal_path = f"{tex_prefix}_normal.png"
    roughness_path = f"{tex_prefix}_roughness.png"

    for p in [mesh_path, diffuse_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    print("Running xatlas UV unwrap...")
    t0 = time.time()
    vmapping, indices, uvs = xatlas.parametrize(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.uint32),
    )
    unwrap_time = time.time() - t0
    print(f"  xatlas done in {unwrap_time:.2f}s")
    print(f"  UV vertices: {len(uvs)}, UV faces: {len(indices)}")

    out_dir = os.path.dirname(os.path.abspath(mesh_path))
    stem = os.path.splitext(os.path.basename(mesh_path))[0]
    obj_out = os.path.join(out_dir, f"{stem}_textured.obj")
    mtl_out = os.path.join(out_dir, f"{stem}_textured.mtl")
    mtl_name = f"{stem}_textured.mtl"
    mat_name = "dresscode_pbr"

    diffuse_rel = os.path.relpath(os.path.abspath(diffuse_path), out_dir)
    normal_rel = os.path.relpath(os.path.abspath(normal_path), out_dir)
    roughness_rel = os.path.relpath(os.path.abspath(roughness_path), out_dir)

    new_verts = mesh.vertices[vmapping]

    print(f"Writing {mtl_out}")
    with open(mtl_out, "w") as f:
        f.write(f"newmtl {mat_name}\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {diffuse_rel}\n")
        if os.path.exists(normal_path):
            f.write(f"bump {normal_rel}\n")
        if os.path.exists(roughness_path):
            f.write(f"map_Ns {roughness_rel}\n")

    print(f"Writing {obj_out}")
    with open(obj_out, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"usemtl {mat_name}\n\n")

        for v in new_verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        for face in indices:
            i0, i1, i2 = face + 1  # OBJ is 1-indexed
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")

    print(f"\nDone. Output:")
    print(f"  {obj_out}")
    print(f"  {mtl_out}")
    print(f"  Textures: {diffuse_rel}, {normal_rel}, {roughness_rel}")


if __name__ == "__main__":
    main()
