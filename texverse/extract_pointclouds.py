"""
Extract point clouds (points + normals) from GLB files for TexGaussian training.
Normalizes to [-1, 1] bounding box (same as render_views.py).

Usage:
    conda activate texgaussian
    python texverse/extract_pointclouds.py
"""
import json
import numpy as np
import trimesh
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLB_DIR = HERE / "glbs"
PC_DIR = HERE / "pointclouds"
NUM_SAMPLES = 200_000


def extract_pointcloud(glb_path, output_path):
    scene = trimesh.load(str(glb_path), force='scene')

    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return False
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    if len(mesh.faces) == 0:
        return False

    # normalize to [-1, 1] (matching render_views.py)
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    center = (bbox_max + bbox_min) / 2
    extent = (bbox_max - bbox_min).max()
    scale = 2.0 / extent if extent > 0 else 1.0

    mesh.vertices = (mesh.vertices - center) * scale

    points, face_indices = trimesh.sample.sample_surface(mesh, NUM_SAMPLES)
    normals = mesh.face_normals[face_indices]

    points = points.astype(np.float32)
    normals = normals.astype(np.float32)

    np.savez(output_path, points=points, normals=normals)

    ply_path = output_path.with_suffix('.ply')
    cloud = trimesh.PointCloud(points)
    cloud.export(str(ply_path))

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default=None, help="Extract a single model by UID")
    args = parser.parse_args()

    manifest = HERE / "verified_garments.jsonl"
    with open(manifest) as f:
        records = [json.loads(line) for line in f]

    if args.single:
        records = [r for r in records if r["model_id"] == args.single]
        if not records:
            records = [{"model_id": args.single}]

    PC_DIR.mkdir(exist_ok=True)

    success = 0
    skipped = 0
    failed = []

    for i, rec in enumerate(records):
        uid = rec["model_id"]
        glb_path = GLB_DIR / f"{uid}.glb"
        out_path = PC_DIR / f"{uid}.npz"

        if out_path.exists():
            skipped += 1
            continue

        if not glb_path.exists():
            failed.append((uid, "GLB not found"))
            continue

        print(f"[{i+1}/{len(records)}] {uid}...", end=" ", flush=True)
        try:
            if extract_pointcloud(glb_path, out_path):
                d = np.load(out_path)
                print(f"OK (pts={d['points'].shape}, bbox=[{d['points'].min(axis=0).round(2)}, {d['points'].max(axis=0).round(2)}])")
                success += 1
            else:
                failed.append((uid, "no faces"))
                print("FAILED (no faces)")
        except Exception as e:
            failed.append((uid, str(e)[:80]))
            print(f"FAILED ({e})")

    print(f"\nDone. Success: {success}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        for uid, reason in failed:
            print(f"  {uid}: {reason}")
    print(f"Total pointclouds on disk: {len(list(PC_DIR.glob('*.npz')))}")


if __name__ == "__main__":
    main()
