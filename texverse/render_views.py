"""
Render multi-view PBR images from GLB files for TexGaussian training.
Run with Blender in headless mode:

    blender --background --python texverse/render_views.py -- \
        --glb_dir texverse/glbs \
        --output_dir texverse/renders \
        [--single UID]       # render one model
        [--num_views N]      # default 64
        [--recheck]          # check existing renders, re-render bad ones
        [--mr-only]          # re-render only MR maps using stored cameras.npz
        [--skip FILE]        # file of UIDs to skip

Per model produces:
    {uid}/{vid}.png        — lit PBR RGBA render (512x512)
    {uid}/{vid}_mr.png     — metallic-roughness as RGB (512x512)
    {uid}/cameras.npz      — camera poses [num_views, 4, 4] c2w OpenGL
"""
import bpy
import numpy as np
import os
import sys
import math
import argparse
from pathlib import Path
from mathutils import Matrix, Vector

NUM_VIEWS = 64
RESOLUTION = 512
FOV_DEG = 30
CYCLES_SAMPLES = 128


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--single", default=None)
    parser.add_argument("--num_views", type=int, default=NUM_VIEWS)
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--mr-only", action="store_true")
    parser.add_argument("--repro", action="store_true",
                        help="Re-render all views from saved cameras.npz + scene_params.json")
    parser.add_argument("--skip", default=None)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for img in bpy.data.images:
        bpy.data.images.remove(img)


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'

    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        prefs.preferences.compute_device_type = 'CUDA'
        prefs.preferences.get_devices()
        for d in prefs.preferences.devices:
            d.use = True


def setup_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.data.type = 'PERSP'
    cam.data.angle = math.radians(FOV_DEG)
    cam.data.clip_start = 0.01
    cam.data.clip_end = 10000
    bpy.context.scene.camera = cam
    return cam


def setup_lighting():
    world = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 0.5
    output = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))


def normalize_scene():
    """Center and scale geometry to [-1, 1] at world origin.
    Returns (center, scale) so they can be saved for exact reproduction."""
    import json as _json
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not meshes:
        return [0.0, 0.0, 0.0], 1.0

    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass

    all_verts = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_verts.append((v.co.x, v.co.y, v.co.z))
    all_verts = np.array(all_verts)

    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    center = (bbox_max + bbox_min) / 2
    extent = float((bbox_max - bbox_min).max())
    scale = 2.0 / extent if extent > 0 else 1.0

    for obj in meshes:
        for v in obj.data.vertices:
            co = np.array([v.co.x, v.co.y, v.co.z])
            co = (co - center) * scale
            v.co = Vector(co)
        obj.data.update()

    bpy.context.view_layer.update()
    return center.tolist(), scale


def apply_stored_normalization(center, scale):
    """Reproduce normalization from saved params on a fresh GLB import."""
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not meshes:
        return
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass
    center = np.array(center)
    for obj in meshes:
        for v in obj.data.vertices:
            co = np.array([v.co.x, v.co.y, v.co.z])
            co = (co - center) * scale
            v.co = Vector(co)
        obj.data.update()
    bpy.context.view_layer.update()


def generate_camera_poses(num_views=NUM_VIEWS):
    elevations = [-15, 0, 15, 30]
    poses = []
    for elev in elevations:
        for i in range(8):
            poses.append((i * 45.0, elev))
    remaining = num_views - len(poses)
    if remaining > 0:
        rng = np.random.RandomState(42)
        for _ in range(remaining):
            poses.append((rng.uniform(0, 360), rng.uniform(-20, 40)))
    return poses[:num_views]


def set_camera_pose(cam, azimuth_deg, elevation_deg, radius, target=None):
    if target is None:
        target = Vector((0, 0, 0))
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = target.x + radius * math.cos(el) * math.cos(az)
    y = target.y + radius * math.cos(el) * math.sin(az)
    z = target.z + radius * math.sin(el)
    cam.location = Vector((x, y, z))
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()


def get_c2w_matrix(cam):
    return np.array(cam.matrix_world, dtype=np.float32)


def read_alpha(image_path):
    img = bpy.data.images.load(str(image_path))
    w, h = img.size[0], img.size[1]
    pixels = np.array(img.pixels[:]).reshape(h, w, img.channels)
    bpy.data.images.remove(img)
    if pixels.shape[2] < 4:
        return np.ones((h, w))
    return pixels[:, :, 3]


def check_clipping(alpha, border=5, threshold=0.02):
    top = alpha[-border:, :].max() > threshold
    bottom = alpha[:border, :].max() > threshold
    left = alpha[:, :border].max() > threshold
    right = alpha[:, -border:].max() > threshold
    return top or bottom or left or right


def get_coverage(alpha, threshold=0.04):
    return float((alpha > threshold).sum() / alpha.size)


def get_scene_bounds():
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    DEFAULT_NAMES = {'Cube', 'Camera', 'Light', 'Point', 'Sun', 'Area', 'Spot'}
    all_verts = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.name in DEFAULT_NAMES:
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            eval_obj.to_mesh_clear()
            continue
        mat = eval_obj.matrix_world
        for v in mesh.vertices:
            world_co = mat @ v.co
            all_verts.append((world_co.x, world_co.y, world_co.z))
        eval_obj.to_mesh_clear()
    if not all_verts:
        return Vector((0, 0, 0)), 1.0
    all_verts = np.array(all_verts)
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    center = (bbox_max + bbox_min) / 2
    extent = (bbox_max - bbox_min).max()
    print(f"    Scene bbox: [{bbox_min.round(3)}] to [{bbox_max.round(3)}], extent={extent:.3f}")
    return Vector(center), extent


def get_mesh_centers():
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    DEFAULT_NAMES = {'Cube', 'Camera', 'Light', 'Point', 'Sun', 'Area', 'Spot'}
    centers = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.name in DEFAULT_NAMES:
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            eval_obj.to_mesh_clear()
            continue
        mat = eval_obj.matrix_world
        verts = np.array([list(mat @ v.co) for v in mesh.vertices])
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        c = (bbox_max + bbox_min) / 2
        ext = (bbox_max - bbox_min).max()
        centers.append((Vector(c), ext, obj.name))
        eval_obj.to_mesh_clear()
    return centers


def _clips_any_view(cam, test_poses, test_path, target, radius, scene):
    for az, el in test_poses:
        set_camera_pose(cam, az, el, radius, target=target)
        scene.render.filepath = test_path
        bpy.ops.render.render(write_still=True)
        if check_clipping(read_alpha(test_path)):
            return True
    return False


def _try_radius_search(cam, test_poses, test_path, target, start_radius, scene):
    set_camera_pose(cam, test_poses[0][0], test_poses[0][1], start_radius, target=target)
    scene.render.filepath = test_path
    bpy.ops.render.render(write_still=True)
    if get_coverage(read_alpha(test_path)) < 0.01:
        return None

    r_hi = start_radius
    while _clips_any_view(cam, test_poses, test_path, target, r_hi, scene):
        r_hi *= 1.3
        print(f"    Zooming out to r={r_hi:.2f}")

    r_lo = r_hi * 0.3
    for i in range(8):
        r_mid = (r_lo + r_hi) / 2
        if _clips_any_view(cam, test_poses, test_path, target, r_mid, scene):
            r_lo = r_mid
            print(f"    r={r_mid:.2f} clips")
        else:
            r_hi = r_mid
            print(f"    r={r_mid:.2f} OK")
        if (r_hi - r_lo) / r_hi < 0.03:
            break

    radius = r_hi
    set_camera_pose(cam, test_poses[0][0], test_poses[0][1], radius, target=target)
    scene.render.filepath = test_path
    bpy.ops.render.render(write_still=True)
    coverage = get_coverage(read_alpha(test_path))
    print(f"    Final: radius={radius:.2f}, coverage={coverage:.1%}")
    return radius


def _cleanup_tmp(tmp_dir):
    for f in Path(tmp_dir).glob("test*.png"):
        f.unlink()
    try:
        Path(tmp_dir).rmdir()
    except OSError:
        pass


def find_optimal_framing(cam, test_poses, tmp_dir, target, extent):
    scene = bpy.context.scene
    orig_samples = scene.cycles.samples
    scene.cycles.samples = 16
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    test_path = str(tmp_dir / "test.png")

    default_radius = extent * 2.5
    result = _try_radius_search(cam, test_poses, test_path, target, default_radius, scene)

    if result is not None:
        scene.cycles.samples = orig_samples
        _cleanup_tmp(tmp_dir)
        return result, target

    print(f"    Model not visible at scene center, trying individual meshes...")
    mesh_centers = get_mesh_centers()
    mesh_centers.sort(key=lambda x: -x[1])
    for center, ext, name in mesh_centers:
        r = ext * 2.5
        print(f"    Trying mesh '{name}' (extent={ext:.1f})...")
        result = _try_radius_search(cam, test_poses, test_path, center, r, scene)
        if result is not None:
            scene.cycles.samples = orig_samples
            _cleanup_tmp(tmp_dir)
            return result, center

    print(f"    WARNING: Could not find good framing, using default")
    scene.cycles.samples = orig_samples
    _cleanup_tmp(tmp_dir)
    return default_radius, target


def swap_materials_to_mr(meshes_mats):
    """Rewire existing material nodes in-place to emit MR as color.
    Channel mapping: R=0, G=roughness, B=metallic."""
    saved = {}
    for obj in [o for o in bpy.data.objects if o.type == 'MESH']:
        saved[obj.name] = []
        for i, mat in enumerate(obj.data.materials):
            if mat is None or not mat.use_nodes:
                saved[obj.name].append(None)
                continue
            tree = mat.node_tree
            nodes = tree.nodes
            links = tree.links
            principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            mat_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if not principled or not mat_output:
                saved[obj.name].append(None)
                continue
            orig_link_from = None
            for link in links:
                if link.to_node == mat_output and link.to_socket.name == 'Surface':
                    orig_link_from = (link.from_node.name, link.from_socket.name)
                    break
            saved[obj.name].append(orig_link_from)
            combine = nodes.new('ShaderNodeCombineRGB')
            combine.inputs['R'].default_value = 0.0
            roughness_input = principled.inputs['Roughness']
            metallic_input = principled.inputs['Metallic']
            if roughness_input.is_linked:
                links.new(roughness_input.links[0].from_socket, combine.inputs['G'])
            else:
                combine.inputs['G'].default_value = roughness_input.default_value
            if metallic_input.is_linked:
                links.new(metallic_input.links[0].from_socket, combine.inputs['B'])
            else:
                combine.inputs['B'].default_value = metallic_input.default_value
            emission = nodes.new('ShaderNodeEmission')
            links.new(combine.outputs['Image'], emission.inputs['Color'])
            for link in list(links):
                if link.to_node == mat_output and link.to_socket.name == 'Surface':
                    links.remove(link)
            links.new(emission.outputs['Emission'], mat_output.inputs['Surface'])
    return saved


def restore_materials(saved):
    for obj in [o for o in bpy.data.objects if o.type == 'MESH']:
        if obj.name not in saved:
            continue
        for i, orig_link_info in enumerate(saved[obj.name]):
            if orig_link_info is None or i >= len(obj.data.materials):
                continue
            mat = obj.data.materials[i]
            if not mat or not mat.use_nodes:
                continue
            tree = mat.node_tree
            nodes = tree.nodes
            links = tree.links
            mat_output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if not mat_output:
                continue
            to_remove = [n for n in nodes if n.type in ('EMISSION', 'COMBRGB')]
            for link in list(links):
                if link.to_node == mat_output and link.to_socket.name == 'Surface':
                    links.remove(link)
            from_node_name, from_socket_name = orig_link_info
            from_node = nodes.get(from_node_name)
            if from_node:
                links.new(from_node.outputs[from_socket_name], mat_output.inputs['Surface'])
            for n in to_remove:
                nodes.remove(n)


def render_model(glb_path, output_dir, num_views=NUM_VIEWS, force=False):
    uid = Path(glb_path).stem.split('_')[0]
    out = Path(output_dir) / uid
    out.mkdir(parents=True, exist_ok=True)

    cameras_file = out / "cameras.npz"
    last_view = out / f"{num_views - 1}_mr.png"
    if not force and cameras_file.exists() and last_view.exists():
        print(f"  Skipping {uid} (already rendered)")
        return True

    clear_scene()
    setup_render()
    setup_lighting()
    cam = setup_camera()

    try:
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
    except Exception as e:
        print(f"  Failed to import {glb_path}: {e}")
        return False

    cube = bpy.data.objects.get('Cube')
    if cube and cube.data and hasattr(cube.data, 'vertices') and len(cube.data.vertices) == 8:
        bpy.data.objects.remove(cube, do_unlink=True)

    norm_center, norm_scale = normalize_scene()

    scene_center, scene_extent = get_scene_bounds()
    camera_poses_params = generate_camera_poses(num_views=num_views)

    test_poses = [(az, el) for el in [-15, 0, 30] for az in [0, 90, 180, 270]]
    print(f"  Finding optimal framing...")
    radius, scene_center = find_optimal_framing(cam, test_poses, out / "_tmp", scene_center, scene_extent)

    all_c2w = np.zeros((num_views, 4, 4), dtype=np.float32)

    print(f"  Rendering albedo views...")
    for vid, (az, el) in enumerate(camera_poses_params):
        set_camera_pose(cam, az, el, radius, target=scene_center)
        all_c2w[vid] = get_c2w_matrix(cam)
        bpy.context.scene.render.filepath = str(out / f"{vid}.png")
        bpy.ops.render.render(write_still=True)

    print(f"  Rendering MR views...")
    saved_mats = swap_materials_to_mr(None)
    for vid, (az, el) in enumerate(camera_poses_params):
        set_camera_pose(cam, az, el, radius, target=scene_center)
        bpy.context.scene.render.filepath = str(out / f"{vid}_mr.png")
        bpy.ops.render.render(write_still=True)
    restore_materials(saved_mats)

    np.savez(cameras_file, poses=all_c2w)

    import json as _json
    (out / "scene_params.json").write_text(_json.dumps({"center": norm_center, "scale": norm_scale}))

    print(f"  Done: {uid} ({num_views} views, radius={radius:.2f})")
    return True


def main():
    args = parse_args()
    glb_dir = Path(args.glb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_uids = set()
    if args.skip and Path(args.skip).exists():
        skip_uids = set(Path(args.skip).read_text().split())
        print(f"Skipping {len(skip_uids)} UIDs")

    def _load_and_setup(d, glb_dir, mode_label):
        """Import GLB and apply stored normalization. Returns (cam, poses) or None."""
        import json as _json
        uid = d.name
        cameras_file = d / 'cameras.npz'
        params_file = d / 'scene_params.json'
        glb_path = glb_dir / f"{uid}.glb"
        if not cameras_file.exists() or not glb_path.exists() or not params_file.exists():
            print(f"  {uid} — missing cameras.npz/scene_params.json/GLB, skipping")
            return None, None
        poses = np.load(cameras_file)['poses']
        params = _json.loads(params_file.read_text())
        clear_scene()
        setup_render()
        setup_lighting()
        cam = setup_camera()
        try:
            bpy.ops.import_scene.gltf(filepath=str(glb_path))
        except Exception as e:
            print(f"  import failed: {e}")
            return None, None
        cube = bpy.data.objects.get('Cube')
        if cube and cube.data and hasattr(cube.data, 'vertices') and len(cube.data.vertices) == 8:
            bpy.data.objects.remove(cube, do_unlink=True)
        apply_stored_normalization(params['center'], params['scale'])
        return cam, poses

    if getattr(args, 'mr_only', False):
        render_dirs = sorted(output_dir.iterdir())
        total = len([d for d in render_dirs if d.is_dir()])
        done = 0
        idx = 0
        for d in render_dirs:
            if not d.is_dir():
                continue
            idx += 1
            uid = d.name
            if uid in skip_uids:
                print(f"[{idx}/{total}] {uid} — skipped")
                continue
            print(f"[{idx}/{total}] {uid} — re-rendering MR maps...")
            cam, poses = _load_and_setup(d, glb_dir, 'MR')
            if cam is None:
                continue
            saved_mats = swap_materials_to_mr(None)
            for vid, c2w in enumerate(poses):
                cam.matrix_world = Matrix(c2w.tolist())
                bpy.context.view_layer.update()
                bpy.context.scene.render.filepath = str(d / f"{vid}_mr.png")
                bpy.ops.render.render(write_still=True)
            restore_materials(saved_mats)
            done += 1
            print(f"  Done ({done} complete)")
        print(f"\nMR re-render complete: {done} models")
        return

    if args.repro:
        render_dirs = sorted(output_dir.iterdir())
        total = len([d for d in render_dirs if d.is_dir()])
        done = 0
        idx = 0
        for d in render_dirs:
            if not d.is_dir():
                continue
            idx += 1
            uid = d.name
            if uid in skip_uids:
                print(f"[{idx}/{total}] {uid} — skipped")
                continue
            print(f"[{idx}/{total}] {uid} — reproducing all views...")
            cam, poses = _load_and_setup(d, glb_dir, 'repro')
            if cam is None:
                continue
            # albedo
            for vid, c2w in enumerate(poses):
                cam.matrix_world = Matrix(c2w.tolist())
                bpy.context.view_layer.update()
                bpy.context.scene.render.filepath = str(d / f"repro_{vid}.png")
                bpy.ops.render.render(write_still=True)
            # MR
            saved_mats = swap_materials_to_mr(None)
            for vid, c2w in enumerate(poses):
                cam.matrix_world = Matrix(c2w.tolist())
                bpy.context.view_layer.update()
                bpy.context.scene.render.filepath = str(d / f"repro_{vid}_mr.png")
                bpy.ops.render.render(write_still=True)
            restore_materials(saved_mats)
            done += 1
            print(f"  Done ({done} complete)")
        print(f"\nRepro complete: {done} models")
        return

    if args.recheck:
        import shutil
        all_glbs = sorted(glb_dir.glob("*.glb"))
        all_uids = [g.stem.split('_')[0] for g in all_glbs]
        print(f"Phase 1: Quick check for clipping and missing renders...")
        good, bad, missing, needs_coverage_check = [], [], [], []
        for uid in all_uids:
            render_dir = output_dir / uid
            view0 = render_dir / "0.png"
            if not render_dir.exists() or not view0.exists():
                missing.append(uid)
                continue
            alpha = read_alpha(str(view0))
            coverage = get_coverage(alpha)
            clipped = check_clipping(alpha)
            if clipped:
                print(f"  {uid}: coverage={coverage:.1%} CLIPPED")
                bad.append(uid)
            else:
                needs_coverage_check.append((uid, coverage))
        print(f"\nClipped: {len(bad)}, Not yet rendered: {len(missing)}, Need coverage check: {len(needs_coverage_check)}")
        if needs_coverage_check:
            print(f"\nPhase 2: Checking if existing renders could be framed tighter...")
            for uid, existing_coverage in needs_coverage_check:
                glb_path = glb_dir / f"{uid}.glb"
                if not glb_path.exists():
                    good.append(uid)
                    continue
                clear_scene()
                setup_render()
                setup_lighting()
                cam = setup_camera()
                try:
                    bpy.ops.import_scene.gltf(filepath=str(glb_path))
                except Exception:
                    good.append(uid)
                    continue
                cube = bpy.data.objects.get('Cube')
                if cube and cube.data and hasattr(cube.data, 'vertices') and len(cube.data.vertices) == 8:
                    bpy.data.objects.remove(cube, do_unlink=True)
                normalize_scene()
                scene_center, scene_extent = get_scene_bounds()
                test_poses = [(az, el) for el in [-15, 0, 30] for az in [0, 90, 180, 270]]
                tmp_dir = output_dir / uid / "_tmp"
                optimal_radius, _ = find_optimal_framing(cam, test_poses, tmp_dir, scene_center, scene_extent)
                test_path = str(tmp_dir / "test.png")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                set_camera_pose(cam, 0, 0, optimal_radius, target=scene_center)
                bpy.context.scene.render.filepath = test_path
                bpy.ops.render.render(write_still=True)
                optimal_coverage = get_coverage(read_alpha(test_path))
                _cleanup_tmp(tmp_dir)
                diff = optimal_coverage - existing_coverage
                if diff > 0.05:
                    print(f"  {uid}: existing={existing_coverage:.1%}, optimal={optimal_coverage:.1%} → RERENDER")
                    bad.append(uid)
                else:
                    print(f"  {uid}: existing={existing_coverage:.1%}, optimal={optimal_coverage:.1%} → OK")
                    good.append(uid)
        print(f"\nSummary — Good: {len(good)}, Rerender: {len(bad)}, New: {len(missing)}")
        to_render = bad + missing
        if not to_render:
            print("All renders look good!")
            return
        print(f"Rendering {len(to_render)} models...")
        for i, uid in enumerate(to_render):
            glb_path = glb_dir / f"{uid}.glb"
            if not glb_path.exists():
                print(f"[{i+1}/{len(to_render)}] {uid} — GLB not found")
                continue
            print(f"[{i+1}/{len(to_render)}] {'Re-rendering' if uid in bad else 'Rendering'} {uid}")
            shutil.rmtree(output_dir / uid, ignore_errors=True)
            render_model(str(glb_path), str(output_dir), num_views=args.num_views, force=True)
        return

    if args.single:
        glb_path = glb_dir / f"{args.single}.glb"
        if not glb_path.exists():
            print(f"ERROR: {glb_path} not found")
            sys.exit(1)
        print(f"Rendering single model: {args.single} ({args.num_views} views)")
        render_model(str(glb_path), str(output_dir), num_views=args.num_views)
        return

    glbs = sorted(glb_dir.glob("*.glb"))
    print(f"Found {len(glbs)} GLB files, rendering {args.num_views} views each")
    success, failed = 0, []
    for i, glb in enumerate(glbs):
        uid = glb.stem.split('_')[0]
        print(f"[{i+1}/{len(glbs)}] {uid}")
        if render_model(str(glb), str(output_dir), num_views=args.num_views):
            success += 1
        else:
            failed.append(uid)
    print(f"\nDone. Success: {success}, Failed: {len(failed)}")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
