# CrowdNet

**LOD-aware generative garment variety for virtual crowds.**

CrowdNet generates clothing for virtual crowds at a level of detail matched to each character's on-screen relevance: foreground characters receive high-resolution geometry and full PBR appearance, while distant characters are generated more cheaply. The aim is to have high variety while keeping per-garment cost low enough to scale to large crowds.

The pipeline uses fast feed-forward garment generation built on [DrapeNet](https://github.com/liren2515/DrapeNet) (garments as unsigned distance fields, meshed with MeshUDF), and explores different methods of physically-based texturing via DressCode and TexGaussian, and level-of-detail scaling across both geometry and texture.

## Components

- **DrapeNet** — feed-forward garment generation from a samplable latent space.
- **DressCode** — tile-based PBR material generation, applied in UV space.
- **TexGaussian** — feed-forward per-Gaussian PBR for relightable Gaussian-splat rendering.

DressCode, TexGaussian, and Relightable 3D Gaussian are included as submodules: `git clone --recursive`, or `git submodule update --init --recursive`.

## Scripts

### Garment generation (`drapenet` env)

| Script | What it does |
|---|---|
| `generate_one.py` | Generate a single garment mesh (smoke test) |
| `sample_variety.py` | Sample many garments from the latent space |
| `resolution_sweep.py` | Sweep MeshUDF grid resolution vs cost for geometry LOD |

### Texturing — TexGaussian (`texgaussian` env)

| Script | What it does |
|---|---|
| `render_pbr.py` | Render PBR gaussian splats using R3DG's GGX specular + environment map lighting |
| `bake_pbr.py` | Bake gaussian PBR textures onto UV maps → single OBJ + MTL with albedo and metallic-roughness texture maps |

### Texturing — DressCode (`dresscode` env)

| Script | What it does |
|---|---|
| `generate_dresscode_textures.py` | Generate PBR texture maps (diffuse, normal, roughness) from a text prompt |
| `apply_dresscode_textures.py` | UV unwrap a bare mesh with xatlas and apply DressCode textures → textured OBJ + MTL |