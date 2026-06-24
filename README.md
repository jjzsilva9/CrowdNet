# CrowdNet

**LOD-aware generative garment variety for virtual crowds.**

CrowdNet generates clothing for virtual crowds at a level of detail matched to each character's on-screen relevance: foreground characters receive high-resolution geometry and full PBR appearance, while distant characters are generated more cheaply. The aim is to have high variety while keeping per-garment cost low enough to scale to large crowds.

The pipeline uses fast feed-forward garment generation built on [DrapeNet](https://github.com/liren2515/DrapeNet) (garments as unsigned distance fields, meshed with MeshUDF), and explores different methods of physically-based texturing via DressCode and TexGaussian, and level-of-detail scaling across both geometry and texture.

## Components

- **DrapeNet** — feed-forward garment generation from a samplable latent space.
- **DressCode** — tile-based PBR material generation, applied in UV space.
- **TexGaussian** — feed-forward per-Gaussian PBR for relightable Gaussian-splat rendering.

DressCode and TexGaussian are included as submodules: `git clone --recursive`, or `git submodule update --init --recursive`.