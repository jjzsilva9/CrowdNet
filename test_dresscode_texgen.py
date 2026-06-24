"""
Standalone test: DressCode PBR texture generation.
Generates diffuse/normal/roughness maps from a text prompt.
Run with: conda activate dresscode && python test_dresscode_texgen.py
"""
import sys
import os
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "DressCode"))

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

MATERIAL_GEN_DIR = os.path.join(REPO_ROOT, "external", "DressCode", "nn", "material_gen")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "textures")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def patch_conv(module):
    if isinstance(module, torch.nn.Conv2d):
        module.padding_mode = "circular"


def main():
    device = "cuda"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    print("Loading VAEs...")
    t0 = time.time()

    vae_diffuse = AutoencoderKL.from_pretrained(
        os.path.join(MATERIAL_GEN_DIR, "refine_vae"),
        subfolder="vae_checkpoint_diffuse",
        revision="fp16",
        local_files_only=True,
        torch_dtype=torch.float16,
    ).half().to(device)

    vae_normal = AutoencoderKL.from_pretrained(
        os.path.join(MATERIAL_GEN_DIR, "refine_vae"),
        subfolder="vae_checkpoint_normal",
        revision="fp16",
        local_files_only=True,
        torch_dtype=torch.float16,
    ).half().to(device)

    vae_roughness = AutoencoderKL.from_pretrained(
        os.path.join(MATERIAL_GEN_DIR, "refine_vae"),
        subfolder="vae_checkpoint_roughness",
        revision="fp16",
        local_files_only=True,
        torch_dtype=torch.float16,
    ).half().to(device)

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MATERIAL_GEN_DIR,
        torch_dtype=torch.float16,
        safety_checker=None,
        vae=vae_diffuse,
        local_files_only=True,
    ).to(device)

    pipe.unet.apply(patch_conv)
    pipe.vae.apply(patch_conv)
    vae_diffuse.apply(patch_conv)
    vae_normal.apply(patch_conv)
    vae_roughness.apply(patch_conv)

    load_time = time.time() - t0
    print(f"Models loaded in {load_time:.1f}s")

    prompts = [
        "Deep grey fabric",
        "red leather",
        "blue denim",
    ]

    for prompt in prompts:
        print(f"\nGenerating textures for: '{prompt}'")
        t1 = time.time()

        with torch.no_grad():
            latents = pipe(
                [prompt], 512, 512,
                output_type="latent",
                return_dict=True,
            )[0]

            pt = vae_diffuse.decode(
                latents / vae_diffuse.config.scaling_factor,
                return_dict=False,
            )[0]
            diffuse = pipe.image_processor.postprocess(
                pt, output_type="pil", do_denormalize=[True],
            )[0]

            pt = vae_normal.decode(
                latents / vae_normal.config.scaling_factor,
                return_dict=False,
            )[0]
            normal = pipe.image_processor.postprocess(
                pt, output_type="pil", do_denormalize=[True],
            )[0]

            pt = vae_roughness.decode(
                latents / vae_roughness.config.scaling_factor,
                return_dict=False,
            )[0]
            roughness = pipe.image_processor.postprocess(
                pt, output_type="pil", do_denormalize=[True],
            )[0]

        gen_time = time.time() - t1
        safe_name = prompt.replace(" ", "_").lower()
        diffuse.save(os.path.join(OUTPUT_DIR, f"{safe_name}_diffuse.png"))
        normal.save(os.path.join(OUTPUT_DIR, f"{safe_name}_normal.png"))
        roughness.save(os.path.join(OUTPUT_DIR, f"{safe_name}_roughness.png"))

        print(f"  Saved to {OUTPUT_DIR}/{safe_name}_*.png")
        print(f"  Generation time: {gen_time:.2f}s")

    print("\nDone. All textures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
