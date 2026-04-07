import argparse
import os
import re
from pathlib import Path

import torch

from diffsynth import (
    ModelManager,
    VideoData,
    WanVideoPipeline,
    apply_benchmark_settings,
    build_run_metadata,
    save_video,
    write_metadata,
)


DEFAULT_MODEL_DIR = "models/Wan2.1-T2V-1.3B"
DEFAULT_DIT_MODEL = "diffusion_pytorch_model.safetensors"
DEFAULT_T5_MODEL = "models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_VAE_MODEL = "Wan2.1_VAE.pth"
DEFAULT_PROMPT = (
    "Documentary photography style. A lively puppy running quickly on a green grass field. "
    "The puppy has brown-yellow fur, ears perked up, with a focused and joyful expression. "
    "Sunlight shines on it, making the fur look extra soft and shiny. The background is an open "
    "grass field, occasionally dotted with wildflowers, with blue sky and white clouds visible in "
    "the distance. Strong perspective, capturing the puppy's dynamic movement and the vitality of "
    "the surrounding grass. Medium shot, side tracking view."
)
DEFAULT_NEGATIVE_PROMPT = (
    "vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, "
    "still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra "
    "fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused "
    "fingers, static frame, cluttered background, three legs, many people in background, walking backwards"
)


def discover_model_paths(model_dir):
    return [
        os.path.join(model_dir, DEFAULT_DIT_MODEL),
        os.path.join(model_dir, DEFAULT_T5_MODEL),
        os.path.join(model_dir, DEFAULT_VAE_MODEL),
    ]


def get_model_paths(model_dir):
    return discover_model_paths(model_dir)


def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for fn in os.listdir(output_dir):
        match = pattern.match(fn)
        if match:
            nums.append(int(match.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")


def main(args):
    settings = apply_benchmark_settings(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        benchmark_preset=args.benchmark_preset,
    )

    model_manager = ModelManager(device="cpu")
    model_manager.load_models(discover_model_paths(args.model_dir), torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    input_video = VideoData(args.input_video, height=settings["height"], width=settings["width"])
    if args.benchmark_preset is not None:
        available_frames = len(input_video)
        if available_frames < settings["num_frames"]:
            raise ValueError(
                f"Reference video only has {available_frames} frames, but benchmark_preset "
                f"`{args.benchmark_preset}` requires {settings['num_frames']} frames."
            )
        input_video.set_length(settings["num_frames"])

    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=settings["num_inference_steps"],
        denoising_strength=args.denoising_strength,
        input_video=input_video,
        seed=args.seed,
        tiled=True,
        height=settings["height"],
        width=settings["width"],
        num_frames=settings["num_frames"],
        sf=args.sf,
        test_latency=args.test_latency,
        latency_dir=args.latency_dir,
        transfer_method=args.transfer_method,
        benchmark_preset=args.benchmark_preset,
        mode=args.mode,
        ttc_enabled=args.ttc_enabled,
        ttc_noise_levels=tuple(args.ttc_noise_levels),
        ttc_step_ratios=tuple(args.ttc_step_ratios),
        ttc_anchor_blend=args.ttc_anchor_blend,
        ttc_anchor_mode=args.ttc_anchor_mode,
        ttc_anchor_ref_weight=args.ttc_anchor_ref_weight,
        ttc_anchor_blend_start=args.ttc_anchor_blend_start,
        ttc_anchor_blend_end=args.ttc_anchor_blend_end,
        ttc_debug=args.ttc_debug,
        guidance_steps=args.guidance_steps,
        msa_enabled=args.msa_enabled,
        msa_optim_start=args.msa_optim_start,
        msa_optim_end=args.msa_optim_end,
        msa_iter=args.msa_iter,
        msa_scale_list=tuple(args.msa_scale_list),
        msa_mask_mode=args.msa_mask_mode,
        msa_mask_power=args.msa_mask_power,
        msa_mask_min=args.msa_mask_min,
        msa_balance_with_amf=args.msa_balance_with_amf,
        msa_debug=args.msa_debug,
    )
    output_path = get_next_video_path(output_dir=args.output_dir)
    save_video(frames, output_path, fps=15, quality=5)

    summary = pipe.last_run_summary or {}
    metadata = build_run_metadata(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        ref_video=args.input_video,
        output_path=output_path,
        seed=args.seed,
        steps=summary.get("num_inference_steps", settings["num_inference_steps"]),
        frames=summary.get("output_num_frames", summary.get("num_frames", len(frames))),
        height=summary.get("height", settings["height"]),
        width=summary.get("width", settings["width"]),
        method=summary.get("transfer_method", args.transfer_method or args.mode or "fastvmt"),
        model_variant=settings.get("model_variant", "Wan2.1-T2V-1.3B"),
        benchmark_preset=args.benchmark_preset,
        extra={
            "requested_num_frames": summary.get("requested_num_frames", settings["num_frames"]),
            "decoded_num_frames": summary.get("decoded_num_frames", len(frames)),
            "output_num_frames": summary.get("output_num_frames", len(frames)),
        },
    )
    write_metadata(Path(output_path).with_suffix(".json"), metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo Text-to-Video Example")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--input_video", type=str, default="data/source.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sf", type=int, default=4)
    parser.add_argument("--test_latency", action="store_true")
    parser.add_argument("--latency_dir", type=str, default=None)
    parser.add_argument(
        "--transfer_method",
        type=str,
        default=None,
        choices=["fastvmt", "ditflow", "moft", "smm", "motionclone", "no_transfer"],
        help="Unified transfer method interface for Wan-native benchmark baselines",
    )
    parser.add_argument(
        "--benchmark_preset",
        type=str,
        default=None,
        choices=["wan14b_32f_832x480", "wan13b_32f_832x480"],
        help="Optional benchmark preset that overrides frames, resolution, and denoising steps",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="effi_AMF",
        choices=["No_transfer", "effi_AMF", "AMF", "MOFT"],
        help="Legacy compatibility mode. Prefer --transfer_method for new benchmark runs.",
    )
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--guidance_steps", type=int, default=10)
    parser.add_argument("--msa_enabled", action="store_true")
    parser.add_argument("--msa_optim_start", type=int, default=0)
    parser.add_argument("--msa_optim_end", type=int, default=1)
    parser.add_argument("--msa_iter", type=int, default=2)
    parser.add_argument("--msa_scale_list", type=float, nargs="+", default=[50.0, 300.0])
    parser.add_argument("--msa_mask_mode", type=str, default="uniform", choices=["uniform", "amf"])
    parser.add_argument("--msa_mask_power", type=float, default=1.0)
    parser.add_argument("--msa_mask_min", type=float, default=0.15)
    parser.add_argument("--msa_balance_with_amf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--msa_debug", action="store_true")
    parser.add_argument("--ttc_enabled", action="store_true")
    parser.add_argument("--ttc_noise_levels", type=int, nargs="+", default=[500, 250])
    parser.add_argument("--ttc_step_ratios", type=float, nargs="+", default=[0.5, 0.25])
    parser.add_argument("--ttc_anchor_blend", type=float, default=1.0)
    parser.add_argument(
        "--ttc_anchor_mode",
        type=str,
        default="hybrid",
        choices=["legacy_input_clean", "pred_x0", "hybrid"],
    )
    parser.add_argument("--ttc_anchor_ref_weight", type=float, default=0.25)
    parser.add_argument("--ttc_anchor_blend_start", type=float, default=0.35)
    parser.add_argument("--ttc_anchor_blend_end", type=float, default=0.12)
    parser.add_argument("--ttc_debug", action="store_true")
    parser.add_argument("--denoising_strength", type=float, default=0.75)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
