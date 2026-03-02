import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
import os, re, argparse


# Default model paths
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


def get_model_paths(model_dir):
    """Get model paths from the specified directory."""
    return [
        os.path.join(model_dir, DEFAULT_DIT_MODEL),
        os.path.join(model_dir, DEFAULT_T5_MODEL),
        os.path.join(model_dir, DEFAULT_VAE_MODEL),
    ]


def get_next_video_path(output_dir="results", prefix="video", ext=".mp4"):
    """
    Find all prefixN.ext files in output_dir and return the next available path.
    For example, if video1.mp4 and video2.mp4 exist, returns results/video3.mp4
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")
def main(args):
    # Load models
    model_manager = ModelManager(device="cpu")
    model_paths = get_model_paths(args.model_dir)
    model_manager.load_models(
        model_paths,
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)


    video = VideoData(args.input_video, height=args.height, width=args.width)
    #video.set_length(args.num_frames) #此行开启后，必须保证输入视频帧数和num_frames一致，否则会报错。
    # Text-to-video
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        denoising_strength=args.denoising_strength,
        input_video=video,
        seed=args.seed,
        tiled=True,
        num_frames=args.num_frames,
        sf=args.sf,
        test_latency=args.test_latency,
        latency_dir=args.latency_dir,
        mode=args.mode,
        ttc_enabled=args.ttc_enabled,
        ttc_noise_levels=tuple(args.ttc_noise_levels),
        ttc_step_ratios=tuple(args.ttc_step_ratios),
        ttc_anchor_blend=args.ttc_anchor_blend,
        ttc_debug=args.ttc_debug,
        guidance_steps=args.guidance_steps,
    )
    save_video(video, get_next_video_path(output_dir=args.output_dir), fps=15, quality=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanVideo Text-to-Video Example")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory containing model files (default: models/Wan2.1-T2V-1.3B)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output videos")
    parser.add_argument("--input_video", type=str, default="data/source.mp4",
                        help="Path to reference video for motion transfer")
    parser.add_argument("--height", type=int, default=480, help="Output video height (default: 480)")
    parser.add_argument("--width", type=int, default=832, help="Output video width (default: 832)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_tile", action="store_true", help="Use tiled processing for large videos")
    parser.add_argument("--sf", type=int, default=4, help="Spatial factor for AMF computation (default: 4)")
    parser.add_argument("--test_latency", action="store_true", help="Test latency of the model")
    parser.add_argument("--latency_dir", type=str, default=None, help="Directory to save latency logs")
    parser.add_argument("--mode", type=str, default="effi_AMF", choices=['No_transfer', 'effi_AMF'],help="Mode for the video generation")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to load/use")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for generation")
    parser.add_argument("--guidance_steps", type=int, default=10, help="How many early steps run AMF guidance (set 0 for low-VRAM smoke test)")
    parser.add_argument("--ttc_enabled", action="store_true", help="Enable path-wise test-time correction")
    parser.add_argument("--ttc_noise_levels", type=int, nargs="+", default=[500, 250], help="TTC target noise levels")
    parser.add_argument("--ttc_step_ratios", type=float, nargs="+", default=[0.5, 0.25], help="TTC fallback ratios if noise-level mapping fails")
    parser.add_argument("--ttc_anchor_blend", type=float, default=1.0, help="Blend factor for TTC anchor injection")
    parser.add_argument("--ttc_debug", action="store_true", help="Print resolved TTC step indices")
    parser.add_argument("--denoising_strength", type=float, default=0.75, help="Denoising strength (default: 1.0)")
    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(args)
