from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..benchmarks import (
    apply_benchmark_settings,
    build_run_metadata,
    get_method_family,
    normalize_stage,
    normalize_transfer_method,
)
from ..data.video import save_video
from ..pipelines.wan_video import WanVideoPipeline
from .common import (
    build_case_artifact_bundle,
    build_size_info,
    load_shared_latents,
    prepare_shared_case_artifacts,
    resolve_train_block_ids,
    write_json,
)
from .det import DeTAdapter, ensure_det_tracks
from .motion_director import MotionDirectorAdapter
from .motion_inversion import MotionInversionAdapter


def build_tuning_method(method: str, *, pipe, settings: dict[str, Any], run_kwargs: dict[str, Any]):
    train_block_ids = resolve_train_block_ids(run_kwargs.get("train_block_ids"), pipe=pipe)
    size_info = build_size_info(settings["height"], settings["width"], settings["num_frames"])
    if method == "motioninversion":
        return MotionInversionAdapter(
            pipe=pipe,
            train_block_ids=train_block_ids,
            max_frames=size_info["frames"],
            grid_size=size_info["tile_size"],
        )
    if method == "motiondirector":
        return MotionDirectorAdapter(
            pipe=pipe,
            train_block_ids=train_block_ids,
            lora_rank=int(run_kwargs.get("motiondirector_lora_rank", 8)),
            lora_alpha=float(run_kwargs.get("motiondirector_lora_alpha", 8.0)),
            temporal_scale=float(run_kwargs.get("temporal_scale", 1.0)),
            spatial_scale=float(run_kwargs.get("spatial_scale", 1.0)),
        )
    if method == "det":
        return DeTAdapter(pipe=pipe, train_block_ids=train_block_ids)
    raise ValueError(f"Unsupported tuning-based method `{method}`.")


def _freeze_pipe_for_tuning(pipe: WanVideoPipeline) -> None:
    for module_name in ["dit", "text_encoder", "vae", "image_encoder", "motion_controller", "vace"]:
        module = getattr(pipe, module_name, None)
        if module is None:
            continue
        module.requires_grad_(False)
        module.eval()
    pipe.dit.train()


def _resolve_track_cache(track_cache: str | Path | None, case: dict[str, str]) -> Path | None:
    if track_cache is None:
        return None
    root = Path(track_cache)
    candidates = [root]
    if root.is_dir():
        candidates = [root / f"{case['case_id']}.npz", root / f"{case['ref']}.npz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@contextlib.contextmanager
def installed_tuner(method_module, pipe, *, inference_mode: bool):
    method_module.install(pipe, inference_mode=inference_mode)
    method_module.to(pipe.device)
    try:
        yield method_module
    finally:
        method_module.remove()


def train_tuning_method(
    pipe: WanVideoPipeline,
    method: str,
    method_module,
    case: dict[str, str],
    bundle,
    *,
    settings: dict[str, Any],
    denoising_strength: float,
    run_kwargs: dict[str, Any],
) -> dict[str, Any]:
    train_steps = int(run_kwargs.get("train_steps", 200))
    train_lr = float(run_kwargs.get("train_lr", 5e-4))
    train_log_interval = int(run_kwargs.get("train_log_interval", 20))
    size_info = build_size_info(settings["height"], settings["width"], settings["num_frames"])
    latents = load_shared_latents(bundle, device=pipe.device, dtype=pipe.torch_dtype)
    if latents.ndim >= 3:
        size_info["frames"] = int(latents.shape[2])
    pipe.scheduler.set_timesteps(
        settings["num_inference_steps"],
        denoising_strength=denoising_strength,
        shift=float(run_kwargs.get("sigma_shift", 7.0)),
        training=True,
    )
    pipe.load_models_to_device(["dit", "text_encoder"])
    _freeze_pipe_for_tuning(pipe)
    prompt_emb = pipe.encode_prompt(case["prompt"], positive=True)
    extra_input = pipe.prepare_extra_input(latents)
    losses = []

    track_path = None
    if method == "det":
        cached_track = _resolve_track_cache(run_kwargs.get("track_cache"), case)
        if cached_track is not None:
            bundle.method_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_track, bundle.track_path)
            track_path = str(bundle.track_path)
        else:
            track_path = ensure_det_tracks(
                bundle,
                source_num_frames=settings["num_frames"],
                latent_frames=size_info["frames"],
                reuse_artifacts=bool(run_kwargs.get("reuse_artifacts", True)),
                grid_size=size_info["tile_size"],
                device=pipe.device,
                prefer_cotracker=bool(run_kwargs.get("det_use_cotracker", True)),
                cotracker_grid_size=int(run_kwargs.get("det_cotracker_grid_size", 24)),
                cotracker_checkpoint=run_kwargs.get("det_cotracker_checkpoint"),
            )

    with installed_tuner(method_module, pipe, inference_mode=False):
        trainable_params = [param for param in method_module.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=train_lr)
        for step in range(train_steps):
            optimizer.zero_grad(set_to_none=True)
            timestep_idx = int(torch.randint(0, len(pipe.scheduler.timesteps), (1,)).item())
            timestep = pipe.scheduler.timesteps[timestep_idx].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise = pipe.generate_noise(latents.shape, seed=None, device=pipe.device, dtype=torch.float32).to(
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)
            target = pipe.scheduler.training_target(latents, noise, timestep)
            model_out, intermediates = pipe.dit(
                noisy_latents,
                timestep=timestep,
                size_info=size_info,
                return_intermediates=True,
                use_gradient_checkpointing=bool(run_kwargs.get("use_gradient_checkpointing", True)),
                **prompt_emb,
                **extra_input,
            )
            denoise_loss = F.mse_loss(model_out.float(), target.float())
            total_loss = denoise_loss
            if method == "det" and track_path is not None:
                total_loss = total_loss + method_module.compute_track_loss(
                    intermediates,
                    size_info,
                    track_path,
                    track_weight=float(run_kwargs.get("det_track_weight", 1.0)),
                )
            total_loss.backward()
            optimizer.step()
            losses.append(float(total_loss.detach().item()))
            if train_log_interval > 0 and ((step + 1) % train_log_interval == 0 or step in {0, train_steps - 1}):
                print(f"[{method}] train step {step + 1}/{train_steps}: loss={float(total_loss.detach().item()):.6f}")

    artifact_paths = method_module.save(bundle.method_dir)
    train_summary = {
        "artifact_dir": str(bundle.method_dir),
        "train_steps": train_steps,
        "train_lr": train_lr,
        "train_blocks": list(getattr(method_module, "block_ids", [])),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "track_path": track_path,
        "checkpoints": artifact_paths,
    }
    write_json(bundle.method_metadata_path, train_summary)
    pipe.load_models_to_device([])
    return train_summary


def infer_tuning_method(
    pipe: WanVideoPipeline,
    method: str,
    method_module,
    case: dict[str, str],
    bundle,
    output_path: str | Path,
    *,
    settings: dict[str, Any],
    denoising_strength: float,
    run_kwargs: dict[str, Any],
) -> dict[str, Any]:
    with installed_tuner(method_module, pipe, inference_mode=True):
        method_module.load(bundle.method_dir, device=pipe.device, dtype=pipe.torch_dtype)
        if method == "motiondirector":
            method_module.set_inference_scales(
                temporal_scale=float(run_kwargs.get("temporal_scale", 1.0)),
                spatial_scale=float(run_kwargs.get("spatial_scale", 0.0)),
            )
        frames = pipe(
            prompt=case["prompt"],
            negative_prompt=case["negative_prompt"],
            input_video=None,
            seed=run_kwargs["seed"],
            height=settings["height"],
            width=settings["width"],
            num_frames=settings["num_frames"],
            num_inference_steps=settings["num_inference_steps"],
            denoising_strength=denoising_strength,
            cfg_scale=run_kwargs["cfg_scale"],
            sigma_shift=run_kwargs["sigma_shift"],
            tiled=True,
            sf=run_kwargs["sf"],
            benchmark_preset=run_kwargs.get("benchmark_preset"),
            transfer_method="no_transfer",
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(output_path), fps=15, quality=5)
    summary = pipe.last_run_summary or {}
    return {
        "output_path": str(output_path),
        "decoded_num_frames": summary.get("decoded_num_frames"),
        "output_num_frames": summary.get("output_num_frames"),
        "requested_num_frames": summary.get("requested_num_frames"),
    }


def run_tuning_case(
    pipe: WanVideoPipeline,
    case: dict[str, str],
    output_dir: str | Path,
    denoising_strength: float,
    **run_kwargs: Any,
) -> dict[str, Any]:
    settings = apply_benchmark_settings(
        height=run_kwargs["height"],
        width=run_kwargs["width"],
        num_frames=run_kwargs["num_frames"],
        num_inference_steps=run_kwargs["num_inference_steps"],
        benchmark_preset=run_kwargs.get("benchmark_preset"),
    )
    method = normalize_transfer_method(run_kwargs.get("transfer_method"), run_kwargs.get("mode"))
    stage = normalize_stage(run_kwargs.get("stage"))
    artifacts_root = run_kwargs.get("artifacts_root", "artifacts")

    prepare_shared_case_artifacts(
        pipe,
        case,
        settings=settings,
        artifacts_root=artifacts_root,
        benchmark_preset=run_kwargs.get("benchmark_preset"),
        reuse_artifacts=bool(run_kwargs.get("reuse_artifacts", True)),
        mask_dir=run_kwargs.get("mask_dir"),
    )
    bundle = build_case_artifact_bundle(
        artifacts_root,
        method,
        case["case_id"],
        case.get("ref_video_path", Path("data") / f"{case['ref']}.mp4"),
    )
    method_module = build_tuning_method(method, pipe=pipe, settings=settings, run_kwargs=run_kwargs).to(pipe.device)

    train_summary = None
    infer_summary = None
    if stage in {"train", "full"}:
        train_summary = train_tuning_method(
            pipe,
            method,
            method_module,
            case,
            bundle,
            settings=settings,
            denoising_strength=denoising_strength,
            run_kwargs=run_kwargs,
        )

    if stage in {"infer", "full"}:
        output_path = Path(output_dir) / method / f"{case['case_id']}.mp4"
        infer_summary = infer_tuning_method(
            pipe,
            method,
            method_module,
            case,
            bundle,
            output_path,
            settings=settings,
            denoising_strength=denoising_strength,
            run_kwargs=run_kwargs,
        )

    metadata = build_run_metadata(
        prompt=case["prompt"],
        negative_prompt=case["negative_prompt"],
        ref_video=str(case.get("ref_video_path", Path("data") / f"{case['ref']}.mp4")),
        output_path=(infer_summary or {}).get("output_path", str(bundle.method_dir)),
        seed=run_kwargs["seed"],
        steps=settings["num_inference_steps"],
        frames=(infer_summary or {}).get("output_num_frames", settings["num_frames"]),
        height=settings["height"],
        width=settings["width"],
        method=method,
        model_variant=settings.get("model_variant", "Wan2.1-T2V-14B"),
        benchmark_preset=run_kwargs.get("benchmark_preset"),
        method_family=get_method_family(method),
        stage=stage,
        extra={
            "case_id": case["case_id"],
            "artifact_dir": str(bundle.method_dir),
            "train_checkpoint": str(bundle.method_dir) if train_summary is not None else None,
            "train_steps": train_summary["train_steps"] if train_summary is not None else int(run_kwargs.get("train_steps", 200)),
            "train_blocks": train_summary["train_blocks"] if train_summary is not None else list(getattr(method_module, "block_ids", [])),
            "mask_dir": str(bundle.mask_dir) if bundle.mask_dir.exists() else None,
            "track_path": (train_summary or {}).get("track_path"),
            "method_artifacts": (train_summary or {}).get("checkpoints"),
        },
    )
    if infer_summary is not None:
        metadata_path = Path(infer_summary["output_path"]).with_suffix(".json")
    else:
        metadata_path = bundle.method_dir / f"{stage}.json"
    write_json(metadata_path, metadata)
    return metadata
