#!/usr/bin/env python
from __future__ import annotations

import gc
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wan_14b_text_to_video as wan
from diffsynth import (
    ModelManager,
    VideoData,
    WanVideoPipeline,
    apply_benchmark_settings,
    build_run_metadata,
    normalize_transfer_method,
    save_video,
    write_metadata,
)


DEFAULT_RUN_KWARGS: dict[str, Any] = {
    "seed": 42,
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "num_inference_steps": 50,
    "cfg_scale": 5.8,
    "sigma_shift": 7.0,
    "mode": "effi_AMF",
    "transfer_method": "fastvmt",
    "benchmark_preset": "wan14b_32f_832x480",
    "group_by_method": True,
    "sf": 4,
    "guidance_steps": 10,
    "msa_optim_start": 0,
    "msa_optim_end": 1,
    "msa_iter": 2,
    "msa_scale_list": (50.0, 300.0),
    "msa_mask_mode": "amf",
    "msa_mask_power": 1.0,
    "msa_mask_min": 0.15,
    "msa_balance_with_amf": True,
    "msa_debug": False,
    "ttc_noise_levels": (500, 250),
    "ttc_step_ratios": (0.5, 0.25),
    "ttc_anchor_blend": 1.0,
    "ttc_anchor_mode": "hybrid",
    "ttc_anchor_ref_weight": 0.25,
    "ttc_anchor_blend_start": 0.35,
    "ttc_anchor_blend_end": 0.85,
    "ttc_debug": False,
    "ttc_enabled": True,
    "msa_enabled": True,
}


def build_pipe(model_dir: str = wan.DEFAULT_MODEL_DIR) -> WanVideoPipeline:
    model_manager = ModelManager(device="cpu")
    model_paths = wan.discover_model_paths(model_dir)
    model_manager.load_models(model_paths, torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    return pipe


def parse_shell_cases(shell_text: str) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for raw_line in shell_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("run_case "):
            continue

        tokens = shlex.split(line)
        if len(tokens) == 4:
            _, ref, prompt, neg = tokens
            case_id = ref
        elif len(tokens) == 5:
            _, ref, variant, prompt, neg = tokens
            case_id = f"{ref}_{variant}"
        else:
            raise ValueError(f"Unexpected run_case format: {line}")

        cases.append(
            {
                "case_id": case_id,
                "ref": ref,
                "prompt": prompt,
                "negative_prompt": neg,
            }
        )

    if not cases:
        raise ValueError("No run_case entries were found in the shell script.")
    return cases


def parse_shell_metadata(shell_text: str) -> dict[str, Any]:
    output_dir_match = re.search(r"^\s*mkdir -p\s+(.+)$", shell_text, re.MULTILINE)
    gpu_match = re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", shell_text)
    denoise_match = re.search(r"--denoising_strength\s+([0-9]*\.?[0-9]+)", shell_text)
    mask_match = re.search(r"--msa_mask_mode\s+(\w+)", shell_text)

    if output_dir_match is None:
        raise ValueError("Could not detect output directory from shell script.")
    if gpu_match is None:
        raise ValueError("Could not detect CUDA_VISIBLE_DEVICES from shell script.")
    if denoise_match is None:
        raise ValueError("Could not detect denoising strength from shell script.")

    return {
        "output_dir": output_dir_match.group(1).strip(),
        "gpu": gpu_match.group(1),
        "denoising_strength": float(denoise_match.group(1)),
        "msa_mask_mode": mask_match.group(1) if mask_match else "amf",
    }


def _resolve_run_label(kwargs: dict[str, Any]) -> str:
    transfer_method = normalize_transfer_method(kwargs.get("transfer_method"), kwargs.get("mode"))
    if transfer_method == "fastvmt":
        if kwargs.get("ttc_enabled") and kwargs.get("msa_enabled"):
            return "fastvmt"
        if kwargs.get("ttc_enabled"):
            return "ttc_only"
        if kwargs.get("msa_enabled"):
            return "msa_only"
        return "baseline"
    return transfer_method


def run_case(
    pipe: WanVideoPipeline,
    case: dict[str, str],
    output_dir: str | Path,
    denoising_strength: float,
    **run_kwargs: Any,
) -> dict[str, Any]:
    run_label = run_kwargs.pop("run_label", None)
    kwargs = dict(DEFAULT_RUN_KWARGS)
    kwargs.update(run_kwargs)

    settings = apply_benchmark_settings(
        height=kwargs["height"],
        width=kwargs["width"],
        num_frames=kwargs["num_frames"],
        num_inference_steps=kwargs["num_inference_steps"],
        benchmark_preset=kwargs.get("benchmark_preset"),
    )
    transfer_method = normalize_transfer_method(kwargs.get("transfer_method"), kwargs.get("mode"))
    method_dir_name = _resolve_run_label(kwargs) if kwargs.get("group_by_method", True) else None

    ref = case["ref"]
    case_id = case["case_id"]
    if run_label is None:
        run_label = method_dir_name or transfer_method
    print(f"===== Running {case_id} / {run_label} =====")

    input_video = VideoData(
        str(Path("data") / f"{ref}.mp4"),
        height=settings["height"],
        width=settings["width"],
    )
    if kwargs.get("benchmark_preset") is not None:
        available_frames = len(input_video)
        if available_frames < settings["num_frames"]:
            raise ValueError(
                f"Reference video only has {available_frames} frames, but benchmark_preset "
                f"`{kwargs['benchmark_preset']}` requires {settings['num_frames']} frames."
            )
        input_video.set_length(settings["num_frames"])

    frames = pipe(
        prompt=case["prompt"],
        negative_prompt=case["negative_prompt"],
        num_inference_steps=settings["num_inference_steps"],
        denoising_strength=denoising_strength,
        cfg_scale=kwargs["cfg_scale"],
        sigma_shift=kwargs["sigma_shift"],
        input_image=None,
        end_image=None,
        input_video=input_video,
        seed=kwargs["seed"],
        tiled=True,
        height=settings["height"],
        width=settings["width"],
        num_frames=settings["num_frames"],
        sf=kwargs["sf"],
        test_latency=False,
        latency_dir=None,
        transfer_method=transfer_method,
        benchmark_preset=kwargs.get("benchmark_preset"),
        mode=kwargs["mode"],
        ttc_enabled=kwargs["ttc_enabled"],
        ttc_noise_levels=tuple(kwargs["ttc_noise_levels"]),
        ttc_step_ratios=tuple(kwargs["ttc_step_ratios"]),
        ttc_anchor_blend=kwargs["ttc_anchor_blend"],
        ttc_anchor_mode=kwargs["ttc_anchor_mode"],
        ttc_anchor_ref_weight=kwargs["ttc_anchor_ref_weight"],
        ttc_anchor_blend_start=kwargs["ttc_anchor_blend_start"],
        ttc_anchor_blend_end=kwargs["ttc_anchor_blend_end"],
        ttc_debug=kwargs["ttc_debug"],
        guidance_steps=kwargs["guidance_steps"],
        msa_enabled=kwargs["msa_enabled"],
        msa_optim_start=kwargs["msa_optim_start"],
        msa_optim_end=kwargs["msa_optim_end"],
        msa_iter=kwargs["msa_iter"],
        msa_scale_list=tuple(kwargs["msa_scale_list"]),
        msa_mask_mode=kwargs["msa_mask_mode"],
        msa_mask_power=kwargs["msa_mask_power"],
        msa_mask_min=kwargs["msa_mask_min"],
        msa_balance_with_amf=kwargs["msa_balance_with_amf"],
        msa_debug=kwargs["msa_debug"],
    )

    case_output_dir = Path(output_dir)
    if method_dir_name is not None:
        case_output_dir = case_output_dir / method_dir_name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_output_dir / f"{case_id}.mp4"
    save_video(frames, str(output_path), fps=15, quality=5)

    summary = pipe.last_run_summary or {}
    metadata = build_run_metadata(
        prompt=case["prompt"],
        negative_prompt=case["negative_prompt"],
        ref_video=str(Path("data") / f"{ref}.mp4"),
        output_path=output_path,
        seed=kwargs["seed"],
        steps=summary.get("num_inference_steps", settings["num_inference_steps"]),
        frames=summary.get("output_num_frames", summary.get("num_frames", len(frames))),
        height=summary.get("height", settings["height"]),
        width=summary.get("width", settings["width"]),
        method=summary.get("transfer_method", transfer_method),
        model_variant=settings.get("model_variant", "Wan2.1-T2V-14B"),
        benchmark_preset=kwargs.get("benchmark_preset"),
        extra={
            "case_id": case_id,
            "run_label": run_label,
            "output_dir": str(case_output_dir),
            "requested_num_frames": summary.get("requested_num_frames", settings["num_frames"]),
            "decoded_num_frames": summary.get("decoded_num_frames", len(frames)),
            "output_num_frames": summary.get("output_num_frames", len(frames)),
        },
    )
    write_metadata(output_path.with_suffix(".json"), metadata)

    del frames
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def run_cases(
    pipe: WanVideoPipeline,
    cases: list[dict[str, str]],
    output_dir: str | Path,
    denoising_strength: float,
    **run_kwargs: Any,
) -> list[dict[str, Any]]:
    metadata_entries = []
    for case in cases:
        metadata_entries.append(run_case(pipe, case, output_dir, denoising_strength, **run_kwargs))
    root_metadata = {
        "cases": metadata_entries,
        "num_cases": len(metadata_entries),
    }
    root_path = Path(output_dir) / "metadata.json"
    write_metadata(root_path, root_metadata)
    print(f"===== All {len(cases)} cases completed =====")
    return metadata_entries


def run_from_shell(
    shell_path: str | Path,
    model_dir: str = wan.DEFAULT_MODEL_DIR,
    **run_kwargs: Any,
) -> None:
    shell_path = Path(shell_path)
    shell_text = shell_path.read_text(encoding="utf-8", errors="ignore")
    metadata = parse_shell_metadata(shell_text)
    cases = parse_shell_cases(shell_text)

    os.environ["CUDA_VISIBLE_DEVICES"] = metadata["gpu"]
    pipe = build_pipe(model_dir)

    merged_kwargs = dict(DEFAULT_RUN_KWARGS)
    merged_kwargs.update(run_kwargs)
    merged_kwargs["msa_mask_mode"] = metadata["msa_mask_mode"]

    run_cases(
        pipe,
        cases,
        output_dir=metadata["output_dir"],
        denoising_strength=metadata["denoising_strength"],
        **merged_kwargs,
    )
