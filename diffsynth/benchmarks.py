from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BENCHMARK_PRESETS: dict[str, dict[str, Any]] = {
    "wan14b_32f_832x480": {
        "model_variant": "Wan2.1-T2V-14B",
        "height": 480,
        "width": 832,
        "num_frames": 32,
        "num_inference_steps": 50,
    },
    "wan13b_32f_832x480": {
        "model_variant": "Wan2.1-T2V-1.3B",
        "height": 480,
        "width": 832,
        "num_frames": 32,
        "num_inference_steps": 50,
    },
}


TRANSFER_METHOD_ALIASES = {
    "fastvmt": "fastvmt",
    "ditflow": "ditflow",
    "moft": "moft",
    "smm": "smm",
    "motionclone": "motionclone",
    "no_transfer": "no_transfer",
    "baseline": "no_transfer",
}


LEGACY_MODE_TO_METHOD = {
    None: "fastvmt",
    "effi_AMF": "fastvmt",
    "AMF": "ditflow",
    "MOFT": "moft",
    "No_transfer": "no_transfer",
}


def normalize_transfer_method(transfer_method: str | None = None, mode: str | None = None) -> str:
    if transfer_method is None:
        transfer_method = LEGACY_MODE_TO_METHOD.get(mode, mode)
    if transfer_method is None:
        transfer_method = "fastvmt"
    transfer_method = str(transfer_method).strip().lower()
    if transfer_method not in TRANSFER_METHOD_ALIASES:
        supported = ", ".join(sorted(TRANSFER_METHOD_ALIASES))
        raise ValueError(f"Unsupported transfer_method `{transfer_method}`. Expected one of: {supported}.")
    return TRANSFER_METHOD_ALIASES[transfer_method]


def resolve_benchmark_preset(benchmark_preset: str | None = None) -> dict[str, Any] | None:
    if benchmark_preset is None:
        return None
    preset_name = str(benchmark_preset).strip().lower()
    if preset_name not in BENCHMARK_PRESETS:
        supported = ", ".join(sorted(BENCHMARK_PRESETS))
        raise ValueError(f"Unsupported benchmark_preset `{benchmark_preset}`. Expected one of: {supported}.")
    return dict(BENCHMARK_PRESETS[preset_name])


def apply_benchmark_settings(
    *,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    benchmark_preset: str | None = None,
) -> dict[str, Any]:
    settings = {
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "num_inference_steps": int(num_inference_steps),
    }
    preset = resolve_benchmark_preset(benchmark_preset)
    if preset is None:
        return settings
    settings.update(
        {
            "height": int(preset["height"]),
            "width": int(preset["width"]),
            "num_frames": int(preset["num_frames"]),
            "num_inference_steps": int(preset["num_inference_steps"]),
        }
    )
    settings["benchmark_preset"] = str(benchmark_preset).strip().lower()
    settings["model_variant"] = preset["model_variant"]
    return settings


def enforce_video_length(video_data, *, num_frames: int, strict: bool = True):
    available_frames = len(video_data)
    if available_frames < int(num_frames):
        raise ValueError(
            f"Reference video only has {available_frames} frames, but the benchmark requires {num_frames} frames."
        )
    if strict or available_frames != int(num_frames):
        video_data.set_length(int(num_frames))
    return video_data


def build_run_metadata(
    *,
    prompt: str,
    negative_prompt: str,
    ref_video: str | None,
    output_path: str | Path,
    seed: int | None,
    steps: int,
    frames: int,
    height: int,
    width: int,
    method: str,
    model_variant: str,
    benchmark_preset: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "ref_video": ref_video,
        "output_path": str(output_path),
        "seed": seed,
        "steps": int(steps),
        "frames": int(frames),
        "resolution": f"{int(width)}x{int(height)}",
        "width": int(width),
        "height": int(height),
        "method": method,
        "model_variant": model_variant,
    }
    if benchmark_preset is not None:
        metadata["benchmark_preset"] = str(benchmark_preset).strip().lower()
    if extra:
        metadata.update(extra)
    return metadata


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
