from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from ..benchmarks import normalize_train_block_ids
from ..data.video import VideoData, save_video
from ..models.utils import load_state_dict

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(slots=True)
class CaseArtifactBundle:
    case_id: str
    method: str
    ref_video_path: Path
    shared_dir: Path
    method_dir: Path
    frames_dir: Path
    processed_video_path: Path
    latent_path: Path
    prompt_path: Path
    shared_metadata_path: Path
    method_metadata_path: Path
    mask_dir: Path
    track_path: Path


def build_case_artifact_bundle(
    artifacts_root: str | Path,
    method: str,
    case_id: str,
    ref_video_path: str | Path,
) -> CaseArtifactBundle:
    artifacts_root = Path(artifacts_root)
    shared_dir = artifacts_root / "shared" / case_id
    method_dir = artifacts_root / method / case_id
    return CaseArtifactBundle(
        case_id=case_id,
        method=method,
        ref_video_path=Path(ref_video_path),
        shared_dir=shared_dir,
        method_dir=method_dir,
        frames_dir=shared_dir / "frames",
        processed_video_path=shared_dir / "ref_32f_832x480.mp4",
        latent_path=shared_dir / "latents.safetensors",
        prompt_path=shared_dir / "prompt.json",
        shared_metadata_path=shared_dir / "metadata.json",
        method_metadata_path=method_dir / "metadata.json",
        mask_dir=shared_dir / "mask",
        track_path=method_dir / "tracks.npz",
    )


def default_grid_size(height: int, width: int) -> tuple[int, int]:
    return int(height) // 16, int(width) // 16


def build_size_info(height: int, width: int, num_frames: int, *, tiled: bool = True) -> dict[str, Any]:
    return {
        "tiled": bool(tiled),
        "tile_size": default_grid_size(height, width),
        "frames": (int(num_frames) - 1) // 4 + 1,
    }


def resolve_train_block_ids(train_block_ids: list[int] | tuple[int, ...] | None, *, pipe) -> tuple[int, ...]:
    return normalize_train_block_ids(train_block_ids, num_layers=len(pipe.dit.blocks))


def save_tensor_artifact(path: str | Path, tensors: dict[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(payload, str(path))


def load_tensor_artifact(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    state_dict = load_state_dict(str(path))
    for name, tensor in state_dict.items():
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        state_dict[name] = tensor
    return state_dict


def save_zero_padded_frames(frames: list[Image.Image], frame_dir: str | Path) -> None:
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(frame_dir / f"{idx:04d}.png")


def load_zero_padded_frames(frame_dir: str | Path) -> list[Image.Image]:
    frame_dir = Path(frame_dir)
    frame_paths = sorted(frame_dir.glob("*.png"))
    return [Image.open(path).convert("RGB") for path in frame_paths]


def _frame_sequence_exists(frame_dir: Path, num_frames: int) -> bool:
    return len(list(frame_dir.glob("*.png"))) == int(num_frames)


def list_image_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        return []
    return sorted(
        [candidate for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES]
    )


def resolve_case_mask_source(case: dict[str, str], mask_dir: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if case.get("mask_dir"):
        candidates.append(Path(case["mask_dir"]))
    if mask_dir is not None:
        root = Path(mask_dir)
        candidates.extend([root / case["case_id"], root / case["ref"], root])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prepare_mask_artifacts(
    case: dict[str, str],
    *,
    mask_dir: str | Path | None,
    target_dir: str | Path,
    height: int,
    width: int,
    num_frames: int,
    reuse_artifacts: bool,
) -> Path | None:
    source = resolve_case_mask_source(case, mask_dir)
    target_dir = Path(target_dir)
    if source is None:
        return None
    if reuse_artifacts and (
        _frame_sequence_exists(target_dir, num_frames)
        or len(list_image_files(target_dir)) == 1
    ):
        return target_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        frame = Image.open(source).convert("L")
        if frame.size != (int(width), int(height)):
            frame = frame.resize((int(width), int(height)), Image.Resampling.NEAREST)
        frame.save(target_dir / "0000.png")
        return target_dir

    if source.is_dir():
        image_files = list_image_files(source)
        if len(image_files) == 1:
            frame = Image.open(image_files[0]).convert("L")
            if frame.size != (int(width), int(height)):
                frame = frame.resize((int(width), int(height)), Image.Resampling.NEAREST)
            frame.save(target_dir / "0000.png")
            return target_dir
        video_data = VideoData(image_folder=str(source), height=height, width=width)
    else:
        video_data = VideoData(video_file=str(source), height=height, width=width)
    if len(video_data) < int(num_frames):
        raise ValueError(
            f"Mask source `{source}` only has {len(video_data)} frames, but {num_frames} frames are required."
        )
    video_data.set_length(int(num_frames))
    frames = [video_data[idx] for idx in range(int(num_frames))]
    save_zero_padded_frames(frames, target_dir)
    return target_dir


def prepare_shared_case_artifacts(
    pipe,
    case: dict[str, str],
    *,
    settings: dict[str, Any],
    artifacts_root: str | Path,
    benchmark_preset: str | None,
    reuse_artifacts: bool = True,
    mask_dir: str | Path | None = None,
    tiled: bool = True,
    tile_size: tuple[int, int] = (30, 52),
    tile_stride: tuple[int, int] = (15, 26),
) -> CaseArtifactBundle:
    bundle = build_case_artifact_bundle(
        artifacts_root=artifacts_root,
        method="shared",
        case_id=case["case_id"],
        ref_video_path=case.get("ref_video_path", Path("data") / f"{case['ref']}.mp4"),
    )
    bundle.shared_dir.mkdir(parents=True, exist_ok=True)

    should_prepare = not (
        reuse_artifacts
        and bundle.processed_video_path.exists()
        and bundle.latent_path.exists()
        and _frame_sequence_exists(bundle.frames_dir, settings["num_frames"])
    )
    if should_prepare:
        ref_video = VideoData(
            str(bundle.ref_video_path),
            height=settings["height"],
            width=settings["width"],
        )
        if len(ref_video) < int(settings["num_frames"]):
            raise ValueError(
                f"Reference video `{bundle.ref_video_path}` only has {len(ref_video)} frames, "
                f"but the benchmark requires {settings['num_frames']}."
            )
        ref_video.set_length(int(settings["num_frames"]))
        frames = [ref_video[idx] for idx in range(int(settings["num_frames"]))]
        save_zero_padded_frames(frames, bundle.frames_dir)
        save_video(frames, str(bundle.processed_video_path), fps=15, quality=5)

        pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            frame_tensors = pipe.preprocess_images(frames)
            frame_tensor = torch.stack(frame_tensors, dim=2).to(dtype=pipe.torch_dtype, device=pipe.device)
            latents = pipe.encode_video(frame_tensor, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        save_tensor_artifact(bundle.latent_path, {"latents": latents})
        pipe.load_models_to_device([])

    copied_mask_dir = prepare_mask_artifacts(
        case,
        mask_dir=mask_dir,
        target_dir=bundle.mask_dir,
        height=settings["height"],
        width=settings["width"],
        num_frames=settings["num_frames"],
        reuse_artifacts=reuse_artifacts,
    )

    prompt_payload = {
        "case_id": case["case_id"],
        "ref": case["ref"],
        "prompt": case["prompt"],
        "negative_prompt": case["negative_prompt"],
        "ref_video": str(bundle.ref_video_path),
        "benchmark_preset": benchmark_preset,
        "frames": int(settings["num_frames"]),
        "height": int(settings["height"]),
        "width": int(settings["width"]),
        "model_variant": settings.get("model_variant", "Wan2.1-T2V-14B"),
    }
    write_json(bundle.prompt_path, prompt_payload)
    write_json(
        bundle.shared_metadata_path,
        {
            **prompt_payload,
            "latents_path": str(bundle.latent_path),
            "frames_dir": str(bundle.frames_dir),
            "processed_video_path": str(bundle.processed_video_path),
            "mask_dir": str(copied_mask_dir) if copied_mask_dir is not None else None,
        },
    )
    return bundle


def load_shared_latents(bundle: CaseArtifactBundle, *, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    state_dict = load_tensor_artifact(bundle.latent_path, device=device, dtype=dtype)
    if "latents" not in state_dict:
        raise ValueError(f"`{bundle.latent_path}` does not contain a `latents` tensor.")
    return state_dict["latents"]


def require_mask_frames(bundle: CaseArtifactBundle, *, num_frames: int) -> Path:
    if not bundle.mask_dir.exists():
        raise ValueError(
            f"DeT requires masks at `{bundle.mask_dir}`, but none were found. "
            "Provide `mask_dir` or per-case `mask_dir` before running DeT."
        )
    mask_count = len(list_image_files(bundle.mask_dir))
    if mask_count != int(num_frames):
        raise ValueError(
            f"DeT requires exactly {num_frames} mask frames in `{bundle.mask_dir}`, but found {mask_count}."
        )
    return bundle.mask_dir


def resolve_det_mask_layout(bundle: CaseArtifactBundle, *, num_frames: int) -> tuple[str, list[Path]]:
    if not bundle.mask_dir.exists():
        raise ValueError(
            f"DeT requires masks at `{bundle.mask_dir}`, but none were found. "
            "Provide `mask_dir` or per-case `mask_dir` before running DeT."
        )
    mask_paths = list_image_files(bundle.mask_dir)
    mask_count = len(mask_paths)
    if mask_count == 1:
        return "seed", mask_paths
    if mask_count == int(num_frames):
        return "dense", mask_paths
    raise ValueError(
        f"DeT expects either 1 seed mask or {num_frames} dense masks in `{bundle.mask_dir}`, "
        f"but found {mask_count}."
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tracks_from_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path))
    return data["tracks"], data["visibility"]
