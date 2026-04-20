from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

from .common import (
    CaseArtifactBundle,
    load_zero_padded_frames,
    resolve_det_mask_layout,
    require_mask_frames,
    resolve_train_block_ids,
    save_tensor_artifact,
    load_tensor_artifact,
    tracks_from_npz,
)


class TemporalDepthwiseKernel(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        nn.init.zeros_(self.conv.weight)
        with torch.no_grad():
            self.conv.weight[:, 0, kernel_size // 2] = 1.0
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, *, frames: int, grid_h: int, grid_w: int) -> torch.Tensor:
        batch = x.shape[0]
        seq = rearrange(x, "b (f h w) c -> (b h w) c f", f=frames, h=grid_h, w=grid_w)
        conv_dtype = self.conv.weight.dtype
        conv_input = seq if seq.dtype == conv_dtype else seq.to(dtype=conv_dtype)
        filtered = self.conv(conv_input)
        if filtered.dtype != x.dtype:
            filtered = filtered.to(dtype=x.dtype)
        filtered = rearrange(filtered, "(b h w) c f -> b (f h w) c", b=batch, h=grid_h, w=grid_w)
        residual_scale = self.residual_scale if self.residual_scale.dtype == x.dtype else self.residual_scale.to(dtype=x.dtype)
        return x + residual_scale * filtered


class DeTAdapter(nn.Module):
    method_name = "det"
    requires_intermediates = True

    def __init__(self, *, pipe, train_block_ids: list[int] | tuple[int, ...] | None = None):
        super().__init__()
        self.block_ids = resolve_train_block_ids(train_block_ids, pipe=pipe)
        self.block_index = {block_id: idx for idx, block_id in enumerate(self.block_ids)}
        self.kernels = nn.ModuleList([TemporalDepthwiseKernel(pipe.dit.dim) for _ in self.block_ids])
        object.__setattr__(self, "_installed_pipe", None)
        self._original_forwards: dict[int, object] = {}

    def install(self, pipe, *, inference_mode: bool = False) -> None:
        del inference_mode
        if self._installed_pipe is pipe:
            return
        object.__setattr__(self, "_installed_pipe", pipe)
        for block_id in self.block_ids:
            block = pipe.dit.blocks[block_id]
            self._original_forwards[block_id] = block.forward

            def patched_forward(block_module, x, context, t_mod, freqs, size_info, *, _block_id=block_id, _adapter=self):
                x = _adapter._original_forwards[_block_id](x, context, t_mod, freqs, size_info)
                kernel = _adapter.kernels[_adapter.block_index[_block_id]]
                return kernel(
                    x,
                    frames=int(size_info["frames"]),
                    grid_h=int(size_info["tile_size"][0]),
                    grid_w=int(size_info["tile_size"][1]),
                )

            block.forward = types.MethodType(patched_forward, block)

    def remove(self) -> None:
        if self._installed_pipe is None:
            return
        for block_id in self.block_ids:
            self._installed_pipe.dit.blocks[block_id].forward = self._original_forwards[block_id]
        self._original_forwards.clear()
        object.__setattr__(self, "_installed_pipe", None)

    def save(self, artifact_dir) -> dict[str, str]:
        kernel_path = Path(artifact_dir) / "temporal_kernel.safetensors"
        save_tensor_artifact(kernel_path, self.kernels.state_dict())
        return {"temporal_kernel": str(kernel_path)}

    def load(self, artifact_dir, *, device, dtype) -> dict[str, str]:
        kernel_path = Path(artifact_dir) / "temporal_kernel.safetensors"
        state_dict = load_tensor_artifact(kernel_path, device=device, dtype=dtype)
        self.kernels.load_state_dict(state_dict, strict=True)
        return {"temporal_kernel": str(kernel_path)}

    def compute_track_loss(
        self,
        intermediates: list[torch.Tensor],
        size_info: dict[str, int],
        track_path: str | Path,
        *,
        track_weight: float = 1.0,
    ) -> torch.Tensor:
        if not intermediates:
            raise ValueError("DeT tracking loss requires intermediate hidden states.")
        last_block_id = self.block_ids[-1]
        features = intermediates[last_block_id]
        frames = int(size_info["frames"])
        grid_h, grid_w = int(size_info["tile_size"][0]), int(size_info["tile_size"][1])
        feature_map = rearrange(features, "b (f h w) c -> b f c h w", f=frames, h=grid_h, w=grid_w)[0]

        tracks, visibility = tracks_from_npz(track_path)
        tracks_tensor = torch.tensor(tracks, device=feature_map.device, dtype=feature_map.dtype)
        visibility_tensor = torch.tensor(visibility, device=feature_map.device, dtype=feature_map.dtype)
        norm_x = (tracks_tensor[..., 0] / max(grid_w - 1, 1)) * 2 - 1
        norm_y = (tracks_tensor[..., 1] / max(grid_h - 1, 1)) * 2 - 1

        sampled_features = []
        for frame_id in range(frames):
            frame_feature = feature_map[frame_id].unsqueeze(0)
            grid = torch.stack([norm_x[:, frame_id], norm_y[:, frame_id]], dim=-1).view(1, -1, 1, 2)
            sampled = F.grid_sample(frame_feature, grid, mode="bilinear", align_corners=True)
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
            sampled_features.append(sampled)
        sampled_features = torch.stack(sampled_features, dim=1)
        anchor = sampled_features[:, :1]
        diff = (sampled_features - anchor).pow(2).mean(dim=-1) * visibility_tensor
        return float(track_weight) * diff.mean()


def _mask_array(mask_path: Path) -> np.ndarray:
    return (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)


def _mask_tensor(mask_path: Path, *, device: str | torch.device) -> torch.Tensor:
    mask = torch.from_numpy(_mask_array(mask_path).astype(np.float32))
    return mask.unsqueeze(0).unsqueeze(0).to(device=device)


def _video_tensor_from_bundle(bundle: CaseArtifactBundle, *, device: str | torch.device) -> torch.Tensor:
    frames = load_zero_padded_frames(bundle.frames_dir)
    tensor = torch.stack(
        [torch.from_numpy(np.array(frame, dtype=np.float32)).permute(2, 0, 1) for frame in frames],
        dim=0,
    )
    return tensor.unsqueeze(0).to(device=device)


def _normalize_track_layout(
    tracks: torch.Tensor | np.ndarray,
    visibility: torch.Tensor | np.ndarray,
    *,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(tracks, torch.Tensor):
        tracks = tracks.detach().cpu().float().numpy()
    if isinstance(visibility, torch.Tensor):
        visibility = visibility.detach().cpu().float().numpy()

    if tracks.ndim != 4 or visibility.ndim != 3:
        raise ValueError(
            f"Unexpected CoTracker output shapes: tracks={tracks.shape}, visibility={visibility.shape}."
        )
    if tracks.shape[0] != 1 or visibility.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1 from CoTracker, got tracks={tracks.shape}, visibility={visibility.shape}."
        )
    tracks = tracks[0]
    visibility = visibility[0]

    if tracks.shape[0] == int(num_frames):
        tracks = np.transpose(tracks, (1, 0, 2))
    elif tracks.shape[1] != int(num_frames):
        raise ValueError(
            f"Unexpected CoTracker frame layout for num_frames={num_frames}: tracks={tracks.shape}, visibility={visibility.shape}."
        )

    if visibility.shape[0] == int(num_frames):
        visibility = np.transpose(visibility, (1, 0))
    elif visibility.shape[1] != int(num_frames):
        raise ValueError(
            f"Unexpected CoTracker visibility layout for num_frames={num_frames}: {visibility.shape}."
        )
    return tracks.astype(np.float32), visibility.astype(np.float32)


def _tracks_to_feature_grid(
    tracks: np.ndarray,
    visibility: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    latent_frames: int,
    grid_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.linspace(0, tracks.shape[1] - 1, num=int(latent_frames)).round().astype(int).tolist()
    tracks = tracks[:, indices]
    visibility = visibility[:, indices]
    tracks[..., 0] = np.clip(
        tracks[..., 0] / max(float(image_width) - 1.0, 1.0) * (float(grid_size[1]) - 1.0),
        0.0,
        float(grid_size[1]) - 1.0,
    )
    tracks[..., 1] = np.clip(
        tracks[..., 1] / max(float(image_height) - 1.0, 1.0) * (float(grid_size[0]) - 1.0),
        0.0,
        float(grid_size[0]) - 1.0,
    )
    return tracks.astype(np.float32), visibility.astype(np.float32)


def _cotracker_tracks_from_seed_mask(
    bundle: CaseArtifactBundle,
    seed_mask_path: Path,
    *,
    source_num_frames: int,
    latent_frames: int,
    grid_size: tuple[int, int],
    device: str | torch.device,
    cotracker_grid_size: int = 24,
    cotracker_checkpoint: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from cotracker.predictor import CoTrackerPredictor
    except ImportError as exc:
        raise ImportError(
            "Real DeT track generation requires `cotracker`. Install it with `pip install -r requirements-benchmark-train.txt`."
        ) from exc

    video = _video_tensor_from_bundle(bundle, device=device)
    seed_mask = _mask_tensor(seed_mask_path, device=device)

    predictor_kwargs = {}
    if cotracker_checkpoint:
        predictor_kwargs["checkpoint"] = cotracker_checkpoint
    predictor = CoTrackerPredictor(**predictor_kwargs)
    predictor = predictor.to(device)
    predictor.eval()

    with torch.no_grad():
        tracks, visibility = predictor(
            video,
            grid_size=int(cotracker_grid_size),
            segm_mask=seed_mask,
            grid_query_frame=0,
        )

    tracks, visibility = _normalize_track_layout(tracks, visibility, num_frames=source_num_frames)
    image_height = int(video.shape[-2])
    image_width = int(video.shape[-1])
    return _tracks_to_feature_grid(
        tracks,
        visibility,
        image_height=image_height,
        image_width=image_width,
        latent_frames=latent_frames,
        grid_size=grid_size,
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1] - 1, mask.shape[0] - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _sample_points_from_mask(mask: np.ndarray, num_points: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Cannot sample DeT tracks from an empty mask.")
    step = max(1, len(xs) // int(num_points))
    xs = xs[::step][:num_points]
    ys = ys[::step][:num_points]
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def _fallback_tracks_from_masks(
    mask_dir: Path,
    source_num_frames: int,
    *,
    target_frames: int | None = None,
    num_points: int = 64,
    grid_size: tuple[int, int] = (30, 52),
) -> tuple[np.ndarray, np.ndarray]:
    mask_paths = sorted(mask_dir.glob("*.png"))
    if len(mask_paths) != int(source_num_frames):
        raise ValueError(f"Expected {source_num_frames} masks in `{mask_dir}`, but found {len(mask_paths)}.")
    if target_frames is None:
        target_frames = int(source_num_frames)
    indices = np.linspace(0, len(mask_paths) - 1, num=int(target_frames)).round().astype(int).tolist()
    masks = [_mask_array(mask_paths[idx]) for idx in indices]
    base_points = _sample_points_from_mask(masks[0], num_points=num_points)
    base_bbox = _bbox_from_mask(masks[0])
    base_w = max(1, base_bbox[2] - base_bbox[0] + 1)
    base_h = max(1, base_bbox[3] - base_bbox[1] + 1)
    rel = np.stack(
        [
            (base_points[:, 0] - base_bbox[0]) / base_w,
            (base_points[:, 1] - base_bbox[1]) / base_h,
        ],
        axis=-1,
    )
    tracks = []
    visibility = []
    for mask in masks:
        x0, y0, x1, y1 = _bbox_from_mask(mask)
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)
        points = np.stack([x0 + rel[:, 0] * w, y0 + rel[:, 1] * h], axis=-1)
        points[:, 0] = np.clip(points[:, 0] / max(mask.shape[1] - 1, 1) * (grid_size[1] - 1), 0, grid_size[1] - 1)
        points[:, 1] = np.clip(points[:, 1] / max(mask.shape[0] - 1, 1) * (grid_size[0] - 1), 0, grid_size[0] - 1)
        tracks.append(points.astype(np.float32))
        visibility.append(np.ones(points.shape[0], dtype=np.float32))
    return np.stack(tracks, axis=1), np.stack(visibility, axis=1)


def ensure_det_tracks(
    bundle: CaseArtifactBundle,
    *,
    source_num_frames: int,
    latent_frames: int,
    reuse_artifacts: bool,
    grid_size: tuple[int, int] = (30, 52),
    device: str | torch.device = "cuda",
    prefer_cotracker: bool = True,
    cotracker_grid_size: int = 24,
    cotracker_checkpoint: str | None = None,
) -> str:
    if reuse_artifacts and bundle.track_path.exists():
        return str(bundle.track_path)
    mask_layout, mask_paths = resolve_det_mask_layout(bundle, num_frames=source_num_frames)
    if mask_layout == "seed":
        if not prefer_cotracker:
            raise ValueError(
                "A single seed mask was provided for DeT, but `prefer_cotracker` is disabled. "
                "Either enable CoTracker or provide dense per-frame masks."
            )
        tracks, visibility = _cotracker_tracks_from_seed_mask(
            bundle,
            mask_paths[0],
            source_num_frames=source_num_frames,
            latent_frames=latent_frames,
            grid_size=grid_size,
            device=device,
            cotracker_grid_size=cotracker_grid_size,
            cotracker_checkpoint=cotracker_checkpoint,
        )
    else:
        require_mask_frames(bundle, num_frames=source_num_frames)
        tracks, visibility = _fallback_tracks_from_masks(
            bundle.mask_dir,
            source_num_frames,
            target_frames=latent_frames,
            grid_size=grid_size,
        )
    bundle.method_dir.mkdir(parents=True, exist_ok=True)
    np.savez(bundle.track_path, tracks=tracks, visibility=visibility)
    return str(bundle.track_path)
