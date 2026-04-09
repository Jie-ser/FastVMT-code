from __future__ import annotations

import types
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

from ..models.wan_video_dit import flash_attention, rope_apply
from .common import load_tensor_artifact, resolve_train_block_ids, save_tensor_artifact


class MotionInversionAdapter(nn.Module):
    method_name = "motioninversion"

    def __init__(
        self,
        *,
        pipe,
        train_block_ids: list[int] | tuple[int, ...] | None = None,
        max_frames: int = 8,
        grid_size: tuple[int, int] = (30, 52),
    ):
        super().__init__()
        self.block_ids = resolve_train_block_ids(train_block_ids, pipe=pipe)
        self.max_frames = int(max_frames)
        self.grid_size = tuple(int(v) for v in grid_size)
        self.block_index = {block_id: idx for idx, block_id in enumerate(self.block_ids)}
        dim = pipe.dit.dim
        spatial_tokens = self.grid_size[0] * self.grid_size[1]
        self.qk_frame_bias = nn.Parameter(torch.zeros(len(self.block_ids), self.max_frames, dim))
        self.v_frame_bias = nn.Parameter(torch.zeros(len(self.block_ids), self.max_frames, dim))
        self.v_spatial_bias = nn.Parameter(torch.zeros(len(self.block_ids), spatial_tokens, dim))
        self._original_forwards: dict[int, object] = {}
        self._installed_pipe = None
        self.inference_mode = False

    def install(self, pipe, *, inference_mode: bool = False) -> None:
        if self._installed_pipe is pipe:
            self.inference_mode = bool(inference_mode)
            return
        self._installed_pipe = pipe
        self.inference_mode = bool(inference_mode)
        for block_id in self.block_ids:
            attn = pipe.dit.blocks[block_id].self_attn
            self._original_forwards[block_id] = attn.forward

            def patched_forward(attn_module, x, freqs, size_info, *, _block_id=block_id, _adapter=self):
                return _adapter._forward_block(_block_id, attn_module, x, freqs, size_info)

            attn.forward = types.MethodType(patched_forward, attn)

    def remove(self) -> None:
        if self._installed_pipe is None:
            return
        for block_id in self.block_ids:
            self._installed_pipe.dit.blocks[block_id].self_attn.forward = self._original_forwards[block_id]
        self._original_forwards.clear()
        self._installed_pipe = None

    def _expand_frame_bias(self, bias: torch.Tensor, frames: int, spatial_tokens: int, *, device, dtype) -> torch.Tensor:
        bias = bias[:frames].to(device=device, dtype=dtype)
        bias = bias.unsqueeze(1).expand(frames, spatial_tokens, bias.shape[-1])
        return rearrange(bias, "f s c -> 1 (f s) c")

    def _expand_spatial_bias(self, bias: torch.Tensor, frames: int, spatial_tokens: int, *, device, dtype) -> torch.Tensor:
        bias = bias[:spatial_tokens].to(device=device, dtype=dtype)
        bias = bias.unsqueeze(0).expand(frames, spatial_tokens, bias.shape[-1])
        return rearrange(bias, "f s c -> 1 (f s) c")

    def _build_qk_bias(self, block_offset: int, size_info: dict[str, int], *, device, dtype) -> torch.Tensor:
        frames = int(size_info["frames"])
        spatial_tokens = int(size_info["tile_size"][0]) * int(size_info["tile_size"][1])
        return self._expand_frame_bias(self.qk_frame_bias[block_offset], frames, spatial_tokens, device=device, dtype=dtype)

    def _build_v_bias(self, block_offset: int, size_info: dict[str, int], *, device, dtype) -> torch.Tensor:
        frames = int(size_info["frames"])
        spatial_tokens = int(size_info["tile_size"][0]) * int(size_info["tile_size"][1])
        frame_bias = self.v_frame_bias[block_offset]
        if self.inference_mode:
            temporal_delta = torch.zeros_like(frame_bias)
            temporal_delta[1:] = frame_bias[1:] - frame_bias[:-1]
            return self._expand_frame_bias(temporal_delta, frames, spatial_tokens, device=device, dtype=dtype)
        spatial_bias = self._expand_spatial_bias(self.v_spatial_bias[block_offset], frames, spatial_tokens, device=device, dtype=dtype)
        return self._expand_frame_bias(frame_bias, frames, spatial_tokens, device=device, dtype=dtype) + spatial_bias

    def _forward_block(self, block_id: int, attn_module, x, freqs, size_info):
        block_offset = self.block_index[block_id]
        q = attn_module.norm_q(attn_module.q(x))
        k = attn_module.norm_k(attn_module.k(x))
        v = attn_module.v(x)
        q = rope_apply(q, freqs, attn_module.num_heads)
        k = rope_apply(k, freqs, attn_module.num_heads)

        qk_bias = self._build_qk_bias(block_offset, size_info, device=x.device, dtype=q.dtype)
        q = q + qk_bias
        k = k + qk_bias
        v = v + self._build_v_bias(block_offset, size_info, device=x.device, dtype=v.dtype)

        if attn_module.save_qk:
            frames = int(size_info["frames"])
            grid_h, grid_w = int(size_info["tile_size"][0]), int(size_info["tile_size"][1])
            q_reshaped = rearrange(
                q,
                "b (f h w) (n d) -> b f h w n d",
                f=frames,
                h=grid_h,
                w=grid_w,
                n=attn_module.num_heads,
                d=attn_module.head_dim,
            ).mean(dim=4)
            k_reshaped = rearrange(
                k,
                "b (f h w) (n d) -> b f h w n d",
                f=frames,
                h=grid_h,
                w=grid_w,
                n=attn_module.num_heads,
                d=attn_module.head_dim,
            ).mean(dim=4)
            attn_module.q_reshape = q_reshaped[0] if q_reshaped.shape[0] == 1 else q_reshaped
            attn_module.k_reshape = k_reshaped[0] if k_reshaped.shape[0] == 1 else k_reshaped

        x_out = flash_attention(q=q, k=k, v=v, num_heads=attn_module.num_heads)
        return attn_module.o(x_out)

    def save(self, artifact_dir) -> dict[str, str]:
        artifact_dir = Path(artifact_dir)
        qk_path = artifact_dir / "motion_qk.safetensors"
        v_path = artifact_dir / "motion_v.safetensors"
        save_tensor_artifact(qk_path, {"qk_frame_bias": self.qk_frame_bias})
        save_tensor_artifact(
            v_path,
            {
                "v_frame_bias": self.v_frame_bias,
                "v_spatial_bias": self.v_spatial_bias,
            },
        )
        return {
            "motion_qk": str(qk_path),
            "motion_v": str(v_path),
        }

    def load(self, artifact_dir, *, device, dtype) -> dict[str, str]:
        artifact_dir = Path(artifact_dir)
        qk_path = artifact_dir / "motion_qk.safetensors"
        v_path = artifact_dir / "motion_v.safetensors"
        qk_state = load_tensor_artifact(qk_path, device=device, dtype=dtype)
        v_state = load_tensor_artifact(v_path, device=device, dtype=dtype)
        self.qk_frame_bias.data.copy_(qk_state["qk_frame_bias"])
        self.v_frame_bias.data.copy_(v_state["v_frame_bias"])
        self.v_spatial_bias.data.copy_(v_state["v_spatial_bias"])
        return {
            "motion_qk": str(qk_path),
            "motion_v": str(v_path),
        }
