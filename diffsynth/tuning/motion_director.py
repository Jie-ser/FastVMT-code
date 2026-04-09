from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .common import load_tensor_artifact, resolve_train_block_ids, save_tensor_artifact


class LoRAWrappedLinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 8.0, path_scale: float = 1.0):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.path_scale = float(path_scale)
        self.scaling = self.alpha / max(1, self.rank)
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.path_scale * self.scaling * self.lora_B(self.lora_A(x))


class MotionDirectorAdapter(nn.Module):
    method_name = "motiondirector"

    def __init__(
        self,
        *,
        pipe,
        train_block_ids: list[int] | tuple[int, ...] | None = None,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        temporal_scale: float = 1.0,
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        self.block_ids = resolve_train_block_ids(train_block_ids, pipe=pipe)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.temporal_scale = float(temporal_scale)
        self.spatial_scale = float(spatial_scale)
        self._installed_pipe = None
        self._original_modules: dict[str, nn.Module] = {}
        self.temporal_wrappers = nn.ModuleDict()
        self.spatial_wrappers = nn.ModuleDict()

    def _register_wrapper(self, prefix: str, key: str, wrapper: nn.Module) -> None:
        if prefix == "temporal":
            self.temporal_wrappers[key] = wrapper
        else:
            self.spatial_wrappers[key] = wrapper

    def _wrap_linear(self, base: nn.Linear, *, prefix: str, key: str, scale: float) -> LoRAWrappedLinear:
        wrapper = LoRAWrappedLinear(base, rank=self.lora_rank, alpha=self.lora_alpha, path_scale=scale)
        self._register_wrapper(prefix, key, wrapper)
        return wrapper

    def install(self, pipe, *, inference_mode: bool = False) -> None:
        del inference_mode
        if self._installed_pipe is pipe:
            return
        self._installed_pipe = pipe
        for block_id in self.block_ids:
            block = pipe.dit.blocks[block_id]
            for name in ["q", "k", "v", "o"]:
                key = f"block_{block_id}_{name}"
                original = getattr(block.self_attn, name)
                self._original_modules[f"self_{key}"] = original
                setattr(
                    block.self_attn,
                    name,
                    self._wrap_linear(original, prefix="temporal", key=key, scale=self.temporal_scale),
                )
            for name in ["q", "k", "v", "o"]:
                key = f"block_{block_id}_{name}"
                original = getattr(block.cross_attn, name)
                self._original_modules[f"cross_{key}"] = original
                setattr(
                    block.cross_attn,
                    name,
                    self._wrap_linear(original, prefix="spatial", key=key, scale=self.spatial_scale),
                )

    def set_inference_scales(self, *, temporal_scale: float = 1.0, spatial_scale: float = 0.0) -> None:
        self.temporal_scale = float(temporal_scale)
        self.spatial_scale = float(spatial_scale)
        for wrapper in self.temporal_wrappers.values():
            wrapper.path_scale = self.temporal_scale
        for wrapper in self.spatial_wrappers.values():
            wrapper.path_scale = self.spatial_scale

    def remove(self) -> None:
        if self._installed_pipe is None:
            return
        for block_id in self.block_ids:
            block = self._installed_pipe.dit.blocks[block_id]
            for name in ["q", "k", "v", "o"]:
                setattr(block.self_attn, name, self._original_modules[f"self_block_{block_id}_{name}"])
            for name in ["q", "k", "v", "o"]:
                setattr(block.cross_attn, name, self._original_modules[f"cross_block_{block_id}_{name}"])
        self._original_modules.clear()
        self._installed_pipe = None

    def save(self, artifact_dir) -> dict[str, str]:
        artifact_dir = Path(artifact_dir)
        temporal_path = artifact_dir / "temporal_lora.safetensors"
        spatial_path = artifact_dir / "spatial_lora.safetensors"
        temporal_state = {
            key: value
            for key, value in self.temporal_wrappers.state_dict().items()
            if ".base." not in key
        }
        spatial_state = {
            key: value
            for key, value in self.spatial_wrappers.state_dict().items()
            if ".base." not in key
        }
        save_tensor_artifact(temporal_path, temporal_state)
        save_tensor_artifact(spatial_path, spatial_state)
        return {
            "temporal_lora": str(temporal_path),
            "spatial_lora": str(spatial_path),
        }

    def load(self, artifact_dir, *, device, dtype) -> dict[str, str]:
        artifact_dir = Path(artifact_dir)
        temporal_path = artifact_dir / "temporal_lora.safetensors"
        spatial_path = artifact_dir / "spatial_lora.safetensors"
        temporal_state = load_tensor_artifact(temporal_path, device=device, dtype=dtype)
        spatial_state = load_tensor_artifact(spatial_path, device=device, dtype=dtype)
        self.temporal_wrappers.load_state_dict(temporal_state, strict=False)
        self.spatial_wrappers.load_state_dict(spatial_state, strict=False)
        return {
            "temporal_lora": str(temporal_path),
            "spatial_lora": str(spatial_path),
        }
