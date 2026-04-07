import types
from typing import Callable, Tuple
import math
from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import gc
import json

from ..benchmarks import apply_benchmark_settings, normalize_transfer_method
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_motion_controller import WanMotionControllerModel
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder', 'motion_controller', 'vace']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        self.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        self.vace = model_manager.fetch_model("wan_video_vace")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, end_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(self.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if self.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, self.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}
    
    
    def encode_control_video(self, control_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        control_video = self.preprocess_images(control_video)
        control_video = torch.stack(control_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
        latents = self.encode_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
        return latents
    
    
    def prepare_controlnet_kwargs(self, control_video, num_frames, height, width, clip_feature=None, y=None, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        if control_video is not None:
            control_latents = self.encode_control_video(control_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if clip_feature is None or y is None:
                clip_feature = torch.zeros((1, 257, 1280), dtype=self.torch_dtype, device=self.device)
                y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=self.torch_dtype, device=self.device)
            else:
                y = y[:, -16:]
            y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames


    def align_output_frame_count(self, frames, target_frames):
        target_frames = int(target_frames)
        current_frames = int(frames.shape[2])
        if target_frames <= 0:
            raise ValueError("target_frames must be positive.")
        if current_frames <= 0:
            raise ValueError("Cannot align an empty frame sequence.")
        if current_frames == target_frames:
            return frames
        if target_frames == 1:
            return frames[:, :, :1].clone()

        positions = torch.linspace(
            0,
            current_frames - 1,
            steps=target_frames,
            device=frames.device,
            dtype=torch.float32,
        )
        left_idx = torch.floor(positions).long()
        right_idx = torch.ceil(positions).long()
        weights = (positions - left_idx.to(dtype=positions.dtype)).view(1, 1, target_frames, 1, 1)

        left_frames = frames.index_select(2, left_idx)
        right_frames = frames.index_select(2, right_idx)
        weights = weights.to(dtype=frames.dtype)
        return left_frames * (1.0 - weights) + right_frames * weights
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    
    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}
    
    
    def prepare_motion_bucket_id(self, motion_bucket_id):
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=self.torch_dtype, device=self.device)
        return {"motion_bucket_id": motion_bucket_id}
    
    
    def prepare_vace_kwargs(
        self,
        latents,
        vace_video=None, vace_mask=None, vace_reference_image=None, vace_scale=1.0,
        height=480, width=832, num_frames=81,
        seed=None, rand_device="cpu",
        tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        if vace_video is not None or vace_mask is not None or vace_reference_image is not None:
            self.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=self.torch_dtype, device=self.device)
            else:
                vace_video = self.preprocess_images(vace_video)
                vace_video = torch.stack(vace_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            if vace_mask is None:
                vace_mask = torch.ones_like(vace_video)
            else:
                vace_mask = self.preprocess_images(vace_mask)
                vace_mask = torch.stack(vace_mask, dim=2).to(dtype=self.torch_dtype, device=self.device)
            
            inactive = vace_video * (1 - vace_mask) + 0 * vace_mask
            reactive = vace_video * vace_mask + 0 * (1 - vace_mask)
            inactive = self.encode_video(inactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            reactive = self.encode_video(reactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = self.preprocess_images([vace_reference_image])
                vace_reference_image = torch.stack(vace_reference_image, dim=2).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = self.encode_video(vace_reference_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=self.torch_dtype, device=self.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
                
                noise = self.generate_noise((1, 16, 1, latents.shape[3], latents.shape[4]), seed=seed, device=rand_device, dtype=torch.float32)
                noise = noise.to(dtype=self.torch_dtype, device=self.device)
                latents = torch.concat((noise, latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return latents, {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return latents, {"vace_context": None, "vace_scale": vace_scale}


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_tile_AMF(self, Q, K, sf, l=21, tau=3.0, tile=(3, 4)):
        """
        Compute the tile-based attention motion flow (AMF).

        Parameters:
            self: The object instance.
            Q (torch.Tensor): Query tensor, shape (F, H, W, D_h).
            K (torch.Tensor): Key tensor, shape (F, H, W, D_h).
            sf (int): The farthest frame distance to consider for motion flow.
            l (int): The side length of the local search window.
            tau (float): Temperature parameter for softmax.
            tile (tuple): The height and width of a tile.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - AMF (torch.Tensor): A stacked tensor of attention motion flow displacement matrices.
                - tracking_loss (torch.Tensor): The computed tracking loss.
        """
        # --- 1. Initialization and Reshaping ---
        f, h, w, D_h = K.shape
        S = h * w  # Total number of spatial locations (pixels)
        
        # Reshape Q and K from (F, H, W, D_h) to (F, S, D_h) for matrix multiplication
        Q = Q.view(f, S, D_h)
        K = K.view(f, S, D_h)

        # Pre-calculate constants and indices if not already done
        if not self.indices_computed:
            half_l = l // 2
            tile_h, tile_w = tile
            num_tiles_h = h // tile_h
            num_tiles_w = w // tile_w
            self.compute_indices(h, w, tile_h, tile_w, l, half_l, num_tiles_h, num_tiles_w, D_h)

        amf_results = []
        tracking_loss_values = []

        # --- 2. Iterate Through Frames to Compute Motion Flow ---
        for i in range(f):
            K_windows_for_tracking = []
            # Compare frame `i` with subsequent frames up to `i + sf`
            for j in range(i, min(f, i + sf)):
                if i == j:
                    # --- 2a. Intra-frame (i == j) Attention (Full Attention) ---
                    # Compute attention matrix over the entire frame
                    A_ij = torch.matmul(Q[i], K[j].transpose(-1, -2)) / (D_h ** 0.5)
                    A_ij = F.softmax(A_ij * tau, dim=-1)

                    # Approximate displacement by taking a weighted average of grid coordinates
                    u_j_approx = torch.sum(A_ij * self.u, dim=-1)
                    v_j_approx = torch.sum(A_ij * self.v, dim=-1)

                    # Calculate block displacement relative to the original grid
                    delta_u = u_j_approx - self.u
                    delta_v = v_j_approx - self.v

                    Delta_ij = torch.stack([delta_u, delta_v], dim=-1).squeeze(0)
                    amf_results.append(Delta_ij)
                else:
                    # --- 2b. Inter-frame (i != j) Attention (Tiled/Windowed Attention) ---
                    K_frame_T = K[j].transpose(-1, -2)
                    
                    # --- First Pass: Approximate displacement to find search windows ---
                    Q_tile_centers = torch.gather(Q[i], 0, self.linear_indices)
                    A_ij_approx = torch.matmul(Q_tile_centers, K_frame_T) / (D_h ** 0.5)
                    A_ij_approx = F.softmax(A_ij_approx * tau, dim=-1)
                    
                    u_j_approx = torch.sum(A_ij_approx * self.u, dim=-1)
                    v_j_approx = torch.sum(A_ij_approx * self.v, dim=-1)

                    # --- Second Pass: Refined attention within predicted windows ---
                    # Define search window centers based on the first pass approximation
                    window_h_centers = u_j_approx.view(self.num_tiles_h, self.num_tiles_w)
                    window_w_centers = v_j_approx.view(self.num_tiles_h, self.num_tiles_w)
                    
                    # Construct the local search windows (l x l) around the predicted centers
                    window_h = window_h_centers.unsqueeze(-1).unsqueeze(-1) + self.h_offsets
                    window_w = window_w_centers.unsqueeze(-1).unsqueeze(-1) + self.w_offsets
                    window_h = window_h.clamp(self.half_l, h - self.half_l - 1).long()
                    window_w = window_w.clamp(self.half_l, w - self.half_l - 1).long()

                    # Gather K vectors from within the computed windows
                    linear_indices = (window_h * w + window_w).view(self.num_tiles_h * self.num_tiles_w, l * l)
                    indices_expanded = linear_indices.unsqueeze(-1).expand(-1, -1, D_h).transpose(-1, -2)
                    K_expanded = K[j].unsqueeze(1).expand(-1, l*l, -1).transpose(-1, -2)
                    K_gathered = torch.gather(K_expanded, 0, indices_expanded)

                    # Save the mean of the K window for the tracking loss
                    K_windows_for_tracking.append(K_gathered.mean(dim=1))
                    
                    # Reshape Q to match tile structure
                    Q_tiled = Q[i].reshape(self.num_tiles_h, self.tile_h, self.num_tiles_w, self.tile_w, D_h).permute(0, 2, 1, 3, 4).reshape(-1, self.tile_h * self.tile_w, D_h)

                    # Compute attention scores within the local windows
                    A_ij_local = torch.matmul(Q_tiled, K_gathered) / (D_h ** 0.5)
                    A_ij_local = F.softmax(A_ij_local.reshape(S, -1) * tau, dim=-1)

                    # Calculate displacement within the local window
                    u_j_local = torch.sum(A_ij_local * self.posi_u_in_window, dim=-1)
                    v_j_local = torch.sum(A_ij_local * self.posi_v_in_window, dim=-1)
                    
                    delta_u = u_j_local - self.orig_u
                    delta_v = v_j_local - self.orig_v
                    
                    Delta_ij = torch.stack([delta_u, delta_v], dim=-1)
                    amf_results.append(Delta_ij)

            # --- 3. Compute Tracking Loss ---
            # This loss penalizes large changes in K-window means over time, promoting smooth motion.
            if len(K_windows_for_tracking) > 1:
                K_tensor = torch.stack(K_windows_for_tracking, dim=0).float()
                # Calculate the difference between consecutive K-window means
                diff = K_tensor[1:] - K_tensor[:-1]
                l2_norm = torch.linalg.vector_norm(diff, ord=2, dim=-1)
                mean_l2_norm = l2_norm.mean()
                
                # Append loss if it's a valid number
                if not torch.isnan(mean_l2_norm):
                    tracking_loss_values.append(mean_l2_norm)

        # --- 4. Final Aggregation ---
        # Stack all displacement matrices into a single tensor
        AMF = torch.stack(amf_results)
        # Compute the final mean tracking loss
        tracking_loss = torch.stack(tracking_loss_values).mean() if tracking_loss_values else torch.tensor(0.0, device=Q.device)
        
        return AMF, tracking_loss

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_tile_amf_loss(self, amf_ref_orin, amf_gen):
        """
        Compute the loss between the original AMF and the generated AMF.
        A weighted L2 norm is used.

        Parameters:
            self: The object instance.
            amf_ref_orin (torch.Tensor): The reference AMF tensor, shape (B, S, 2).
            amf_gen (torch.Tensor): The generated AMF tensor, shape (B, S, 2).
            
        Returns:
            torch.Tensor: The computed loss value (a scalar tensor).
        """
        # Ensure the shapes of reference and generated AMFs match
        assert amf_ref_orin.shape == amf_gen.shape, "AMF reference and generated tensors must have the same shape"
        
        # Reshape tensors from (B, S, 2) to (B, S*2) to compute norm over all components
        num_components = 2 * amf_ref_orin.shape[1]
        amf_ref_flat = amf_ref_orin.view(-1, num_components)
        amf_gen_flat = amf_gen.view(-1, num_components)
        
        # Calculate the difference, detaching the reference to avoid computing its gradients
        diff = amf_ref_flat.detach() - amf_gen_flat
        
        # Compute the squared L2 norm for each item in the batch
        squared_l2_norm = torch.norm(diff, p=2, dim=-1) ** 2
        
        # Apply weights and compute the final mean loss
        loss = (squared_l2_norm * self.weights).mean()
        
        return loss

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_dense_AMF(self, Q, K, sf, tau=1.0):
        """
        Compute dense AMF without the sliding-window approximation.
        This is used for the Wan-native DiTFlow adaptation.
        """
        f, h, w, d = K.shape
        s = h * w
        q_flat = Q.view(f, s, d)
        k_flat = K.view(f, s, d)
        u = (torch.arange(s, device=Q.device) // w).float()
        v = (torch.arange(s, device=Q.device) % w).float()
        motion = []
        for i in range(f):
            for j in range(i, min(f, i + sf)):
                scores = torch.matmul(q_flat[i], k_flat[j].transpose(-1, -2)) / (d ** 0.5)
                scores = F.softmax(scores * tau, dim=-1)
                u_j = torch.sum(scores * u.view(1, -1), dim=-1)
                v_j = torch.sum(scores * v.view(1, -1), dim=-1)
                motion.append(torch.stack([u_j - u, v_j - v], dim=-1))
        return torch.stack(motion)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_sparse_AMF(self, Q, K, sf, topk=8, tau=1.0):
        """
        Compute a sparse temporal attention motion descriptor for the
        Wan-native MotionClone adaptation.
        """
        f, h, w, d = K.shape
        s = h * w
        q_flat = Q.view(f, s, d)
        k_flat = K.view(f, s, d)
        u = (torch.arange(s, device=Q.device) // w).float()
        v = (torch.arange(s, device=Q.device) % w).float()
        topk = int(max(1, min(topk, s)))
        motion = []
        for i in range(f):
            for j in range(i, min(f, i + sf)):
                scores = torch.matmul(q_flat[i], k_flat[j].transpose(-1, -2)) / (d ** 0.5)
                topk_values, topk_indices = torch.topk(scores, k=topk, dim=-1)
                sparse_scores = F.softmax(topk_values * tau, dim=-1)
                u_sparse = u[topk_indices]
                v_sparse = v[topk_indices]
                u_j = torch.sum(sparse_scores * u_sparse, dim=-1)
                v_j = torch.sum(sparse_scores * v_sparse, dim=-1)
                motion.append(torch.stack([u_j - u, v_j - v], dim=-1))
        return torch.stack(motion)

    def _set_attention_capture(self, block_id: int, enabled: bool = True):
        target = int(block_id)
        for idx, block in enumerate(self.dit.blocks):
            block.self_attn.save_qk = bool(enabled and idx == target)

    def _reshape_sequence_tokens(self, tokens: torch.Tensor, size_info: dict):
        if tokens is None:
            return None
        frames = int(size_info["frames"])
        grid_h = int(size_info["tile_size"][0])
        grid_w = int(size_info["tile_size"][1])
        hidden = rearrange(tokens, "b (f h w) c -> b f h w c", f=frames, h=grid_h, w=grid_w)
        return hidden.contiguous()

    def _extract_guidance_state(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_kwargs,
        image_emb,
        extra_input,
        size_info,
        block_id: int = 14,
    ):
        self._set_attention_capture(block_id, True)
        velocity, intermediates = self.dit(
            latents,
            timestep=timestep,
            preserve_space=True,
            size_info=size_info,
            return_intermediates=True,
            **prompt_kwargs,
            **image_emb,
            **extra_input,
        )
        capture_idx = min(int(block_id), len(intermediates) - 1)
        hidden = self._reshape_sequence_tokens(intermediates[capture_idx], size_info)
        attn = self.dit.blocks[capture_idx].self_attn
        state = {
            "velocity": velocity,
            "hidden": hidden,
            "q": attn.q_reshape,
            "k": attn.k_reshape,
        }
        self._set_attention_capture(block_id, False)
        return state

    def _compute_smm_feature(self, hidden: torch.Tensor, pool_size: int = 4):
        if hidden is None:
            raise ValueError("SMM guidance requires hidden features, but none were captured.")
        if hidden.shape[0] != 1:
            raise ValueError("SMM guidance currently expects batch size 1.")
        feature = rearrange(hidden[0], "f h w c -> f c h w").float()
        feature = F.adaptive_avg_pool2d(feature, output_size=(pool_size, pool_size))
        feature = feature - feature.mean(dim=(2, 3), keepdim=True)
        feature = feature[1:] - feature[:-1]
        flat = feature.flatten(1)
        feature = feature / flat.norm(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1)
        return feature.to(dtype=self.torch_dtype, device=self.device)

    def _compute_moft_feature(self, hidden: torch.Tensor, topk_idx=None, channel_ratio: float = 0.125):
        if hidden is None:
            raise ValueError("MOFT guidance requires hidden features, but none were captured.")
        if hidden.shape[0] != 1:
            raise ValueError("MOFT guidance currently expects batch size 1.")
        feature = rearrange(hidden[0], "f h w c -> f c h w").float().mean(dim=(-1, -2))
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature[1:] - feature[:-1]
        if topk_idx is None:
            scores = feature.abs().mean(dim=0)
            num_channels = scores.shape[0]
            keep = max(1, int(num_channels * float(channel_ratio)))
            topk_idx = torch.topk(scores, k=min(keep, num_channels), dim=0).indices
        feature = feature.index_select(1, topk_idx)
        feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feature.to(dtype=self.torch_dtype, device=self.device), topk_idx
    
    
    def compute_indices(self, h, w, tile_h, tile_w, l, half_l, num_tiles_h, num_tiles_w, D_h):
        device = self.device
        u = torch.arange(h*w, device=device).unsqueeze(0) // w
        v = torch.arange(h*w, device=device).unsqueeze(0) % w
        
        posi_u_in_window = (torch.arange(l**2, device=device) // l).unsqueeze(0)
        posi_v_in_window = (torch.arange(l**2, device=device) % l).unsqueeze(0)
        
        #rows = torch.arange(0, h, tile_h, device='cuda').long()
        #cols = torch.arange(0, w, tile_w, device='cuda').long()
        rows = torch.arange(tile_h // 2, h + tile_h // 2, tile_h, device=device).long()
        cols = torch.arange(tile_w // 2, w + tile_w // 2, tile_w, device=device).long()
        
        h_offsets = torch.arange(-half_l, half_l + 1, device=device).view(1, 1, l, 1).expand(-1,-1,l,l)
        w_offsets = torch.arange(-half_l, half_l + 1, device=device).view(1, 1, 1, l).expand(-1,-1,l,l)
        
        orig_u = torch.arange(tile_h * tile_w, device=device).repeat(num_tiles_h * num_tiles_w) // tile_w
        orig_v = torch.arange(tile_h * tile_w, device=device).repeat(num_tiles_h * num_tiles_w) % tile_w
        #num_cals = f * sf - (1 + sf) * sf // 2
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing='ij')
        grid_row = grid_row.flatten() # (num_tiles_h * num_tiles_w,)
        grid_col = grid_col.flatten()  # (num_tiles_h * num_tiles
        linear_indices = grid_row * w + grid_col  
        linear_indices = linear_indices.view(-1) 
        linear_indices = linear_indices.unsqueeze(-1).expand(-1, D_h)
        self.u = u
        self.v = v
        #self.posi_u_in_window = posi_u_in_window
        #self.posi_v_in_window = posi_v_in_window
        # Sparsed
        mask = (torch.arange(l**2, device=device).unsqueeze(0) % 2 == 0).float()
        self.posi_u_in_window = posi_u_in_window * mask
        self.posi_v_in_window = posi_v_in_window * mask
        self.orig_u = orig_u
        self.orig_v = orig_v
        self.h_offsets = h_offsets
        self.w_offsets = w_offsets
        self.linear_indices = linear_indices
        self.num_tiles_h = num_tiles_h
        self.num_tiles_w = num_tiles_w
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.half_l = half_l
        self.indices_computed = True
    
    
    def clean_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()  

    def _resolve_ttc_indices(self, timesteps, ttc_noise_levels=(500, 250), ttc_step_ratios=(0.5, 0.25)):
        # Convert to a detached CPU tensor so index search is deterministic and cheap.
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.float32)
        ts = timesteps.detach().float().cpu()
        num_steps = int(ts.shape[0])
        indices = []

        # Primary strategy: map requested TTC noise levels (e.g. 500/250) to nearest scheduler steps.
        for noise_level in ttc_noise_levels or []:
            target = float(noise_level)
            idx = int(torch.argmin((ts - target).abs()).item())
            # Exclude the last step because TTC needs a valid "next" timestep (j + 1).
            if 0 <= idx < num_steps - 1:
                indices.append(idx)

        # Fallback strategy: use ratios of the starting timestep if explicit levels do not match.
        if len(indices) == 0:
            t_start = float(ts[0].item())
            for ratio in ttc_step_ratios or []:
                target = t_start * float(ratio)
                idx = int(torch.argmin((ts - target).abs()).item())
                if 0 <= idx < num_steps - 1:
                    indices.append(idx)

        return sorted(set(indices))

    def _get_ttc_anchor(
        self,
        ttc_anchor_clean,
        input_video,
        x0_pred,
        ttc_anchor_mode="hybrid",
        ttc_anchor_ref_weight=0.25,
    ):
        # Reuse cached anchor once created so the anchor remains stable across TTC hits.
        if ttc_anchor_clean is not None:
            return ttc_anchor_clean

        mode = str(ttc_anchor_mode).strip().lower()
        pred_anchor = x0_pred[:, :, :1].clone().detach()

        ref_anchor = None
        if input_video is not None and hasattr(self, "clean_latents") and self.clean_latents is not None:
            ref_anchor = self.clean_latents[:, :, :1].clone().detach()

        if mode == "legacy_input_clean":
            return ref_anchor if ref_anchor is not None else pred_anchor
        if mode == "pred_x0":
            return pred_anchor
        if mode == "hybrid":
            if ref_anchor is None:
                return pred_anchor
            ref_w = float(max(0.0, min(1.0, ttc_anchor_ref_weight)))
            return ref_w * ref_anchor + (1.0 - ref_w) * pred_anchor
        raise ValueError("Invalid ttc_anchor_mode. Expected one of: legacy_input_clean, pred_x0, hybrid")

    def _inject_anchor_on_noisy_latents(self, latents_noisy, anchor_clean, timestep_next, blend=1.0, rand_device="cpu"):
        # Keep blend in [0, 1] to avoid invalid interpolation weights.
        blend = float(max(0.0, min(1.0, blend)))
        # Map clean anchor to the same noise level as latents_next before blending.
        anchor_noise = self.generate_noise(anchor_clean.shape, device=rand_device, dtype=torch.float32)
        anchor_noise = anchor_noise.to(dtype=self.torch_dtype, device=self.device)
        anchor_noisy = self.scheduler.add_noise(anchor_clean, anchor_noise, timestep=timestep_next)
        latents_noisy = latents_noisy.clone()
        # Only inject on the first temporal slice; other frames stay unchanged.
        latents_noisy[:, :, :1] = (1 - blend) * latents_noisy[:, :, :1] + blend * anchor_noisy
        return latents_noisy

    def _ttc_pathwise_update(
        self,
        latents: torch.Tensor,
        current_timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        current_noise_pred: torch.Tensor,
        prompt_emb_posi,
        prompt_emb_nega,
        cfg_scale: float,
        image_emb: dict,
        extra_input: dict,
        size_info: dict,
        usp_kwargs: dict,
        motion_kwargs: dict,
        vace_kwargs: dict,
        ttc_anchor_clean: torch.Tensor,
        ttc_anchor_blend: float = 1.0,
        rand_device: str = "cpu",
    ):
        # Step A: estimate current clean latent x0 from current step prediction.
        x0_current = self.scheduler.step(current_noise_pred, current_timestep, latents, to_final=True)
        # Step B: resample to next timestep noise level to construct TTC correction state.
        noise_1 = self.generate_noise(latents.shape, device=rand_device, dtype=torch.float32)
        noise_1 = noise_1.to(dtype=self.torch_dtype, device=self.device)
        latents_next = self.scheduler.add_noise(x0_current, noise_1, timestep=next_timestep)
        latents_next = self._inject_anchor_on_noisy_latents(
            latents_next, ttc_anchor_clean, timestep_next=next_timestep, blend=ttc_anchor_blend, rand_device=rand_device
        )

        # Step C: run model at next timestep on corrected state; disable TeaCache to avoid cache-step mismatch.
        noise_pred_posi = model_fn_wan_video(
            self.dit, motion_controller=self.motion_controller, vace=self.vace,
            x=latents_next, timestep=next_timestep, size_info=size_info,
            **prompt_emb_posi, **image_emb, **extra_input,
            tea_cache=None, **usp_kwargs, **motion_kwargs, **vace_kwargs,
        )
        if cfg_scale != 1.0:
            noise_pred_nega = model_fn_wan_video(
                self.dit, motion_controller=self.motion_controller, vace=self.vace,
                x=latents_next, timestep=next_timestep, size_info=size_info,
                **prompt_emb_nega, **image_emb, **extra_input,
                tea_cache=None, **usp_kwargs, **motion_kwargs, **vace_kwargs,
            )
            noise_pred_corr = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred_corr = noise_pred_posi

        # Step D: project corrected prediction back to x0, then sample a final latent at next timestep.
        x0_corr = self.scheduler.step(noise_pred_corr, next_timestep, latents_next, to_final=True)
        noise_2 = self.generate_noise(latents.shape, device=rand_device, dtype=torch.float32)
        noise_2 = noise_2.to(dtype=self.torch_dtype, device=self.device)
        latents_next = self.scheduler.add_noise(x0_corr, noise_2, timestep=next_timestep)
        return latents_next

    def _compute_msa_velocity_loss(self, velocity_opt, velocity_anchor, msa_mask=None):
        anchor = velocity_anchor.detach()
        if msa_mask is not None:
            diff = (velocity_opt - anchor) * msa_mask
        else:
            diff = velocity_opt - anchor
        return torch.mean(diff.float() * diff.float())

    def _build_msa_mask(self, amf_ref_orin, target_shape, size_info, msa_mask_mode="uniform", msa_mask_power=1.0, msa_mask_min=0.15):
        b, _, _, h, w = target_shape
        if msa_mask_mode == "uniform" or amf_ref_orin is None:
            return torch.ones((b, 1, 1, h, w), dtype=self.torch_dtype, device=self.device)

        try:
            spatial_mag = torch.linalg.vector_norm(amf_ref_orin.float(), ord=2, dim=-1).mean(dim=0)
            low_h, low_w = int(size_info["tile_size"][0]), int(size_info["tile_size"][1])
            if low_h * low_w != spatial_mag.numel():
                raise ValueError("AMF spatial shape does not match tile_size.")

            mask = spatial_mag.view(1, 1, low_h, low_w)
            mask = mask - mask.amin(dim=(-2, -1), keepdim=True)
            mask = mask / mask.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            mask = torch.pow(mask.clamp(min=0.0), float(msa_mask_power))
            mask = mask * (1.0 - float(msa_mask_min)) + float(msa_mask_min)
            mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
            mask = mask.unsqueeze(2).to(dtype=self.torch_dtype, device=self.device)
            if b > 1:
                mask = mask.expand(b, -1, -1, -1, -1)
            return mask
        except Exception:
            # Fallback keeps optimization stable when AMF mask inference fails.
            return torch.ones((b, 1, 1, h, w), dtype=self.torch_dtype, device=self.device)
        
    def guidance_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb_posi,
        prompt_null,
        image_emb,
        extra_input,
        noise,
        size_info,
        sf,
        transfer_method,
        seed,
        step_id,
        interval: int = 3,
        guidance_block_id: int = 14,
        motionclone_topk: int = 8,
        smm_pool_size: int = 4,
        moft_channel_ratio: float = 0.125,
        msa_enabled: bool = False,
        msa_optim_start: int = 0,
        msa_optim_end: int = 1,
        msa_iter: int = 2,
        msa_scale_list=(50.0, 300.0),
        msa_mask_mode: str = "uniform",
        msa_mask_power: float = 1.0,
        msa_mask_min: float = 0.15,
        msa_balance_with_amf: bool = True,
        msa_debug: bool = False,
    ) -> torch.Tensor:
        self.clean_memory()
        initial_lr = 0.003
        final_lr = 0.002
        default_total_steps = 10
        transfer_method = normalize_transfer_method(transfer_method=transfer_method)
        fastvmt_active = transfer_method == "fastvmt"
        msa_step_active = (
            fastvmt_active
            and bool(msa_enabled)
            and hasattr(self, "clean_latents")
            and self.clean_latents is not None
            and int(msa_optim_start) <= int(step_id) <= int(msa_optim_end)
        )
        total_steps = int(max(1, msa_iter if msa_step_active else default_total_steps))
        effective_interval = 1 if not fastvmt_active or msa_step_active else max(1, int(interval))
        current_step = 0

        optimized_latents = latents.clone().detach().requires_grad_(True)
        optimizer = torch.optim.AdamW([optimized_latents], lr=initial_lr)
        self.scale_range = np.linspace(0.007, 0.004, 50)
        self.cached_grad = None

        msa_anchor_velocity = None
        msa_mask = None
        msa_scale = None
        if msa_step_active:
            local_msa_step_id = int(step_id) - int(msa_optim_start)
            if local_msa_step_id < 0 or local_msa_step_id >= len(msa_scale_list):
                msa_step_active = False
                if msa_debug:
                    print(f"MSA disabled at step={step_id}: scale list index {local_msa_step_id} is out of range.")
            else:
                msa_scale = float(msa_scale_list[local_msa_step_id])

        with torch.no_grad():
            noise = self.generate_noise(latents.shape, device=latents.device, dtype=torch.float32, seed=seed)
            noise = noise.to(dtype=self.torch_dtype, device=self.device)
            ref_latents = self.scheduler.add_noise(self.clean_latents, noise, timestep=timestep)
            ref_state = self._extract_guidance_state(
                ref_latents,
                timestep=timestep,
                prompt_kwargs=prompt_null,
                image_emb=image_emb,
                extra_input=extra_input,
                size_info=size_info,
                block_id=guidance_block_id,
            )

            ref_signal = None
            ref_aux = {}
            if transfer_method == "fastvmt":
                ref_signal, _ = self.compute_tile_AMF(ref_state["q"], ref_state["k"], sf=sf, l=21, tau=1.0, tile=(3, 4))
            elif transfer_method == "ditflow":
                ref_signal = self.compute_dense_AMF(ref_state["q"], ref_state["k"], sf=sf, tau=1.0)
            elif transfer_method == "motionclone":
                ref_signal = self.compute_sparse_AMF(ref_state["q"], ref_state["k"], sf=sf, topk=motionclone_topk, tau=1.0)
            elif transfer_method == "smm":
                ref_signal = self._compute_smm_feature(ref_state["hidden"], pool_size=smm_pool_size)
            elif transfer_method == "moft":
                ref_signal, ref_aux["topk_idx"] = self._compute_moft_feature(
                    ref_state["hidden"], topk_idx=None, channel_ratio=moft_channel_ratio
                )
            else:
                raise ValueError(f"Unsupported transfer_method `{transfer_method}` during guidance.")

            if msa_step_active:
                msa_anchor_velocity = ref_state["velocity"].detach()
                msa_mask = self._build_msa_mask(
                    ref_signal if transfer_method == "fastvmt" else None,
                    target_shape=latents.shape,
                    size_info=size_info,
                    msa_mask_mode=msa_mask_mode,
                    msa_mask_power=msa_mask_power,
                    msa_mask_min=msa_mask_min,
                )
                if msa_debug:
                    print(
                        f"MSA active at step={step_id}: iter={total_steps}, scale={msa_scale:.4f}, "
                        f"mask_mode={msa_mask_mode}, mask_min={float(msa_mask.min().item()):.4f}, "
                        f"mask_max={float(msa_mask.max().item()):.4f}"
                    )

        self.clean_memory()
        detached_prompt_emb_posi_new = {k: v.detach() if hasattr(v, "detach") else v for k, v in prompt_emb_posi.items()}
        for j in tqdm(range(total_steps)):
            lr = initial_lr - (initial_lr - final_lr) * (current_step / total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    self.dit.train()
                    for param in self.dit.parameters():
                        param.requires_grad_(False)

                    should_recompute = (self.cached_grad is None) or (j % effective_interval == 0)
                    if should_recompute:
                        gen_state = self._extract_guidance_state(
                            optimized_latents,
                            timestep=timestep,
                            prompt_kwargs=detached_prompt_emb_posi_new,
                            image_emb=image_emb,
                            extra_input=extra_input,
                            size_info=size_info,
                            block_id=guidance_block_id,
                        )

                        if transfer_method == "fastvmt":
                            gen_signal, track_loss = self.compute_tile_AMF(
                                gen_state["q"], gen_state["k"], sf=sf, l=21, tau=1.0, tile=(3, 4)
                            )
                            motion_loss = self.compute_tile_amf_loss(ref_signal, gen_signal)
                            loss_core = 5 * motion_loss + track_loss
                            if msa_step_active and msa_anchor_velocity is not None and msa_scale is not None:
                                msa_loss = self._compute_msa_velocity_loss(gen_state["velocity"], msa_anchor_velocity, msa_mask)
                                if msa_balance_with_amf:
                                    balance = (
                                        loss_core.detach().abs() / msa_loss.detach().abs().clamp_min(1e-6)
                                    ).clamp(min=0.05, max=20.0)
                                    msa_weight = float(msa_scale) * balance
                                else:
                                    msa_weight = float(msa_scale)
                                loss = loss_core + msa_weight * msa_loss
                                if msa_debug and (j == 0 or j == total_steps - 1):
                                    print(
                                        f"MSA debug step={step_id}, iter={j + 1}/{total_steps}: "
                                        f"amf={float(motion_loss.detach().item()):.6f}, "
                                        f"track={float(track_loss.detach().item()):.6f}, "
                                        f"msa={float(msa_loss.detach().item()):.6f}"
                                    )
                            else:
                                loss = loss_core
                        elif transfer_method == "ditflow":
                            gen_signal = self.compute_dense_AMF(gen_state["q"], gen_state["k"], sf=sf, tau=1.0)
                            loss = 5 * self.compute_tile_amf_loss(ref_signal, gen_signal)
                        elif transfer_method == "motionclone":
                            gen_signal = self.compute_sparse_AMF(
                                gen_state["q"], gen_state["k"], sf=sf, topk=motionclone_topk, tau=1.0
                            )
                            loss = 5 * self.compute_tile_amf_loss(ref_signal, gen_signal)
                        elif transfer_method == "smm":
                            gen_signal = self._compute_smm_feature(gen_state["hidden"], pool_size=smm_pool_size)
                            loss = F.mse_loss(gen_signal.float(), ref_signal.detach().float())
                        elif transfer_method == "moft":
                            gen_signal, _ = self._compute_moft_feature(
                                gen_state["hidden"],
                                topk_idx=ref_aux["topk_idx"],
                                channel_ratio=moft_channel_ratio,
                            )
                            loss = F.mse_loss(gen_signal.float(), ref_signal.detach().float())
                        else:
                            raise ValueError(f"Unsupported transfer_method `{transfer_method}` during optimization.")
                    else:
                        loss = (optimized_latents * self.cached_grad).sum()

                    loss.backward()
                    self.cached_grad = optimized_latents.grad.clone().detach()
                    optimizer.step()
                    self.clean_memory()
            current_step += 1
            print("lr", lr)
        return optimized_latents
    
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        input_video=None,
        control_video=None,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        denoising_strength=0.92,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.8,
        num_inference_steps=50,
        sigma_shift=7.0,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        sf=4,
        test_latency=False,
        latency_dir=None,
        transfer_method=None,
        benchmark_preset=None,
        benchmark_strict=True,
        mode=None,
        guidance_block_id=14,
        motionclone_topk=8,
        smm_pool_size=4,
        moft_channel_ratio=0.125,
        ttc_enabled=False,
        ttc_noise_levels=(500, 250),
        ttc_step_ratios=(0.5, 0.25),
        ttc_anchor_blend=1.0,
        ttc_anchor_mode="hybrid",
        ttc_anchor_ref_weight=0.25,
        ttc_anchor_blend_start=None,
        ttc_anchor_blend_end=None,
        ttc_debug=False,
        guidance_steps=10,
        msa_enabled=False,
        msa_optim_start=0,
        msa_optim_end=1,
        msa_iter=2,
        msa_scale_list=(50.0, 300.0),
        msa_mask_mode="uniform",
        msa_mask_power=1.0,
        msa_mask_min=0.15,
        msa_balance_with_amf=True,
        msa_debug=False,
    ):
        self.indices_computed = False
        self.indices_expanded = []
        self.last_run_summary = None
        transfer_method = normalize_transfer_method(transfer_method=transfer_method, mode=mode)
        if transfer_method == "fastvmt":
            print("You are using the FastVMT transfer method with Wan-native sliding-window AMF guidance.")
        elif transfer_method == "ditflow":
            print("You are using the DiTFlow transfer method with Wan-native dense AMF guidance.")
        elif transfer_method == "smm":
            print("You are using the SMM transfer method with Wan-native space-time feature guidance.")
        elif transfer_method == "moft":
            print("You are using the MOFT transfer method with Wan-native motion-channel guidance.")
        elif transfer_method == "motionclone":
            print("You are using the MotionClone transfer method with Wan-native sparse temporal attention guidance.")
        elif transfer_method == "no_transfer":
            print("You are using the no_transfer method, which does not apply motion transfer guidance.")
        else:
            raise ValueError(f"Unsupported transfer_method `{transfer_method}`.")

        benchmark_settings = apply_benchmark_settings(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            benchmark_preset=benchmark_preset,
        )
        height = int(benchmark_settings["height"])
        width = int(benchmark_settings["width"])
        num_frames = int(benchmark_settings["num_frames"])
        num_inference_steps = int(benchmark_settings["num_inference_steps"])
        benchmark_preset = benchmark_settings.get("benchmark_preset", benchmark_preset)

        if transfer_method != "no_transfer" and input_video is None:
            raise ValueError(f"transfer_method `{transfer_method}` requires a reference `input_video`.")

        msa_mask_mode = str(msa_mask_mode).strip().lower()
        if msa_mask_mode not in ["uniform", "amf"]:
            raise ValueError("msa_mask_mode must be one of: uniform, amf")
        msa_mask_power = float(max(0.1, msa_mask_power))
        msa_mask_min = float(max(0.0, min(1.0, msa_mask_min)))
        msa_iter = int(max(1, msa_iter))
        msa_balance_with_amf = bool(msa_balance_with_amf)
        msa_enabled = bool(msa_enabled)
        msa_scale_list = tuple(float(scale) for scale in msa_scale_list)

        if transfer_method != "fastvmt" and ttc_enabled:
            print(f"TTC is disabled because transfer_method='{transfer_method}' does not use FastVMT path correction.")
            ttc_enabled = False
        if transfer_method != "fastvmt" and msa_enabled:
            print(f"MSA is disabled because transfer_method='{transfer_method}' does not use FastVMT structure anchoring.")
            msa_enabled = False

        if msa_enabled and transfer_method == "no_transfer":
            print("MSA is disabled because `transfer_method='no_transfer'`.")
            msa_enabled = False
        if msa_enabled and guidance_steps <= 0:
            print("MSA is disabled because `guidance_steps <= 0`.")
            msa_enabled = False
        if msa_enabled and input_video is None:
            print("MSA is disabled because reference `input_video` is required for MSA anchor.")
            msa_enabled = False

        max_guidance_index = max(0, int(guidance_steps) - 1)
        msa_optim_start = int(max(0, msa_optim_start))
        msa_optim_end = int(max(0, msa_optim_end))
        msa_optim_start = min(msa_optim_start, max_guidance_index)
        msa_optim_end = min(msa_optim_end, max_guidance_index)
        if msa_enabled and msa_optim_end < msa_optim_start:
            print("MSA is disabled because `msa_optim_end < msa_optim_start` after clamping.")
            msa_enabled = False
        if msa_enabled:
            expected_msa_scales = msa_optim_end - msa_optim_start + 1
            if len(msa_scale_list) != expected_msa_scales:
                raise ValueError(
                    f"Expected {expected_msa_scales} msa_scale values for step range "
                    f"[{msa_optim_start}, {msa_optim_end}], but got {len(msa_scale_list)}."
                )
            if msa_debug:
                print(
                    f"MSA config: range=[{msa_optim_start}, {msa_optim_end}], iter={msa_iter}, "
                    f"mask={msa_mask_mode}, power={msa_mask_power}, min={msa_mask_min}, "
                    f"balance_with_amf={msa_balance_with_amf}, scales={msa_scale_list}"
                )
        
        # Parameter check
        height, width = self.check_resize_height_width(height, width)

        # Auto-infer num_frames from input_video if provided
        if input_video is not None:
            if benchmark_preset is not None:
                available_frames = len(input_video)
                if available_frames < num_frames:
                    raise ValueError(
                        f"Reference input_video only has {available_frames} frames, but benchmark_preset "
                        f"`{benchmark_preset}` requires {num_frames} frames."
                    )
                if benchmark_strict or available_frames != num_frames:
                    input_video.set_length(num_frames)
            inferred_frames = len(input_video)
            if benchmark_preset is None and num_frames != inferred_frames:
                print(f"num_frames auto-adjusted from {num_frames} to {inferred_frames} based on input_video length.")
                num_frames = inferred_frames
            elif benchmark_preset is not None and inferred_frames != num_frames:
                raise ValueError(
                    f"Benchmark preset `{benchmark_preset}` requires {num_frames} frames after protocol enforcement, "
                    f"but got {inferred_frames}."
                )
        elif num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        # size_info
        latent_frames = (num_frames - 1) // 4 + 1
        size_info = {'tiled': tiled, 'tile_size': tile_size, 'frames': latent_frames}
        # This line is added to preserve size information of the latent, which will then be passed
        # to reshape the q, k
        
        weights = np.linspace(1, 0.8, num=sf)
        for i in range(1, latent_frames):
            if i + sf < latent_frames:        
                weights = np.concatenate([weights, np.linspace(1, 0.8, num=sf)], axis=0)
            else:
                weights = np.concatenate([weights, np.linspace(1, 0.8, num=sf)[:latent_frames-i]], axis=0)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        
            
        if input_video is not None:                
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            self.clean_latents = latents.clone().detach()
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
            #latents = noise
        else:
            latents = noise
          
        #self.orin_latents = latents
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_null = self.encode_prompt("", positive=True)
        prompt_emb_nega = None
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, end_image, num_frames, height, width, **tiler_kwargs)
        else:
            image_emb = {}
            
        # ControlNet
        if control_video is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.prepare_controlnet_kwargs(control_video, num_frames, height, width, **image_emb, **tiler_kwargs)
            
        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # VACE
        latents, vace_kwargs = self.prepare_vace_kwargs(
            latents, vace_video, vace_video_mask, vace_reference_image, vace_scale,
            height=height, width=width, num_frames=num_frames, seed=seed, rand_device=rand_device, **tiler_kwargs
        )
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()
        # Default empty set keeps TTC branch disabled unless explicitly enabled and resolved below.
        ttc_indices = set()
        # New TTC blend schedule with backward-compatible fallback to legacy single blend.
        ttc_blend_start = ttc_anchor_blend if ttc_anchor_blend_start is None else float(ttc_anchor_blend_start)
        ttc_blend_end = ttc_blend_start if ttc_anchor_blend_end is None else float(ttc_anchor_blend_end)
        ttc_blend_start = float(max(0.0, min(1.0, ttc_blend_start)))
        ttc_blend_end = float(max(0.0, min(1.0, ttc_blend_end)))
        if ttc_enabled and transfer_method == "fastvmt":
            # Resolve TTC trigger steps once before denoising loop.
            ttc_indices = set(self._resolve_ttc_indices(self.scheduler.timesteps, ttc_noise_levels, ttc_step_ratios))
            if ttc_debug:
                # Print both indices and actual timestep values for quick verification.
                resolved_timesteps = [float(self.scheduler.timesteps[i].item()) for i in sorted(ttc_indices)]
                print(f"TTC enabled. Step indices: {sorted(ttc_indices)}, timesteps: {resolved_timesteps}")
                print(
                    f"TTC anchor config: mode={ttc_anchor_mode}, ref_weight={ttc_anchor_ref_weight}, "
                    f"blend_start={ttc_blend_start}, blend_end={ttc_blend_end}"
                )


        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event = torch.cuda.Event(enable_timing=True)

        #start_event.record()
        # Denoise
        
        
        self.load_models_to_device(["dit", "motion_controller", "vace"])

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            i_for_guidance = 0
            ttc_anchor_clean = None
            ttc_total_hits = len(ttc_indices)
            ttc_hit_count = 0
            if test_latency:
                start_event_guidance = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                start_event_gen = torch.cuda.Event(enable_timing=True)
                end_event_gen = torch.cuda.Event(enable_timing=True)
                end_event_guidance = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
                
                start_event_gen.record()
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                if transfer_method != "no_transfer" and i_for_guidance < guidance_steps:
                    if test_latency:
                        start_event_guidance[i_for_guidance].record()
                    
                        latents = self.guidance_step(
                            latents,
                            timestep,
                            prompt_emb_posi,
                            prompt_null,
                            image_emb,
                            extra_input,
                            noise,
                            size_info,
                            sf=sf,
                            transfer_method=transfer_method,
                            seed=seed,
                            interval=3,
                            step_id=progress_id,
                            guidance_block_id=guidance_block_id,
                            motionclone_topk=motionclone_topk,
                            smm_pool_size=smm_pool_size,
                            moft_channel_ratio=moft_channel_ratio,
                            msa_enabled=msa_enabled,
                            msa_optim_start=msa_optim_start,
                            msa_optim_end=msa_optim_end,
                            msa_iter=msa_iter,
                            msa_scale_list=msa_scale_list,
                            msa_mask_mode=msa_mask_mode,
                            msa_mask_power=msa_mask_power,
                            msa_mask_min=msa_mask_min,
                            msa_balance_with_amf=msa_balance_with_amf,
                            msa_debug=msa_debug,
                        )
                        end_event_guidance[i_for_guidance].record()
                    else:
                        latents = self.guidance_step(
                            latents,
                            timestep,
                            prompt_emb_posi,
                            prompt_null,
                            image_emb,
                            extra_input,
                            noise,
                            size_info,
                            sf=sf,
                            transfer_method=transfer_method,
                            seed=seed,
                            interval=3,
                            step_id=progress_id,
                            guidance_block_id=guidance_block_id,
                            motionclone_topk=motionclone_topk,
                            smm_pool_size=smm_pool_size,
                            moft_channel_ratio=moft_channel_ratio,
                            msa_enabled=msa_enabled,
                            msa_optim_start=msa_optim_start,
                            msa_optim_end=msa_optim_end,
                            msa_iter=msa_iter,
                            msa_scale_list=msa_scale_list,
                            msa_mask_mode=msa_mask_mode,
                            msa_mask_power=msa_mask_power,
                            msa_mask_min=msa_mask_min,
                            msa_balance_with_amf=msa_balance_with_amf,
                            msa_debug=msa_debug,
                        )
                    i_for_guidance += 1

                # Inference
                noise_pred_posi = model_fn_wan_video(
                    self.dit, motion_controller=self.motion_controller, vace=self.vace,
                    x=latents, timestep=timestep, size_info=size_info,
                    **prompt_emb_posi, **image_emb, **extra_input,
                    **tea_cache_posi, **usp_kwargs, **motion_kwargs, **vace_kwargs,
                )
                if cfg_scale != 1.0:
                    noise_pred_nega = model_fn_wan_video(
                        self.dit, motion_controller=self.motion_controller, vace=self.vace,
                        x=latents, timestep=timestep, size_info=size_info,
                        **prompt_emb_nega, **image_emb, **extra_input,
                        **tea_cache_nega, **usp_kwargs, **motion_kwargs, **vace_kwargs,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                if ttc_enabled and transfer_method == "fastvmt" and progress_id in ttc_indices and progress_id + 1 < len(self.scheduler.timesteps):
                    if ttc_total_hits <= 1:
                        current_ttc_blend = ttc_blend_start
                    else:
                        blend_alpha = float(ttc_hit_count) / float(ttc_total_hits - 1)
                        current_ttc_blend = ttc_blend_start + (ttc_blend_end - ttc_blend_start) * blend_alpha

                    x0_current = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, to_final=True)
                    ttc_anchor_clean = self._get_ttc_anchor(
                        ttc_anchor_clean, input_video, x0_current, ttc_anchor_mode, ttc_anchor_ref_weight
                    )
                    next_timestep = self.scheduler.timesteps[progress_id + 1].unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                    if ttc_debug:
                        print(
                            f"TTC hit {ttc_hit_count + 1}/{ttc_total_hits} at step={progress_id}, "
                            f"t={float(self.scheduler.timesteps[progress_id].item()):.4f}, blend={current_ttc_blend:.4f}"
                        )
                    latents = self._ttc_pathwise_update(
                        latents=latents,
                        current_timestep=timestep,
                        next_timestep=next_timestep,
                        current_noise_pred=noise_pred,
                        prompt_emb_posi=prompt_emb_posi,
                        prompt_emb_nega=prompt_emb_nega,
                        cfg_scale=cfg_scale,
                        image_emb=image_emb,
                        extra_input=extra_input,
                        size_info=size_info,
                        usp_kwargs=usp_kwargs,
                        motion_kwargs=motion_kwargs,
                        vace_kwargs=vace_kwargs,
                        ttc_anchor_clean=ttc_anchor_clean,
                        ttc_anchor_blend=current_ttc_blend,
                        rand_device=rand_device,
                    )
                    ttc_hit_count += 1
                    continue

                # Scheduler
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            self.clean_memory()
        if test_latency:     
            end_event_gen.record()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            time_guidance = [start_event_guidance[i].elapsed_time(end_event_guidance[i]) / 1000 for i in range(10)]
            time_gen = start_event_gen.elapsed_time(end_event_gen) / 1000
            results = {
                "seed": seed,
                "time_guidance": time_guidance,
                "time_gen": time_gen,
            }
            with open(latency_dir, "a") as f:
                json.dump(results, f, ensure_ascii=False)
                f.write('\n') 
        if vace_reference_image is not None:
            latents = latents[:, :, 1:]
        
        del prompt_emb_posi, prompt_null
        if prompt_emb_nega is not None:
            del prompt_emb_nega
        #del self.dit, self.motion_controller, self.vace
        self.clean_memory()
        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        decoded_num_frames = int(frames.shape[2])
        output_num_frames = decoded_num_frames
        if benchmark_preset is not None and decoded_num_frames != num_frames:
            print(
                f"Benchmark preset `{benchmark_preset}` requires {num_frames} output frames, "
                f"but Wan VAE decoded {decoded_num_frames}. Applying temporal resampling to enforce protocol."
            )
            frames = self.align_output_frame_count(frames, target_frames=num_frames)
            output_num_frames = int(frames.shape[2])
            if output_num_frames != num_frames:
                raise ValueError(
                    f"Failed to align decoded frames for benchmark preset `{benchmark_preset}`: "
                    f"expected {num_frames}, got {output_num_frames}."
                )
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])
        self.last_run_summary = {
            "transfer_method": transfer_method,
            "benchmark_preset": benchmark_preset,
            "height": height,
            "width": width,
            "requested_num_frames": num_frames,
            "decoded_num_frames": decoded_num_frames,
            "output_num_frames": output_num_frames,
            "num_frames": output_num_frames,
            "num_inference_steps": num_inference_steps,
        }

        return frames


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    x: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    size_info: dict = None,
    **kwargs,
):
    return_intermediates = kwargs.pop("return_intermediates", False)
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = dit.patchify(x)
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    intermediates = [] if return_intermediates else None
    for block_id, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs, size_info)
        if vace_context is not None and block_id in vace.vace_layers_mapping:
            x = x + vace_hints[vace.vace_layers_mapping[block_id]] * vace_scale
        if return_intermediates:
            intermediates.append(x)
    if tea_cache is not None:
        tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    x = dit.unpatchify(x, (f, h, w))
    if not return_intermediates:
        return x
    else:
        return x, intermediates
