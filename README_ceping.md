# FastVMT Benchmark Guide for Non-Ours Methods

This document explains how to evaluate the seven non-Ours methods currently exposed by this repository:

- `ditflow`
- `moft`
- `smm`
- `motionclone`
- `motioninversion`
- `motiondirector`
- `det`

`Ours` corresponds to `fastvmt`, which is intentionally not covered here.

## 1. Method Summary

These seven methods are split into two groups.

Training-free methods:

- `ditflow`
- `moft`
- `smm`
- `motionclone`

Tuning-based methods:

- `motioninversion`
- `motiondirector`
- `det`

The unified entry script is:

```bash
python examples/wan_14b_text_to_video.py
```

Supported method names are defined by `--transfer_method`:

```bash
fastvmt ditflow moft smm motionclone no_transfer motioninversion motiondirector det
```

For batch evaluation, the shared helper is:

```bash
examples/run_batch_common.py
```

## 2. Environment Setup

Create and activate the environment:

```bash
conda create -n fastvmt python=3.10
conda activate fastvmt
pip install -e .
```

Install extra dependencies when needed.

For DeT seed-mask generation:

```bash
pip install -r requirements-benchmark-mask.txt
```

For tuning-based methods:

```bash
pip install -r requirements-benchmark-train.txt
```

Notes:

- `requirements-benchmark-mask.txt` adds Hugging Face Grounding DINO + SAM dependencies.
- `requirements-benchmark-train.txt` adds `lightning`, `peft`, `accelerate`, `opencv-python`, `decord`, and `cotracker`.
- `det` usually benefits from `cotracker` when only a first-frame seed mask is provided.

## 3. Model Preparation

Download Wan checkpoints first. The default 14B path used by the main script is:

```bash
models/Wan2.1-T2V-14B
```

Recommended download command:

```bash
python examples/download_model.py --model 14b
```

If you want the 1.3B model instead:

```bash
python examples/download_model.py --model 1.3b
```

## 4. Reference Video Preparation

Put reference videos under `data/`, for example:

```text
data/ref1.mp4
data/ref2.mp4
data/source.mp4
```

For benchmark-style runs, make sure the reference videos are already resized or at least compatible with the benchmark preset you use.

The common preset used by recent benchmark scripts is:

```text
wan14b_32f_832x480
```

This preset enforces:

- resolution: `832x480`
- frames: `32`
- inference steps: `50`

If your raw test videos are not already prepared, you can use:

```bash
python examples/prepare_test_videos.py
```

This script batch-converts source videos into a benchmark-friendly format by sampling frames, resizing, and writing processed videos.

## 5. Single-Case Evaluation

### 5.1 Training-Free Methods

The following commands run directly with a reference video:

#### DiTFlow

```bash
python examples/wan_14b_text_to_video.py \
  --transfer_method ditflow \
  --input_video data/ref1.mp4 \
  --prompt "your prompt" \
  --negative_prompt "your negative prompt" \
  --benchmark_preset wan14b_32f_832x480 \
  --denoising_strength 0.88 \
  --output_dir results/ditflow
```

#### MOFT

```bash
python examples/wan_14b_text_to_video.py \
  --transfer_method moft \
  --input_video data/ref1.mp4 \
  --prompt "your prompt" \
  --negative_prompt "your negative prompt" \
  --benchmark_preset wan14b_32f_832x480 \
  --denoising_strength 0.88 \
  --output_dir results/moft
```

#### SMM

```bash
python examples/wan_14b_text_to_video.py \
  --transfer_method smm \
  --input_video data/ref1.mp4 \
  --prompt "your prompt" \
  --negative_prompt "your negative prompt" \
  --benchmark_preset wan14b_32f_832x480 \
  --denoising_strength 0.88 \
  --output_dir results/smm
```

#### MotionClone

```bash
python examples/wan_14b_text_to_video.py \
  --transfer_method motionclone \
  --input_video data/ref1.mp4 \
  --prompt "your prompt" \
  --negative_prompt "your negative prompt" \
  --benchmark_preset wan14b_32f_832x480 \
  --denoising_strength 0.88 \
  --output_dir results/motionclone
```

Notes for training-free methods:

- They use the reference video directly through `--input_video`.
- They do not need separate training artifacts.
- `--ttc_enabled` and `--msa_enabled` are meaningful for `fastvmt`, not for these four methods.

### 5.2 Tuning-Based Methods

The following methods first prepare shared artifacts and then train and infer through the tuning runner.

#### Motion Inversion

```bash
python examples/wan_14b_text_to_video.py \
  --transfer_method motioninversion \
  --input_video data/ref1.mp4 \
  --prompt "your prompt" \
  --negative_prompt "your negative prompt" \
  --benchmark_preset wan14b_32f_832x480 \
  --denoising_strength 0.88 \
  --stage full \
  --train_steps 200 \
  --output_dir results/motioninversion
```

#### Motion Director

```bash
python examples/wan_14b_text_to_video.py \
  --transfer_method motiondirector \
  --input_video data/ref1.mp4 \
  --prompt "your prompt" \
  --negative_prompt "your negative prompt" \
  --benchmark_preset wan14b_32f_832x480 \
  --denoising_strength 0.88 \
  --stage full \
  --train_steps 200 \
  --motiondirector_lora_rank 8 \
  --motiondirector_lora_alpha 8.0 \
  --temporal_scale 1.0 \
  --spatial_scale 0.0 \
  --output_dir results/motiondirector
```

#### DeT

```bash
python examples/wan_14b_text_to_video.py \
  --transfer_method det \
  --input_video data/ref1.mp4 \
  --prompt "your prompt" \
  --negative_prompt "your negative prompt" \
  --benchmark_preset wan14b_32f_832x480 \
  --denoising_strength 0.88 \
  --stage full \
  --train_steps 200 \
  --mask_dir det_seed_masks_38/ref1 \
  --output_dir results/det
```

Important notes for tuning-based methods:

- They write reusable artifacts under `artifacts/` by default.
- `--stage prepare`, `--stage train`, `--stage infer`, and `--stage full` are all supported.
- `motiondirector` and `motioninversion` do not use the reference video at inference time the same way training-free methods do. They learn method-specific adapters from the reference sequence first.
- `det` needs masks or track artifacts in addition to the reference video.

## 6. DeT Mask Preparation

If you evaluate `det`, prepare masks first.

The repository provides:

```bash
python examples/prepare_det_seed_masks.py
```

Typical usage:

```bash
python examples/prepare_det_seed_masks.py \
  --shell_script run_ablation38.sh \
  --subject_prompts_json examples/det_subject_prompts_38.json \
  --video_root data \
  --output_dir det_seed_masks_38 \
  --debug_dir det_seed_masks_38_debug \
  --benchmark_preset wan14b_32f_832x480
```

What it does:

- parses benchmark cases from a shell script containing `run_case ...`
- loads the first frame of each reference video
- uses Grounding DINO to detect the subject from a text prompt
- uses SAM to segment the subject
- writes a seed mask `0000.png` for each case
- optionally writes overlay visualizations for debugging

If you already have cached tracks, you can also pass:

```bash
--track_cache path/to/cache_or_npz
```

when running `det`.

## 7. Batch Evaluation

For repeatable benchmark batches, prefer the shared batch interface:

```python
from run_batch_common import build_pipe, parse_shell_cases, run_cases
```

A minimal example:

```python
from pathlib import Path
from examples.run_batch_common import build_pipe, parse_shell_cases, run_cases

shell_text = Path("run_ablation38.sh").read_text(encoding="utf-8", errors="ignore")
cases = parse_shell_cases(shell_text)
pipe = build_pipe()

run_cases(
    pipe,
    cases,
    output_dir="results_batch",
    denoising_strength=0.88,
    transfer_method="ditflow",
    benchmark_preset="wan14b_32f_832x480",
    height=480,
    width=832,
    num_inference_steps=50,
    group_by_method=False,
    ttc_enabled=False,
    msa_enabled=False,
)
```

The repository already includes example batch launchers for some training-free methods:

- `examples/run_batch_38_ditflow_CUDA0.py`
- `examples/run_batch_38_moft_CUDA1.py`
- `examples/run_batch_38_smm_CUDA2.py`
- `examples/run_batch_38_motionclone_CUDA3.py`

These are good starting points if you want to run the same benchmark split on multiple GPUs.

## 8. Recommended Evaluation Workflow

### For DiTFlow / MOFT / SMM / MotionClone

1. Prepare reference videos in `data/`.
2. Confirm the benchmark preset and denoising strength.
3. Run either a single-case command or a batch script.
4. Collect generated videos and sidecar metadata JSON files from `results/`.

### For Motion Inversion / Motion Director

1. Prepare reference videos in `data/`.
2. Install training dependencies from `requirements-benchmark-train.txt`.
3. Run with `--stage full` first.
4. If needed, split work into `prepare`, `train`, and `infer`.
5. Reuse `artifacts/` for repeated inference.

### For DeT

1. Prepare reference videos in `data/`.
2. Install both mask-preparation and training dependencies if needed.
3. Generate seed masks with `examples/prepare_det_seed_masks.py`.
4. Run `det` with `--mask_dir` or provide `--track_cache`.
5. Check `artifacts/` and output videos after training and inference finish.

## 9. Output Layout

Typical outputs include:

- generated videos under `results/...`
- per-video metadata JSON files next to the generated videos
- reusable training artifacts under `artifacts/...`
- optional DeT masks under `det_seed_masks_*/`
- optional DeT debug overlays under `det_seed_masks_*_debug/`

## 10. Common Issues

`ModuleNotFoundError: No module named 'torch'`

- Your current Python environment is not the intended FastVMT environment.

`Reference video only has ... frames`

- Your input video is shorter than the selected benchmark preset requires.
- Either prepare the video first or do not force the benchmark preset.

`Grounding DINO could not detect ...`

- Your subject prompt in the DeT prompt JSON is too vague or mismatched to the first frame.

`det` runs but does not generate meaningful guidance

- Check that `mask_dir` is correct.
- Check the generated `0000.png` masks and debug overlays.
- If available, try using cached tracks or `cotracker`.

`motiondirector` or `motioninversion` is too slow

- These methods are tuning-based. They are expected to be much slower than training-free methods because they include optimization.

## 11. Quick Reference

Training-free:

```text
ditflow, moft, smm, motionclone
```

Tuning-based:

```text
motioninversion, motiondirector, det
```

Main CLI:

```bash
python examples/wan_14b_text_to_video.py --transfer_method <method_name> ...
```

Benchmark preset commonly used by batch scripts:

```text
wan14b_32f_832x480
```
