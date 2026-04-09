#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageColor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffsynth import VideoData
from diffsynth.benchmarks import apply_benchmark_settings


DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM_MODEL = "facebook/sam-vit-huge"


def parse_shell_cases(shell_text: str) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for raw_line in shell_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("run_case "):
            continue
        tokens = shlex.split(line)
        if len(tokens) == 4:
            _, ref, prompt, negative_prompt = tokens
            case_id = ref
        elif len(tokens) == 5:
            _, ref, variant, prompt, negative_prompt = tokens
            case_id = f"{ref}_{variant}"
        else:
            raise ValueError(f"Unexpected run_case format: {line}")
        cases.append(
            {
                "case_id": case_id,
                "ref": ref,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
        )
    if not cases:
        raise ValueError("No run_case entries were found in the shell script.")
    return cases


def load_subject_prompts(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_models(args):
    try:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "Mask preparation requires Hugging Face SAM + Grounding DINO support. "
            "Install `pip install -r requirements-benchmark-mask.txt` first."
        ) from exc

    detector_processor = AutoProcessor.from_pretrained(args.grounding_model)
    detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(args.device)
    detector_model.eval()

    sam_processor = SamProcessor.from_pretrained(args.sam_model)
    sam_model = SamModel.from_pretrained(args.sam_model).to(args.device)
    sam_model.eval()
    return detector_processor, detector_model, sam_processor, sam_model


def detect_boxes(
    image: Image.Image,
    subject_prompt: str,
    detector_processor,
    detector_model,
    *,
    device: torch.device,
    box_thresholds: list[float],
    text_threshold: float,
) -> tuple[np.ndarray, list[str], list[float]]:
    clean_prompt = subject_prompt.strip()
    if not clean_prompt:
        raise ValueError("Empty subject prompt is not allowed.")
    if not clean_prompt.endswith("."):
        clean_prompt = f"{clean_prompt}."

    inputs = detector_processor(images=image, text=clean_prompt, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        outputs = detector_model(**inputs)

    target_sizes = [image.size[::-1]]
    last_result = None
    for threshold in box_thresholds:
        result = detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=float(threshold),
            text_threshold=float(text_threshold),
            target_sizes=target_sizes,
        )[0]
        last_result = result
        if len(result["boxes"]) > 0:
            boxes = result["boxes"].detach().cpu().float().numpy()
            labels = [str(label) for label in result["labels"]]
            scores = [float(score) for score in result["scores"].detach().cpu().float().tolist()]
            return boxes, labels, scores

    labels = [str(label) for label in last_result["labels"]] if last_result is not None else []
    scores = (
        [float(score) for score in last_result["scores"].detach().cpu().float().tolist()]
        if last_result is not None and "scores" in last_result
        else []
    )
    raise ValueError(
        f"Grounding DINO could not detect `{subject_prompt}`. Last labels={labels}, scores={scores}."
    )


def segment_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    sam_processor,
    sam_model,
    *,
    device: torch.device,
) -> np.ndarray:
    input_boxes = [[box.tolist() for box in boxes]]
    inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    inputs = _to_device(inputs, device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
    )
    masks_tensor = masks[0].float()
    iou_scores = outputs.iou_scores.detach().cpu()[0]
    union_mask = torch.zeros_like(masks_tensor[0, 0], dtype=torch.bool)
    for box_idx in range(masks_tensor.shape[0]):
        best_mask_idx = int(torch.argmax(iou_scores[box_idx]).item())
        union_mask |= masks_tensor[box_idx, best_mask_idx] > 0
    return union_mask.numpy().astype(np.uint8)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(path)


def save_debug_overlay(image: Image.Image, mask: np.ndarray, boxes: np.ndarray, path: str | Path) -> None:
    rgb = np.array(image.convert("RGB"))
    overlay = rgb.copy()
    color = np.array(ImageColor.getrgb("#00B050"), dtype=np.uint8)
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * color).astype(np.uint8)
    debug_pixels = overlay.copy()
    for box in boxes.astype(int):
        x0, y0, x1, y1 = box.tolist()
        x0 = max(0, min(debug_pixels.shape[1] - 1, x0))
        x1 = max(0, min(debug_pixels.shape[1] - 1, x1))
        y0 = max(0, min(debug_pixels.shape[0] - 1, y0))
        y1 = max(0, min(debug_pixels.shape[0] - 1, y1))
        debug_pixels[y0:y0 + 2, x0:x1 + 1] = (255, 80, 80)
        debug_pixels[max(y1 - 1, 0):y1 + 1, x0:x1 + 1] = (255, 80, 80)
        debug_pixels[y0:y1 + 1, x0:x0 + 2] = (255, 80, 80)
        debug_pixels[y0:y1 + 1, max(x1 - 1, 0):x1 + 1] = (255, 80, 80)
    Image.fromarray(debug_pixels).save(path)


def resolve_frame_path(
    case: dict[str, str],
    *,
    artifacts_root: str | Path | None,
    video_root: str | Path,
    height: int,
    width: int,
) -> Image.Image:
    if artifacts_root is not None:
        frame_path = Path(artifacts_root) / "shared" / case["case_id"] / "frames" / "0000.png"
        if frame_path.exists():
            return Image.open(frame_path).convert("RGB")
    video_path = Path(video_root) / f"{case['ref']}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Reference video not found: {video_path}")
    video_data = VideoData(video_file=str(video_path), height=height, width=width)
    return video_data[0]


def main(args) -> None:
    settings = apply_benchmark_settings(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        benchmark_preset=args.benchmark_preset,
    )
    args.height = settings["height"]
    args.width = settings["width"]
    args.num_frames = settings["num_frames"]
    args.num_inference_steps = settings["num_inference_steps"]
    args.device = torch.device(args.device)

    cases = parse_shell_cases(Path(args.shell_script).read_text(encoding="utf-8", errors="ignore"))
    subject_prompts = load_subject_prompts(args.subject_prompts_json)
    missing_prompts = [case["case_id"] for case in cases if case["case_id"] not in subject_prompts]
    if missing_prompts:
        raise ValueError(
            "Missing subject prompts for cases: " + ", ".join(missing_prompts) +
            ". Add them to the JSON mapping before generating DeT masks."
        )

    detector_processor, detector_model, sam_processor, sam_model = load_models(args)
    box_thresholds = [float(value) for value in args.box_thresholds]

    output_root = Path(args.output_dir)
    debug_root = Path(args.debug_dir) if args.debug_dir else None
    summary: list[dict[str, Any]] = []

    for case in cases:
        case_dir = output_root / case["case_id"]
        mask_path = case_dir / "0000.png"
        metadata_path = case_dir / "mask_meta.json"
        if not args.overwrite and mask_path.exists() and metadata_path.exists():
            summary.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue

        image = resolve_frame_path(
            case,
            artifacts_root=args.artifacts_root,
            video_root=args.video_root,
            height=args.height,
            width=args.width,
        )
        subject_prompt = subject_prompts[case["case_id"]]
        boxes, labels, scores = detect_boxes(
            image,
            subject_prompt,
            detector_processor,
            detector_model,
            device=args.device,
            box_thresholds=box_thresholds,
            text_threshold=args.text_threshold,
        )
        mask = segment_boxes(
            image,
            boxes,
            sam_processor,
            sam_model,
            device=args.device,
        )
        area_ratio = float(mask.mean())
        if area_ratio <= 0.0:
            raise ValueError(f"Generated an empty mask for {case['case_id']}.")

        case_dir.mkdir(parents=True, exist_ok=True)
        save_mask(mask, mask_path)

        metadata = {
            "case_id": case["case_id"],
            "ref": case["ref"],
            "subject_prompt": subject_prompt,
            "boxes": boxes.round(2).tolist(),
            "labels": labels,
            "scores": [round(score, 4) for score in scores],
            "mask_area_ratio": round(area_ratio, 6),
            "image_width": image.size[0],
            "image_height": image.size[1],
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(metadata)

        if debug_root is not None:
            debug_case_dir = debug_root / case["case_id"]
            debug_case_dir.mkdir(parents=True, exist_ok=True)
            save_debug_overlay(image, mask, boxes, debug_case_dir / "overlay.png")

        print(
            f"[prepare_det_seed_masks] {case['case_id']}: "
            f"boxes={len(boxes)}, mask_area_ratio={area_ratio:.4f}, prompt={subject_prompt}"
        )

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare official-style first-frame seed masks for DeT.")
    parser.add_argument("--shell_script", type=str, default="run_ablation38.sh")
    parser.add_argument("--subject_prompts_json", type=str, default="examples/det_subject_prompts_38.json")
    parser.add_argument("--video_root", type=str, default="data")
    parser.add_argument("--artifacts_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="det_seed_masks_38")
    parser.add_argument("--debug_dir", type=str, default="det_seed_masks_38_debug")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grounding_model", type=str, default=DEFAULT_GROUNDING_MODEL)
    parser.add_argument("--sam_model", type=str, default=DEFAULT_SAM_MODEL)
    parser.add_argument("--box_thresholds", type=float, nargs="+", default=[0.35, 0.25, 0.15])
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--benchmark_preset", type=str, default="wan14b_32f_832x480")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    main(parser.parse_args())
