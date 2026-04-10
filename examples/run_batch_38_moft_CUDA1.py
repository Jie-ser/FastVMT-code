#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_batch_common import build_pipe, parse_shell_cases, run_cases


METHOD = "moft"
GPU_ID = "1"
DENOISING_STRENGTH = 0.88
BENCHMARK_PRESET = "wan14b_32f_832x480"
SHELL_PATH = Path(__file__).resolve().parents[1] / "run_ablation38.sh"


def load_cases() -> list[dict[str, str]]:
    shell_text = SHELL_PATH.read_text(encoding="utf-8", errors="ignore")
    return parse_shell_cases(shell_text)


def main() -> None:
    cases = load_cases()
    pipe = build_pipe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results38_compare") / f"{METHOD}_cuda{GPU_ID}" / timestamp
    run_cases(
        pipe,
        cases,
        output_dir=str(output_dir),
        denoising_strength=DENOISING_STRENGTH,
        transfer_method=METHOD,
        benchmark_preset=BENCHMARK_PRESET,
        height=480,
        width=832,
        num_inference_steps=50,
        group_by_method=False,
        ttc_enabled=False,
        msa_enabled=False,
    )


if __name__ == "__main__":
    main()
