# LLM Molecular Graph Generation with GRPO

CS594 Reinforcement Learning at UIC. This project fine-tunes a 7B language model to generate molecular graphs structurally similar to those in the MUTAG dataset, using a two-stage pipeline of supervised fine-tuning followed by Group Relative Policy Optimization (GRPO).

## Overview

- **Dataset**: MUTAG, 188 nitroaromatic compounds with on average 17 nodes and 19 edges per graph.
- **Base model**: Qwen2.5-7B-Instruct, frozen, loaded with 4-bit NF4 quantization.
- **Adapter**: LoRA at rank 16, applied to all attention and MLP linear projections (~40M trainable parameters, ~0.9% of the model).
- **Pipeline**: SFT warmup (40 examples, 1 epoch) followed by GRPO for 100 steps.
- **Reward**: Dense shaped reward combining format check, parse check, graph validity, and weighted structural quality across six metrics.

## Results

| Stage | Mean Overall Score (n=15 eval graphs) |
|-------|---------------------------------------|
| Untrained baseline | 0.771 |
| After SFT | 0.833 |
| After SFT + GRPO | 0.837 |

Training ran on a single RTX Pro 3000 Blackwell laptop GPU (8 GB VRAM) and completed in approximately five hours including evaluation.

## Setup

### Quick start (Windows)

```bat
setup.bat
```

### Manual setup

1. Create a virtual environment:
```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Linux/Mac
```

2. Install PyTorch with CUDA support. For CUDA 12.8:
```bash
   pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

3. Install everything else:
```bash
   pip install -r requirements.txt
```

## Usage

### Inference only (untrained model)

```bash
python pipeline_grpo_opu.py
```

This loads Qwen2.5-7B-Instruct, generates a graph for the first MUTAG example, and prints a structural comparison.

### Run the full training pipeline

```bash
python pipeline_grpo_opu.py --train --sft_warmup --max_steps 100
```

This runs SFT warmup on 40 graphs, then GRPO for 100 steps. Outputs are written to `grpo_output/`, including the LoRA adapter, training reward log, and per-stage evaluation dumps.

### Generate plots

```bash
python make_plots.py
```

Reads the JSON dumps in `grpo_output/` and produces the four figures used in the report (per-metric bars, reward curve, score distribution, and qualitative graph comparison).

### Compute dataset statistics

```bash
python mutag_stats.py
```

Prints summary statistics for the MUTAG dataset.

## File structure

```
pipeline_grpo_opu.py    Main training pipeline
make_plots.py           Generates result figures from grpo_output/
mutag_stats.py          Dataset statistics
requirements.txt        Python dependencies (excluding torch)
setup.bat               Windows one-click setup
grpo_output/            Training outputs (logs, checkpoints, plots)
data/                   MUTAG dataset cache (auto-downloaded)
```

## Notes

- **Memory**: The pipeline is calibrated for 8 GB VRAM through 4-bit quantization, LoRA adapters, gradient checkpointing, and small per-device batch sizes. If you have more memory, you can raise `per_device_train_batch_size`, `gradient_accumulation_steps`, or `num_generations` in `pipeline_grpo_opu.py` for smoother training.
- **Windows + Python 3.13**: The pipeline includes a UTF-8 shim at the top of `pipeline_grpo_opu.py` to work around a cp1252 decoding issue when TRL loads its bundled Jinja chat templates. This is harmless on Linux and macOS.
- **Sparse vs dense reward**: An earlier version of the reward function was all-or-nothing (zero for any failure, around 0.7 for valid graphs). This caused the policy to collapse within five GRPO steps because of how GRPO normalizes advantages within a group. The current shaped reward awards partial credit at each verification stage, which keeps advantages bounded and training stable.