# LLMGraphGenRL — Graph Generation with LLMs & GRPO

> **CS594 Course Final Project**  
> Exploring reinforcement learning–aligned large language models for structured graph generation across three distinct domains.

---

## Overview

This repository contains the final project for **CS594**, investigating whether **Group Relative Policy Optimization (GRPO)** can effectively align large language models for the task of graph generation. Rather than relying solely on supervised fine-tuning, we leverage RL-based training signals to teach LLMs to produce valid, meaningful graphs across three fundamentally different graph domains.

Each team member owned an independent sub-project, housed in its own folder. Together, the three tracks form a unified study of how GRPO scales across graph types with varying structural constraints and domain-specific validity requirements.

---

## Repository Structure

```
LLMGraphGenRL_GRPO/
├── Citation_Netowrks/       # Track 1 — Citation network graph generation
├── MolecularGeneration/     # Track 2 — Molecular graph generation
├── SceneGraphs/             # Track 3 — Scene graph generation from images
└── README.md
```

---

## The Three Tracks

### 📄 Track 1 — Citation Networks (`Citation_Netowrks/`)

**Goal:** Generate realistic academic citation graphs where nodes represent papers and edges represent citations.

Citation graphs have rich semantic structure — the generated graphs must reflect realistic connectivity patterns (e.g., in-degree distributions, clustering by topic). This track investigates whether GRPO reward signals based on graph-structural validity can guide an LLM to produce plausible citation network topologies.

**Key aspects:**
- Node/edge generation conditioned on textual paper metadata
- Structural reward functions (degree distribution, clustering coefficient)
- Evaluation against real citation network benchmarks

---

### 🧪 Track 2 — Molecular Generation (`MolecularGeneration/`)

**Goal:** Generate valid, novel molecular graphs represented as SMILES strings or graph adjacency structures, targeting drug-discovery applications.

Molecular graphs have strict hard validity constraints — every generated molecule must be chemically valid. GRPO is used to optimize for verifiable chemical correctness (valency, aromaticity, ring closure) rather than just log-likelihood, pushing the LLM to generate molecules that are simultaneously valid, unique, and novel.

**Key aspects:**
- SMILES-based and graph-based molecular representations
- Validity, uniqueness, and novelty (VUN) as reward signals
- Benchmarks on standard molecular generation datasets

---

### 🖼️ Track 3 — Scene Graphs (`SceneGraphs/`)

**Goal:** Generate structured scene graphs from visual inputs — predicting objects, attributes, and their relationships as a graph.

Scene graph generation (SGG) is a multimodal task requiring both visual understanding and structured output. This track applies GRPO atop a visual-instruction-tuned LLM, using rule-based rewards (e.g., bounding box IoU, relationship triplet validity) that supervised fine-tuning alone cannot effectively capture.

**Key aspects:**
- Multimodal LLM fine-tuning (vision + language)
- Rule-based RL rewards for object detection and relationship prediction
- Evaluation on standard SGG benchmarks

---

## Methodology

All three tracks share a common RL training backbone:

```
Pretrained / Instruction-Tuned LLM
          │
          ▼
  Supervised Fine-Tuning (SFT)      ← domain-specific graph data
          │
          ▼
  GRPO Reinforcement Learning        ← verifiable, rule-based rewards
          │
          ▼
  Aligned Graph-Generating LLM
```

**GRPO (Group Relative Policy Optimization)** generates a group of candidate outputs per prompt, scores them with a reward function, and uses the relative rankings within the group to compute policy gradients — without needing a separate value network. This makes it well-suited for structured generation tasks where rewards are discrete and verifiable.

---

## Getting Started

Each track is self-contained. Navigate into the relevant folder and follow its own notebook/instructions:

```bash
# Clone the repo
git clone https://github.com/shadmankabir27/LLMGraphGenRL_GRPO.git
cd LLMGraphGenRL_GRPO

# Go to the track you want to explore
cd Citation_Netowrks/     # or MolecularGeneration/ or SceneGraphs/
```

Each folder contains Jupyter notebooks (`.ipynb`) and any supporting Python scripts. Dependencies vary per track — check inside each folder for environment setup details.

**General requirements (varies by track):**
- Python 3.9+
- PyTorch
- Hugging Face `transformers` & `trl`
- Domain-specific libraries (e.g., `rdkit` for molecules, `pyg` for graph utilities)

---

## Team

| Track | Domain | Folder |
|---|---|---|
| Homaira Huda Shomee | Citation Networks | `Citation_Netowrks/` |
| Shadman Kabir | Molecular Generation | `MolecularGeneration/` |
| Mehul Mathur | Scene Graphs | `SceneGraphs/` |

---

## Course

**CS594** — Reinforcement Learning Course Final Project  
All work in this repository was produced for academic purposes.

---

## References

- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* (GRPO)
- You et al. (2018). *Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation.*
- Vignac et al. (2023). *DiGress: Discrete Denoising Diffusion for Graph Generation.*
- R1-SGG: *Compile Scene Graphs with Reinforcement Learning* (2025).
