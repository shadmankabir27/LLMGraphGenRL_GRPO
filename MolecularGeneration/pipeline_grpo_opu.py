"""
=============================================================================
LLM Molecular Graph Generation -- GRPO Training Pipeline (FIXED)
CS594 Reinforcement Learning -- UIC
=============================================================================

This is a rewrite of pipeline_grpo.py with fixes for the catastrophic
policy collapse that happened after a handful of GRPO steps (random Chinese
characters, runs of zeros, unparseable output).

Applied fixes:
  1. Proper QLoRA setup: prepare_model_for_kbit_training() before LoRA
  2. bf16 end-to-end (bnb compute dtype + training dtype now match)
  3. Dense shaped reward (format + parse + valid + quality) instead of
     an all-or-nothing cliff -- eliminates the main source of advantage
     explosion in GRPO
  4. Reward function uses per-completion target lookup (correct for modern
     TRL), with a defensive fallback for older TRL versions
  5. GRPOConfig uses max_completion_length, larger effective batch via
     gradient accumulation, stronger KL (beta), saner max_grad_norm
  6. Optional SFT warmup (--sft_warmup) anchors the JSON format before GRPO
  7. Per-step reward logging with a callback so you can see collapse early

HOW TO RUN
  Inference only:   python pipeline_grpo_opu.py
  Verify GRPO:      python pipeline_grpo_opu.py --train --max_steps 5
  With SFT warmup:  python pipeline_grpo_opu.py --train --sft_warmup --max_steps 50
  Full training:    python pipeline_grpo_opu.py --train --sft_warmup --max_steps 500

REQUIREMENTS
  pip install --upgrade trl peft datasets transformers bitsandbytes accelerate
"""

# =============================================================================
# WINDOWS UTF-8 SHIM  (must run BEFORE importing trl/transformers/anything)
# =============================================================================
# TRL >=0.19 loads its bundled Jinja chat templates via pathlib.Path.read_text()
# without passing encoding=. On Windows + Python 3.13 the default codec is
# cp1252, which cannot decode the UTF-8 bytes in deepseekv3.jinja (byte 0x81
# is undefined in cp1252). The import then crashes with UnicodeDecodeError
# before training ever starts.
#
# We force every Path.read_text() / builtin open() to default to UTF-8.
# This is only needed on Windows but is harmless on Linux/macOS.
import sys as _sys
if _sys.platform == "win32":
    import pathlib as _pathlib
    _orig_read_text = _pathlib.Path.read_text

    def _utf8_read_text(self, encoding=None, errors=None, newline=None):
        if encoding is None:
            encoding = "utf-8"
        try:
            return _orig_read_text(self, encoding=encoding,
                                   errors=errors, newline=newline)
        except TypeError:
            # older pathlib without `newline` kw
            return _orig_read_text(self, encoding=encoding, errors=errors)

    _pathlib.Path.read_text = _utf8_read_text

    # Belt-and-braces: also flip the process-wide default where possible.
    import os as _os
    _os.environ.setdefault("PYTHONUTF8", "1")
    _os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# =============================================================================

import argparse
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.datasets import TUDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# SECTION 1: MODEL LOADING  (FIX 2: bf16 end-to-end)
# =============================================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def load_model():
    """
    Loads Qwen and picks the best precision strategy for the hardware.

    Fix: When 4-bit quantization is used, compute dtype is now bfloat16 to
    match the bf16=True we pass to GRPOConfig. Mixing fp16 (bnb compute) with
    bf16 (autocast in the trainer) was a main source of numerical instability.
    """

    print(f"Loading model: {MODEL_NAME}")
    print("First run downloads ~15GB. Cached afterwards.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware: {gpu_name}")

        # bf16 is supported on Ampere (RTX 30xx, A100) and newer.
        supports_bf16 = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
        print(f"Using compute dtype: {compute_dtype}")

        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,   # <-- fixed
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=compute_dtype,              # <-- fixed
            )
            print("Load strategy: 4-bit NF4 quantization\n")

        except Exception as e:
            print(f"4-bit quantization failed ({e}).")
            print("Falling back to full-precision bf16/fp16...\n")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=compute_dtype,
            )

    elif torch.backends.mps.is_available():
        print("Hardware: Apple Silicon (MPS)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to("mps")

    else:
        print("Hardware: CPU only -- inference will be slow.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

    print("Model loaded.\n")
    return model, tokenizer


# =============================================================================
# SECTION 2: MUTAG DATASET
# =============================================================================

def load_mutag_dataset():
    dataset = TUDataset(root="data", name="MUTAG")
    print(f"MUTAG loaded: {len(dataset)} graphs, {dataset.num_classes} classes")
    return dataset


def pyg_graph_to_edge_list(data):
    nodes = list(range(data.num_nodes))
    seen, edges = set(), []
    for a, b in data.edge_index.t().tolist():
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            edges.append([key[0], key[1]])
    return nodes, edges


# =============================================================================
# SECTION 3: GRAPH PROPERTIES
# =============================================================================

def compute_graph_properties(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes else 0
    max_degree = max(degrees) if degrees else 0
    is_connected = nx.is_connected(G) if num_nodes > 0 else False

    clustering_coefficient = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    cycle_basis = nx.cycle_basis(G)
    num_5_cycles = sum(1 for c in cycle_basis if len(c) == 5)
    num_6_cycles = sum(1 for c in cycle_basis if len(c) == 6)
    avg_cycle_length = (
        round(sum(len(c) for c in cycle_basis) / len(cycle_basis), 2)
        if cycle_basis else 0
    )

    diameter = nx.diameter(G) if is_connected else -1

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "num_components": nx.number_connected_components(G),
        "is_connected": is_connected,
        "clustering_coefficient": round(clustering_coefficient, 4),
        "transitivity": round(transitivity, 4),
        "diameter": diameter,
        "graph_object": G,
        "num_5_cycles": num_5_cycles,
        "num_6_cycles": num_6_cycles,
        "avg_cycle_length": avg_cycle_length,
    }


def print_properties(props, label="Graph"):
    print(f"\n--- {label} ---")
    print(f"  Nodes:                  {props['num_nodes']}")
    print(f"  Edges:                  {props['num_edges']}")
    print(f"  Average Degree:         {props['avg_degree']}")
    print(f"  Max Degree:             {props['max_degree']}")
    print(f"  Connected:              {props['is_connected']}")
    print(f"  Clustering Coefficient: {props['clustering_coefficient']}")
    print(f"  Diameter:               {props['diameter']}")
    print(f"  5-cycles:               {props['num_5_cycles']}")
    print(f"  6-cycles:               {props['num_6_cycles']}")
    print(f"  Avg cycle length:       {props['avg_cycle_length']}")


# =============================================================================
# SECTION 4: PROMPT BUILDING
# =============================================================================

def build_prompt(target_props):
    return f"""You are a graph generator. Your task is to generate an undirected graph that matches the following structural properties as closely as possible.

Target properties:
- Number of nodes: {target_props['num_nodes']}
- Number of edges: {target_props['num_edges']}
- Average degree: {target_props['avg_degree']}
- Maximum degree: {target_props['max_degree']}
- The graph should be connected (all nodes reachable from any other node)
- Clustering coefficient: {target_props['clustering_coefficient']}
- Diameter: {target_props['diameter']}

Important structural guidance:
- This graph represents an aromatic molecule from the MUTAG dataset.
- Build one or two small rings of exactly 5 or 6 nodes, connected by short
  chain-like extensions (paths of 2-4 nodes).
- Think: a hexagon or pentagon with a short tail, like benzene with a side chain.
- DO NOT generate one large ring containing all nodes.
- DO NOT generate a graph that looks like a circle or oval.
- DO NOT generate any triangles (3-node rings).

Rules:
- Nodes are labelled 0 to {target_props['num_nodes'] - 1}
- No self-loops, no duplicate edges, undirected

Output ONLY valid JSON in exactly this format, no other text:
{{"nodes": [0, 1, 2, ...], "edges": [[0,1], [1,2], ...]}}

No explanation, no markdown, no code blocks. Output only the raw JSON object."""


# =============================================================================
# SECTION 5: LLM INFERENCE
# =============================================================================

def call_llm(prompt, model, tokenizer, max_new_tokens=1024):
    messages = [
        {"role": "system",
         "content": "You are a graph generator that outputs only valid JSON."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # CRITICAL: put the model in eval mode for generation. After GRPO
    # training, the model is left in train mode with LoRA dropout and
    # gradient checkpointing still active -- dropout at generation time
    # randomizes adapter weights on every token and produces degenerate
    # output ("status_t * 11.200000..."). Save the prior state so training
    # can resume correctly if run_single is called between GRPO phases.
    was_training = model.training

    # On PEFT-wrapped models, `is_gradient_checkpointing` lives on the
    # inner base model, not the outer wrapper. Walk down to find it.
    def _gc_on(m):
        for attr_chain in [
            ("is_gradient_checkpointing",),
            ("base_model", "model", "is_gradient_checkpointing"),
            ("model", "is_gradient_checkpointing"),
        ]:
            obj = m
            ok = True
            for a in attr_chain:
                if hasattr(obj, a):
                    obj = getattr(obj, a)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, bool):
                return obj
        return False

    gc_was_enabled = _gc_on(model)

    model.eval()
    if gc_was_enabled and hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    print("Generating graph...")
    start = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,           # safe now that gc is off
            )
    finally:
        # Restore prior training state so GRPO can continue if it wants to
        if was_training:
            model.train()
        if gc_was_enabled and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()

    print(f"Generated in {round(time.time() - start, 1)}s")

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# =============================================================================
# SECTION 6: RESPONSE PARSING
# =============================================================================

def parse_llm_response(response_text, verbose=True):
    if not response_text:
        return None
    if verbose:
        print(f"\n--- Raw LLM Response (first 500 chars) ---")
        print(response_text[:500])
        print("--- End of preview ---\n")

    try:
        data = json.loads(response_text.strip())
        if "nodes" in data and "edges" in data:
            return data
    except json.JSONDecodeError:
        pass

    for pattern in [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{[^{}]*"nodes"[^{}]*"edges"[^{}]*\})',
        r'(\{.*\})',
    ]:
        for m in re.findall(pattern, response_text, re.DOTALL):
            try:
                data = json.loads(m.strip())
                if "nodes" in data and "edges" in data:
                    return data
            except json.JSONDecodeError:
                continue

    if verbose:
        print("WARNING: Could not parse LLM response as JSON.")
    return None


# =============================================================================
# SECTION 7: GRAPH VALIDATION
# =============================================================================

def validate_and_clean_graph(data, target_num_nodes, verbose=True):
    if data is None:
        return None, None

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list) or len(nodes) == 0:
        return None, None
    if not isinstance(edges, list):
        return None, None

    clean_edges, seen = [], set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        try:
            a, b = int(edge[0]), int(edge[1])
        except (ValueError, TypeError):
            continue
        if a == b or a not in nodes or b not in nodes:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        clean_edges.append([a, b])

    if verbose:
        print(f"\nValidation: {len(nodes)} nodes (target {target_num_nodes}),"
              f" {len(clean_edges)} clean edges")
    return nodes, clean_edges


# =============================================================================
# SECTION 8: EVALUATION / QUALITY SCORE
# =============================================================================

WEIGHTS = {
    "num_nodes":              0.05,
    "num_edges":              0.15,
    "is_connected":           0.10,
    "clustering_coefficient": 0.15,
    "diameter":               0.25,
    "avg_cycle_length":       0.30,
}
# Sum: 1.00


def evaluate(target_props, generated_props, verbose=True):
    """Per-metric score = max(0, 1 - |gen - target| / target)."""
    scores = {}

    def rel_score(t, g):
        if t == 0:
            return 1.0 if g == 0 else 0.0
        return max(0.0, 1.0 - abs(g - t) / abs(t))

    scores["num_nodes"]  = round(rel_score(target_props["num_nodes"],
                                           generated_props["num_nodes"]), 3)
    scores["num_edges"]  = round(rel_score(target_props["num_edges"],
                                           generated_props["num_edges"]), 3)
    scores["is_connected"] = 1.0 if (
        target_props["is_connected"] == generated_props["is_connected"]
    ) else 0.0

    t_cc, g_cc = target_props["clustering_coefficient"], generated_props["clustering_coefficient"]
    scores["clustering_coefficient"] = round(
        max(0.0, 1.0 - abs(g_cc - t_cc) / t_cc) if t_cc > 0
        else (1.0 if g_cc == 0.0 else max(0.0, 1.0 - g_cc)), 3
    )

    t_d, g_d = target_props["diameter"], generated_props["diameter"]
    if t_d == -1 and g_d == -1:
        scores["diameter"] = 1.0
    elif t_d == -1 or g_d == -1:
        scores["diameter"] = 0.0
    else:
        scores["diameter"] = round(rel_score(t_d, g_d), 3)

    scores["avg_cycle_length"] = round(
        rel_score(target_props["avg_cycle_length"],
                  generated_props["avg_cycle_length"]), 3
    )

    overall = sum(scores[m] * WEIGHTS[m] for m in WEIGHTS)
    scores["overall"] = round(overall, 3)

    if verbose:
        print("\n" + "=" * 72)
        print("EVALUATION")
        print("=" * 72)
        for m in WEIGHTS:
            t = target_props.get(m, "-")
            g = generated_props.get(m, "-")
            print(f"  {m:<28} target={str(t):>8}  gen={str(g):>8}  "
                  f"score={scores[m]:>5.3f}  w={WEIGHTS[m]:.2f}")
        print(f"  {'WEIGHTED OVERALL':<28} "
              f"{'':>8}  {'':>12} {scores['overall']:>5.3f}")
        print("=" * 72)

    return scores


# =============================================================================
# SECTION 9: VISUALIZATION
# =============================================================================

def visualize_graphs(target_props, generated_props, graph_index=0, save_path=None):
    if save_path is None:
        save_path = f"graph_{graph_index}_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"MUTAG Graph #{graph_index} -- Target vs Generated",
                 fontsize=14, fontweight="bold")

    for ax, props, title, color in [
        (axes[0], target_props, "TARGET", "steelblue"),
        (axes[1], generated_props, "GENERATED", "tomato"),
    ]:
        pos = nx.spring_layout(props["graph_object"], seed=42)
        nx.draw_networkx(
            props["graph_object"], pos=pos, ax=ax, with_labels=True,
            node_color=color, node_size=500, font_color="white",
            font_size=8, edge_color="gray", width=1.5,
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {save_path}")


# =============================================================================
# SECTION 10: SINGLE-GRAPH RUNNER
# =============================================================================

def run_single(dataset, model, tokenizer, graph_index=0, save_viz=True):
    print("=" * 65)
    print(f"SINGLE GRAPH PIPELINE -- Graph #{graph_index}")
    print("=" * 65)

    pyg_data = dataset[graph_index]
    target_nodes, target_edges = pyg_graph_to_edge_list(pyg_data)
    target_props = compute_graph_properties(target_nodes, target_edges)
    print_properties(target_props, label="Target")

    response = call_llm(build_prompt(target_props), model, tokenizer)
    parsed = parse_llm_response(response)
    if parsed is None:
        print("Failed: could not parse LLM response")
        return None

    gen_nodes, gen_edges = validate_and_clean_graph(
        parsed, target_props["num_nodes"]
    )
    if gen_nodes is None:
        print("Failed: invalid graph")
        return None

    gen_props = compute_graph_properties(gen_nodes, gen_edges)
    print_properties(gen_props, label="Generated")

    scores = evaluate(target_props, gen_props)

    if save_viz:
        visualize_graphs(target_props, gen_props, graph_index=graph_index)

    return scores


# =============================================================================
# SECTION 10b: BATCH EVAL DUMP (for plotting)
# =============================================================================

def run_eval_dump(dataset, model, tokenizer, label, out_path, num_graphs=20,
                  store_samples=2):
    """
    Generates graphs for `num_graphs` MUTAG examples and stores per-metric
    means + the full distribution of overall scores to a JSON file consumed
    by make_plots.py. Stores `store_samples` full graph examples (target +
    generated edges) so we can render before/after diagrams later.
    """
    print("=" * 65)
    print(f"BATCH EVAL DUMP -- {label}  (n={num_graphs})")
    print("=" * 65)

    per_metric_scores = {m: [] for m in WEIGHTS}
    overall_scores = []
    samples = []
    n = min(num_graphs, len(dataset))

    for idx in range(n):
        pyg_data = dataset[idx]
        target_nodes, target_edges = pyg_graph_to_edge_list(pyg_data)
        target_props = compute_graph_properties(target_nodes, target_edges)

        try:
            response = call_llm(build_prompt(target_props), model, tokenizer)
        except Exception as e:
            print(f"  graph {idx}: generation failed ({e})")
            overall_scores.append(0.0)
            for m in per_metric_scores:
                per_metric_scores[m].append(0.0)
            continue

        parsed = parse_llm_response(response, verbose=False)
        gen_nodes, gen_edges = (None, None)
        if parsed is not None:
            gen_nodes, gen_edges = validate_and_clean_graph(
                parsed, target_props["num_nodes"], verbose=False
            )

        if gen_nodes is None or not gen_edges:
            overall_scores.append(0.0)
            for m in per_metric_scores:
                per_metric_scores[m].append(0.0)
            print(f"  graph {idx}: parse/validate failed -> overall=0.000")
        else:
            gen_props = compute_graph_properties(gen_nodes, gen_edges)
            scores = evaluate(target_props, gen_props, verbose=False)
            overall_scores.append(scores["overall"])
            for m in per_metric_scores:
                per_metric_scores[m].append(scores.get(m, 0.0))
            print(f"  graph {idx}: overall={scores['overall']:.3f}")

            # Store the first few full samples for graph rendering.
            if len(samples) < store_samples:
                samples.append({
                    "graph_index": idx,
                    "target_nodes": target_nodes,
                    "target_edges": target_edges,
                    "gen_nodes": gen_nodes,
                    "gen_edges": gen_edges,
                    "overall": scores["overall"],
                })

    payload = {
        "label": label,
        "n": n,
        "mean_overall": (sum(overall_scores) / len(overall_scores))
                        if overall_scores else 0.0,
        "per_metric_mean": {m: (sum(v) / len(v) if v else 0.0)
                            for m, v in per_metric_scores.items()},
        "overall_per_graph": overall_scores,
        "samples": samples,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_path}  (mean overall = {payload['mean_overall']:.3f})\n")
    return payload




def apply_lora(model):
    """
    Applies LoRA for memory-efficient fine-tuning.

    Fix: When the base is 4-bit quantized, we MUST call
    prepare_model_for_kbit_training() before get_peft_model(). This:
      - casts layer norms / lm_head to fp32 for numerical stability
      - enables gradient checkpointing correctly
      - makes input embeddings require grad so gradients flow to adapters

    Without this, training on a 4-bit base produces NaN gradients within
    a handful of steps and destroys the output distribution.
    """
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    is_quantized = getattr(model, "is_loaded_in_4bit", False) or \
                   getattr(model, "is_loaded_in_8bit", False)

    if is_quantized:
        print("Base is quantized -- running prepare_model_for_kbit_training...")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        # Explicit, non-reentrant gradient checkpointing is more stable.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # older transformers don't accept the kwargs argument
            model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,     # was 0.05; zero dropout keeps train/eval identical
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: trainable {trainable:,} "
          f"({100 * trainable / total:.3f}%) of {total:,}\n")

    # Paranoid sanity check on dtypes -- common source of subtle bugs.
    adapter_dtypes = {p.dtype for n, p in model.named_parameters()
                      if p.requires_grad}
    print(f"Adapter dtypes: {adapter_dtypes}")
    return model


def save_lora_adapter(model, save_dir="grpo_output/lora_adapter"):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"LoRA adapter saved to: {save_dir}/")


# =============================================================================
# SECTION 12: GRPO DATASET
# =============================================================================

def build_grpo_dataset(dataset):
    from datasets import Dataset

    print("Building GRPO dataset from all MUTAG graphs...")
    rows = []
    for i in range(len(dataset)):
        pyg_data = dataset[i]
        nodes, edges = pyg_graph_to_edge_list(pyg_data)
        props = compute_graph_properties(nodes, edges)
        prompt_text = build_prompt(props)

        messages = [
            {"role": "system",
             "content": "You are a graph generator that outputs only valid JSON."},
            {"role": "user", "content": prompt_text},
        ]
        rows.append({
            "prompt": messages,
            "target_props": json.dumps({
                k: v for k, v in props.items() if k != "graph_object"
            }),
        })

    grpo_dataset = Dataset.from_list(rows)
    print(f"Dataset built: {len(grpo_dataset)} prompts\n")
    return grpo_dataset


# =============================================================================
# SECTION 13: REWARD FUNCTION  (FIX 3: dense reward, no cliff)
# =============================================================================

NUM_GENERATIONS = 4


def make_reward_fn():
    """
    Dense shaped reward (max = 1.0):
      +0.10 -- output contains JSON-like structure (braces + nodes + edges)
      +0.10 -- parses as JSON with nodes/edges
      +0.10 -- validates as a non-empty graph
      +0.70 * quality -- weighted structural score vs target

    This eliminates the all-or-nothing cliff where parse failures got 0.0
    while valid graphs got ~0.7. With a cliff, GRPO's advantage normalization
    explodes: a single good sample among 3 failures gets huge +advantage and
    the failures get huge -advantage, pushing probability mass onto arbitrary
    tokens and collapsing the output distribution.

    The dense signal means:
      - Every sample gets SOME credit for moving in the right direction.
      - Advantages are bounded in roughly [-1.0, +1.0] rather than [-sigma, sigma]
        driven by a bimodal distribution.
      - The optimizer can nudge, not shove.
    """

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        target_props_list = kwargs.get("target_props", None)

        # Some older TRL versions pass target_props once per prompt instead
        # of once per completion. Detect and handle both.
        per_completion = (
            target_props_list is not None
            and len(target_props_list) == len(completions)
        )

        for i, completion in enumerate(completions):
            try:
                # Extract text regardless of completion format
                if isinstance(completion, list) and completion:
                    text = completion[0].get("content", "")
                elif isinstance(completion, dict):
                    text = completion.get("content", "")
                else:
                    text = str(completion)

                r = 0.0

                # Stage 1: format reward -- cheap sanity, awards partial credit
                has_braces = "{" in text and "}" in text
                has_keys = "nodes" in text and "edges" in text
                if has_braces and has_keys:
                    r += 0.10

                # Stage 2: parse reward
                parsed = parse_llm_response(text, verbose=False)
                if parsed is None:
                    rewards.append(r)
                    continue
                r += 0.10

                # Look up the correct target
                if target_props_list is None:
                    rewards.append(r)
                    continue
                idx = i if per_completion else i // NUM_GENERATIONS
                if idx >= len(target_props_list):
                    rewards.append(r)
                    continue
                target_props = json.loads(target_props_list[idx])
                expected_nodes = target_props.get("num_nodes", 17)

                # Stage 3: validity reward
                gen_nodes, gen_edges = validate_and_clean_graph(
                    parsed, expected_nodes, verbose=False
                )
                if gen_nodes is None or len(gen_edges) == 0:
                    rewards.append(r)
                    continue
                r += 0.10

                # Stage 4: structural quality (main reward)
                gen_props = compute_graph_properties(gen_nodes, gen_edges)
                scores = evaluate(target_props, gen_props, verbose=False)
                r += 0.70 * float(scores["overall"])

                rewards.append(r)

            except Exception as e:
                # Never let reward computation throw -- log and give 0.
                print(f"[reward] exception on completion {i}: {e}")
                rewards.append(0.0)

        return rewards

    return reward_fn


# =============================================================================
# SECTION 14: TRAINING CALLBACK  (FIX 7: early-collapse detection)
# =============================================================================

class RewardSanityCallback:
    """
    Logs reward stats every step AND persists them to grpo_output/reward_log.json
    so make_plots.py can build the reward-curve figure later.
    """

    def __init__(self):
        from transformers import TrainerCallback
        self._Base = TrainerCallback

    def build(self):
        parent = self._Base
        log_path = os.path.join("grpo_output", "reward_log.json")
        os.makedirs("grpo_output", exist_ok=True)
        # Truncate any prior log so plots reflect this run only.
        with open(log_path, "w") as f:
            json.dump([], f)

        class _Impl(parent):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                reward = logs.get("reward") or logs.get("rewards/chosen") \
                         or logs.get("train/reward")
                reward_std = logs.get("reward_std") or logs.get("rewards/std")
                completion_len = logs.get("completions/mean_length") \
                                 or logs.get("completion_length")
                step = state.global_step
                bits = [f"step={step}"]
                if reward is not None:
                    bits.append(f"reward={reward:.3f}")
                if reward_std is not None:
                    bits.append(f"reward_std={reward_std:.3f}")
                if completion_len is not None:
                    bits.append(f"completion_len={completion_len:.0f}")
                if len(bits) > 1:
                    print("[sanity] " + "  ".join(bits))
                # Persist to disk so plotting can read it later.
                if reward is not None:
                    try:
                        with open(log_path, "r") as f:
                            data = json.load(f)
                    except Exception:
                        data = []
                    data.append({
                        "step": step,
                        "reward": float(reward),
                        "reward_std": float(reward_std) if reward_std is not None else 0.0,
                        "completion_len": float(completion_len) if completion_len is not None else 0.0,
                    })
                    with open(log_path, "w") as f:
                        json.dump(data, f, indent=2)

        return _Impl()


# =============================================================================
# SECTION 15: OPTIONAL SFT WARMUP  (FIX 6)
# =============================================================================

def sft_warmup(model, tokenizer, dataset, num_examples=40, num_epochs=1,
               learning_rate=1e-4):
    """
    Short supervised fine-tune on (prompt, ground-truth JSON) pairs built
    from MUTAG graphs themselves. Anchors the JSON output format before
    GRPO starts perturbing the policy. Typically takes a minute or two.
    """
    print("=" * 65)
    print("SFT WARMUP -- anchoring JSON output format")
    print("=" * 65)
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    # Build (prompt_ids, target_ids) pairs.
    examples = []
    num_examples = min(num_examples, len(dataset))
    for i in range(num_examples):
        pyg_data = dataset[i]
        nodes, edges = pyg_graph_to_edge_list(pyg_data)
        props = compute_graph_properties(nodes, edges)
        prompt_text = build_prompt(props)
        target_json = json.dumps({"nodes": nodes, "edges": edges})

        messages = [
            {"role": "system",
             "content": "You are a graph generator that outputs only valid JSON."},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": target_json},
        ]
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_only = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        examples.append((prompt_only, full))

    model.train()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    total_steps = num_epochs * len(examples)
    step = 0
    start = time.time()
    for epoch in range(num_epochs):
        for prompt_only, full in examples:
            full_ids = tokenizer(full, return_tensors="pt",
                                 truncation=True, max_length=2048
                                 ).input_ids.to(model.device)
            prompt_ids = tokenizer(prompt_only, return_tensors="pt",
                                   truncation=True, max_length=2048
                                   ).input_ids.to(model.device)
            prompt_len = prompt_ids.shape[1]

            # Mask out the prompt tokens in the loss.
            labels = full_ids.clone()
            labels[0, :prompt_len] = -100

            outputs = model(input_ids=full_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            step += 1
            if step % 5 == 0 or step == total_steps:
                print(f"  sft step {step}/{total_steps}  loss={loss.item():.3f}")

    print(f"SFT warmup done in {round(time.time() - start, 1)}s\n")
    # Note: do NOT call model.eval() here. main() will run the post-SFT
    # batch eval which calls call_llm(), and call_llm() handles the
    # eval/train switch internally. After eval, GRPO needs the model in
    # train mode -- if we eval() here and then GRPOTrainer doesn't
    # re-enable training (which it doesn't always do explicitly), the LoRA
    # adapter weights won't update during RL.
    return model


# =============================================================================
# SECTION 16: GRPO TRAINING  (FIX 4+5: correct config, bigger effective batch)
# =============================================================================

def run_grpo_training(model, tokenizer, grpo_dataset, max_steps=5):
    import inspect

    from trl import GRPOConfig, GRPOTrainer

    print("=" * 65)
    print("GRPO TRAINING")
    print("=" * 65)
    print(f"  Steps:             {max_steps}")
    print(f"  Candidates/prompt: {NUM_GENERATIONS}")
    print(f"  Dataset size:      {len(grpo_dataset)} graphs")

    # Print TRL version so it's obvious in the log what we're dealing with.
    try:
        import trl as _trl
        print(f"  TRL version:       {_trl.__version__}")
    except Exception:
        pass

    # Our "wish list" -- the ideal set of GRPO args. Some may not exist in
    # whichever TRL version is installed, so we filter against the actual
    # __init__ signature of GRPOConfig before constructing it.
    #
    # Smaller batch than before: 4 per device with gradient_accumulation_steps=2
    # means 1 group (4 completions) per forward pass, 2 groups per optimizer
    # update. This keeps wall-clock per step sane while still averaging over
    # multiple groups for GRPO variance reduction.
    wish = dict(
        output_dir="grpo_output",
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=NUM_GENERATIONS,
        learning_rate=5e-7,
        beta=0.2,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_steps=1,
        save_steps=max(max_steps, 1),
        bf16=torch.cuda.is_available(),
        fp16=False,
        push_to_hub=False,
        remove_unused_columns=False,
        temperature=0.9,
        max_prompt_length=1024,
        max_completion_length=512,
    )

    # Filter wish list to only params this TRL version supports.
    try:
        sig = inspect.signature(GRPOConfig.__init__)
        supported = set(sig.parameters.keys())
        filtered = {k: v for k, v in wish.items() if k in supported}
        dropped = [k for k in wish if k not in supported]
        if dropped:
            print(f"  Note: TRL version does not support these args, "
                  f"dropping: {dropped}")
    except (ValueError, TypeError) as e:
        print(f"  Could not inspect GRPOConfig signature ({e}); "
              f"passing all args.")
        filtered = wish

    # Sanity: we MUST have max_completion_length to prevent runaway generation.
    if "max_completion_length" not in filtered:
        print("  WARNING: max_completion_length not accepted by this TRL -- "
              "generation will not be capped. Upgrade TRL: pip install -U trl")

    try:
        training_args = GRPOConfig(**filtered)
    except TypeError as e:
        # Last-ditch fallback: strip anything that might still be wrong
        # and build an absolutely minimal config.
        print(f"  GRPOConfig still rejected args ({e}).")
        print(f"  Falling back to minimal config. Training will be slow "
              f"and uncapped -- upgrade TRL.")
        training_args = GRPOConfig(
            output_dir="grpo_output",
            max_steps=max_steps,
            num_generations=NUM_GENERATIONS,
            per_device_train_batch_size=4,
            logging_steps=1,
            bf16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=grpo_dataset,
        reward_funcs=make_reward_fn(),
    )

    # Attach the sanity-check callback
    cb = RewardSanityCallback().build()
    trainer.add_callback(cb)

    print("Starting training...\n")
    start = time.time()
    trainer.train()
    print(f"\nTraining done in {round(time.time() - start, 1)}s")

    save_lora_adapter(model)
    return trainer


# =============================================================================
# SECTION 17: MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Run GRPO training")
    parser.add_argument("--sft_warmup", action="store_true",
                        help="Run a short SFT pass to anchor JSON format "
                             "before GRPO (strongly recommended)")
    parser.add_argument("--max_steps", type=int, default=5,
                        help="GRPO training steps (5=verify, 500=real)")
    parser.add_argument("--sft_examples", type=int, default=40,
                        help="Number of MUTAG graphs used for SFT warmup")
    parser.add_argument("--graph_index", type=int, default=0,
                        help="Which MUTAG graph to use for inference eval")
    parser.add_argument("--eval_graphs", type=int, default=20,
                        help="Number of MUTAG graphs to use for batch eval "
                             "dumps used by make_plots.py (set 0 to skip)")
    args = parser.parse_args()

    print("=" * 65)
    print("LLM MOLECULAR GRAPH GENERATION -- GRPO PIPELINE (FIXED)")
    print("CS594 Reinforcement Learning -- UIC")
    print("=" * 65)
    if args.train:
        mode = f"GRPO training ({args.max_steps} steps)"
        if args.sft_warmup:
            mode += " with SFT warmup"
    else:
        mode = "Inference only"
    print(f"Mode: {mode}\n")

    # --- Load ---
    print("Step 1: Loading model...")
    model, tokenizer = load_model()
    print("Step 2: Applying LoRA...")
    model = apply_lora(model)
    print("Step 3: Loading MUTAG dataset...")
    dataset = load_mutag_dataset()

    # --- Baseline eval (single + batch dump) ---
    print("\nStep 4a: Baseline single-graph eval...")
    baseline_scores = run_single(dataset, model, tokenizer,
                                 graph_index=args.graph_index,
                                 save_viz=True)
    if baseline_scores:
        print(f"\nBaseline single-graph overall: {baseline_scores['overall']}")

    if args.eval_graphs > 0:
        print("\nStep 4b: Baseline batch eval dump (for plots)...")
        run_eval_dump(dataset, model, tokenizer, "Baseline",
                      "grpo_output/eval_baseline.json",
                      num_graphs=args.eval_graphs)

    if not args.train:
        print("\nRun with --train to start GRPO training.")
        print("  Verify:         python pipeline_grpo_fixed.py --train --max_steps 5")
        print("  With SFT warmup: python pipeline_grpo_fixed.py --train --sft_warmup --max_steps 50")
        print("  Full run:       python pipeline_grpo_fixed.py --train --sft_warmup --max_steps 500")
        return

    # --- Optional SFT warmup ---
    if args.sft_warmup:
        print("\nStep 5a: SFT warmup...")
        model = sft_warmup(model, tokenizer, dataset,
                           num_examples=args.sft_examples)

        print("Step 5b: Post-SFT single-graph eval...")
        post_sft_scores = run_single(dataset, model, tokenizer,
                                     graph_index=args.graph_index,
                                     save_viz=False)
        if post_sft_scores:
            print(f"Post-SFT overall: {post_sft_scores['overall']}")

        if args.eval_graphs > 0:
            print("\nStep 5c: Post-SFT batch eval dump (for plots)...")
            run_eval_dump(dataset, model, tokenizer, "SFT only",
                          "grpo_output/eval_sft.json",
                          num_graphs=args.eval_graphs)

    # --- GRPO ---
    print("\nStep 6: Building GRPO dataset...")
    grpo_dataset = build_grpo_dataset(dataset)

    print("Step 7: Running GRPO training...")
    run_grpo_training(model, tokenizer, grpo_dataset,
                      max_steps=args.max_steps)

    # --- Post-training eval ---
    print("\nStep 8a: Post-training single-graph eval...")
    post_scores = run_single(dataset, model, tokenizer,
                             graph_index=args.graph_index,
                             save_viz=True)

    if args.eval_graphs > 0:
        print("\nStep 8b: Post-GRPO batch eval dump (for plots)...")
        run_eval_dump(dataset, model, tokenizer, "SFT + GRPO",
                      "grpo_output/eval_grpo.json",
                      num_graphs=args.eval_graphs)

    if post_scores and baseline_scores:
        print("\n" + "=" * 65)
        print("BEFORE vs AFTER TRAINING (single graph)")
        print("=" * 65)
        print(f"  {'Metric':<28} {'Before':>8} {'After':>8} {'Change':>8}")
        print(f"  {'-' * 54}")
        for m in ["num_nodes", "num_edges", "is_connected",
                  "clustering_coefficient", "diameter",
                  "avg_cycle_length", "overall"]:
            b = baseline_scores.get(m, 0)
            a = post_scores.get(m, 0)
            diff = round(a - b, 3)
            arrow = "up" if diff > 0 else ("down" if diff < 0 else "--")
            print(f"  {m:<28} {b:>8} {a:>8}  {arrow}{abs(diff):>6.3f}")
        print("=" * 65)
        print("\nAdapter saved to: grpo_output/lora_adapter/")
        print("Run `python make_plots.py` to generate report figures.")


if __name__ == "__main__":
    main()