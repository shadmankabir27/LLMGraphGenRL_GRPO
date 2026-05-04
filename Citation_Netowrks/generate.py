import json
import os
import re
from typing import Tuple

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from huggingface_hub import login
    _hf_token = os.getenv("HF_TOKEN")
    if _hf_token:
        try:
            login(token=_hf_token, add_to_git_credential=False)
        except Exception:
            pass
except ImportError:
    pass

_HF_MODEL = None
_HF_TOKENIZER = None
_HF_MODEL_NAME = None

def build_graph_generation_prompt(target_properties: dict) -> str:
    n = target_properties["num_nodes"]
    return f"""You are a graph generator.

Generate an UNDIRECTED simple graph as a JSON object that approximately matches the target structural properties below. The graph represents a small subgraph of a citation network: papers are nodes, and edges are citation links. Citation subgraphs are typically sparse, with some clustering (papers that cite each other often share citations) and short paths within communities.

Target properties:
- Domain name: {target_properties["domain_name"]}
- Number of nodes: {n}
- Number of edges: {target_properties["num_edges"]}
- Average clustering coefficient: {target_properties["clustering_coefficient"]:.4f}
- Transitivity (global clustering): {target_properties["transitivity"]:.4f}
- Average degree: {target_properties["avg_degree"]:.4f}
- Maximum degree: {target_properties["max_degree"]}
- Number of triangles: {target_properties["num_triangles"]}
- Average shortest path length: {target_properties["avg_shortest_path_length"]:.4f}
- Is connected: {str(target_properties["is_connected"]).lower()}

Output requirements:
1. Output ONLY valid JSON. No markdown fences, no commentary, no explanation.
2. Use exactly this schema:
{{"nodes": [0, 1, 2, ...], "edges": [[0,1], [1,2], ...]}}
3. Undirected, simple graph: no self-loops, no duplicate edges, each edge listed once.
4. Nodes must be integers from 0 to {n - 1}. The "nodes" array must contain EXACTLY {n} entries: 0, 1, ..., {n - 1}. Do NOT include any node id >= {n}.
5. End the JSON with a closing curly brace. The output must be a complete, parseable JSON object.
6. To get the requested clustering and triangle count, include short closed loops, especially triangles. Do not produce a star or chain.

Return only the JSON object.""".strip()

def extract_json_object(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        try:
            json.loads(json_str)
            return json_str
        except Exception:
            pass
    candidate = _attempt_repair_truncated_json(text)
    if candidate is not None:
        return candidate

    raise ValueError("No valid JSON found in model output.")


def _attempt_repair_truncated_json(text: str):
    start = text.find("{")
    if start == -1:
        return None

    s = text[start:]
    depth_curly = 0
    depth_square = 0
    in_string = False
    escape = False
    last_safe = -1
    last_safe_curly = 0
    last_safe_square = 0

    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1
        elif ch == "," and depth_curly >= 1 and depth_square == 1:

            last_safe = i
            last_safe_curly = depth_curly
            last_safe_square = depth_square
    if depth_curly > 0 or depth_square > 0:
        if last_safe == -1:
            return None
        truncated = s[:last_safe]
        truncated = truncated + ("]" * last_safe_square) + ("}" * last_safe_curly)
        try:
            json.loads(truncated)
            return truncated
        except Exception:
            return None

    return None
def _load_hf_model(model_name: str, device_map: str = "auto"):
    global _HF_MODEL, _HF_TOKENIZER, _HF_MODEL_NAME
    if _HF_MODEL is not None and _HF_MODEL_NAME == model_name:
        return _HF_MODEL, _HF_TOKENIZER

    print(f"[generate] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )

    model.eval()
    print(f"[generate] Loaded {model_name} on device(s) {model.device}, dtype {dtype}")

    _HF_MODEL = model
    _HF_TOKENIZER = tokenizer
    _HF_MODEL_NAME = model_name
    return model, tokenizer


def _generate_with_hf(
    prompt_text: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    hf_device: str,
) -> str:
    torch.manual_seed(seed)
    model, tokenizer = _load_hf_model(model_name, hf_device)

    messages = [
        {"role": "system", "content": "You generate graph JSON objects."},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)

def _generate_with_openai(
    prompt_text: str,
    model_name: str,
    temperature: float,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You generate graph JSON objects."},
            {"role": "user", "content": prompt_text},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""

def generate_raw_response(
    target_properties: dict,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.7,
    seed: int = 42,
    max_new_tokens: int = 8192,
    hf_device: str = "auto",
) -> Tuple[str, str]:
    prompt_text = build_graph_generation_prompt(target_properties)

    if "gpt" in model_name.lower():
        raw_response = _generate_with_openai(prompt_text, model_name, temperature)
    else:
        raw_response = _generate_with_hf(
            prompt_text,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            hf_device=hf_device,
        )

    return prompt_text, raw_response
def generate_graph_json_with_llm(
    target_properties: dict,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.7,
    seed: int = 42,
    max_new_tokens: int = 8192,
    hf_device: str = "auto",
) -> Tuple[str, str, str]:
    prompt_text, raw_response = generate_raw_response(
        target_properties=target_properties,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
        hf_device=hf_device,
    )
    print("\n" + "=" * 50)
    print("RAW LLM OUTPUT (first 500 chars):")
    print(raw_response[:500])
    print("=" * 50 + "\n")
    generated_json = extract_json_object(raw_response)
    return generated_json, prompt_text, raw_response
