import argparse
import json
import os
import time

from generate import generate_raw_response, extract_json_object
from verifier import evaluate_generated_graph_json, METRIC_WEIGHTS
from utils import ensure_dir, save_text


def parse_args():
    p = argparse.ArgumentParser(description="Citation graph generation pipeline (server)")

    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to a JSON file produced by build_dataset.py "
                        "(e.g. data/cora_test.json).")

    p.add_argument("--model_name", type=str, default="gpt-4.1-mini",
                   help="OpenAI (e.g. gpt-4.1-mini, gpt-4o-mini) or HF "
                        "(e.g. Qwen/Qwen2.5-7B-Instruct).")

    p.add_argument("--hf_device", type=str, default="auto")
    p.add_argument("--max_new_tokens", type=int, default=8192,
                   help="Max generated tokens for HF models. "
                        "50-node subgraphs with ~150 edges fit in ~3K tokens; "
                        "leave headroom for verbose output.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of subgraphs (debug).")
    p.add_argument("--output_dir", type=str, default="outputs_run")

    return p.parse_args()


def aggregate(per_sample_results):
    n = len(per_sample_results)
    if n == 0:
        return {}

    metric_sums = {m: 0.0 for m in METRIC_WEIGHTS}
    reward_sum = 0.0
    success_count = 0

    for r in per_sample_results:
        if not r["success"]:
            continue
        success_count += 1
        reward_sum += r["reward"]
        for m in METRIC_WEIGHTS:
            metric_sums[m] += r["comparison"][m]["score"]

    return {
        "num_samples": n,
        "successful": success_count,
        "success_rate": success_count / n,
        "avg_reward_over_all": reward_sum / n,
        "avg_reward_over_success": (reward_sum / success_count) if success_count else 0.0,
        "per_metric_avg_score": {
            m: (metric_sums[m] / success_count) if success_count else 0.0
            for m in METRIC_WEIGHTS
        },
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    if args.limit is not None:
        dataset = dataset[:args.limit]

    print("=" * 80)
    print(f"Dataset:        {args.dataset_path}  ({len(dataset)} subgraphs)")
    print(f"Model:          {args.model_name}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"Output:         {args.output_dir}")
    print(f"Tracked metrics: {list(METRIC_WEIGHTS.keys())}")
    print("=" * 80)

    per_sample_results = []
    t0 = time.time()

    for i, row in enumerate(dataset):
        print(f"\n--- [{i + 1}/{len(dataset)}] subgraph id={row['id']} ---")
        target_props = row["target_properties"]
        sample_dir = os.path.join(args.output_dir, f"sample_{row['id']:03d}")
        ensure_dir(sample_dir)

        with open(os.path.join(sample_dir, "target_properties.json"), "w") as f:
            json.dump(target_props, f, indent=2)
        with open(os.path.join(sample_dir, "target_edges.json"), "w") as f:
            json.dump(row["edges"], f)
        try:
            prompt_text, raw_response = generate_raw_response(
                target_properties=target_props,
                model_name=args.model_name,
                temperature=args.temperature,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                hf_device=args.hf_device,
            )
            save_text(os.path.join(sample_dir, "prompt.txt"), prompt_text)
            save_text(os.path.join(sample_dir, "raw_response.txt"), raw_response)
            print(f"  [raw saved] response len = {len(raw_response)} chars; "
                  f"first 200: {raw_response[:200]!r}")
        except Exception as e:
            result = {
                "id": row["id"],
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e), "stage": "generation"},
                "reward": 0.0,
                "comparison": None,
                "generated_properties": None,
            }
            per_sample_results.append(result)
            with open(os.path.join(sample_dir, "verification_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"  GENERATION FAILED: {type(e).__name__}: {e}")
            continue
        try:
            generated_json = extract_json_object(raw_response)
            save_text(os.path.join(sample_dir, "generated_graph.json"), generated_json)
        except Exception as e:
            result = {
                "id": row["id"],
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e), "stage": "json_parse"},
                "reward": 0.0,
                "comparison": None,
                "generated_properties": None,
            }
            per_sample_results.append(result)
            with open(os.path.join(sample_dir, "verification_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"  JSON PARSE FAILED: {type(e).__name__}: {e}")
            print(f"  --> inspect {os.path.join(sample_dir, 'raw_response.txt')}")
            continue

        # ---- step 3: evaluate ----
        result = evaluate_generated_graph_json(
            json_text=generated_json,
            target_properties=target_props,
            output_dir=sample_dir,
        )
        result["id"] = row["id"]
        per_sample_results.append(result)

        with open(os.path.join(sample_dir, "verification_result.json"), "w") as f:
            json.dump(result, f, indent=2)

        if result["success"]:
            print(f"  reward = {result['reward']:.4f}")
        else:
            err = result.get("error", {}) or {}
            print(f"  EVAL FAILED: {err.get('type')}: {str(err.get('message', ''))[:200]}")

    elapsed = time.time() - t0
    print(f"\nFinished {len(per_sample_results)} samples in {elapsed:.1f}s")

    summary = aggregate(per_sample_results)
    summary["model_name"] = args.model_name
    summary["dataset_path"] = args.dataset_path
    summary["max_new_tokens"] = args.max_new_tokens

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    results_path = os.path.join(args.output_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(per_sample_results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved all per-sample results to {results_path}")


if __name__ == "__main__":
    main()
