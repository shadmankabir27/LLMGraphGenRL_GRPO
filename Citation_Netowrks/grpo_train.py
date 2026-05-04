

import argparse
import csv
import json
import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate import build_graph_generation_prompt, extract_json_object
from utils import ensure_dir
from verifier import METRIC_WEIGHTS, evaluate_generated_graph_json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output_dir", type=str, default="outputs_grpo")

    p.add_argument("--limit_train", type=int, default=None,
                   help="Cap number of training subgraphs (smoke test).")
    p.add_argument("--limit_test", type=int, default=10,
                   help="How many test subgraphs to use during periodic eval. "
                        "Final eval still runs on the full test set.")

    p.add_argument("--group_size", type=int, default=4,
                   help="G: candidates generated per prompt.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--beta_kl", type=float, default=0.02)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Sampling temperature during training (higher = more diverse groups).")
    p.add_argument("--eval_temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--eval_every", type=int, default=-1,
                   help="Eval every N global steps; -1 = only at end of each epoch.")
    p.add_argument("--skip_baseline_eval", action="store_true",
                   help="Skip the pre-training baseline eval. Use this if you've "
                        "already measured the baseline elsewhere.")
    p.add_argument("--skip_endepoch_eval", action="store_true",
                   help="Skip the end-of-epoch eval. Use for chained intermediate "
                        "chunks where only the final chunk needs to evaluate.")
    p.add_argument("--resume_from_adapter", type=str, default=None,
                   help="Path to a saved LoRA adapter to resume training from. "
                        "If unset, train from base model.")
    p.add_argument("--train_start", type=int, default=0,
                   help="Index of first training sample to use (inclusive).")
    p.add_argument("--train_end", type=int, default=None,
                   help="Index of last training sample, exclusive. None = end of dataset.")
    p.add_argument("--full_test_eval_at_end", action="store_true",
                   help="At the very end of training, run eval on the FULL test "
                        "set (ignoring --limit_test). Use only on the final chunk.")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)

    return p.parse_args()

def compute_reward_from_text(raw_response: str, target_props: dict) -> float:
    try:
        json_text = extract_json_object(raw_response)
    except Exception:
        return 0.0

    try:
        result = evaluate_generated_graph_json(json_text, target_props)
        if result.get("success"):
            return 0.1 + 0.9 * float(result["reward"])
        return 0.1
    except Exception:
        return 0.1
def compute_completion_logprobs(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    outputs = model(input_ids=full_ids)
    logits = outputs.logits  

    shift_logits = logits[:, :-1, :]  
    shift_labels = full_ids[:, 1:]     

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1) 

   
    return token_logp[0, prompt_len - 1:]


def find_completion_end(completion_ids: torch.Tensor, eos_token_id: int) -> int:
    eos_pos = (completion_ids == eos_token_id).nonzero(as_tuple=False)
    if eos_pos.numel() > 0:
        return int(eos_pos[0].item()) + 1  
    return int(completion_ids.shape[0])

def grpo_step(
    model, tokenizer, prompt_text: str, target_props: dict, args, device,
) -> dict:
    messages = [
        {"role": "system", "content": "You generate graph JSON objects."},
        {"role": "user", "content": prompt_text},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    print(f"    [stage 1/5] generating G={args.group_size} candidates "
          f"(prompt_len={prompt_len}, max_new={args.max_new_tokens})...",
          flush=True)
    t_gen = time.time()
    model.eval()
    with torch.no_grad():
        gen_outputs = model.generate(
            **prompt_inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.group_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    t_score = time.time()
    rewards = []
    full_ids_per_gen = []
    for g in range(args.group_size):
        full_seq = gen_outputs[g]                             
        completion = full_seq[prompt_len:]
        actual_compl_len = find_completion_end(completion, tokenizer.eos_token_id)
        actual_full_len = prompt_len + actual_compl_len
        full_ids_g = full_seq[:actual_full_len].unsqueeze(0)  

        raw_response = tokenizer.decode(
            full_seq[prompt_len:prompt_len + actual_compl_len],
            skip_special_tokens=True,
        )
        reward = compute_reward_from_text(raw_response, target_props)
        rewards.append(reward)
        full_ids_per_gen.append(full_ids_g)
    print(f"    [stage 2/5] scored in {time.time()-t_score:.1f}s, "
          f"rewards={[f'{r:.3f}' for r in rewards]}",
          flush=True)
    del gen_outputs
    torch.cuda.empty_cache()

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    if rewards_t.std() > 1e-6:
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    else:
        advantages = rewards_t - rewards_t.mean()
    advantages = advantages.detach()
          flush=True)
    t_lp = time.time()
    model.train()

    pol_logp_means = []
    kl_means = []

    for g in range(args.group_size):
        full_ids = full_ids_per_gen[g].to(device)
        pol_per_token = compute_completion_logprobs(model, full_ids, prompt_len)
        with torch.no_grad():
            with model.disable_adapter():
                ref_per_token = compute_completion_logprobs(model, full_ids, prompt_len)

        ref_per_token = ref_per_token.detach()
        log_ratio = pol_per_token - ref_per_token
        kl_per_token = torch.exp(-log_ratio) + log_ratio - 1.0
        pol_logp_means.append(pol_per_token.mean())
        kl_means.append(kl_per_token.mean())
        del pol_per_token, ref_per_token, log_ratio, kl_per_token, full_ids

    pol_logp_means = torch.stack(pol_logp_means)  
    kl_means = torch.stack(kl_means)            

    policy_loss = -(advantages * pol_logp_means).mean()
    kl_loss = args.beta_kl * kl_means.mean()
    loss = policy_loss + kl_loss

    return {
        "loss": loss,
        "policy_loss": float(policy_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "rewards": rewards,
        "reward_mean": float(rewards_t.mean().item()),
        "reward_std": float(rewards_t.std().item()),
        "reward_max": float(rewards_t.max().item()),
        "reward_min": float(rewards_t.min().item()),
    }


def evaluate_on_test(model, tokenizer, test_data, args, device, max_samples: int) -> dict:
    model.eval()
    test_subset = test_data[:max_samples] if max_samples is not None else test_data

    rewards = []
    per_metric = {m: [] for m in METRIC_WEIGHTS}
    success_count = 0

    for i, row in enumerate(test_subset):
        t_one = time.time()
        target_props = row["target_properties"]
        prompt_text = build_graph_generation_prompt(target_props)
        messages = [
            {"role": "system", "content": "You generate graph JSON objects."},
            {"role": "user", "content": prompt_text},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.eval_temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = gen[0][inputs["input_ids"].shape[1]:]
        raw_response = tokenizer.decode(completion, skip_special_tokens=True)

        sample_reward = 0.0
        try:
            json_text = extract_json_object(raw_response)
            result = evaluate_generated_graph_json(json_text, target_props)
            if result.get("success"):
                sample_reward = float(result["reward"])
                rewards.append(sample_reward)
                success_count += 1
                for m in METRIC_WEIGHTS:
                    per_metric[m].append(result["comparison"][m]["score"])
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

       
        del gen, inputs
        torch.cuda.empty_cache()

        print(f"    [eval {i+1}/{len(test_subset)}] "
              f"reward={sample_reward:.3f}  ({time.time()-t_one:.1f}s)",
              flush=True)

    n = max(1, len(test_subset))
    return {
        "n_samples": len(test_subset),
        "success_rate": success_count / n,
        "avg_reward": sum(rewards) / n,
        "per_metric_avg": {
            m: (sum(v) / len(v) if v else 0.0) for m, v in per_metric.items()
        },
    }

def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(args.train_path) as f:
        train_data = json.load(f)
    with open(args.test_path) as f:
        test_data = json.load(f)

    if args.limit_train is not None:
        train_data = train_data[:args.limit_train]
    train_end = args.train_end if args.train_end is not None else len(train_data)
    train_data = train_data[args.train_start:train_end]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtype, device_map="auto",
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype, device_map="auto",
        )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        base_model.config.use_cache = False

    if args.resume_from_adapter:
        from peft import PeftModel
        print(f"Loading existing LoRA adapter from {args.resume_from_adapter} ...")
        model = PeftModel.from_pretrained(
            base_model,
            args.resume_from_adapter,
            is_trainable=True,  
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()

    device = next(model.parameters()).device
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    csv_path = os.path.join(args.output_dir, "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "step_in_epoch", "global_step",
            "reward_min", "reward_mean", "reward_max", "reward_std",
            "loss", "policy_loss", "kl_loss", "elapsed_sec",
        ])

    eval_log = [{"step": 0, "phase": "baseline", "summary": baseline}]

    torch.manual_seed(args.seed)
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        for step, row in enumerate(train_data):
            target_props = row["target_properties"]
            prompt_text = build_graph_generation_prompt(target_props)

            stats = grpo_step(model, tokenizer, prompt_text, target_props, args, device)

            optimizer.zero_grad()
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()

            global_step += 1
            elapsed = time.time() - t0
            print(
                f"[ep {epoch + 1} step {step + 1}/{len(train_data)}] "
                f"reward={stats['reward_mean']:.3f}±{stats['reward_std']:.3f} "
                f"max={stats['reward_max']:.3f} "
                f"loss={float(stats['loss'].detach().item()):.4f} "
                f"pol={stats['policy_loss']:.4f} kl={stats['kl_loss']:.4f} "
                f"elapsed={elapsed/60:.1f}m"
            )

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch + 1, step + 1, global_step,
                    f"{stats['reward_min']:.4f}",
                    f"{stats['reward_mean']:.4f}",
                    f"{stats['reward_max']:.4f}",
                    f"{stats['reward_std']:.4f}",
                    f"{float(stats['loss'].detach().item()):.4f}",
                    f"{stats['policy_loss']:.4f}",
                    f"{stats['kl_loss']:.4f}",
                    f"{elapsed:.1f}",
                ])

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                summ = evaluate_on_test(
                    model, tokenizer, test_data, args, device, args.limit_test
                )
                eval_log.append(
                    {"step": global_step, "phase": "mid", "summary": summ}
                )
        if args.skip_endepoch_eval:
            print(f"\n[end of epoch {epoch + 1}] (eval skipped via "
                  f"--skip_endepoch_eval)")
        else:
            summ = evaluate_on_test(
                model, tokenizer, test_data, args, device, args.limit_test
            )
            print(f"\n[end of epoch {epoch + 1}] avg_reward={summ['avg_reward']:.4f} "
                  f"success={summ['success_rate']:.2f}")
            eval_log.append({
                "step": global_step, "phase": f"end_epoch_{epoch + 1}",
                "summary": summ,
            })

        ckpt_dir = os.path.join(args.output_dir, f"adapter_epoch_{epoch + 1}")
        ensure_dir(ckpt_dir)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved adapters to {ckpt_dir}")
    final_dir = os.path.join(args.output_dir, "adapter_final")
    ensure_dir(final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if args.full_test_eval_at_end:
        print("\n=== Final eval on FULL test set ===")
        final_summ = evaluate_on_test(
            model, tokenizer, test_data, args, device, len(test_data)
        )
        eval_log.append({"step": global_step, "phase": "final_full",
                         "summary": final_summ})
        print(json.dumps(final_summ, indent=2))
    else:
        print("\n=== Skipping final full-test eval (--full_test_eval_at_end "
              "not set). Adapter saved; run evaluate_adapter.py separately. ===")

    with open(os.path.join(args.output_dir, "eval_log.json"), "w") as f:
        json.dump(eval_log, f, indent=2)

    print(f"\nDone. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
