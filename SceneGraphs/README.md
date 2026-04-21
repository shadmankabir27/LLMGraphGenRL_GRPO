**Overview**

This README explains how scene-graph generation works using the Qwen multi-modal LLM (Qwen MLLM), how inputs and outputs are represented, the reward signals used by the GRPO trainer, and how GRPO training is configured in this repo.

### Download dataset in datasets folder
Link - https://drive.google.com/drive/folders/1Xy7a_CR2fY7KuOWaqmNZnO5DI_Fb7Wb1?usp=sharing

**Generation (Qwen MLLM)**

- **Pipeline**: We use the Qwen visual-language model via an `AutoProcessor` and the helper functions in `mllm_generation.py` to create model-ready inputs and to decode outputs.
- **Message format**: The code builds chat-style messages with a system message and a user message containing both an image and a text prompt (see `build_messages`). The user message contains an image entry and a text entry.
- **Input construction**: `build_inputs(processor, messages, device)` applies Qwen's chat template to form the text prompt, extracts vision tensors via `qwen_vl_utils.process_vision_info`, and calls the processor to get tensors moved to `device`.
- **Generation**: Model generation uses `model.generate(...)`. The repo trims prompt tokens and decodes only the generated suffix (see `generate_and_parse` and `generate_pipeline`).

**Inputs & Outputs**

- **Inputs**:
  - **Image**: PIL image(s) embedded into the chat `messages` and processed into tensors by `process_vision_info`.
  - **Prompt**: a text prompt (the repo uses a `PROMPT_TEMPLATE`) describing the requested scene-graph format and constraints.
  - **Batch**: during GRPO training the data collator returns a batch with keys `prompt`, `image_data`, and `solution`.

- **Outputs**:
  - The model produces text. We expect the assistant to emit a JSON-like answer wrapped in `<answer>...</answer>` tags (or at least an inline JSON object). The `extract_answer_content` helper extracts the fragment.
  - The expected JSON structure:
    - `objects`: list of `{"id": "name.number", "bbox": [x1, y1, x2, y2]}`
    - `relationships`: list of `{"subject": "name.number", "predicate": "relation", "object": "name.number"}`
  - When parsing fails, the generation helpers return a dict with `_raw_text` and `_extracted` to aid debugging.

See `mllm_generation.py` for the implementation: [mllm_generation.py](SceneGraphs/mllm_generation.py)

**Evaluation / Rewards for GRPO**

The evaluation and reward code lives in `mllm_eval.py`. Key points:

- **Format check reward (`format_reward`)**: returns 1.0 when the parsed output includes both top-level `objects` and `relationships`. If format fails, reward is 0 (early filter).

- **Hard recall (`hard_recall`)**:
  - Matches ground-truth triplets (subject,predicate,object) exactly by id and predicate.
  - For a matched relation, both subject and object bboxes must exceed an IoU threshold (default 0.5).
  - Returns matched / total_gt_relations.

- **Relaxed recall (`hard_recall_relax`)**:
  - Uses name similarity (SequenceMatcher) to match object classes (allows e.g., plural/synonym variations) and predicate similarity.
  - Also requires bbox IoU > threshold for matched pairs.

- **Total reward (`total_reward`)**: computed as

  - format contribution: multiplied by 2.0
  - hard recall contribution: multiplied by 0.4
  - relaxed recall contribution: multiplied by 0.4

  So:

  total_reward = 2.0 * format + 0.4 * hard_recall + 0.4 * hard_recall_relax

  (See `mllm_eval.py` for exact implementation and helper utilities.)

Link: [mllm_eval.py](SceneGraphs/mllm_eval.py)

**GRPO Training Setup**

This repo implements a GRPO-style trainer adapted for the Qwen visual-LM. Main components:

- **Config**: `GRPOConfigVL` (extends `TrainingArguments`) holds GRPO-specific hyperparameters such as
  - `num_generations` (G), `max_completion_length`, `beta`, `temperature`, and `top_p`.

- **Custom Trainer (`GRPOTrainerVL`)**:
  - `generate_completions`: for each example it builds messages, calls `build_inputs`, runs `model.generate` multiple times (G samples), and collects prompt ids, completion ids, and decoded texts.
  - `compute_logprobs`: re-runs the model over concatenated prompt+completion to compute token-level log-probabilities for the completion tokens (used to compute sequence logprob).
  - `compute_rewards`: expands ground-truth solutions to match generated samples and calls the reward function (`grpo_reward`) to obtain scalar rewards per generation.
  - `compute_loss`: forms groups of G generations, computes advantages (reward minus group mean), computes sequence logprobs, and returns the GRPO loss: -mean(advantages * logprobs).

- **Rollout utilities**: A `qwen_vl_rollout` helper pads and returns tensors for prompt ids, completion ids, and per-token logprobs — useful for rollout-based components or diagnostics. A `QwenVLGRPOTrainer` subclass also exists that stores `self._current_inputs` to let rollout functions access the raw batch.

- **Data collator**: `grpo_data_collator(features)` returns a batch dict with keys `prompt`, `image_data`, and `solution` used by the trainer.

- **Model & LoRA**: the notebook shows loading `Qwen/Qwen2.5-VL-3B-Instruct` via `Qwen2_5_VLForConditionalGeneration` and optionally wrapping the model with PEFT/LoRA using the provided `LoraConfig` for parameter-efficient fine-tuning.

- **Training flow** (notebook `MLLM_Train_GRPO.ipynb`):
  1. Build dataset → each row contains `prompt`, `image_data`, and `solution` (GT objects/relationships).
  2. Instantiate `GRPOConfigVL` with small `per_device_train_batch_size` and `num_generations` for initial experiments.
  3. Create `GRPOTrainerVL` with model, `processing_class` (the processor), `reward_funcs` pointing to `grpo_reward` (which wraps `total_reward`), and `data_collator`.
  4. Call `trainer.train()` to run GRPO optimization.

See the training notebook for runnable examples and parameters: [MLLM_Train_GRPO.ipynb](SceneGraphs/MLLM_Train_GRPO.ipynb)

**Files to Inspect**

- `mllm_generation.py` — input, generation, and parsing helpers. ([mllm_generation.py](SceneGraphs/mllm_generation.py))
- `mllm_eval.py` — reward functions and evaluation helpers. ([mllm_eval.py](SceneGraphs/mllm_eval.py))
- `MLLM_Train_GRPO.ipynb` — notebook demonstrating dataset prep, model loading, GRPO config, and training. ([MLLM_Train_GRPO.ipynb](SceneGraphs/MLLM_Train_GRPO.ipynb))

**Quick notes & tips**

- When adding prompts, keep the output constraints strict (JSON + `<answer>` tag) to improve format parsing and the `format_reward`.
- For debugging parse failures, inspect `_raw_text` and `_extracted` returned by the generation helpers.
- Start GRPO with small `num_generations` (e.g., 2) and small batch sizes to validate the pipeline before scaling.

If you'd like, I can: run a quick lint of these files, update the notebook with example output, or add example JSON outputs to this README.
