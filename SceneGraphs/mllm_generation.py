from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re
import json
from typing import Any, Dict


def build_messages(image, prompt: str, system_prompt=None):
    """Build the chat-style message list the Qwen2-VL processor expects.

    This mirrors the repository: the user message contains an image entry and a text entry.
    """

    sys_prompt = system_prompt if system_prompt else "You are a helpful assistant"

    return [
        {
            "role": "system",
            "content": sys_prompt
        },
         {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

 
# Building inputs


def build_inputs(processor: AutoProcessor, messages, device: torch.device):
    """Convert chat messages + images into model-ready tensors.

    Steps:
    1. Convert messages into a single chat text via `apply_chat_template`.
    2. Extract vision tensors via `process_vision_info`.
    3. Call `processor(...)` with text, images, videos to get tensors.
    """
    # Apply chat template to turn the messages into the expected prompt string
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # process_vision_info (from qwen_vl_utils) turns the messages image(s) into tensors
    image_inputs, video_inputs = process_vision_info(messages)

    # print("IMAGE INPUTS")
    # print(image_inputs)

    # Build model inputs (text + image tensors)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, text

 
# Generation

def extract_answer_content(text: str) -> str:
    """Extract JSON content from model output: first try <answer>...</answer>, then code fences."""
    # clean up simple markdown fences
    text = text.replace("```", " ")
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0).strip() if m else text


def generate_and_parse(model, processor: AutoProcessor, inputs: Dict[str, torch.Tensor], gen_args: Dict) -> Any:
    """Run generation and parse the scene-graph JSON from the generated text.

    The repository trims prompt tokens and decodes only the generated suffix. We follow
    the same approach: for each batch element, remove the input prompt length.
    """
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **gen_args
            # max_new_tokens=config.get('max_new_tokens', 2048),
            # temperature=config.get('temprature', 0.01),
            # top_k=config.get('top_k', 1),
            # top_p=config.get('top_p', 0.001),
            # repetition_penalty=config.get('repetition_penalty', 1.0),
            # do_sample=config.get('do_sample', False),
        )

    # Trim prompt tokens: generated_ids is [batch, seq], inputs['input_ids'] is the prompt
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        # Fallback: decode whole generated sequence
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = decoded[0]
    else:
        # Trim the prompt portion from each sequence
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Extract JSON-like answer and parse
    json_text = extract_answer_content(text)
    try:
        return json.loads(json_text)
    except Exception:
        # if parsing fails, return raw text and the extracted fragment for debugging
        return {"_raw_text": text, "_extracted": json_text}


# make function to generate a scene graph given a sample and model
def generate_pipeline(sample, model, processor, config, sys_prompt=None):
  prompt = sample["prompt_close"]

  messages = build_messages(sample["image"], prompt, system_prompt=sys_prompt)
  inputs, prompt_text = build_inputs(processor, messages, model.device)

  scene_graph = generate_and_parse(model, processor, inputs, config)
  return scene_graph




