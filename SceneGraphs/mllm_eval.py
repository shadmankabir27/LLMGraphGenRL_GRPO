from datasets import load_from_disk, load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import re
import json
from typing import Tuple, Any, Dict
from PIL import Image
import matplotlib.pyplot as plt
from pprint import pprint
import ast

import networkx as nx
import matplotlib.pyplot as plt

from SceneGraphs.mllm_generation import generate_pipeline

def load_sample(ds, index: int = 0):
    """Load a VG sample from a HuggingFace-disk dataset.
    """
    sample = ds[index]
    return {
        "image": sample["image"].convert("RGB"),
        "objects": sample["objects"],
        "relationships": sample["relationships"],
        "prompt_open": sample.get("prompt_open", ""),
        "prompt_close": sample.get("prompt_close", "")
    }

psg_val = load_from_disk("./datasets/psg_test_sg")
print(psg_val)

## sample the psg dataset
sample_psg = load_sample(psg_val, index=1)
pprint(sample_psg)
plt.imshow(sample_psg["image"])
plt.show()

def get_class(obj: str):
  cls = obj.split(".")[0]
  return cls

all_classes = []
all_predicates = []

# plot distribution of objects and predicates
for idx in range(len(psg_val)):
    sample = load_sample(psg_val, index=idx)
    objects = ast.literal_eval(sample['objects'])
    relationships = ast.literal_eval(sample['relationships'])

    all_classes.extend([get_class(obj['id']) for obj in objects])
    all_predicates.extend([rel['predicate'] for rel in relationships])



all_classes = list(set(all_classes))
all_predicates = list(set(all_predicates))

print(all_classes[:5])
print(all_predicates[:5])


print("Total unique classes: ", len(all_classes))
print("Total unique predicates: ", len(all_predicates))

# plot triplets of relationships

def plot_graph(rels):
    # rels: list of {"subject": "...", "predicate": "...", "object": "..."}
    G = nx.DiGraph()
    for rel in rels:
        G.add_edge(rel["subject"], rel["object"], label=rel["predicate"])

    # Layout and figure size
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.8, seed=0)  # k controls spacing

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color="#87CEEB", node_size=1200)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, width=1.5)

    # Node labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Edge labels (predicates)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.5,
        rotate=False,  # easier to read
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    centrality_score = nx.betweenness_centrality(G)
    print("Centrality Score:", centrality_score)

sample = load_sample(psg_val, index=1000)
plot_graph(ast.literal_eval(sample['relationships']))

plt.imshow(sample["image"])
plt.show()


# model_name = "Qwen/Qwen2-VL-2B-Instruct"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("DEVICE = ", model.device)
model.eval()




# ---------- Build messages, inputs ----------

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

scene_graph = generate_pipeline(sample, model, processor, sys_prompt=SYSTEM_PROMPT)

print("PREDICTED SCENE GRAPH")
pprint(scene_graph)





