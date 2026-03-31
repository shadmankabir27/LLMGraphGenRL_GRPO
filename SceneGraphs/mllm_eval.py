import re
import json
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt
from pprint import pprint
import ast

import networkx as nx
import matplotlib.pyplot as plt


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

def get_class(obj: str):
  cls = obj.split(".")[0]
  return cls

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


### Reward Functions

# format reward
def format_reward(pred_json):
  # get keys of json
  keys = list(pred_json.keys())

  if ("objects" in keys) and ("relationships" in keys):
    return 1.0

  if "_raw_text" in keys:
    text = pred_json["_raw_text"]
    if re.search(r'"objects"', text, re.I) and re.search(r'"relationships"', text, re.I):
        return 1.0
    return 0.0

  if "_extracted" in keys:
    text = pred_json["_extracted"]
    if re.search(r'"objects"', text, re.I) and re.search(r'"relationships"', text, re.I):
        return 1.0
    return 0.0

  return 0.0

# hard recall
def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def gt_triplets(gt_graph):
    # gt_graph["objects"] and gt_graph["relationships"]
    id_to_box = {}
    for o in gt_graph["objects"]:
        oid = o["id"] if isinstance(o, dict) else o[0]
        bbox = o["bbox"] if isinstance(o, dict) else o[1]
        id_to_box[oid] = bbox

    trips = []
    for r in gt_graph["relationships"]:
        trips.append({
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
        })
    return id_to_box, trips

def hard_recall(pred_graph, gt_graph, iou_thr=0.5):
    pred_objs = {o["id"]: o["bbox"] for o in pred_graph.get("objects", [])}
    gt_objs, gt_rels = gt_triplets(gt_graph)

    matched = 0
    total = len(gt_rels)
    if total == 0:
        return 0.0

    for gr in gt_rels:
        ps = gr["subject"]
        po = gr["object"]
        pred_match = False
        for pr in pred_graph.get("relationships", []):
            if pr["predicate"] != gr["predicate"]:
                continue
            if pr["subject"] != ps or pr["object"] != po:
                continue

            print("same atleast")
            if ps in pred_objs and po in pred_objs and ps in gt_objs and po in gt_objs:
                if bbox_iou(pred_objs[ps], gt_objs[ps]) > iou_thr and bbox_iou(pred_objs[po], gt_objs[po]) > iou_thr:
                    pred_match = True
                    break
        if pred_match:
            matched += 1

    return matched / total


# hard recall + relax
from difflib import SequenceMatcher

def name_sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def hard_recall_relax(pred_graph, gt_graph, iou_thr=0.5, sim_thr=0.8):
    pred_objs = {o["id"]: o["bbox"] for o in pred_graph.get("objects", [])}
    gt_objs, gt_rels = gt_triplets(gt_graph)

    matched = 0
    total = len(gt_rels)
    if total == 0:
        return 0.0

    for gr in gt_rels:
        gs = get_class(gr["subject"])
        go = get_class(gr["object"])
        gp = gr["predicate"]

        best = False
        for pr in pred_graph.get("relationships", []):
            ps = get_class(pr["subject"])
            po = get_class(pr["object"])
            pp = pr["predicate"]

            if name_sim(ps, gs) < sim_thr or name_sim(po, go) < sim_thr or name_sim(pp, gp) < sim_thr:
                continue

            if pr["subject"] in pred_objs and pr["object"] in pred_objs and gr["subject"] in gt_objs and gr["object"] in gt_objs:
                if bbox_iou(pred_objs[pr["subject"]], gt_objs[gr["subject"]]) > iou_thr and bbox_iou(pred_objs[pr["object"]], gt_objs[gr["object"]]) > iou_thr:
                    best = True
                    break

        if best:
            matched += 1

    return matched / total


def total_reward(pred_graph, gt_graph):
  r_format = format_reward(pred_graph)
  r_hr = hard_recall(pred_graph, gt_graph)
  r_hrr = hard_recall_relax(pred_graph, gt_graph)
  if r_format == 0.0:
    return 0.0

  return r_format * 2.0 + r_hr * 0.4 + r_hrr * 0.4

