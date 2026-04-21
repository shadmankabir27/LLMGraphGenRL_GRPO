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

#   if "_raw_text" in keys:
#     text = pred_json["_raw_text"]
#     if re.search(r'"objects"', text, re.I) and re.search(r'"relationships"', text, re.I):
#         return 1.0
#     return 0.0

#   if "_extracted" in keys:
#     text = pred_json["_extracted"]
#     if re.search(r'"objects"', text, re.I) and re.search(r'"relationships"', text, re.I):
#         return 1.0
#     return 0.0

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
    if isinstance(gt_graph["objects"], str):
        gt_graph["objects"] = ast.literal_eval(gt_graph["objects"])

    if isinstance(gt_graph["relationships"], str):
        gt_graph["relationships"] = ast.literal_eval(gt_graph["relationships"])

    for o in gt_graph["objects"]:
        # print("checking gt object ", o)

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

def soft_match_predicates(pr1, pr2):
  pr1_words = pr1.split()
  pr2_words = pr2.split()
  # filter out empty
  pr1_words = [w for w in pr1_words if w]
  pr2_words = [w for w in pr2_words if w]

  # get common words
  common_words = set(pr1_words) & set(pr2_words)
  if len(common_words) > 0:
    return True
  return False

def hard_recall(pred_graph, gt_graph):
    pred_objs = {o["id"]: o["bbox"] for o in pred_graph.get("objects", [])}
    gt_objs, gt_rels = gt_triplets(gt_graph)

    matched = 0
    total = len(gt_rels)
    if total == 0:
        return 0.0

    for gr in gt_rels:
        ps = get_class(gr["subject"])
        po = get_class(gr["object"])
        pred_match = False
        for pr in pred_graph.get("relationships", []):

            if pr.get("predicate", "") != gr["predicate"] and not (soft_match_predicates(pr.get("predicate", ""), gr["predicate"])):
                continue

            # print("predicate atleast - ", gr['predicate'], " x " , pr.get("predicate", ""))
            # # subject and object
            # print("pred subj ", get_class(pr.get("subject", "")), " gt subj ", ps)
            # print("pred obj ", get_class(pr.get("object", "")), " gt obj ", po)

            if get_class(pr.get("subject", "")) != ps or get_class(pr.get("object", "")) != po:
                continue

            # print("same atleast")
            # if ps in pred_objs and po in pred_objs and ps in gt_objs and po in gt_objs:
            pred_match = True
            break
                # if bbox_iou(pred_objs[ps], gt_objs[ps]) > iou_thr and bbox_iou(pred_objs[po], gt_objs[po]) > iou_thr:
                #     pred_match = True
                #     break
        if pred_match:
            matched += 1

    return matched / total


# hard recall + relax
# hard recall + relax
from difflib import SequenceMatcher

def name_sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def hard_recall_relax(pred_graph, gt_graph, sim_thr=0.5):
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

        # print("\n== for rl ===" , gr)

        best = False
        for pr in pred_graph.get("relationships", []):
            ps = get_class(pr.get("subject", ""))
            po = get_class(pr.get("object", ""))
            pp = pr.get("predicate", "")

            s_sim = name_sim(ps, gs)
            o_sim = name_sim(po, go)

            if soft_match_predicates(pp, gp):
                p_sim = 1.0
            else:
              p_sim = name_sim(pp, gp)



            if s_sim < sim_thr or o_sim < sim_thr or p_sim < sim_thr:
                continue

            # print(f" sim subj {gs} - {ps}", s_sim)
            # print(f"sim obj {go} - {po}", o_sim)
            # print(f"sim pred {gp} - {pp}", p_sim)


            # if pr["subject"] in pred_objs and pr["object"] in pred_objs and gr["subject"] in gt_objs and gr["object"] in gt_objs:
            #     if bbox_iou(pred_objs[pr["subject"]], gt_objs[gr["subject"]]) > iou_thr and bbox_iou(pred_objs[pr["object"]], gt_objs[gr["object"]]) > iou_thr:
            best = True
            break

        if best:
            matched += 1

    return matched / total


# nodes match

def nodes_match_reward(pred_graph, gt_graph):
  pred_objs = [o["id"] for o in pred_graph.get("objects", [])]
  if len(pred_objs) == 0:
    return 0.0

  gt_objs, _ = gt_triplets(gt_graph)
  gt_classes = set([get_class(o_id) for o_id in gt_objs])

  matched = 0
  for p_obj in pred_objs:
    if get_class(p_obj) in gt_classes:
      matched += 1

  return matched / len(pred_objs)


def total_reward(pred_graph, gt_graph):
  r_format = format_reward(pred_graph)
  r_n_match = nodes_match_reward(pred_graph, gt_graph)
  r_hr = hard_recall(pred_graph, gt_graph)
  r_hrr = hard_recall_relax(pred_graph, gt_graph)
  # if r_format == 0.0:
  #   return 0.0

  total = r_format * 2.0 + r_hr * 0.4 + r_hrr * 0.4 + r_n_match * 0.2
  return {
      "total": total,
      "format": r_format,
      "nodes_match": r_n_match,
      "hard_recall": r_hr,
      "hard_recall_relax": r_hrr
  }

