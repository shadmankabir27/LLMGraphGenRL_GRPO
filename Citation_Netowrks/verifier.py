import json
import os
import traceback
import networkx as nx
from properties import extract_graph_properties_direct
from utils import relative_error

METRIC_WEIGHTS = {
    "num_nodes":                 0.05,
    "num_edges":                 0.10,
    "is_connected":              0.10,
    "clustering_coefficient":    0.15,
    "transitivity":              0.15,
    "avg_degree":                0.15,
    "max_degree":                0.10,
    "num_triangles":             0.10,
    "avg_shortest_path_length":  0.10,
}
assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"
METRIC_SCORING_MODE = {
    "num_nodes":                 "relative",
    "num_edges":                 "relative",
    "is_connected":              "boolean",
    "clustering_coefficient":    "bounded",
    "transitivity":              "bounded",
    "avg_degree":                "relative",
    "max_degree":                "relative",
    "num_triangles":             "zero_safe",
    "avg_shortest_path_length":  "relative",
}


def score_boolean(target, generated) -> float:
    return 1.0 if bool(target) == bool(generated) else 0.0


def score_relative(target, generated) -> float:
    """For unbounded numeric counts. score = max(0, 1 - relative_error)."""
    err = relative_error(target, generated)
    return max(0.0, 1.0 - err)


def score_zero_safe(target, generated) -> float:
    """
    Same as score_relative, but if target == 0:
      - generated == 0  -> 1.0
      - generated  > 0  -> graceful decay 1 / (1 + generated)
    Avoids the division-by-eps blowup when target == 0.
    """
    t = float(target)
    g = float(generated)
    if abs(t) < 1e-12:
        return 1.0 if abs(g) < 1e-12 else 1.0 / (1.0 + abs(g))
    return score_relative(t, g)


def score_bounded(target, generated, scale: float = 1.0) -> float:
    """
    For metrics in a fixed bounded range. score = max(0, 1 - |t - g| / scale).
    For metrics in [0, 1] use scale=1.0.
    """
    err = abs(float(target) - float(generated)) / scale
    return max(0.0, 1.0 - err)


def score_metric(name: str, target, generated) -> float:
    mode = METRIC_SCORING_MODE.get(name, "relative")
    if mode == "boolean":
        return score_boolean(target, generated)
    if mode == "bounded":
        return score_bounded(target, generated, scale=1.0)
    if mode == "zero_safe":
        return score_zero_safe(target, generated)
    return score_relative(target, generated)

def graph_from_json_text(json_text: str) -> nx.Graph:
    obj = json.loads(json_text)

    if not isinstance(obj, dict):
        raise ValueError("Generated output is not a JSON object.")
    if "nodes" not in obj or "edges" not in obj:
        raise ValueError("JSON must contain 'nodes' and 'edges' keys.")

    nodes = obj["nodes"]
    edges = obj["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("'nodes' and 'edges' must both be lists.")

    G = nx.Graph()
    for node in nodes:
        if not isinstance(node, int):
            raise ValueError(f"Node {node!r} is not an int.")
        G.add_node(node)

    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"Each edge must be a [u, v] pair, got {edge!r}.")
        u, v = edge
        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError(f"Edge endpoints must be ints, got {edge!r}.")
        if u == v:
            continue
        G.add_edge(u, v)

    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def compare_properties(target: dict, generated: dict) -> dict:
    comparison = {}
    weighted_total = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        t = target[metric]
        g = generated[metric]
        s = score_metric(metric, t, g)
        comparison[metric] = {
            "target": t,
            "generated": g,
            "score": s,
            "weight": weight,
        }
        weighted_total += weight * s
    comparison["weighted_reward"] = weighted_total
    return comparison


def evaluate_generated_graph_json(
    json_text: str,
    target_properties: dict,
    output_dir: str = None,
):
    result = {
        "success": False,
        "error": None,
        "generated_properties": None,
        "comparison": None,
        "reward": 0.0,
    }
    try:
        G_gen = graph_from_json_text(json_text)

        if G_gen.number_of_nodes() == 0:
            raise ValueError("Generated graph has no nodes.")
        if G_gen.number_of_edges() == 0:
            raise ValueError("Generated graph has no edges.")

        gen_props = extract_graph_properties_direct(
            G_gen, graph_name=target_properties["domain_name"]
        )
        comparison = compare_properties(target_properties, gen_props)

        result["success"] = True
        result["generated_properties"] = gen_props
        result["comparison"] = comparison
        result["reward"] = comparison["weighted_reward"]

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "generated_graph.json"), "w") as f:
                f.write(json_text)
            nx.write_edgelist(
                G_gen,
                os.path.join(output_dir, "generated_graph.edgelist"),
                data=False,
            )

    except Exception as e:
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

    return result
