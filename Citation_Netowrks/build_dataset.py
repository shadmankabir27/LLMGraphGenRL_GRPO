import argparse
import json
import os
import random

from properties import load_graph_from_edgelist, extract_graph_properties_direct
from subgraph_sampler import sample_louvain_subgraphs, relabel_subgraph
from utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Build subgraph dataset (train/test JSON) from a citation network."
    )
    p.add_argument("--input_path", type=str, required=True,
                   help="Edgelist file, e.g. data/cora.edgelist")
    p.add_argument("--graph_name", type=str, default="cora",
                   help="Domain name embedded into each subgraph's properties.")
    p.add_argument("--num_subgraphs", type=int, default=200)
    p.add_argument("--target_size", type=int, default=200)
    p.add_argument("--size_tolerance", type=int, default=15)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print(f"Loading {args.input_path} ...")
    G = load_graph_from_edgelist(args.input_path, directed=False)
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    sampled = sample_louvain_subgraphs(
        G,
        num_subgraphs=args.num_subgraphs,
        target_size=args.target_size,
        size_tolerance=args.size_tolerance,
        seed=args.seed,
    )

    rows = []
    for i, (sub, comm_id) in enumerate(sampled):
        sub_relabeled = relabel_subgraph(sub)
        props = extract_graph_properties_direct(
            sub_relabeled,
            graph_name=f"{args.graph_name}_sub_{i}",
        )
        rows.append({
            "id": i,
            "community_id": comm_id,
            "target_properties": props,
            "edges": [list(e) for e in sub_relabeled.edges()],
        })

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_train = int(len(rows) * args.train_frac)
    train_rows = rows[:n_train]
    test_rows = rows[n_train:]

    train_path = os.path.join(args.output_dir, f"{args.graph_name}_train.json")
    test_path = os.path.join(args.output_dir, f"{args.graph_name}_test.json")

    with open(train_path, "w") as f:
        json.dump(train_rows, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_rows, f, indent=2)

    
    sizes = [r["target_properties"]["num_nodes"] for r in rows]
    edges = [r["target_properties"]["num_edges"] for r in rows]
    clusts = [r["target_properties"]["clustering_coefficient"] for r in rows]
    trans = [r["target_properties"]["transitivity"] for r in rows]
    spls = [r["target_properties"]["avg_shortest_path_length"] for r in rows]
    n_connected = sum(1 for r in rows if r["target_properties"]["is_connected"])

    print(f"\n=== Dataset summary ===")
    print(f"  total subgraphs:   {len(rows)}")
    print(f"  train / test:      {len(train_rows)} / {len(test_rows)}")
    print(f"  nodes:             min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")
    print(f"  edges:             min={min(edges)}, max={max(edges)}, mean={sum(edges)/len(edges):.1f}")
    print(f"  connected:         {n_connected}/{len(rows)}")
    print(f"  avg clustering:    min={min(clusts):.3f}, max={max(clusts):.3f}, mean={sum(clusts)/len(clusts):.3f}")
    print(f"  transitivity:      min={min(trans):.3f}, max={max(trans):.3f}, mean={sum(trans)/len(trans):.3f}")
    print(f"  avg path length:   min={min(spls):.2f}, max={max(spls):.2f}, mean={sum(spls)/len(spls):.2f}")
    print(f"\nWrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()

#python build_dataset.py --input_path data/cora.edgelist --graph_name cora