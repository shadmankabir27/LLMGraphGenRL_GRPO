"""
=============================================================================
MUTAG Dataset Statistics Analysis
CS594 Reinforcement Learning — UIC
=============================================================================

This script loads all 188 graphs from the MUTAG dataset and computes
comprehensive statistics across the entire dataset. Use this to understand
the "ground truth" distribution that our LLM is trying to match.

Statistics computed per graph, then aggregated (avg + median):
  1. num_nodes           — number of atoms
  2. num_edges           — number of bonds
  3. is_connected        — whether all atoms are reachable
  4. clustering_coeff    — local triangle density
  5. diameter            — longest shortest path
  6. avg_cycle_length    — average length of fundamental cycles

HOW TO RUN:
    python mutag_stats.py
"""

import random
import statistics
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset


# =============================================================================
# SECTION 1: LOAD DATASET
# =============================================================================

def load_mutag():
    dataset = TUDataset(root="data", name="MUTAG")
    print(f"MUTAG loaded: {len(dataset)} graphs, {dataset.num_classes} classes\n")
    return dataset


def pyg_to_edges(data):
    """Convert PyG graph to plain node/edge lists."""
    nodes = list(range(data.num_nodes))
    raw_edges = data.edge_index.t().tolist()
    seen = set()
    edges = []
    for a, b in raw_edges:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            edges.append([key[0], key[1]])
    return nodes, edges


# =============================================================================
# SECTION 2: COMPUTE STATS FOR ONE GRAPH
# =============================================================================

def compute_stats(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_connected = nx.is_connected(G)
    clustering = nx.average_clustering(G)

    # Diameter — only defined for connected graphs
    if is_connected:
        diameter = nx.diameter(G)
    else:
        diameter = None   # will be excluded from diameter stats

    # Average cycle length via minimum cycle basis
    cycle_basis = nx.cycle_basis(G)
    if cycle_basis:
        avg_cycle_length = sum(len(c) for c in cycle_basis) / len(cycle_basis)
    else:
        avg_cycle_length = None   # no cycles at all — excluded from avg

    return {
        "num_nodes":        num_nodes,
        "num_edges":        num_edges,
        "is_connected":     is_connected,
        "clustering":       clustering,
        "diameter":         diameter,
        "avg_cycle_length": avg_cycle_length,
    }


# =============================================================================
# SECTION 3: AGGREGATE HELPERS
# =============================================================================

def avg(values):
    return round(sum(values) / len(values), 4) if values else None

def med(values):
    return round(statistics.median(values), 4) if values else None

def pct(count, total):
    return round(100 * count / total, 1)


# =============================================================================
# SECTION 4: MAIN ANALYSIS
# =============================================================================

def main():
    dataset = load_mutag()
    total = len(dataset)

    # Collect raw stats per graph
    all_nodes        = []
    all_edges        = []
    all_connected    = []
    all_clustering   = []
    all_diameter     = []    # only connected graphs
    all_cycle_length = []    # only graphs with at least one cycle

    disconnected_graphs = []

    print("Processing all 188 graphs...")
    print("(This takes a few seconds — no GPU needed)\n")

    for i in range(total):
        data = dataset[i]
        nodes, edges = pyg_to_edges(data)
        s = compute_stats(nodes, edges)

        all_nodes.append(s["num_nodes"])
        all_edges.append(s["num_edges"])
        all_connected.append(s["is_connected"])
        all_clustering.append(s["clustering"])

        if s["diameter"] is not None:
            all_diameter.append(s["diameter"])
        else:
            disconnected_graphs.append(i)

        if s["avg_cycle_length"] is not None:
            all_cycle_length.append(s["avg_cycle_length"])

    # -------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("MUTAG DATASET STATISTICS (all 188 graphs)")
    print("=" * 60)

    # 1. Nodes
    print(f"\n1. NUMBER OF NODES (atoms per molecule)")
    print(f"   Average:  {avg(all_nodes)}")
    print(f"   Median:   {med(all_nodes)}")
    print(f"   Min:      {min(all_nodes)}")
    print(f"   Max:      {max(all_nodes)}")

    # 2. Edges
    print(f"\n2. NUMBER OF EDGES (bonds per molecule)")
    print(f"   Average:  {avg(all_edges)}")
    print(f"   Median:   {med(all_edges)}")
    print(f"   Min:      {min(all_edges)}")
    print(f"   Max:      {max(all_edges)}")

    # 3. Connectivity
    num_connected    = sum(all_connected)
    num_disconnected = total - num_connected
    print(f"\n3. CONNECTIVITY")
    print(f"   Connected graphs:    {num_connected} / {total} "
          f"({pct(num_connected, total)}%)")
    print(f"   Disconnected graphs: {num_disconnected} / {total} "
          f"({pct(num_disconnected, total)}%)")
    if disconnected_graphs:
        print(f"   Disconnected graph indices: {disconnected_graphs}")

    # 4. Clustering coefficient
    print(f"\n4. CLUSTERING COEFFICIENT")
    print(f"   Average:  {avg(all_clustering)}")
    print(f"   Median:   {med(all_clustering)}")
    print(f"   Min:      {round(min(all_clustering), 4)}")
    print(f"   Max:      {round(max(all_clustering), 4)}")
    zero_clustering = sum(1 for c in all_clustering if c == 0.0)
    print(f"   Graphs with 0.0 clustering: {zero_clustering} / {total} "
          f"({pct(zero_clustering, total)}%)")

    # 5. Diameter
    print(f"\n5. DIAMETER (computed on {len(all_diameter)} connected graphs)")
    print(f"   Average:  {avg(all_diameter)}")
    print(f"   Median:   {med(all_diameter)}")
    print(f"   Min:      {min(all_diameter)}")
    print(f"   Max:      {max(all_diameter)}")

    # 6. Average cycle length
    print(f"\n6. AVERAGE CYCLE LENGTH "
          f"(computed on {len(all_cycle_length)} graphs that have cycles)")
    print(f"   Average:  {avg(all_cycle_length)}")
    print(f"   Median:   {med(all_cycle_length)}")
    print(f"   Min:      {round(min(all_cycle_length), 4)}")
    print(f"   Max:      {round(max(all_cycle_length), 4)}")
    no_cycles = total - len(all_cycle_length)
    print(f"   Graphs with no cycles: {no_cycles} / {total} "
          f"({pct(no_cycles, total)}%)")

    # -------------------------------------------------------------------------
    # SUMMARY TABLE — useful for copy-pasting into your report
    # -------------------------------------------------------------------------
    print(f"\n\n{'=' * 60}")
    print("SUMMARY TABLE (copy into report)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'Average':>10} {'Median':>10}")
    print(f"  {'-' * 52}")
    print(f"  {'num_nodes':<30} {str(avg(all_nodes)):>10} {str(med(all_nodes)):>10}")
    print(f"  {'num_edges':<30} {str(avg(all_edges)):>10} {str(med(all_edges)):>10}")
    print(f"  {'clustering_coefficient':<30} {str(avg(all_clustering)):>10} {str(med(all_clustering)):>10}")
    print(f"  {'diameter (connected only)':<30} {str(avg(all_diameter)):>10} {str(med(all_diameter)):>10}")
    print(f"  {'avg_cycle_length (w/ cycles)':<30} {str(avg(all_cycle_length)):>10} {str(med(all_cycle_length)):>10}")
    connected_str = f"{pct(num_connected, total)}% connected"
    print(f"  {'is_connected':<30} {connected_str:>10}")
    print(f"{'=' * 60}")
    print("\nDone!")

    # -------------------------------------------------------------------------
    # VISUALIZATION — randomly sample 8 graphs and save as a single PNG
    # -------------------------------------------------------------------------
    visualize_sample(dataset, n=8, save_path="mutag_sample_graphs.png")


def visualize_sample(dataset, n=8, save_path="mutag_sample_graphs.png"):
    """
    Randomly samples n graphs from the MUTAG dataset and saves them
    as a single PNG in a 2-row x 4-column grid.

    Each subplot shows:
      - The graph drawn with spring layout
      - Title: graph index + mutagenic label
      - Caption: key stats (nodes, edges, diameter, avg cycle length)
    """
    total = len(dataset)
    indices = random.sample(range(total), n)
    indices.sort()   # sort so they appear in index order left-to-right

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "MUTAG Dataset — 8 Randomly Sampled Graphs",
        fontsize=16,
        fontweight="bold",
        y=1.01
    )

    axes_flat = axes.flatten()

    for plot_idx, graph_idx in enumerate(indices):
        ax = axes_flat[plot_idx]

        data = dataset[graph_idx]
        nodes, edges = pyg_to_edges(data)
        s = compute_stats(nodes, edges)
        label = data.y.item()

        G = s.get("graph_object") if "graph_object" in s else None
        # Rebuild G since compute_stats doesn't return graph_object
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, seed=42)

        node_color = "steelblue" if label == 1 else "mediumpurple"
        nx.draw_networkx(
            G,
            pos=pos,
            ax=ax,
            with_labels=True,
            node_color=node_color,
            node_size=400,
            font_color="white",
            font_size=7,
            edge_color="gray",
            width=1.5
        )

        mutagenic_str = "Mutagenic" if label == 1 else "Non-mutagenic"
        ax.set_title(
            f"Graph #{graph_idx}  ({mutagenic_str})",
            fontsize=10,
            fontweight="bold",
            pad=8
        )
        ax.set_axis_off()

        diam_str = str(s["diameter"]) if s["diameter"] is not None else "N/A"
        cycle_str = (
            str(round(s["avg_cycle_length"], 1))
            if s["avg_cycle_length"] is not None
            else "none"
        )
        caption = (
            f"Nodes: {s['num_nodes']}  |  Edges: {s['num_edges']}\n"
            f"Connected: {s['is_connected']}  |  Diameter: {diam_str}\n"
            f"Clustering: {round(s['clustering'], 3)}  |  "
            f"Avg Cycle Len: {cycle_str}"
        )
        ax.text(
            0.5, -0.04,
            caption,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7.5,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved to: {save_path}")
    print("Blue = mutagenic, Purple = non-mutagenic")


if __name__ == "__main__":
    main()
