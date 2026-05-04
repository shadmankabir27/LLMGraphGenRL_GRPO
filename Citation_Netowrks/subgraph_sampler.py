import random
from typing import List, Set, Tuple

import networkx as nx
import networkx.algorithms.community as nx_comm


def bfs_sample(
    G: nx.Graph,
    seed_node: int,
    target_size: int,
    allowed_nodes: Set[int] = None,
) -> List[int]:
    visited = [seed_node]
    visited_set = {seed_node}
    queue = [seed_node]

    while queue and len(visited) < target_size:
        node = queue.pop(0)
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)
        for nbr in neighbors:
            if nbr in visited_set:
                continue
            if allowed_nodes is not None and nbr not in allowed_nodes:
                continue
            visited.append(nbr)
            visited_set.add(nbr)
            queue.append(nbr)
            if len(visited) >= target_size:
                break

    return visited


def sample_louvain_subgraphs(
    G: nx.Graph,
    num_subgraphs: int = 50,
    target_size: int = 50,
    size_tolerance: int = 15,
    seed: int = 42,
) -> List[Tuple[nx.Graph, int]]:
    rng = random.Random(seed)
    random.seed(seed)

    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    print(f"[sampler] Running Louvain on {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges...")
    communities = nx_comm.louvain_communities(G, seed=seed)
    communities = sorted(communities, key=len, reverse=True)
    sizes_preview = [len(c) for c in communities[:15]]
    print(f"[sampler] Found {len(communities)} communities. "
          f"Top sizes: {sizes_preview}")

    subgraphs: List[Tuple[nx.Graph, int]] = []
    min_size = max(5, target_size - size_tolerance)
    max_size = target_size + size_tolerance

    for comm_idx, community in enumerate(communities):
        if len(subgraphs) >= num_subgraphs:
            break

        comm_list = list(community)
        comm_set = set(comm_list)

        if min_size <= len(comm_list) <= max_size:
            sub = G.subgraph(comm_list).copy()
            if sub.number_of_edges() > 0:
                subgraphs.append((sub, comm_idx))

        elif len(comm_list) > max_size:
            
            num_chunks = max(1, len(comm_list) // target_size)
            num_chunks = min(num_chunks, num_subgraphs - len(subgraphs))
            for _ in range(num_chunks):
                if len(subgraphs) >= num_subgraphs:
                    break
                seed_node = rng.choice(comm_list)
                sampled = bfs_sample(G, seed_node, target_size, allowed_nodes=comm_set)
                if len(sampled) >= min_size:
                    sub = G.subgraph(sampled).copy()
                    if sub.number_of_edges() > 0:
                        subgraphs.append((sub, comm_idx))

    fallback_attempts = 0
    max_fallback_attempts = num_subgraphs * 5
    while len(subgraphs) < num_subgraphs and fallback_attempts < max_fallback_attempts:
        fallback_attempts += 1
        seed_node = rng.choice(list(G.nodes()))
        sampled = bfs_sample(G, seed_node, target_size)
        if len(sampled) >= min_size:
            sub = G.subgraph(sampled).copy()
            if sub.number_of_edges() > 0:
                subgraphs.append((sub, -1))

    print(f"[sampler] Sampled {len(subgraphs)} subgraphs.")
    return subgraphs


def relabel_subgraph(sub: nx.Graph) -> nx.Graph:
    mapping = {old: new for new, old in enumerate(sorted(sub.nodes()))}
    return nx.relabel_nodes(sub, mapping)