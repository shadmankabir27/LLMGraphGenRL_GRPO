import networkx as nx


def load_graph_from_edgelist(path: str, directed: bool = False) -> nx.Graph:
    if directed:
        G = nx.read_edgelist(path, comments="#", nodetype=int, create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(path, comments="#", nodetype=int, create_using=nx.Graph())
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def extract_graph_properties_direct(G: nx.Graph, graph_name: str = "unknown") -> dict:
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {
            "domain_name": graph_name,
            "num_nodes": 0,
            "num_edges": 0,
            "is_connected": False,
            "clustering_coefficient": 0.0,
            "transitivity": 0.0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "num_triangles": 0,
            "avg_shortest_path_length": 0.0,
        }

    is_connected = nx.is_connected(G)
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / num_nodes
    max_degree = max(degrees.values()) if degrees else 0

    num_triangles = sum(nx.triangles(G).values()) // 3

    # Avg shortest path length: on full graph if connected, else on largest CC.
    if is_connected:
        avg_spl = nx.average_shortest_path_length(G)
    else:
        components = list(nx.connected_components(G))
        if components:
            largest_cc = max(components, key=len)
            sub = G.subgraph(largest_cc)
            avg_spl = nx.average_shortest_path_length(sub) if sub.number_of_nodes() > 1 else 0.0
        else:
            avg_spl = 0.0

    return {
        "domain_name": graph_name,
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "is_connected": bool(is_connected),
        "clustering_coefficient": float(avg_clustering),
        "transitivity": float(transitivity),
        "avg_degree": float(avg_degree),
        "max_degree": int(max_degree),
        "num_triangles": int(num_triangles),
        "avg_shortest_path_length": float(avg_spl),
    }
