import networkx as nx

def subgraph(G: nx.DiGraph, 
             nodes: list, 
             root: str = 'CL:0000000')-> nx.DiGraph:
    """Constructs a subgraph from a larger graph based on a provided list of leaf nodes

    Considers cell types as the leaf nodes of the subgraph when
    finding the most common ancestor (root node)

    Args:
        G (nx.DiGraph): "Master" directed acylic graph containing all 
            possible cell ontology relationships
        nodes (list): cell ontologies of interest - leaf nodes of the graph 
        root (str): root node of the graph

    Returns:
        subgraph (nx.DiGraph): subgraph containing cell ontology relationships
            for only the nodes provided
    """
    if root in nodes:
        raise Exception("Error. Root node cannot be a target node")
    
    node_unions = []
    for node in nodes:
        # For each node in the list, get all of their paths from the root
        node_paths = [set(path) for path in nx.all_simple_paths(G, source=root, target=node)]
        
        if len(node_paths) == 0:
            raise Exception("No paths found. Please enter valid target nodes")
        
        # Then, take the union of those paths to return all possible nodes visited. Store this result in node_unions
        node_union = set.union(*node_paths)
        # Then, add this to a list (node_unions) containing each node's union 
        node_unions.append(node_union)

    # Find a common path by taking the intersection of all unions
    union_inter = set.intersection(*node_unions)

    node_path_lengths = {}
    # Find the distance from each node in union_inter to the root
    for node in union_inter:
        length = nx.shortest_path_length(G, source=root, target=node)
        node_path_lengths[node] = length

    # Get node(s) with the largest path length. This is the lowest common ancestor of [nodes]
    max_value = max(node_path_lengths.values())
    all_LCA = [k for k,v in node_path_lengths.items() if v == max_value]
    
    # Reconstruct a subgraph (DiGraph) from LCA to each node by finding the shortest path
    nodes_union_sps = []
    # for each LCA, reconstruct their graph
    for LCA in all_LCA:
        for node in nodes:
            node_paths = [set(path) for path in nx.all_simple_paths(G, source=LCA, target=node)]
            
            # If target node happens to be a LCA, the node path will be empty
            # skip it
            if len(node_paths) == 0:
                continue

            node_union = set.union(*node_paths)

            # instead, take ALL the paths, and then union of all of those paths
            nodes_union_sps.append(node_union)

    # Take the union of these paths (from each LCA) to return all nodes in the subgraph
    subgraph_nodes = set.union(*nodes_union_sps)
   
    # Create a subgraph
    subgraph = G.subgraph(subgraph_nodes)

    return subgraph