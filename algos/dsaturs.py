import networkx as nx

#starting dsaturs algorithm
def update_saturation(graph, node, colors, saturation):
    #this function should update the saturation degree of the neighbors of the node
    for neighbor in graph.neighbors(node):
        if colors[neighbor] is None and colors[node] is not None:
            saturation[neighbor] += 1


def select_most_saturated(graph, colors, saturation, nodes_by_degree):
    #selects the most saturated uncolored node
    max_saturation = -1
    selected_node = None
    for node in saturation:
        if colors[node] is None and saturation[node] > max_saturation:
            max_saturation = saturation[node]
            selected_node = node
    #tiebreaker by degree if needed
    if max_saturation == -1:
        for node in nodes_by_degree:
            if colors[node] is None:
                return node
    return selected_node

def dsatur(graph, precolored):
    #initialize colors with precolored nodes and saturation degrees
    colors = {node: precolored.get(node, None) for node in graph.nodes}
    saturation = {node: 0 for node in graph.nodes}
    #update saturation for precolored nodes
    for node, color in precolored.items():
        update_saturation(graph, node, colors, saturation)
    #sort nodes by degree
    nodes_by_degree = sorted(graph.nodes, key=lambda x: -graph.degree(x))

    while None in colors.values():
        #select the most saturated uncolored node
        node = select_most_saturated(graph, colors, saturation, nodes_by_degree)
        #if all nodes are colored, break the loop
        if node is None:
            break
        available_colors = set(range(len(graph)))
        #remove colors used by neighbors
        for neighbor in graph.neighbors(node):
            if colors[neighbor] is not None:
                available_colors.discard(colors[neighbor])
        #assign the first available color
        colors[node] = min(available_colors)
        #update saturation degrees
        update_saturation(graph, node, colors, saturation)
    return colors
