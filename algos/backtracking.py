import networkx as nx

def is_safe(graph, vertex, color, colorArr):
    for neighbor in graph.neighbors(vertex):
        if colorArr.get(neighbor) == color:
            return False
    return True

def select_next_vertex(graph, colorArr):
    max_saturation = -1
    selected_node = None
    for node in graph.nodes:
        if colorArr[node] is None:
            saturated_colors = set(colorArr.get(neighbor) for neighbor in graph.neighbors(node))
            saturation = len([color for color in saturated_colors if color is not None])
            if saturation > max_saturation or (saturation == max_saturation and len(list(graph.neighbors(node))) > len(list(graph.neighbors(selected_node)))):
                max_saturation = saturation
                selected_node = node
    return selected_node

def backtrackDsaturs_helper(graph, m, vertex, colorArr, precolored):
    if vertex is None:
        return True

    for color in range(1, m + 1):
        if vertex in precolored:
            if backtrackDsaturs_helper(graph, m, select_next_vertex(graph, colorArr), colorArr, precolored):
                return True
        elif is_safe(graph, vertex, color, colorArr):
            colorArr[vertex] = color
            if backtrackDsaturs_helper(graph, m, select_next_vertex(graph, colorArr), colorArr, precolored):
                return True
            colorArr[vertex] = None

    return False

def backtrackDsaturs(graph, m, precolored):
    color_map = {node: precolored.get(node, None) for node in graph.nodes}
    if backtrackDsaturs_helper(graph, m, select_next_vertex(graph, color_map), color_map, precolored):
        return color_map
    return []
