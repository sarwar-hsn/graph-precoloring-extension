import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def greedy_coloring_with_precoloring(G, order, precolored):
    color_map = precolored.copy()  # Start with the precolored nodes
    for node in order:
        if node not in color_map:  # Skip if the node is already precolored
            used_colors = set(color_map.get(neighbour) for neighbour in G.neighbors(node) if neighbour in color_map)
            available_colors = set(range(len(G)))  # A set of all possible colors
            available_colors -= used_colors  # Remove the colors used by neighbors
            color_map[node] = min(available_colors)  # Assign the smallest available color
    return color_map



def greedy(G,precolored):
    nodes_ordered = [node for node in G.nodes]
    return greedy_coloring_with_precoloring(G,nodes_ordered,precolored)
    

if __name__ == '__main__':
    # Create a graph
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'),('A', 'C'), ('A', 'D'),
        ('B', 'D'), ('B', 'E'), 
        ('C', 'D'),('C', 'G'),('C', 'F'), 
        ('D', 'G'),('D', 'H'),('D', 'E'),
        ('E', 'I'),('E','H'),
        ('F', 'G'),('F', 'J'),
        ('G', 'H'),('G', 'J'),
        ('H', 'I'),('H', 'J'),
        ('I', 'J'),
    ])

    # Get the coloring
    
    precolored = {}  # Example pre-coloring
    color_map = greedy(G, precolored)
    print(color_map)

    # Normalize colors for colormap
    max_color = max(color_map.values(), default=1)  # Prevent division by zero
    node_colors = [color_map.get(node, 0) / max_color for node in G.nodes]

    # Prepare pre-colored node colors for the "before coloring" graph
    precolored_node_colors = ['lightgray' if node not in precolored else plt.cm.tab20(precolored[node] / max_color) for node in G.nodes]

    # Set up the plot for two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.3)

    # Draw the graph before coloring with pre-colored nodes
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color=precolored_node_colors, node_size=500, ax=axes[0])
    axes[0].set_title("Graph Before Coloring")

    # Draw the graph after coloring using a colormap
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, ax=axes[1], cmap=plt.cm.tab20)
    axes[1].set_title("Graph After Greedy Coloring")

    plt.show()



  





        
        