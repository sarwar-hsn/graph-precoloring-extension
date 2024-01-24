import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import random
import math
import time
import ast
from algos import *



def process_uploaded_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        edge_list = ast.literal_eval(content)
        if isinstance(edge_list, list) and all(isinstance(edge, tuple) for edge in edge_list):
            return edge_list
        else:
            return None
    except Exception as e:
        return None

def random_precolor(G,n):
    """Takes a graph and n. n is the number of random node to be precolored"""
    precolored_nodes = random.sample(G.nodes(), n)
    precolored = {node: i for i, node in enumerate(precolored_nodes)}
    return precolored

def generate_random_graph(n,p):
    """This will generate random graph with n being 
    number of nodes and p being probability of edges"""
    G = nx.erdos_renyi_graph(n, p)
    return G

def plot_result(G, precolored, color_map, algorithm: str):
    if len(color_map) == 0:
        return None, None  # Return None for both figures if there's no color map

    # Normalization of color
    max_color = max(color_map.values(), default=1)

    # Prepare node colors
    node_colors = [color_map.get(node, 0) / max_color for node in G.nodes]
    precolored_node_colors = ['lightgray' if node not in precolored else plt.cm.tab20(precolored[node] / max_color) for node in G.nodes]

    # Positions for all nodes
    pos = nx.spring_layout(G)

    # Create first figure: Graph before coloring
    fig_before = plt.figure(figsize=(6, 7))
    nx.draw(G, pos, with_labels=True, node_color=precolored_node_colors, node_size=500)
    plt.title("Graph Before Coloring")

    # Create second figure: Graph after coloring
    fig_after = plt.figure(figsize=(6, 7))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, cmap=plt.cm.tab20)
    plt.title(f"Graph After {algorithm} Coloring")

    return fig_before, fig_after


def plot_sudoku_board(precolor, size, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    colors = list(mcolors.TABLEAU_COLORS)  # Using tableau colors
    num_colors = len(colors)
    for i in range(size):
        for j in range(size):
            cell_value = precolor.get(i * size + j)
            if cell_value is not None:
                color = colors[(cell_value - 1) % num_colors]
            else:
                color = 'white'
            rect = patches.Rectangle((j, size - 1 - i), 1, 1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            if cell_value is not None:
                ax.text(j + 0.5, size - 1 - i + 0.5, str(cell_value), 
                        va='center', ha='center', color='black' if color != 'black' else 'white')
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def find_subgrid_dimensions(n):
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i
        
def create_sudoku_graph(size):
    subgrid_row_size, subgrid_col_size = find_subgrid_dimensions(size)
    G = nx.Graph()
    for i in range(size):
        for j in range(size):
            node = i * size + j
            G.add_node(node)

            # Add edges for same row and column
            for k in range(size):
                if k != j:
                    G.add_edge(node, i * size + k)
                if k != i:
                    G.add_edge(node, k * size + j)

            # Add edges for same subgrid
            start_row, start_col = subgrid_row_size * (i // subgrid_row_size), subgrid_col_size * (j // subgrid_col_size)
            for r in range(start_row, start_row + subgrid_row_size):
                for c in range(start_col, start_col + subgrid_col_size):
                    if (r, c) != (i, j):
                        G.add_edge(node, r * size + c)

    return G

def plot_sudoku_graph(G,precolored,size):
    max_color = size  # Prevent division by zero
    node_colors = [precolored.get(node, 0) / max_color for node in G.nodes]
    precolored_node_colors = ['lightgray' if node not in precolored else plt.cm.tab20(precolored[node] / max_color) for node in G.nodes]
    fig = plt.figure(figsize=(12, 6))
    # Draw the graph before coloring with pre-colored nodes
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color=precolored_node_colors, node_size=500, cmap=plt.cm.tab20)
    plt.title("Sudoku Graph")
    return fig

def analysis(min_node, max_node, step_value, ep_start, ep_end, ep_stepper, random_precolored_node):
    results = []
    total_iterations = ((max_node - min_node) // step_value) * len(np.arange(ep_start, ep_end, ep_stepper)) * 3  # 3 for the three algorithms
    current_iteration = 0
    progress_bar = st.progress(0)

    for n in range(min_node, max_node, step_value): 
        for p in np.arange(ep_start, ep_end, ep_stepper):
            G = generate_random_graph(n, p)
            precolored = random_precolor(G, random_precolored_node)
            for algorithm in [greedy, dsatur]:
                start_time = time.time()
                color_map = algorithm(G, precolored)
                end_time = time.time()
                results.append({
                    'algorithm': algorithm.__name__,
                    'nodes': n,
                    'edge_prob': p,
                    'colors_used': max(color_map.values()),
                    'execution_time': end_time - start_time
                })
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations,text=f"Running {algorithm.__name__} with nodes:{n} edge probability:{p}")
            if len(results) >= 2:
                last_two_results = results[-2:]
                k = max(last_two_results[0]['colors_used'], last_two_results[1]['colors_used'])
            else:
                k = None
            if k:
                start_time = time.time()
                color_map = backtrackDsaturs(G, k, precolored)
                end_time = time.time()
                colors_used = max(color_map.values()) if color_map else None
                result = {
                    'algorithm': 'backtrackDsaturs',
                    'nodes': n,
                    'edge_prob': p,
                    'execution_time': end_time - start_time
                }
                if colors_used:
                    result['colors_used'] = colors_used
                results.append(result)
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations,text=f"Running Backtracking Dsaturs with nodes:{n} edge probability:{p}")
    progress_bar.progress(100)
    df = pd.DataFrame(results)
    return df


def plot_analysis(df):
    grouped_df = df.groupby(['algorithm', 'nodes']).agg({
        'execution_time': 'mean',
        'colors_used': 'max'
    }).reset_index()

    # First Plot: Execution Time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for algorithm in grouped_df['algorithm'].unique():
        algorithm_data = grouped_df[grouped_df['algorithm'] == algorithm]
        ax1.plot(algorithm_data['nodes'], algorithm_data['execution_time'], label=algorithm)

    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Average Execution Time (s)')
    ax1.set_title('Comparison of Execution Time for Different Algorithms')
    ax1.legend()

    # Second Plot: Maximum Colors Used
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for algorithm in grouped_df['algorithm'].unique():
        algorithm_data = grouped_df[grouped_df['algorithm'] == algorithm]
        ax2.plot(algorithm_data['nodes'], algorithm_data['colors_used'], label=algorithm)

    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Maximum Colors Used')
    ax2.set_title('Comparison of Maximum Colors Used for Different Algorithms')
    ax2.legend()

    plt.tight_layout()

    return fig1, fig2


#st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: #1da2dc;margin-bottom:50px;'>Graph PreColoring Extension</h1>", unsafe_allow_html=True)
tab1, tab2,tab3 = st.tabs(["Run Algorithms", "Solve Sudoku","Run Analysis"])



with tab1:
    selected_algorithm = st.selectbox("SELECT AN ALGORITHM", ['Greedy','Dsaturs','Hybrid-BacktrackingDsaturs'])

    uploaded_file = st.file_uploader(label="Upload File With List of Tuples Representing Edges")
    if uploaded_file:
        edge_list = process_uploaded_file(uploaded_file)
        if edge_list is None:
            st.warning("Wrong Format of the file")
        else:
            precolorcol,kcol = st.columns(2)
            with precolorcol:
                precolored_nodes = st.number_input("No of precolored Nodes",value=3)
            if selected_algorithm == 'Hybrid-BacktrackingDsaturs':
                with kcol:
                    k = st.number_input("Number of colors you want",value=4)
        
    else:
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            nodes = st.number_input("Enter No. of Nodes",value=10)
        with col2:
            edge_probability = st.number_input("Enter Edge Probability",value=.5,min_value=.1,max_value=1.0)
        with col3:
            precolored_nodes = st.number_input("No of precolored Nodes",value=round(nodes*.30))
        with col4:
            if selected_algorithm == 'Hybrid-BacktrackingDsaturs':
                k = st.number_input("Number of colors you want",value=round(nodes*.24))
    
    if st.button("RUN THE ALGORITHM", use_container_width=True,type="primary"):
        if uploaded_file:
            G = nx.Graph()
            G.add_edges_from(edge_list)
        else:
            G = generate_random_graph(n=nodes, p=edge_probability)
            
        precolored = random_precolor(G, precolored_nodes)
        if selected_algorithm == 'Greedy':
            color_map = greedy(G, precolored)
        if selected_algorithm == "Dsaturs":
            color_map = dsatur(G,precolored)
        if selected_algorithm == "Hybrid-BacktrackingDsaturs":
            color_map = backtrackDsaturs(G,k,precolored)
        
        if len(color_map) <= 0:
            st.warning(f"{selected_algorithm} Failed to obtain your coloring requirement")
        else:
            fig_before,fig_after = plot_result(color_map=color_map, G=G, algorithm=selected_algorithm, precolored=precolored)
            with st.spinner(f"Running {selected_algorithm} algorithm... Please wait."):
                fbefore,fafter = st.columns(2)
                with fbefore:
                    with st.container(height=450,border=True):
                        st.pyplot(fig_before)
                with fafter:
                    with st.container(height=450,border=True):
                        st.pyplot(fig_after)
                    



with tab2:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        grid_size = st.number_input("Enter Board Size", value=3, min_value=3, max_value=9)

    st.session_state['board_generated'] = True  # flag to indicate the board has been generated

    if st.session_state.get('board_generated', False):
        # Create the Sudoku board form
        with st.form(key='sudoku_board'):
            # Create columns for each row of the Sudoku board
            for i in range(grid_size):
                cols = st.columns(grid_size)
                for j, col in enumerate(cols):
                    with col:
                        # Create a number input for each cell with a hidden label
                        st.number_input(f"", value=0, min_value=0, max_value=grid_size, key=f"cell-{i}-{j}", step=1, label_visibility="collapsed")

            # Submit button for the form
            submitted = st.form_submit_button(label='Submit')

        if submitted:
            G = create_sudoku_graph(grid_size)
            #process the input values after submission
            sudoku_precolored = {}
            for idx in range(grid_size):
                for jdx in range(grid_size):
                    cell_value = st.session_state[f"cell-{idx}-{jdx}"]
                    if cell_value != 0:
                        linear_index = idx * grid_size + jdx
                        sudoku_precolored[linear_index] = cell_value

            st.pyplot(plot_sudoku_graph(G,sudoku_precolored,grid_size))
            color_map = backtrackDsaturs(G, grid_size, sudoku_precolored)
            st.pyplot(plot_sudoku_board(sudoku_precolored,grid_size,"Graph Coloring Representation of the Board"))
            if len(color_map) <=0:
                st.warning("Oops!!! Couldn't Find Any Solution.")
            else:
                st.pyplot(plot_sudoku_board(color_map,grid_size,"Solution Board"))



with tab3:
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        min_node = st.number_input("Min Num of Nodes",value=50)
    with col2:
        max_node = st.number_input("Max Num of Nodes",value=50)
    with col3:
        node_stepper = st.number_input("Stepper in Iteration",value=50)
    with col4:
        random_precolored_node = st.number_input("Random Precolored Nodes",value=10)
    col1,col2,col3 = st.columns(3)
    with col1:
        min_ep = st.number_input("Min Edge Probability",value=0.2)
    with col2:
        max_ep = st.number_input("Max Edge Probability",value=0.8)
    with col3:
        ep_stepper = st.number_input("Stepper in Edge Probability",value=0.2)


    if st.button("Run Analysis",type="primary",use_container_width=True):
        with st.spinner(text="Running Analysis Please Wait..."):
            df = analysis(min_node,max_node,node_stepper,min_ep,max_ep,ep_stepper,random_precolored_node)
            exc_fig,color_fig=plot_analysis(df)
            st.pyplot(exc_fig)
            st.pyplot(color_fig)
            st.write(df)