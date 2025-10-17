import json as j
import random as r
from collections import deque

memory = ".\\media\\test"

# save test
def save_matrix(path, matrix):
    """
    Save the adjacency matrix to a JSON file.
    
    Args:
        path (str): The path (relative or absolute) to save the JSON file.
        matrix (list[list[dict|None]]): The adjacency matrix representing the graph.
    
    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    try:
        with open(path, "w", encoding="utf-8") as file:
            j.dump(matrix, file, indent=4)
        return True
    except Exception as e:
        print("Error saving file:", e)
        return False


def load_matrix(path):
    """
    Load an adjacency matrix from a JSON file.

    Args:
        path (str): The path to the JSON file (relative or absolute).

    Returns:
        list[list[dict|None]]: The adjacency matrix if loaded successfully, 
                               or None if an error occurs.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            matrix = j.load(file)
        return matrix
    except Exception as e:
        print("Error loading file:", e)
        return None

#annex fonction
def calculate_cost(distance, duration):
    """
    Calculate the cost of a road based on its distance and duration.

    Args:
        distance (float): Distance of the road.
        duration (float): Duration of travel along the road.

    Returns:
        float: Calculated cost of the road.
    """
    return distance * 0.5 + duration * 0.2


def perturb_value(value, percent=10):
    """
    Return a slightly modified value within a given percentage range.

    Args:
        value (float): Original value.
        percent (float): Maximum percentage deviation (default is 10%).

    Returns:
        float: Perturbed value.
    """
    variation = value * percent / 100
    return round(r.uniform(value - variation, value + variation), 2)

#create matrix
def is_connected(matrix):
    """
    Check whether the graph represented by the adjacency matrix is connected.

    Args:
        matrix (list[list[dict|None]]): Adjacency matrix.

    Returns:
        bool: True if the graph is connected, False otherwise.
    """
    size = len(matrix)
    visited = [False] * size
    queue = deque([0])
    visited[0] = True

    while queue:
        node = queue.popleft()
        for neighbor, road in enumerate(matrix[node]):
            if road is not None and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return all(visited)


def create_edge(matrix, i, j):
    """
    Create a bidirectional road (edge) between two nodes in the adjacency matrix.

    Args:
        matrix (list[list[dict|None]]): The adjacency matrix.
        i (int): Index of the first node.
        j (int): Index of the second node.
    """
    distance = r.uniform(5, 100)
    duration = r.uniform(5, 120)
    toll = r.randint(1, 20)
    cost = calculate_cost(distance, duration)

    road = {
        "toll": round(toll, 2),
        "distance": round(distance, 2),
        "duration": round(duration, 2),
        "cost": round(cost, 2)
    }

    reversed_road = {
        "toll": round(toll, 2),
        "distance": perturb_value(distance),
        "duration": perturb_value(duration),
        "cost": perturb_value(cost)
    }

    matrix[i][j] = road
    matrix[j][i] = reversed_road


def create_random_conected_matrix(size):
    """
    Generate a connected graph represented by an adjacency matrix with random edges.

    Args:
        size (int): Number of nodes in the graph.

    Returns:
        list[list[dict|None]]: Adjacency matrix representing the connected graph.
    """
    while True:
        matrix = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                if r.choice([True, False]):
                    create_edge(matrix, i, j)

        if is_connected(matrix):
            return matrix

#create the characteristic
def create_node_state(size):
    product = ["metal","wood","meat", "honey"]
    ans =  []
    for i in range(size):
        hour_start = r.randint(8, 20)
        minute_start = r.choice([0, 15, 30, 45])
        hour_end = r.randint(hour_start + 1, 23)
        minute_end = r.choice([0, 15, 30, 45])
        ans.append({"state": (r.choice(["stock","shop"]),r.choice(product)),"schedule":{"start":(hour_start,minute_start),"end":(hour_end,minute_end)}})
    return ans

#save a test after create it
def create_test(size):
    """
    Create a connected random graph and save it to a JSON file.

    Args:
        size (int): Number of nodes in the graph.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    mat = create_random_conected_matrix(size)
    state = create_node_state(size)
    memo = {"node_state":state,"matrix":mat}
    path = memory + f"\\test_{size}.json"
    status = save_matrix(path, memo)
    return status

#print the graph
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph_from_matrix(matrix):
    """
    Draw a graph from an adjacency matrix, displaying each edge's attributes on separate lines.

    Args:
        matrix (list[list[dict|None]]): Adjacency matrix of the graph.
    """
    G = nx.Graph()
    n = len(matrix)
    G.add_nodes_from(range(n))

    # Add edges with attributes
    for i in range(n):
        for j in range(i + 1, n):
            edge = matrix[i][j]
            if edge is not None:
                G.add_edge(i, j, **edge)

    pos = nx.spring_layout(G, seed=42)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='black', alpha=0.7)

    # Create multiline labels for each edge
    edge_labels = {}
    for i, j, data in G.edges(data=True):
        label_lines = [f"{key}: {value}" for key, value in data.items()]
        edge_labels[(i, j)] = "\n".join(label_lines)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()

#text
if __name__ == "__main__":
    create_test(5)
    memo = load_matrix(".\\media\\test\\test_5.json")
    print(memo["node_state"],"\n")
    print(memo["matrix"])
    draw_graph_from_matrix(memo["matrix"])
