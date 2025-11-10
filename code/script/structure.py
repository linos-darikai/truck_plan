import random as r
from collections import namedtuple
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dill
import math as m
import vrplib
from path_finding import hill_climbing


##########################################################################################################
###########################################################################################################
###########################################################################################################
#region Truck
class Truck:
    """Simplified truck for single product, uniform capacity."""
    def __init__(self, truck_id, max_capacity=100, modifier=1.0):
        self.truck_id = truck_id
        self.truck_type = f"Truck_{truck_id}"  # Keep for compatibility
        self.max_capacity = max_capacity
        self.modifier = modifier
        
        # Keep these for compatibility with old code
        self.max_volume = max_capacity
        self.max_weight = max_capacity
        self.allowed_products = None  # No restrictions
    
    def __repr__(self):
        return f"Truck(id={self.truck_id}, capacity={self.max_capacity}, modifier={self.modifier})"



# region GRAPH
class Node:
    def __init__(self, node_id, demand=None):
        """
        Node in VRP graph.
        
        Args:
            node_id: Integer ID of node (0 = depot)
            demand: Integer demand (single product) or dict (multiple products)
        """
        self.node_id = node_id
        
        if demand is not None:
            self.demand = demand
        else:
            # Depot has no demand
            self.demand = 0 if node_id == 0 else 1
    
    def __repr__(self):
        return f"Node({self.node_id}, demand={self.demand})"
    
def random_node(products):
    if not products:
        raise ValueError("Product dictionary is empty. Cannot create random node.")
    
    nb_products = r.randint(1,min(len(products),5))
    selected_products = r.sample(list(products.keys()), nb_products)

    demand = {p: r.randint(1, 20) for p in selected_products}
    n = Node(demand = demand)
    return n
       
class Graph:
    def __init__(self):
        self.time_line = 1440
        self.graph = None
        self.nodes = []
        self.instance = None
        pass

    def __str__(self):
        """
        Display the graph into string
        """
        self.plot_instance_graph(t = 0)
        return ""


    #compute the value of edge
    def create_time_function(self, period, n_terms = 4, amp_range = (1,5)):
        """
        Create a positive periodic time function using cosine components:
        f(t) = a0 + Σ a_i * cos(w_i t + phi_i)
        """
        a_coeffs = [r.uniform(*amp_range) for _ in range(n_terms)]
        k_values = [r.uniform(1, 5) for _ in range(n_terms)]
        phi_values = [r.uniform(0, 2 * np.pi) for _ in range(n_terms)]
        w_values = [2 * np.pi * k / period for k in k_values]
        offset = sum(a_coeffs) + r.uniform(1.0, 5.0) #distance has to be accounted for here

        def f(t):
            val = offset
            for a, w, phi in zip(a_coeffs, w_values, phi_values):
                val += a * np.cos(w * t + phi)
            return val
        return f

    def distance_function (self, coord1, coord2):
        """
        create a positiove constant function as value the distance between coord1 and coord2
        """
        x = (coord1[0]-coord2[0])**2
        y = (coord1[1]-coord2[1])**2
        def  f(t):
            return m.sqrt(x+y)
        return f


    def load_from_vrplib(self, file_path):
        """
        Load VRP instance from vrplib file.
        
        Args:
            file_path: Path to .vrp file
        
        Returns:
            dict: Instance info (num_nodes, total_demand, capacity, num_vehicles)
        """
        print(f"\nLoading VRP instance: {file_path}")
        
        # Read instance
        instance = vrplib.read_instance(file_path)
        self.instance = instance
        
        # Get dimensions
        n = len(instance['node_coord'])
        print(f"Nodes: {n}")
        
        # Create nodes with demands
        self.nodes = []
        
        if 'demand' in instance:
            demands = instance['demand']
            for i in range(n):
                demand = demands[i] if i < len(demands) else 0
                node = Node(node_id=i, demand=demand)
                self.nodes.append(node)
        else:
            # No demand info - create default
            self.nodes.append(Node(node_id=0, demand=0))  # Depot
            for i in range(1, n):
                self.nodes.append(Node(node_id=i, demand=r.randint(5, 20)))
        
        # Calculate total demand
        total_demand = sum(node.demand for node in self.nodes[1:])
        print(f"Total demand: {total_demand}")
        
        # Create edge functions based on Euclidean distances
        coords = instance['node_coord']
        self.graph = [[None] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate distance
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    distance = m.sqrt(dx**2 + dy**2)
                    
                    # Create time function
                    # Option 1: Use your existing create_time_function
                    self.graph[i][j] = self.create_time_function(self.time_line)
                    
                    # Option 2: Use distance-based function (simpler)
                    # self.graph[i][j] = lambda t, d=distance: d * (1 + 0.1 * np.sin(2*np.pi*t/1440))
        
        # Extract instance info
        vehicle_capacity = instance.get('capacity', 100)
        
        # Extract number of vehicles from name (e.g., "A-n32-k5" -> 5)
        num_vehicles = None
        if 'name' in instance:
            name = instance['name']
            if '-k' in name.lower():
                try:
                    num_vehicles = int(name.lower().split('-k')[1].split('.')[0])
                except:
                    pass
        
        return {
            'num_nodes': n,
            'total_demand': total_demand,
            'vehicle_capacity': vehicle_capacity,
            'num_vehicles': num_vehicles
        }
    #print
    def print_demand(self):
        """
        print the list of the demand
        """
        for i,node in enumerate(self.nodes):
            print(f"{i+1:03} : {node.demand}")
        return
    
    def print_function(self,na,nb):
        """
        print the cost function link to the edge between the nodes na and nb.
        """
        if self.graph[na][nb] == None:
            raise ValueError(f"This edge ({na}, {nb}) doesn't exist or has no function.")
        
        x = np.linspace(0, 24, 200)
        y = np.array([self.graph[na][nb](xi) for xi in x])

        plt.plot(x, y)
        plt.xlabel('Time (t)')
        plt.xlim(0, 24)
        plt.ylabel('Cost')
        plt.ylim(0, np.max(y) * 1.1 )
        plt.title(f'Graph of the function between {na-1} {nb-1}')
        plt.grid(True)
        plt.show()
        
    def plot_instance_graph(self, t=None):
        """
        Visualize a VRP instance as a complete graph using real coordinates.
        
        instance: dict from vrplib.read_instance()
        t:        optional time parameter to evaluate time-dependent edges
        """
        if self.instance is not None and "node_coord" in self.instance:
            coords = self.instance["node_coord"]
            n = len(coords)
            pos = {i+1: (coords[i][0], coords[i][1]) for i in range(n)}
        elif self.graph is not None:
            n = len(self.graph)
            pos = nx.circular_layout(range(1, n+1))
        else:
            raise ValueError("Graph data not found. Please load or generate a graph first.")

        G = nx.DiGraph()

        # Add nodes with IDs 1..N
        for i in range(n):
            G.add_node(i+1)  # node IDs 1-based

        # Add edges
        for i in range(n):
            for j in range(n):
                if i != j and self.graph[i][j] is not None:
                    u, v = i+1, j+1  # convert 0-based index to 1-based node ID
                    if t is not None:
                        # Calculate label value
                        val = self.graph[i][j](t) if callable(self.graph[i][j]) else self.graph[i][j]
                        label = round(val, 2)
                        G.add_edge(u, v, name=label)
                    else:
                        G.add_edge(u, v)

        # Draw graph nodes
        plt.figure("VRP Graph", figsize=(10, 8))
        nx.draw(
            G,
            pos=pos,
            with_labels=True,
            node_color="lightblue",
            node_size=900,
            arrowsize=20
        )

        # Draw edge labels if t is given
        if t is not None:
            edge_labels = nx.get_edge_attributes(G, 'name')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("VRP Instance with Coordinates")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


    #random
    def is_strongly_connected(self):
        """Check if the directed graph is strongly connected."""
        n = len(self.nodes)

        def bfs(start):
            visited = [False]*n
            queue = deque([start])
            visited[start]=True
            while queue:
                node = queue.popleft()
                for neighbor, val in enumerate(self.graph[node]):
                    if val is not None:
                        if neighbor < start:
                            return True
                        if not visited[neighbor]:
                            visited[neighbor]=True
                            queue.append(neighbor)
            return all(visited)

        for i in range(n):
            if not bfs(i):
                return False
        return True

    def create_connected_matrix(self, productes, nb_nodes):
        """Generate a random strongly connected directed graph with time-dependent weights."""
        self.nodes = [random_node(productes) for _ in range(nb_nodes)]
        self.graph = [[None for _ in range(nb_nodes)] for _ in range(nb_nodes)]
        for i in range(nb_nodes):
            for j in range(nb_nodes):
                if i != j and r.random()<0.6:
                    self.graph[i][j] = self.create_time_function(self.time_line)
        while not self.is_strongly_connected():
            i, j = r.sample(range(nb_nodes), 2)
            if j not in self.graph[i]:
                self.graph[i][j] = self.create_time_function(self.time_line)
        return
# endregion

# region INSTANCE
def create_random_instance(nb_nodes = 5, nb_truck = 2):
    products_dict = {}
    generate_random_product(products_dict)

    g = Graph()
    g.create_connected_matrix(products_dict, nb_nodes)


    trucks = generate_list_random_truck(products_dict, nb_truck)

    instance = {"product": products_dict, "graph": g, "trucks": trucks}
    return instance

def save_instance(instance, filename="instance"):
    """
    Save a generated instance (products, graph, trucks, etc.) to a file using dill.

    Args:
        instance (dict): The instance to save.
        filename (str): The name of the file to save it to.
    """
    path = "..//media//test//" + filename + ".pkl"
    with open(path, "wb") as f:
        dill.dump(instance, f)

def load_instance(filename="instance"):
    """
    Load a previously saved instance from a dill file.

    Args:
        filename (str): The name of the file to load.

    Returns:
        dict: The loaded instance.
    """
    path = "..//media//test//" + filename + ".pkl"
    with open(path, "rb") as f:
        instance = dill.load(f)
    return instance
# endregion

# ============================
#        MAIN / TEST
# ============================

# --- TEST BLOCK ---
if __name__ == "__main__":
    # --- Products ---
    g = Graph()
    g.load_from_file('A-n32-k5.vrp')
    #g.plot_graph_functions()
    g.plot_instance_graph()
    # --- Example trucks ---
    trucks = [
        Truck("T1", allowed_products={"P1", "P2"}, max_volume=100, max_weight=120),
        Truck("T2", allowed_products={"P1", "P2"}, max_volume=100, max_weight=120),
        Truck("T3", allowed_products={"P1", "P2"}, max_volume=100, max_weight=120)
    ]

    # --- Example products ---
    products = {}
    add_product_to_list(products, "P1", volume=10, weight=5, delivery_time=0.5)
    add_product_to_list(products, "P2", volume=20, weight=10, delivery_time=0.8)

    # --- Run Hill Climbing ---
    best_solution, best_score = hill_climbing(g.graph, trucks, products, max_iterations=500)
    print("\nBest Score:", best_score)
    print("Best Solution:", best_solution)