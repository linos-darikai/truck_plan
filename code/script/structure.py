import random as r
from collections import namedtuple
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dill

# region TRUCK MANAGEMENT
class Truck:
    """Represents a truck that can carry specific types of products with capacity limits."""
    def __init__(self, truck_type, allowed_products=None, max_volume=100, max_weight=120, modifier=1):
        self.truck_type = truck_type
        self.max_volume = max_volume
        self.max_weight = max_weight
        self.used_volume = 0
        self.used_weight = 0
        self.cargo = {}  # dict: product -> quantity
        self.modifier = modifier  # time modifier

        if allowed_products is None:
            self.allowed_products = set()  # no restriction
        else:
            self.allowed_products = set(allowed_products)

    def __str__(self):
        """Return a formatted string showing the truck's cargo and remaining capacity."""
        lines = [f"Truck '{self.truck_type}' cargo:"]
        for product, qty in self.cargo.items():
            lines.append(f"{qty} x {product.name} (V:{product.volume}, W:{product.weight})")
        lines.append(f"Remaining volume: {self.remaining_capacity()['volume']}, "
                     f"Remaining weight: {self.remaining_capacity()['weight']}")
        return "\n".join(lines)

    __repr__ = __str__

    def remaining_capacity(self):
        """Return remaining volume and weight capacity of the truck."""
        return {
            "volume": self.max_volume - self.used_volume,
            "weight": self.max_weight - self.used_weight
        }

    def add_product(self, product, quantity=1):
        """Add a product to the truck if within capacity and allowed type."""
        if self.allowed_products and product.name not in self.allowed_products:
            raise ValueError(f"{product.name} cannot be transported by {self.truck_type}")

        total_volume = product.volume * quantity
        total_weight = product.weight * quantity

        remaining = self.remaining_capacity()
        if total_volume > remaining["volume"]:
            raise ValueError(f"Cannot add {quantity} x {product.name}: volume exceeded")
        if total_weight > remaining["weight"]:
            raise ValueError(f"Cannot add {quantity} x {product.name}: weight exceeded")

        self.cargo[product] = self.cargo.get(product, 0) + quantity
        self.used_volume += total_volume
        self.used_weight += total_weight
        print(f"{quantity} x {product.name} added to the truck {self.truck_type}.")
# endregion
class Graph:
    def __init__(self):
        self.time_line = 24
        self.graph = None
        pass
    def __str__(self):
        """
        Display the graph into string
        """
        return
    def load_from_csv(self):
        """
        This is a function that is to load the graph from the CSV from the website 
        we need to model that situation into our graph and be able to load.
        Check the data from CSV
        """
        return
    
    def create_time_function(self, period, n_terms = 4, amp_range = (1,5)):
        """
        Create a positive periodic time function using cosine components:
        f(t) = a0 + Σ a_i * cos(w_i t + phi_i)
        """
        a_coeffs = [r.uniform(*amp_range) for _ in range(n_terms)]
        k_values = [r.uniform(1, 5) for _ in range(n_terms)]
        phi_values = [r.uniform(0, 2 * np.pi) for _ in range(n_terms)]
        w_values = [2 * np.pi * k / period for k in k_values]
        offset = sum(a_coeffs) + r.uniform(1.0, 5.0)

        def f(t):
            val = offset
            for a, w, phi in zip(a_coeffs, w_values, phi_values):
                val += a * np.cos(w * t + phi)
            return val
        return f

    def is_strongly_connected(self):
        """Check if the directed graph is strongly connected."""
        def bfs(start):
            visited = [False]*len(self.graph)
            queue = deque([start])
            visited[start]=True
            while queue:
                node = queue.popleft()
                for neighbor in self.graph[node]:
                    if neighbor < start:
                        return True
                    if not visited[neighbor]:
                        visited[neighbor]=True
                        queue.append(neighbor)
            return all(visited)

        for i in range(len(self.graph)):
            if not bfs(i):
                return False
        return True

    def create_connected_matrix(self, n_nodes=5, period = 24):
        """Generate a random strongly connected directed graph with time-dependent weights."""
        graph = [{} for _ in range(n_nodes)]
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and r.random()<0.6:
                    graph[i][j] = self.create_time_function(period)
        while not self.is_strongly_connected(graph):
            i, j = r.sample(range(n_nodes), 2)
            if j not in graph[i]:
                graph[i][j] = self.create_time_function(period)
        self.graph = graph
        return graph

    #print
    def plot_graph_functions(self, period = 24):
        """Plot all time-dependent edge functions of the graph."""
        n_nodes = len(self.graph)
        t_values = np.linspace(0, period, 200)

        fig, axes = plt.subplots(n_nodes, n_nodes, figsize=(3*n_nodes, 3*n_nodes), num = "Time functions")

        # Ensure axes is 2D array for consistent indexing
        if n_nodes == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                ax = axes[i, j]
                if i == j or j not in self.graph[i]:
                    y_values = np.zeros_like(t_values)
                else:
                    f = graph[i][j]
                    y_values = np.array([f(t) for t in t_values])

                ax.plot(t_values, y_values)
                ax.set_title(f"{i} → {j}")
                ax.set_ylim(0, max(y_values.max(), 1)*1.2)
                ax.grid(True)

        plt.tight_layout()

    def plot_graph_image(self, t = None):
        """Display the directed graph with optional edge labels at a specific time t."""
        plt.figure("Graph Image")
        G = nx.DiGraph()
        n_nodes = len(self.graph)
        
        # Ajouter les nœuds
        for i in range(n_nodes):
            G.add_node(i)
        
        # Ajouter les arcs avec labels
        edge_labels = {}
        for i, neighbors in enumerate(self.graph):
            for j, f in neighbors.items():
                G.add_edge(i, j)
                if t is not None:
                    edge_labels[(i,j)] = f"{f(t):.1f}"

        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1000, arrowsize=20)
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    #save
    def save_graph_pickle(self, path):
        """Save the graph object to a file using pickle."""
        with open(path, "wb") as f:
            dill.dump(self.graph, f)

    def load_graph_pickle(self, path):
        """Load the graph object from a pickle file."""
        with open(path, "rb") as f:
            return dill.load(f)












# region PRODUCT MANAGEMENT
Product = namedtuple("Product", ["name", "volume", "weight","delivery_time"])

def add_product_to_list(products_dict, name, volume, weight, delivery_time):
    """Add a new product to the product dictionary."""
    if name in products_dict:
        raise ValueError(f"Product '{name}' already exists")

    products_dict[name] = Product(name=name, volume=volume, weight=weight, delivery_time=delivery_time)
# endregion


# region GRAPH & TIME FUNCTION
#creation
#DEMAND MANAGEMENT  ISMISSING-----------------------------------



#endregion


# ============================
#        MAIN / TEST
# ============================

if __name__ == "__main__":
    # --- Products ---
    products = {}
    add_product_to_list(products, "Oak wood", volume=5, weight=8, delivery_time=1/6)
    add_product_to_list(products, "Pine wood", volume=3, weight=5, delivery_time=1)
    add_product_to_list(products, "Iron", volume=7, weight=10, delivery_time=1.5)
    add_product_to_list(products, "White paint", volume=2, weight=3, delivery_time=0.5)
    add_product_to_list(products, "Plaster", volume=4, weight=6, delivery_time=0.25)

    # --- Trucks ---
    heavy_duty_allowed = ["Oak wood", "Pine wood", "Iron"]
    chemical_truck_allowed = ["White paint", "Plaster"]

    truck1 = Truck("Heavy Duty", allowed_products=heavy_duty_allowed, max_volume=50, max_weight=60, modifier=1)
    truck2 = Truck("Chemical Truck", allowed_products=chemical_truck_allowed, max_volume=40, max_weight=50, modifier=1.2)

    try:
        truck1.add_product(products["Oak wood"], quantity=3)
        truck1.add_product(products["Pine wood"], quantity=2)
        truck2.add_product(products["White paint"], quantity=4)
    except ValueError as e:
        print("Error:", e)

    print("\nTruck 1 contents:")
    print(truck1)
    print("\nTruck 2 contents:")
    print(truck2)

    # --- Weighted graph test ---
    n_nodes = 7
    G = create_oriented_connected_matrix(n_nodes = n_nodes)

    # Display weights at a given instant
    t_test = 10
    print("\nWeights at t =", t_test)
    for i in range(len(G)):
        for j, f in G[i].items():
            print(f"Weight {i} → {j} = {round(f(t_test), 2)}")

    # Plot all the time-dependent weights
    plot_graph_functions(G)
    plot_graph_image(G)
    plt.show()

    save_graph_pickle(G, r"code\media\test\graph_7")
    G = load_graph_pickle(r"code\media\test\graph_7")

    plot_graph_functions(G)
    plot_graph_image(G)
    plt.show()
