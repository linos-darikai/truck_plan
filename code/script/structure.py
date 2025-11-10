import random as r
from collections import namedtuple
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dill
import math as m
import vrplib

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

def generate_random_truck(products):
    """
    Generate a random truck based on the given list of products.
    Each truck has a random type, allowed products, capacity limits, and modifier.
    """
    truck_types = [
        "Light Truck",
        "Medium Truck",
        "Heavy Truck",
        "Refrigerated Truck",
        "Tanker Truck",
        "Flatbed Truck"
    ]

    # Random truck type
    truck_type = r.choice(truck_types)

    # Random allowed products
    if len(products) == 0:
        raise ValueError("Product list is empty. Cannot generate a truck.")
    nb_allowed = r.randint(1, min(len(products), 4))
    allowed_products = r.sample(list(products.keys()), nb_allowed)

    # Random capacity limits
    max_volume = r.randint(80, 250)
    max_weight = r.randint(100, 400)

    # Random modifier (time or cost)
    modifier = round(r.uniform(0.8, 1.5), 2)

    # Create the truck
    truck = Truck(
        truck_type=truck_type,
        allowed_products=allowed_products,
        max_volume=max_volume,
        max_weight=max_weight,
        modifier=modifier
    )

    return truck

def generate_list_random_truck(products, nb_truck):
    """
    Generate a list of randomly configured trucks.

    Args:
        products (dict): Dictionary of products available for transport.
        nb_truck (int): Number of trucks to generate.

    Returns:
        list: A list of Truck objects.
    """
    if nb_truck <= 0:
        raise ValueError("The number of trucks must be greater than 0.")
    if len(products) == 0:
        raise ValueError("Product list is empty. Cannot generate trucks.")

    trucks = []
    for _ in range(nb_truck):
        truck = generate_random_truck(products)
        trucks.append(truck)

    return trucks
# endregion

# region PRODUCT MANAGEMENT
Product = namedtuple("Product", ["name", "volume", "weight","delivery_time"])

def add_product_to_list(products_dict, name, volume, weight, delivery_time):
    """Add a new product to the product dictionary."""
    if name in products_dict:
        raise ValueError(f"Product '{name}' already exists")

    products_dict[name] = Product(name=name, volume=volume, weight=weight, delivery_time=delivery_time)

#random
def generate_random_product(products_dict):
    """
    Generate a random product and add it to the products_dict.
    """
    nb_products = r.randint(5, 10)

    name_list = [
    "Chaise", 
    "Table", 
    "Bureau", 
    "Étagère", 
    "Armoire", 
    "Porte", 
    "Fenêtre", 
    "Lampadaire", 
    "Coffre", 
    "Chariot", 
    "Palette", 
    "Machine CNC", 
    "Convoyeur", 
    "Tapis roulant", 
    "Échelle", 
    "Véhicule utilitaire", 
    "Réservoir", 
    "Pompe", 
    "Compresseur", 
    "Équipement de soudure", 
    "Chariot élévateur", 
    "Conteneur", 
    "Rouleau industriel", 
    "Scie à ruban", 
    "Perceuse industrielle"
]
    name_allready_chose = []

    for _ in range(nb_products):

        name = r.choice(name_list)
        i = 1
        name_m = f"{name}{i}"
        
        while name_m in name_allready_chose:
            i += 1
            name_m = f"{name}{i}"

        name_allready_chose.append(name_m)

        volume = round(r.uniform(0.5, 7), 2)
        weight = round(r.uniform(0.5, 10), 2)
        delivery_time = round(r.uniform(0, 0.5), 1)

        add_product_to_list(products_dict, name_m, volume, weight, delivery_time)
    return
# endregion

# region GRAPH
class Node: 
    def __init__(self, demand = None, products_dict):
        if demand != None:
            self.demand = demand # we can add types here
        else:
            if not products_dict:
                raise RuntimeError("Aucun produit n'a été ajouté ! Veuillez ajouter au moins un produit avant de lancer le système.")
            else:
                self.demand = {next(iter(products_dict)):1}
    
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
        self.time_line = 24
        self.matrix = None
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


    #load from the instance given
    def load_from_file(self, file_name):
        """
        This is a function that is to load the graph from the CSV from the website 
        we need to model that situation into our graph and be able to load.
        Check the data from CSV
        """
        path = f"../media/instances/{file_name}"
        instance = vrplib.read_instance(path)
        self.instance = instance
        print(instance)
        n = len(instance['node_coord'])
        graph = [[None for _ in range(n)] for _ in range(n)]
        nodes = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    graph[i][j] = self.distance_function(instance['node_coord'][i],instance['node_coord'][j]) #use the distance as weight
                else:
                    graph[i][j] = None
            new_node = Node()
            nodes.append(new_node)
        self.matrix = graph
        self.nodes = nodes
        return


    #print
    def print_demand(self):
        """
        print the list of the demand
        """
        for i,node in enumerate(g.nodes):
            print(f"{i+1:03} : {node.demand}")
        return
    
    def print_function(self,na,nb):
        """
        print the cost function link to the edge between the nodes na and nb.
        """
        if self.matrix[na][nb] == None:
            raise ValueError(f"This edge ({na}, {nb}) doesn't exist or has no function.")
        
        x = np.linspace(0, 24, 200)
        y = np.array([self.matrix[na][nb](xi) for xi in x])

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
        self.nodes = [random_node(productes, productes) for _ in range(nb_nodes)]
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

if __name__ == "__main__":
    i = create_random_instance(5,2)
    save_instance(i,"N5B2")
    i["graph"].plot_instance_graph()
    i_bis = load_instance("N5B2")
    i_bis["graph"].plot_instance_graph()


