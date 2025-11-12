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
