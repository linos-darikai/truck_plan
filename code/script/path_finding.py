import structure
import random as r

#time printer
def print_hour(hour):
    """
    Convert a fractional hour into hours and minutes and print it.
    """
    print(f"{int(hour//1)}h {int(hour % 1* 60)}min")
    return


#evaluation
def evaluation(graph, trucks, products, solution):
    """
    Evaluate a solution and return the maximum total delivery time among all trucks.
    Does NOT verify feasibility.
    
    solution: list of truck paths, each path is a list of tuples:
        (node_index, products_delivered_dict, leaving_time)
    """
    time_each_path = []

    for i, path in enumerate(solution):
        total_path_time = 0
        if not path:
            time_each_path.append(0)
            continue

        current_node = path[0][0]

        for stop in path[1:]:
            next_node, delivered, leaving_time = stop
            # Travel time
            edge_function = graph[current_node][next_node]
            travel_value = edge_function(leaving_time)
            # Delivery time
            delivery_value = sum(qty * products[prod].delivery_time for prod, qty in delivered.items())
            # Apply truck modifier
            total_path_time += trucks[i].modifier * travel_value + delivery_value
            current_node = next_node

        time_each_path.append(total_path_time)

    return max(time_each_path)

def feasability(graph, trucks, products, solution):
    """
    Verify if a proposed solution is feasible.

    Conditions checked:
    1. The path between nodes exists in the graph.
    2. Trucks never exceed their weight or volume capacity.
    3. Trucks only deliver what they have loaded.
    4. All delivery demands at each node are fully satisfied.
    """

    # --- initialize total deliveries tracker per node
    deliveries_done = {node: {p: 0 for p in products} for node in range(len(graph))}

    # --- iterate over each truck
    for t_idx, truck in enumerate(trucks):
        used_volume = 0
        used_weight = 0
        cargo = {p: 0 for p in products}  # what's currently on the truck

        path = solution[t_idx]
        if not path:
            continue

        # --- check edges
        current_node = path[0][0]
        for stop_idx in range(1, len(path)):
            next_node, delivered, leaving_time = path[stop_idx]

            # 1️⃣ check if edge exists
            if next_node not in graph[current_node]:
                return False, f"Truck {truck.truck_type}: No edge {current_node} → {next_node}"

            # 2️⃣ check deliveries
            for pname, qty in delivered.items():
                if pname not in products:
                    return False, f"Truck {truck.truck_type}: Unknown product '{pname}'"

                # truck permission
                if truck.allowed_products and pname not in truck.allowed_products:
                    return False, f"Truck {truck.truck_type}: Not allowed to carry '{pname}'"

                # can't deliver more than onboard
                if cargo[pname] < qty:
                    return False, f"Truck {truck.truck_type}: Tried to deliver more '{pname}' than loaded"

                # deliver
                cargo[pname] -= qty
                deliveries_done[next_node][pname] += qty

            current_node = next_node

        # 3️⃣ check truck capacity (for all items it carried)
        used_volume = sum(products[p].volume * cargo[p] for p in cargo)
        used_weight = sum(products[p].weight * cargo[p] for p in cargo)
        if used_volume > truck.max_volume:
            return False, f"Truck {truck.truck_type}: Volume exceeded ({used_volume}/{truck.max_volume})"
        if used_weight > truck.max_weight:
            return False, f"Truck {truck.truck_type}: Weight exceeded ({used_weight}/{truck.max_weight})"

    # 4️⃣ check that all deliveries are completed
    for node, node_demands in demands.items():
        for pname, needed_qty in node_demands.items():
            if deliveries_done[node][pname] < needed_qty:
                return False, f"Node {node}: Missing delivery of {needed_qty - deliveries_done[node][pname]} {pname}"

    return True, "OK ✅"



#random possible solution
def random_possible_solution(graph, trucks, products):
    """
    Generate a random feasible-looking solution.
    Each truck visits a random sequence of nodes and delivers random allowed products.
    """
    solution = []
    n_nodes = len(graph)
    
    for truck in trucks:
        path = []
        visited_nodes = r.sample(range(n_nodes), n_nodes)  # random node order
        for node in visited_nodes:
            # Random delivery quantities within allowed products
            delivered = {}
            for prod_name in truck.allowed_products:
                delivered[prod_name] = r.randint(0, 2)  # small quantity for testing
            leaving_time = r.uniform(0, 24)  # random leaving time
            path.append((node, delivered, leaving_time))
        solution.append(path)
    
    return solution

#random possible mutation
#add node to the cycle of 1 node
def transform_in_compleat_graph(graph):
    return

def cycle_mutation(graph, trucks, products, solution):
    return
#change the number of delivery object of 1 node
def delivery_mutation(graph, trucks, products, solution):
    return
#change the leaving time of 1 node
def leaving_time_mutation(graph, trucks, products, solution):
    return

#global mutation
def random_possible_mutation(graph, trucks, products, curent_solution):


    return

#hillpath   

#tabou


