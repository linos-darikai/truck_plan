import structure
import random as r
import copy as c

"""
Solution new structure 
solution = [
    {
        'truck_id': 0,
        'total_load': 35,  # Sum of all deliveries
        'route': [
            {'node': 0, 'arrival': 0, 'service': 0, 'departure': 0, 'deliver': 0, 'load_after': 35},
            {'node': 3, 'arrival': 10.0, 'service': 0.5, 'departure': 10.5, 'deliver': 15, 'load_after': 20},
            {'node': 7, 'arrival': 25.0, 'service': 0.3, 'departure': 25.3, 'deliver': 20, 'load_after': 0},
            {'node': 0, 'arrival': 40.0, 'service': 0, 'departure': 40.0, 'deliver': 0, 'load_after': 0}
        ]
    },
    {...}
]
"""


#time printer
def get_path_travel_time(graph, truck, path, units='hours'):
    """
    Calculate the total *travel time* for a single truck's path, 
    excluding delivery times.

    Args:
        graph (list): The graph matrix with edge functions.
        truck (Truck): The truck object assigned to this path.
        path (list): A list of tuples:
            (node_index, products_delivered_dict, leaving_time)
        units (str): The desired output format ('hours' or 'minutes').
    
    Returns:
        float or int: The total travel time in the specified units.
                     (float for 'hours', int for 'minutes')
    """
    total_travel_time_hours = 0
    # No travel time if path is empty or just has the starting node
    if not path or len(path) < 2:
        return 0 

    current_node = path[0][0]

    # Iterate from the first stop onwards
    for stop in path[1:]:
        next_node, _, leaving_time = stop # Delivery info is ignored
        
        # Check if edge exists
        edge_function = graph[current_node][next_node]
        if edge_function is None:
            print(f"Warning: No edge from {current_node} to {next_node}. Skipping segment.")
            continue # Skip this segment

        # 1. Calculate Travel time
        travel_value = edge_function(leaving_time)
        
        # 2. Apply truck modifier and add to total time
        total_travel_time_hours += truck.modifier * travel_value
        
        # Update current node for next iteration
        current_node = next_node

    # Return based on desired units
    if units == 'minutes':
        return int(total_travel_time_hours * 60)
    
    # Default to hours
    return total_travel_time_hours

def calculate_path_time(graph, truck, path, service_time=0.5):
    """
    Calculate total time for a truck's path.
    
    Args:
        graph: Graph object with edge_functions matrix
        truck: Truck object
        path: List of tuples (node_id, deliver_qty, departure_time)
             OR list of dicts (if using Option B)
        service_time: Time to deliver at each customer
    
    Returns:
        float: Total time in minutes
    """
    if not path or len(path) < 2:
        return 0
    
    total_time = 0
    
    # If using tuple format (Option A)
    for i in range(len(path) - 1):
        current_node, deliver_qty, departure_time = path[i]
        next_node, _, _ = path[i + 1]
        
        # Get edge function
        edge_func = graph.graph[current_node][next_node]
        if edge_func is None:
            raise ValueError(f"No edge {current_node} -> {next_node}")
        
        # Calculate travel time using DEPARTURE time from current node (correct!)
        travel_time = edge_func(departure_time) * truck.modifier
        
        # Add service time at next node (if not depot)
        service = service_time if next_node != 0 else 0
        
        total_time += travel_time + service
    
    return total_time

#evaluation
def evaluation(graph, trucks, solution, service_time=0.5):
    """
    Evaluate solution quality (minimize maximum route time).
    
    Args:
        graph: Graph object
        trucks: List of Truck objects
        solution: List of paths (one per truck)
        service_time: Service time per customer
    
    Returns:
        float: Maximum time among all routes (makespan)
    """
    if not solution:
        return float('inf')
    
    max_time = 0
    
    for truck_idx, path in enumerate(solution):
        truck = trucks[truck_idx]
        
        # Calculate time for this route
        route_time = calculate_path_time(graph, truck, path, service_time)
        max_time = max(max_time, route_time)
    
    return max_time

def feasability(graph, trucks, solution):
    """
    Check if solution is feasible.
    
    Checks:
    1. All routes start and end at depot (node 0)
    2. No truck exceeds capacity
    3. All customer demands satisfied exactly once
    4. All edges exist
    
    Args:
        graph: Graph object
        trucks: List of Truck objects  
        solution: List of paths
    
    Returns:
        (bool, str): (is_feasible, message)
    """
    n_nodes = len(graph.nodes)
    
    # Track deliveries per node
    deliveries = [0] * n_nodes
    
    for truck_idx, path in enumerate(solution):
        truck = trucks[truck_idx]
        
        if not path or len(path) < 2:
            continue
        
        # Extract node sequence
        if isinstance(path[0], tuple):
            nodes = [p[0] for p in path]
            delivers = [p[1] for p in path]
        else:  # Dict format
            nodes = [stop['node'] for stop in path]
            delivers = [stop['deliver'] for stop in path]
        
        # Check 1: Starts at depot
        if nodes[0] != 0:
            return False, f"Truck {truck_idx}: Doesn't start at depot"
        
        # Check 2: Ends at depot
        if nodes[-1] != 0:
            return False, f"Truck {truck_idx}: Doesn't end at depot"
        
        # Check 3: Edges exist
        for i in range(len(nodes) - 1):
            curr, next = nodes[i], nodes[i + 1]
            if graph.graph[curr][next] is None:
                return False, f"Truck {truck_idx}: No edge {curr} -> {next}"
        
        # Check 4: Capacity
        total_load = sum(delivers[1:-1])  # Exclude depot visits
        if total_load > truck.max_capacity:
            return False, f"Truck {truck_idx}: Capacity exceeded ({total_load}/{truck.max_capacity})"
        
        # Track deliveries
        for node, qty in zip(nodes, delivers):
            deliveries[node] += qty
    
    # Check 5: All demands satisfied
    for node_idx in range(1, n_nodes):  # Skip depot
        node = graph.nodes[node_idx]
        demand = node.demand if isinstance(node.demand, int) else sum(node.demand.values())
        
        if deliveries[node_idx] < demand:
            return False, f"Node {node_idx}: Under-delivered ({deliveries[node_idx]}/{demand})"
        if deliveries[node_idx] > demand:
            return False, f"Node {node_idx}: Over-delivered ({deliveries[node_idx]}/{demand})"
    
    return True, "Solution is feasible ✅"

#random possible solution
def random_possible_solution(graph, trucks, products):#need to check
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
def cycle_mutation(solution):
    """for a compleat graph"""

    new_solution = c.deepcopy(solution)

    truck_id = r.randint(0, len(new_solution)-1)
    route = new_solution[truck_id]
    nb_deliveries = len(route) - 2

    if nb_deliveries == 0:
        while nb_deliveries != 0:
            truck_id = r.randint(0, len(new_solution)-1)
            route = new_solution[truck_id]
            nb_deliveries = len(route) - 2

    elif nb_deliveries == 1:
        action = r.choice(["move"])
    else:
        action = r.choice(["move", "swap"])

    # --- MOVE ---
    if action == "move":
        truck2 = r.randint(0, len(new_solution)-1)
        while truck2 == truck_id:
            truck2 = r.randint(0, len(new_solution)-1)
        route2 = new_solution[truck2]

        node_idx = r.randint(1, len(route)-2)
        node = route.pop(node_idx)

        if len(route2) <= 2:
            insert_idx = 1
        else:
            insert_idx = r.randint(1, len(route2)-1)

        route2.insert(insert_idx, node)


        new_solution[truck_id] = route
        new_solution[truck2] = route2
        return new_solution

    # --- SWAP ---
    elif action == "swap":
        i, j = sorted(r.sample(range(1, len(route)-1), 2))
        route[i], route[j] = route[j], route[i]
        new_solution[truck_id] = route
        return new_solution


#change the number of delivery object of 1 node
def delivery_mutation(graph, trucks, products, solution):
    return

#change the leaving time of 1 node
def leaving_time_mutation(graph, trucks, products, solution):
    """modify the leaving time of a radom node."""

    new_solution = c.deepcopy(solution)

    while True:
        truck_id = r.randint(0, len(new_solution) - 1)
        route = new_solution[truck_id]

        if len(route) <= 2:
            # We dont have a node to modify, we try with an other truck
            continue

        # --- selection ---
        node_idx = r.randint(1, len(route) - 2)
        node = route[node_idx]
        prev_node = route[node_idx - 1]
        next_node = route[node_idx + 1] if node_idx + 1 < len(route) else None

        prev_id, _, prev_leave = prev_node
        node_id, node_deliver, _ = node

        # --- Lower bound ---
        travel_prev = graph.graph[prev_id][node_id](prev_leave) * trucks[truck_id].modifier
        service_cur = sum(products[p].delivery_time * qty for p, qty in node_deliver.items())
       
        lower_bound = prev_leave + travel_prev + service_cur

        # --- Upper bound ---
        if next_node is not None:
            next_id, _, next_leave = next_node
            service_next = sum(products[p].delivery_time * qty for p, qty in next_node.items())
            upper_bound = next_leave - service_next
        else:
            upper_bound = lower_bound + 1440

        #Créer une liste de temps possibles sauf l’actuel
        possible_times = [t for t in range(int(lower_bound), int(upper_bound) + int(max(graph.graph[node_id][next_id])), 5)
                          if upper_bound - t - graph.graph[node_id][next_id](t) > 0]

        # --- validity of the interval ---
        if possible_times != []:
            continue

        # --- Mutation effective ---
        new_leave_time = r.choice(possible_times)
        new_node = (node_id, node_deliver, new_leave_time)
        route[node_idx] = new_node
        new_solution[truck_id] = route
        return new_solution

#global mutation
def random_possible_mutation(graph, trucks, products, current_solution):
    """
    Randomly choose one type of mutation (cycle, delivery, or leaving time)
    and apply it to the current solution.
    """
    mutation_functions = [
        cycle_mutation,
        delivery_mutation,
        leaving_time_mutation
    ]

    # Pick one mutation randomly
    chosen_mutation = r.choice(mutation_functions)

    # Apply it and return the new solution
    new_solution = chosen_mutation(graph, trucks, products, current_solution)

    return new_solution

#hillclimbing 
def hill_climbing(graph, trucks, products, max_iterations=1000):
    """
    Hill Climbing for VRP using your existing mutation functions.

    Parameters:
        graph      : adjacency matrix or dict with time functions
        trucks     : list of Truck objects
        products   : dict of Product objects
        max_iterations : number of iterations to perform
    Returns:
        best_solution : list of truck paths
        best_score    : evaluation score of the best solution
    """
    # Step 1: Generate an initial random solution
    current_solution = random_possible_solution(graph, trucks, products)
    best_solution = current_solution
    best_score = evaluation(graph, trucks, products, best_solution)

    for iteration in range(max_iterations):
        # Step 2: Generate a neighbor using a random mutation
        neighbor_solution = random_possible_mutation(graph, trucks, products, current_solution)

        # Step 3: Evaluate the neighbor
        neighbor_score = evaluation(graph, trucks, products, neighbor_solution)

        # Step 4: If neighbor is better, move to neighbor
        if neighbor_score < best_score:  # assuming lower is better (time/cost)
            best_solution = neighbor_solution
            best_score = neighbor_score
            current_solution = neighbor_solution
            # Optional: print progress
            print(f"Iteration {iteration+1}: Improved score = {best_score}")
        else:
            # Stay at current_solution if no improvement
            current_solution = current_solution

    return best_solution, best_score
  

#tabou