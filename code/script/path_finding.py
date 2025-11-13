from structure import *
import random as r
import copy 
import time

"""
Solution new structure 
solution = [
    {
        'truck_id': 0,
        'total_load': {1: 35}, // change into dict(contemplate) # Sum of all deliveries
        'route': [
            {'node': 0, 'arrival': 0, 'service': 0, 'departure': 0, 'deliver': {1: 0}, 'load_after': {1: 35}},
            {'node': 3, 'arrival': 10.0, 'service': 0.5, 'departure': 10.5, 'deliver': {1: 15}, 'load_after': {1: 20}},
            {'node': 7, 'arrival': 25.0, 'service': 0.3, 'departure': 25.3, 'deliver': {1: 20}, 'load_after': {1: 0}},
            {'node': 0, 'arrival': 40.0, 'service': 0, 'departure': 40.0, 'deliver': {1: 0}, 'load_after': {1: 0}}
        ]
    },
    {...}
]
"""

###########################################################################################################
###########################################################################################################
###########################################################################################################
# region AUXILIAR
def create_route_dict(truck_id, node_sequence, graph, truck, service_time=0.5):
    """
    Create a complete route dictionary with all timing information.
    Compatible with your Graph class structure.
    
    Args:
        truck_id: ID of the truck
        node_sequence: List of node IDs [0, 3, 7, 5, 0]
        graph: Your Graph object (with graph.graph and graph.nodes)
        truck: Your Truck object
        service_time: Time to deliver at each customer
    
    Returns:
        Dict with route information
    """
    route = []
    current_time = 0
    current_load = 0
    
    # Calculate total load needed for this route
    total_load = {}
    for node_id in node_sequence:
        if node_id != 0:  # Skip depot
            node = graph.nodes[node_id]
            if isinstance(node.demand, dict):
                for key in node.demand:
                    if key in total_load.keys():
                        total_load.update({key: total_load[key] + node.demand[key]})
                    else:
                        total_load[key] = node.demand[key]
            else:
                total_load.update({1: 0})
    
    for i, node_id in enumerate(node_sequence):
        node = graph.nodes[node_id]
        
        # Get demand for this node
        if node_id == 0:
            demand = {1: 0}
        elif isinstance(node.demand, dict):
            demand = node.demand
        else:
            demand = {}
        
        # Service time (0 at depot, constant at customers)
        service = service_time if node_id != 0 else 0
        
        # Create stop info
        stop = {
            'node': node_id,
            'arrival': current_time,
            'service': service,
            'departure': current_time + service,
            'deliver': demand,
            'load_after': {}  # Will calculate below
        }
        
        # Calculate load after this delivery
        if i == 0:
            # At depot, load everything
            current_load = total_load
        else:
            # Deliver at customer
            side_dict = {}
            for key in current_load.keys():
                delivered = demand.get(key, 0)  
                if key > 0:
                    side_dict[key] = current_load[key] - delivered
                else:
                    side_dict[key] = 0
            current_load = side_dict
        
        stop['load_after'] = current_load
        route.append(stop)
        
        # Calculate travel time to next node
        if i < len(node_sequence) - 1:
            next_node_id = node_sequence[i + 1]
            
            # Get edge function from your graph structure
            edge_func = graph.graph[node_id][next_node_id]
            if edge_func is None:
                raise ValueError(f"No edge from {node_id} to {next_node_id}")
            
            # Calculate travel time
            travel_time = edge_func(stop['departure']) * truck.modifier
            current_time = stop['departure'] + travel_time
       
    return {
        'truck_id': truck_id,
        'total_load': total_load,
        'route': route
    }

def calculate_path_time(graph, truck, route_dict, service_time=0.5):
    """
    Calculate total time for a truck's path.
    Now works with dict format.
    """
    if not route_dict or 'route' not in route_dict:
        return 0
    
    route = route_dict['route']
    
    if len(route) < 2:
        return 0
    
    total_time = 0
    
    for i in range(len(route) - 1):
        current_stop = route[i]
        next_stop = route[i + 1]
        
        current_node = current_stop['node']
        next_node = next_stop['node']
        departure_time = current_stop['departure']
        
        # Get edge function
        edge_func = graph.graph[current_node][next_node]
        if edge_func is None:
            raise ValueError(f"No edge {current_node} -> {next_node}")
        
        # Calculate travel time using DEPARTURE time from current node
        travel_time = edge_func(departure_time) * truck.modifier
        
        # Add service time at next node
        service = service_time if next_node != 0 else 0
        
        total_time += travel_time + service
    
    return total_time
# endregion

###########################################################################################################
###########################################################################################################
###########################################################################################################
# region EVALUATION AND FEASIBILITY
def evaluation(graph, trucks, solution, service_time=0.5):
    """Evaluate solution quality (minimize maximum route time)."""
    if not solution:
        return float('inf')
    
    max_time = 0
    
    for route_dict in solution:
        truck = trucks[route_dict['truck_id']]
        
        # Calculate time for this route
        route_time = calculate_path_time(graph, truck, route_dict, service_time)
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
        solution: List of route dicts
    
    Returns:
        (bool, str): (is_feasible, message)
    """
    n_nodes = len(graph.nodes)
    
    # Track deliveries per node (as dictionaries to handle multi-product)
    deliveries = [{}] * n_nodes  # ❌ PROBLEM 1: This creates references to the SAME dict
    # ✅ FIX:
    deliveries = [{} for _ in range(n_nodes)]  # Each node gets its own dict
    
    for truck_idx, route_dict in enumerate(solution):
        if not route_dict or 'route' not in route_dict:
            continue
        
        route = route_dict['route']
        truck = trucks[route_dict['truck_id']]  # ✅ Use truck_id from route_dict
        
        if not route or len(route) < 2:
            continue
        
        # Extract node sequence and deliveries from the route
        nodes = [stop['node'] for stop in route]
        delivers = [stop['deliver'] for stop in route]
        
        # Check 1: Starts at depot
        if nodes[0] != 0:
            return False, f"Truck {truck_idx}: Doesn't start at depot (starts at {nodes[0]})"
        
        # Check 2: Ends at depot
        if nodes[-1] != 0:
            return False, f"Truck {truck_idx}: Doesn't end at depot (ends at {nodes[-1]})"
        
        # Check 3: Edges exist
        for i in range(len(nodes) - 1):
            curr, next_node = nodes[i], nodes[i + 1]
            if graph.graph[curr][next_node] is None:
                return False, f"Truck {truck_idx}: No edge {curr} -> {next_node}"
        
        # Check 4: Capacity
        total_load = sum([sum(d.values()) for d in delivers[1:-1]])  # Exclude depot visits
        if total_load > truck.max_capacity:
            return False, f"Truck {truck_idx}: Capacity exceeded ({total_load}/{truck.max_capacity})"
        
        # Track deliveries - accumulate dictionaries
        for node, delivery_dict in zip(nodes, delivers):
            if node != 0:  # Don't track depot deliveries
                for product_id, qty in delivery_dict.items():
                    if product_id in deliveries[node]:
                        deliveries[node][product_id] += qty
                    else:
                        deliveries[node][product_id] = qty
    
    # Check 5: All demands satisfied
    for node_idx in range(1, n_nodes):  # Skip depot
        node = graph.nodes[node_idx]
        
        if isinstance(node.demand, dict):
            demand = node.demand
        else:
            demand = {1: node.demand} if node.demand else {1: 0}  # ✅ Handle int demand
        
        # ❌ PROBLEM 2: Can't compare dicts with !=
        # ✅ FIX: Compare each product
        delivered = deliveries[node_idx]
        
        # Check all products in demand
        for product_id, demand_qty in demand.items():
            delivered_qty = delivered.get(product_id, 0)
            if delivered_qty != demand_qty:
                return False, f"Node {node_idx}: Product {product_id} under-delivered ({delivered_qty}/{demand_qty})"
        
        # Check for over-delivery (products delivered but not demanded)
        for product_id in delivered:
            if product_id not in demand:
                return False, f"Node {node_idx}: Product {product_id} delivered but not demanded ({delivered[product_id]}/0)"
    
    return True, "Solution is feasible ✅"
# endregion

###########################################################################################################
###########################################################################################################
###########################################################################################################
# region FIRST SOLUTION
def rest_demand(graph, customer_set):
    demand_dict = {}
    for node_idx in customer_set:
        node = graph.nodes[node_idx]
        if hasattr(node, "demand") and isinstance(node.demand, dict):
            demand_dict[node_idx] = dict(node.demand)  # copie pour pouvoir modifier
        else:
            demand_dict[node_idx] = {}
    return demand_dict

def generate_feasible_random_solution(graph, trucks, products, service_time=0.5):
    """
    Génère une solution aléatoire feasible avec livraisons partielles et retours au dépôt
    pour recharger si nécessaire.
    Chaque client peut être servi par plusieurs camions si sa demande est trop grande.
    """
    n_nodes = len(graph.nodes)
    rest_customer = set(range(1, n_nodes))
    rest_demand_dict = rest_demand(graph, rest_customer)

    solution = []
    truck_idx = 0

    while rest_customer:
        truck = trucks[truck_idx % len(trucks)]
        remaining_capacity = truck.max_capacity
        route = [0] 
        customers = list(rest_customer)

        while customers:
            current_node = r.choice(customers)
            node_demand = rest_demand_dict[current_node]
            deliver_now = {}

            for pid, qty in node_demand.items():
                if pid not in truck.allowed_products:
                    continue
                prod_capacity = products[pid]["capacity"]
                max_qty_by_capacity = remaining_capacity // prod_capacity
                deliver_qty = min(qty, max_qty_by_capacity)
                if deliver_qty > 0:
                    deliver_now[pid] = deliver_qty
                    remaining_capacity -= deliver_qty * prod_capacity

            if deliver_now:
                route.append(current_node)
                for pid, qty in deliver_now.items():
                    rest_demand_dict[current_node][pid] -= qty
                    if rest_demand_dict[current_node][pid] == 0:
                        del rest_demand_dict[current_node][pid]
                if not rest_demand_dict[current_node]:
                    rest_customer.remove(current_node)

            if remaining_capacity == 0:
                route.append(0)                 
                remaining_capacity = truck.max_capacity
                route.append(0)                 

            customers = [c for c in rest_customer if c not in route]

        if route[-1] != 0:
            route.append(0)

        route_dict = create_route_dict(truck.truck_id, route, graph, truck, service_time=service_time)
        solution.append(route_dict)
        truck_idx += 1

    return solution
# endregion

###########################################################################################################
###########################################################################################################
###########################################################################################################
# region MUTATION

def swap_within_route_mutation(solution, graph, trucks, service_time=0.5):
    """Swap two customers within the same route."""
    new_solution = copy.deepcopy(solution)
    
    # Find routes with at least 2 customers
    valid_routes = []
    for i, route_dict in enumerate(new_solution):
        customers = [s['node'] for s in route_dict['route'] if s['node'] != 0]
        if len(customers) >= 2:
            valid_routes.append(i)
    
    if not valid_routes:
        return new_solution
    
    # Select random route
    route_idx = r.choice(valid_routes)
    route_dict = new_solution[route_idx]
    route = route_dict['route']
    
    # Get customer positions (not depot)
    customer_positions = [i for i, s in enumerate(route) if s['node'] != 0]
    
    # Swap two random customers
    i, j = r.sample(customer_positions, 2)
    
    # Extract node sequence
    node_sequence = [s['node'] for s in route]
    node_sequence[i], node_sequence[j] = node_sequence[j], node_sequence[i]
    
    # Recalculate route using create_route_dict
    truck = trucks[route_dict['truck_id']]
    new_route_dict = create_route_dict(truck.truck_id, node_sequence, graph, truck, service_time)
    new_solution[route_idx] = new_route_dict
    
    return new_solution

def move_customer_mutation(solution, graph, trucks, service_time=0.5):
    """Move a customer from one route to another."""
    if len(solution) < 2:
        return None  # Need at least 2 trucks
    
    new_solution = copy.deepcopy(solution)
    
    # Find source route with at least 1 customer
    source_candidates = []
    for i, route_dict in enumerate(new_solution):
        customers = [s['node'] for s in route_dict['route'] if s['node'] != 0]
        if len(customers) > 0:
            source_candidates.append(i)
    
    if not source_candidates:
        return None
    
    source_idx = r.choice(source_candidates)
    source_route = new_solution[source_idx]['route']
    
    # Select random customer
    customer_positions = [i for i, s in enumerate(source_route) if s['node'] != 0]
    cust_pos = r.choice(customer_positions)
    customer_node = source_route[cust_pos]['node']
    customer_demand = graph.nodes[customer_node].demand
    
    
    # Select destination route
    dest_idx = r.choice([i for i in range(len(solution)) if i != source_idx])
    dest_route = new_solution[dest_idx]['route']
    dest_truck = trucks[dest_idx]
    
    # Check capacity
    current_load = new_solution[dest_idx]['total_load']
    if sum(current_load.values()) + sum(customer_demand.values()) > dest_truck.max_capacity:
        return None  # Exceeds capacity
    
    # Remove from source
    source_sequence = [s['node'] for s in source_route]
    source_sequence.remove(customer_node)
    
    # Add to destination
    dest_sequence = [s['node'] for s in dest_route]
    insert_pos = r.randint(1, len(dest_sequence) - 1)
    dest_sequence.insert(insert_pos, customer_node)
    
    # Recalculate both routes using create_route_dict
    source_truck = trucks[source_idx]
    new_solution[source_idx] = create_route_dict(
        source_truck.truck_id, source_sequence, graph, source_truck, service_time
    )
    new_solution[dest_idx] = create_route_dict(
        dest_truck.truck_id, dest_sequence, graph, dest_truck, service_time
    )
    
    return new_solution

def reverse_segment_mutation(solution, graph, trucks, service_time=0.5):
    """Reverse a segment of a route (2-opt style)."""
    new_solution = copy.deepcopy(solution)
    
    # Find routes with at least 2 customers
    valid_routes = []
    for i, route_dict in enumerate(new_solution):
        customers = [s['node'] for s in route_dict['route'] if s['node'] != 0]
        if len(customers) >= 2:
            valid_routes.append(i)
    
    if not valid_routes:
        return new_solution
    
    route_idx = r.choice(valid_routes)
    route = new_solution[route_idx]['route']
    
    # Get customer positions
    customer_positions = [i for i, s in enumerate(route) if s['node'] != 0]
    
    if len(customer_positions) < 2:
        return new_solution
    
    # Select two positions and reverse between them
    i, j = sorted(r.sample(customer_positions, 2))
    
    # Extract sequence and reverse segment
    node_sequence = [s['node'] for s in route]
    node_sequence[i:j+1] = reversed(node_sequence[i:j+1])
    
    # Recalculate route using create_route_dict
    truck = trucks[route_idx]
    new_solution[route_idx] = create_route_dict(
        truck.truck_id, node_sequence, graph, truck, service_time
    )
    
    return new_solution

# endregion

###########################################################################################################
###########################################################################################################
###########################################################################################################
# region HILL CLIMBING & TABULIST

def apply_random_mutation(solution, graph, trucks, service_time=0.5):
    """Apply a random mutation that preserves feasibility."""
    mutations = [
        swap_within_route_mutation,
        move_customer_mutation,
        reverse_segment_mutation
    ]
    
    # Try mutations in random order
    r.shuffle(mutations)
    
    for mutation_func in mutations:
        new_solution = mutation_func(solution, graph, trucks, service_time)
        
        if new_solution is not None:
            # Check feasibility
            is_feasible, _ = feasability(graph, trucks, new_solution)
            if is_feasible:
                return new_solution
    
    # If all fail, return original
    return solution

#hillclimbing 
def hill_climbing(graph, trucks, max_iterations=1000):
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
    current_solution = generate_feasible_initial_solution(graph, trucks)
    best_solution = current_solution
    best_score = evaluation(graph, trucks, best_solution)

    for iteration in range(max_iterations):
        # Step 2: Generate a neighbor using a random mutation
        neighbor_solution = apply_random_mutation(current_solution,graph, trucks)

        # Step 3: Evaluate the neighbor
        neighbor_score = evaluation(graph, trucks, neighbor_solution)

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
  
# ============================================================================
# TABU SEARCH COMPONENTS
# ============================================================================

def get_solution_hash(solution):
    """
    Create a hash representing the structure of a solution.
    Used to identify if we've seen this solution before in tabu search.
    
    Returns:
        tuple: Hashable representation of customer assignments to trucks
    """
    # For each truck, get the sorted list of customers it serves
    move_signature = []
    for route_dict in solution:
        customers = tuple(sorted([stop['node'] for stop in route_dict['route'] if stop['node'] != 0]))
        move_signature.append(customers)
    return tuple(sorted(move_signature))


def get_all_neighbors(solution, graph, trucks, service_time=0.5, max_neighbors=200):
    """
    Hybrid approach: Exhaustive for cheap operations, sampled for expensive ones.
    """
    neighbors = []
    
    # 1. EXHAUSTIVE: All swaps within routes (cheap, not many)
    for route_idx, route_dict in enumerate(solution):
        route = route_dict['route']
        customer_positions = [i for i, s in enumerate(route) if s['node'] != 0]
        
        if len(customer_positions) >= 2:
            for i in range(len(customer_positions)):
                for j in range(i + 1, len(customer_positions)):
                    # Use swap mutation logic
                    temp_solution = copy.deepcopy(solution)
                    temp_route = temp_solution[route_idx]['route']
                    node_sequence = [s['node'] for s in temp_route]
                    pos_i, pos_j = customer_positions[i], customer_positions[j]
                    node_sequence[pos_i], node_sequence[pos_j] = node_sequence[pos_j], node_sequence[pos_i]
                    
                    truck = trucks[route_idx]
                    temp_solution[route_idx] = create_route_dict(
                        truck.truck_id, node_sequence, graph, truck, service_time
                    )
                    
                    is_feas, _ = feasability(graph, trucks, temp_solution)
                    if is_feas:
                        neighbors.append((temp_solution, "swap", f"swap_r{route_idx}"))
                        if max_neighbors and len(neighbors) >= max_neighbors:
                            return neighbors
    
    # 2. SAMPLED: Move operations (expensive, many possibilities)
    num_move_samples = min(50, max_neighbors - len(neighbors)) if max_neighbors else 50
    for _ in range(num_move_samples):
        neighbor = move_customer_mutation(solution, graph, trucks, service_time)
        if neighbor is not None:
            is_feas, _ = feasability(graph, trucks, neighbor)
            if is_feas:
                neighbors.append((neighbor, "move", "move_sampled"))
                if max_neighbors and len(neighbors) >= max_neighbors:
                    return neighbors
    
    # 3. EXHAUSTIVE: All reversals (cheap, not many)
    for route_idx, route_dict in enumerate(solution):
        route = route_dict['route']
        customer_positions = [i for i, s in enumerate(route) if s['node'] != 0]
        
        if len(customer_positions) >= 2:
            for i in range(len(customer_positions)):
                for j in range(i + 1, min(i + 5, len(customer_positions))):  # Limit segment size
                    temp_solution = copy.deepcopy(solution)
                    node_sequence = [s['node'] for s in temp_solution[route_idx]['route']]
                    pos_i, pos_j = customer_positions[i], customer_positions[j]
                    node_sequence[pos_i:pos_j+1] = reversed(node_sequence[pos_i:pos_j+1])
                    
                    truck = trucks[route_idx]
                    temp_solution[route_idx] = create_route_dict(
                        truck.truck_id, node_sequence, graph, truck, service_time
                    )
                    
                    is_feas, _ = feasability(graph, trucks, temp_solution)
                    if is_feas:
                        neighbors.append((temp_solution, "reverse", f"reverse_r{route_idx}"))
                        if max_neighbors and len(neighbors) >= max_neighbors:
                            return neighbors
    
    return neighbors

# ============================================================================
# SINGLE TABU SEARCH RUN
# ============================================================================

def tabu_search(graph, trucks, initial_solution=None, max_iterations=500, 
                tabu_tenure=20, service_time=0.5, verbose=False):
    """
    Single run of tabu search algorithm.
    
    Args:
        graph: Graph object
        trucks: List of Truck objects
        initial_solution: Starting solution (None = generate new)
        max_iterations: Number of iterations
        tabu_tenure: How long a move stays tabu
        service_time: Service time at each customer
        verbose: Print progress
    
    Returns:
        (best_solution, best_score, iterations_used)
    """
    # Generate or use provided initial solution
    if initial_solution is None:
        current_solution = generate_feasible_initial_solution(graph, trucks, service_time)
    else:
        current_solution = copy.deepcopy(initial_solution)
    
    # Verify feasibility
    is_feasible, msg = feasability(graph, trucks, current_solution)
    if not is_feasible:
        raise RuntimeError(f"Initial solution infeasible: {msg}")
    
    # Initialize best
    best_solution = copy.deepcopy(current_solution)
    best_score = evaluation(graph, trucks, best_solution, service_time)
    current_score = best_score
    
    # Tabu list: stores (move_hash, iteration_when_made_tabu)
    tabu_list = {}
    
    # Statistics
    improvements = 0
    iterations_since_improvement = 0
    tabu_overrides = 0
    
    # Tabu search loop
    for iteration in range(max_iterations):
        # Clean up old tabu entries
        tabu_list = {k: v for k, v in tabu_list.items() if iteration - v < tabu_tenure}
        
        # Generate neighbors (limit for speed)
        neighbors = get_all_neighbors(current_solution, graph, trucks, service_time, max_neighbors=100)
        
        if not neighbors:
            if verbose:
                print(f"  [TS] Iteration {iteration+1}: No neighbors found, stopping")
            break
        
        # Find best non-tabu neighbor
        best_neighbor = None
        best_neighbor_score = float('inf')
        
        for neighbor_solution, move_type, move_details in neighbors:
            neighbor_score = evaluation(graph, trucks, neighbor_solution, service_time)
            neighbor_hash = get_solution_hash(neighbor_solution)
            
            is_tabu = neighbor_hash in tabu_list
            
            # Aspiration criterion: accept tabu move if better than best ever
            if is_tabu and neighbor_score < best_score:
                is_tabu = False
                tabu_overrides += 1
            
            # Track best non-tabu neighbor
            if not is_tabu and neighbor_score < best_neighbor_score:
                best_neighbor = neighbor_solution
                best_neighbor_score = neighbor_score
        
        if best_neighbor is None:
            if verbose:
                print(f"  [TS] Iteration {iteration+1}: All neighbors tabu, stopping")
            break
        
        # Move to best neighbor
        current_solution = best_neighbor
        current_score = best_neighbor_score
        
        # Add to tabu list
        move_hash = get_solution_hash(current_solution)
        tabu_list[move_hash] = iteration
        
        # Update best if improved
        if current_score < best_score:
            best_solution = copy.deepcopy(current_solution)
            best_score = current_score
            improvements += 1
            iterations_since_improvement = 0
            
            if verbose:
                print(f"  [TS] Iteration {iteration+1}: New best = {best_score:.2f}")
        else:
            iterations_since_improvement += 1
        
        # Early stopping if no improvement
        if iterations_since_improvement > max_iterations // 4:
            if verbose:
                print(f"  [TS] Stopping early: no improvement for {iterations_since_improvement} iterations")
            break
    
    return best_solution, best_score, iteration + 1


# ============================================================================
# MULTI-START TABU SEARCH
# ============================================================================

def multi_start_tabu_search(graph, trucks, num_starts=5, iterations_per_start=200,
                            tabu_tenure=20, service_time=0.5, verbose=True,
                            time_limit=None):
    """
    Multi-start tabu search: Run tabu search multiple times with different
    initial solutions and return the best result.
    
    Args:
        graph: Graph object
        trucks: List of Truck objects
        num_starts: Number of different starting points
        iterations_per_start: Iterations for each tabu search run
        tabu_tenure: Tabu list tenure
        service_time: Service time at each customer
        verbose: Print progress
        time_limit: Maximum time in seconds (None = no limit)
    
    Returns:
        (best_solution, best_score, statistics_dict)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MULTI-START TABU SEARCH")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Number of starts: {num_starts}")
        print(f"  Iterations per start: {iterations_per_start}")
        print(f"  Tabu tenure: {tabu_tenure}")
        print(f"  Time limit: {time_limit if time_limit else 'None'}")
    
    start_time = time.time()
    
    # Track overall best
    global_best_solution = None
    global_best_score = float('inf')
    
    # Statistics
    all_scores = []
    all_iterations = []
    improvements_per_start = []
    
    # Run multiple tabu searches
    for start_idx in range(num_starts):
        # Check time limit
        if time_limit and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"\n⏱️  Time limit reached after {start_idx} starts")
            break
        
        if verbose:
            print(f"\n--- Start {start_idx + 1}/{num_starts} ---")
        
        # Generate different initial solution for each start
        # Add randomness by shuffling customer selection order
        if start_idx == 0:
            # First start: use standard nearest neighbor
            initial_solution = generate_feasible_initial_solution(graph, trucks, service_time)
        else:
            # Subsequent starts: add randomness
            # Simple approach: generate with randomized nearest neighbor
            initial_solution = generate_feasible_initial_solution(graph, trucks, service_time)
            
            # Apply a few random mutations to diversify
            for _ in range(r.randint(5, 15)):
                initial_solution = apply_random_mutation(initial_solution, graph, trucks, service_time)
        
        initial_score = evaluation(graph, trucks, initial_solution, service_time)
        
        if verbose:
            print(f"  Initial score: {initial_score:.2f}")
        
        # Run tabu search from this starting point
        solution, score, iterations_used = tabu_search(
            graph, trucks,
            initial_solution=initial_solution,
            max_iterations=iterations_per_start,
            tabu_tenure=tabu_tenure,
            service_time=service_time,
            verbose=verbose
        )
        
        # Track statistics
        all_scores.append(score)
        all_iterations.append(iterations_used)
        improvement = initial_score - score
        improvements_per_start.append(improvement)
        
        if verbose:
            print(f"  Final score: {score:.2f} (improvement: {improvement:.2f})")
        
        # Update global best
        if score < global_best_score:
            global_best_solution = copy.deepcopy(solution)
            global_best_score = score
            
            if verbose:
                print(f"  ⭐ NEW GLOBAL BEST: {global_best_score:.2f}")
    
    elapsed_time = time.time() - start_time
    
    # Compile statistics
    statistics = {
        'total_starts': len(all_scores),
        'best_score': global_best_score,
        'worst_score': max(all_scores) if all_scores else 0,
        'average_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'std_score': (sum((s - sum(all_scores)/len(all_scores))**2 for s in all_scores) / len(all_scores))**0.5 if all_scores else 0,
        'total_iterations': sum(all_iterations),
        'average_iterations': sum(all_iterations) / len(all_iterations) if all_iterations else 0,
        'total_time': elapsed_time,
        'improvements': improvements_per_start
    }
    
    # Final report
    if verbose:
        print("\n" + "=" * 70)
        print("MULTI-START TABU SEARCH RESULTS")
        print("=" * 70)
        print(f"Global best score: {global_best_score:.2f}")
        print(f"Total starts completed: {statistics['total_starts']}")
        print(f"Score range: {statistics['worst_score']:.2f} - {global_best_score:.2f}")
        print(f"Average score: {statistics['average_score']:.2f} (±{statistics['std_score']:.2f})")
        print(f"Total iterations: {statistics['total_iterations']}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per start: {elapsed_time/len(all_scores):.2f} seconds")
        
        # Show improvement distribution
        print(f"\nImprovements per start:")
        for i, imp in enumerate(improvements_per_start):
            print(f"  Start {i+1}: {imp:.2f} improvement")
    
    return global_best_solution, global_best_score, statistics


# ============================================================================
# ADAPTIVE MULTI-START (Advanced version)
# ============================================================================

def adaptive_multi_start_tabu_search(graph, trucks, time_budget=300,
                                     min_starts=3, max_starts=20,
                                     service_time=0.5, verbose=True):
    """
    Adaptive multi-start: Dynamically adjust number of starts and iterations
    based on time budget and improvement rate.
    
    Args:
        graph: Graph object
        trucks: List of Truck objects
        time_budget: Total time budget in seconds
        min_starts: Minimum number of starts
        max_starts: Maximum number of starts
        service_time: Service time at each customer
        verbose: Print progress
    
    Returns:
        (best_solution, best_score, statistics_dict)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ADAPTIVE MULTI-START TABU SEARCH")
        print("=" * 70)
        print(f"Time budget: {time_budget} seconds")
    
    start_time = time.time()
    
    # Adaptive parameters
    base_iterations = 100
    iterations_per_start = base_iterations
    tabu_tenure = 20
    
    global_best_solution = None
    global_best_score = float('inf')
    
    all_scores = []
    improvements_per_start = []
    
    start_idx = 0
    no_improvement_count = 0
    
    while start_idx < max_starts:
        elapsed = time.time() - start_time
        
        # Check if we've used up time budget
        if elapsed >= time_budget:
            if verbose:
                print(f"\n⏱️  Time budget exhausted")
            break
        
        # Stop if minimum starts done and no recent improvements
        if start_idx >= min_starts and no_improvement_count >= 3:
            if verbose:
                print(f"\n🛑 Stopping: no improvement in last 3 starts")
            break
        
        # Adjust iterations based on remaining time
        time_remaining = time_budget - elapsed
        estimated_time_per_iter = 0.5  # Rough estimate
        iterations_per_start = max(50, int(time_remaining / estimated_time_per_iter / (max_starts - start_idx)))
        
        if verbose:
            print(f"\n--- Start {start_idx + 1} ---")
            print(f"  Time remaining: {time_remaining:.1f}s")
            print(f"  Iterations: {iterations_per_start}")
        
        # Generate initial solution
        if start_idx == 0:
            initial_solution = generate_feasible_initial_solution(graph, trucks, service_time)
        else:
            initial_solution = generate_feasible_initial_solution(graph, trucks, service_time)
            for _ in range(r.randint(3, 10)):
                initial_solution = apply_random_mutation(initial_solution, graph, trucks, service_time)
        
        initial_score = evaluation(graph, trucks, initial_solution, service_time)
        
        # Run tabu search
        solution, score, _ = tabu_search(
            graph, trucks,
            initial_solution=initial_solution,
            max_iterations=iterations_per_start,
            tabu_tenure=tabu_tenure,
            service_time=service_time,
            verbose=False
        )
        
        all_scores.append(score)
        improvement = initial_score - score
        improvements_per_start.append(improvement)
        
        if verbose:
            print(f"  Result: {score:.2f} (improvement: {improvement:.2f})")
        
        # Update global best
        if score < global_best_score:
            global_best_solution = copy.deepcopy(solution)
            global_best_score = score
            no_improvement_count = 0
            
            if verbose:
                print(f"  ⭐ NEW BEST: {global_best_score:.2f}")
        else:
            no_improvement_count += 1
        
        start_idx += 1
    
    elapsed_time = time.time() - start_time
    
    statistics = {
        'total_starts': start_idx,
        'best_score': global_best_score,
        'total_time': elapsed_time,
        'improvements': improvements_per_start
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("ADAPTIVE RESULTS")
        print("=" * 70)
        print(f"Best score: {global_best_score:.2f}")
        print(f"Starts completed: {start_idx}")
        print(f"Total time: {elapsed_time:.2f}s")
    
    return global_best_solution, global_best_score, statistics

# endregion

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    # Load instance
    g = Graph()
    g.load_from_vrplib('../media/instances/A-n32-k5.vrp')
    
    # Create trucks (simplified)
    trucks = [
        Truck(truck_id=i, max_capacity=200, modifier=1.0)
        for i in range(30)
    ]
    
    # Generate initial solution
    solution = generate_feasible_initial_solution(g, trucks, service_time=0.5)
    
    # Check it's feasible
    is_feas, msg = feasability(g, trucks, solution)
    print(f"Initial solution feasible: {is_feas}")
    print(f"Message: {msg}")
    
    # Try a mutation
    new_solution = apply_random_mutation(solution, g, trucks, service_time=0.5)
    
    # Check mutated solution
    is_feas, msg = feasability(g, trucks, new_solution)
    print(f"Mutated solution feasible: {is_feas}")
    
    # Run hill climbing
    best_sol, best_score = hill_climbing(g, trucks, max_iterations=1000)
    print(f"Best score: {best_score}")
    adaptive_multi_start_tabu_search(g, trucks)
    multi_start_tabu_search(g, trucks)
