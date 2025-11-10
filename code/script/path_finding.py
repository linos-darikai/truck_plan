import structure
import random as r
import copy 

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
    total_load = 0
    for node_id in node_sequence:
        if node_id != 0:  # Skip depot
            node = graph.nodes[node_id]
            # Handle both int demand and dict demand
            if isinstance(node.demand, int):
                total_load += node.demand
            elif isinstance(node.demand, dict):
                total_load += sum(node.demand.values())
            else:
                total_load += 0
    
    for i, node_id in enumerate(node_sequence):
        node = graph.nodes[node_id]
        
        # Get demand for this node
        if node_id == 0:
            demand = 0
        elif isinstance(node.demand, int):
            demand = node.demand
        elif isinstance(node.demand, dict):
            demand = sum(node.demand.values())
        else:
            demand = 0
        
        # Service time (0 at depot, constant at customers)
        service = service_time if node_id != 0 else 0
        
        # Create stop info
        stop = {
            'node': node_id,
            'arrival': current_time,
            'service': service,
            'departure': current_time + service,
            'deliver': demand,
            'load_after': 0  # Will calculate below
        }
        
        # Calculate load after this delivery
        if i == 0:
            # At depot, load everything
            current_load = total_load
        else:
            # Deliver at customer
            current_load -= demand
        
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
    
    # Calculate max load (for capacity checking)
    max_load = max(stop['load_after'] for stop in route)
    
    return {
        'truck_id': truck_id,
        'total_load': total_load,
        'max_load': max_load,
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
###############################################################################################
###############################################################################################
###############################################################################################
####################### EVALUATION AND FEASIBILITY#############################################
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

###############################################################################################
###############################################################################################
###############################################################################################
##################################################################



def generate_feasible_initial_solution(graph, trucks, service_time=0.5):
    """
    Generate feasible initial solution using nearest neighbor.
    
    Strategy:
    - Start from depot
    - Greedily add nearest customer that fits capacity
    - Return to depot
    
    Returns:
        List of route dicts (one per truck)
    """
    n_nodes = len(graph.nodes)
    customers = set(range(1, n_nodes))  # Exclude depot
    solution = []
    
    truck_idx = 0
    
    while customers and truck_idx < len(trucks):
        truck = trucks[truck_idx]
        
        # Build route for this truck
        route = [0]  # Start at depot
        current_node = 0
        current_load = 0
        current_time = 0
        
        while customers:
            # Find nearest customer that fits
            best_customer = None
            best_distance = float('inf')
            
            for customer in customers:
                # Get demand (handle both int and dict)
                node = graph.nodes[customer]
                demand = node.demand if isinstance(node.demand, int) else sum(node.demand.values())
                
                # Check capacity
                if current_load + demand <= truck.max_capacity:
                    # Calculate distance
                    dist = graph.graph[current_node][customer](current_time)
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer
            
            if best_customer is None:
                break  # No more customers fit
            
            # Add to route
            node = graph.nodes[best_customer]
            demand = node.demand if isinstance(node.demand, int) else sum(node.demand.values())
            
            # Calculate times
            travel_time = graph.graph[current_node][best_customer](current_time) * truck.modifier
            current_time += travel_time + service_time
            current_load += demand
            
            route.append(best_customer)
            customers.remove(best_customer)
            current_node = best_customer
        
        # Return to depot
        route.append(0)
        
        # ========================================
        # SIMPLIFIED: Just use create_route_dict()
        # ========================================
        route_dict = create_route_dict(truck.truck_id, route, graph, truck, service_time)
        solution.append(route_dict)
        # ========================================
        
        truck_idx += 1
    
    if customers:
        raise RuntimeError(f"Cannot assign all customers: {len(customers)} remaining")
    
    return solution




#########################################################################
############## MUTATION #################################################





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
    
    # Handle dict demand
    if isinstance(customer_demand, dict):
        customer_demand = sum(customer_demand.values())
    
    # Select destination route
    dest_idx = r.choice([i for i in range(len(solution)) if i != source_idx])
    dest_route = new_solution[dest_idx]['route']
    dest_truck = trucks[dest_idx]
    
    # Check capacity
    current_load = new_solution[dest_idx]['total_load']
    if current_load + customer_demand > dest_truck.max_capacity:
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






####################################################################################################
####################################################################################################
####################################################################################################




#########################################################################################################
#########################################################################################################
#######################  Hill climbing and Tabu search###################################################

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
  


#########################################################################################################
#########################################################################################################
#########################################################################################################

