import random

from structure import *
import random as r
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Needed for custom legend
import numpy as np  # Ensure numpy is imported




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



# --- END ADD ---

OPTIMUM_COSTS = {
    'A-n32-k5': 784,
    'X-n101-k25': 27591,
    'X-n153-k22': 26524,
    'X-n513-k21': 20033,
    'X-n979-k58': 40899,
    'X-n1001-k43': 66692,
    # Add more known instances here
}


###############################################################################################
###############################################################################################
###############################################################################################
####################### PLOTTING AND ROUTE CREATION #############################################

def plot_vrp_solution(graph, solution, title="VRP Solution"):
    """
    Plots the routes from the final solution on a 2D plane.

    Args:
        graph: Your Graph object (must contain node coordinates in graph.nodes,
               assumed to be a list where index == node ID).
        solution: List of route dicts.
        title: Title of the plot.
    """
    if not graph.nodes:
        print("Error: Graph has no nodes to plot.")
        return

    plt.figure(figsize=(10, 8))

    # --- 1. Extract Coordinates ---
    coords = {}
    for node_id, node in enumerate(graph.nodes):
        if hasattr(node, 'x') and hasattr(node, 'y'):
            coords[node_id] = (node.x, node.y)
        else:
            print(f"Warning: Node {node_id} is missing coordinates (x, y). Skipping.")

    if 0 not in coords:
        print("Error: Depot (Node 0) coordinates not found.")
        return

    depot_coord = coords[0]

    # --- 2. Plot Depot and Customers ---
    plt.plot(depot_coord[0], depot_coord[1], 's', color='red', markerfacecolor='red',
             markersize=10, label='Depot (Node 0)', zorder=3)

    customer_coords = [c for nid, c in coords.items() if nid != 0]
    plt.plot([c[0] for c in customer_coords], [c[1] for c in customer_coords],
             'o', color='blue', markersize=6, alpha=0.7, label='Customer Nodes', zorder=2)

    # --- 3. Plot Routes ---

    # --- FIX 1: Correctly get the colormap object ---
    # The new API doesn't take a number as the second argument.
    cmap = plt.colormaps.get_cmap('viridis')
    # Avoid division by zero if solution list is empty or has 1 route
    num_routes = max(1, len(solution))

    for i, route_dict in enumerate(solution):
        route = route_dict.get('route', [])
        truck_id = route_dict.get('truck_id', i)

        if len(route) < 2:
            continue

        route_nodes = [stop['node'] for stop in route]

        try:
            route_coords = [coords[node_id] for node_id in route_nodes]
        except KeyError as e:
            print(f"Error: Node ID {e} in truck {truck_id} route not found in coordinates dictionary. Skipping route.")
            continue

        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        # --- FIX 2: Use the colormap with a normalized value (0.0 to 1.0) ---
        # We calculate a fraction for the current route index 'i'
        color_val = i / (num_routes - 1) if num_routes > 1 else 0.5

        plt.plot(xs, ys, color=cmap(color_val), linestyle='-', linewidth=1.5, alpha=0.7,
                 label=f'Truck {truck_id} Route', zorder=1)

    # --- 4. Final Formatting ---
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l.startswith('Truck') and 'Truck Route' not in unique_labels:
            unique_labels['Truck Route'] = h
        elif not l.startswith('Truck'):
            unique_labels[l] = h

    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize='small')

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.show()

def plot_cost_improvement(scores_history, title="Cost Improvement Over Iterations"):
    """
    Plots the cost (score) improvement over the course of the algorithm run.

    Args:
        scores_history: List of scores recorded at each step where an improvement was made.
        title: Title for the plot.
    """
    if not scores_history:
        print("No score history provided to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Generate iteration index for each recorded score
    iterations = range(1, len(scores_history) + 1)

    plt.plot(iterations, scores_history, marker='o', linestyle='-', color='teal', alpha=0.8)

    plt.title(title)
    plt.xlabel("Improvement Step Index")
    plt.ylabel("Maximum Route Time (Score)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



def plot_solution_gantt(solution, title="VRP Route Schedule (Gantt Chart)"):
    """
    Creates a Gantt chart visualizing the schedule for each truck's route,
    showing travel, service times, and node visit sequence.

    Args:
        solution: The solution list, where each item is a route dictionary.
        title: The title for the chart.
    """
    if not solution:
        print("Cannot plot Gantt chart: No solution provided.")
        return

    # --- 1. Setup Figure and Colors ---
    fig_height = max(5, 0.5 * len(solution) + 2)
    plt.figure(figsize=(15, fig_height))

    colors = {
        'Travel': 'green',
        'Service': 'yellow'
    }
    color_green = 'green'
    color_yellow = 'yellow'
    color_black = 'black'

    # --- 2. Process Data (Sort and get max_time) ---
    try:
        # Sort routes by truck_id for a consistent plot
        sorted_solution = sorted(solution, key=lambda r: r.get('truck_id', 0))
    except TypeError:
        print("Warning: Could not sort routes by truck_id. Plotting in given order.")
        sorted_solution = solution

    num_routes = len(sorted_solution)
    y_labels = []

    # --- First Pass: Get max_time for x-axis scaling ---
    max_time = 0
    for route_dict in sorted_solution:
        route = route_dict.get('route', [])
        if route:
            max_time = max(max_time, route[-1]['arrival'])

    if max_time == 0: max_time = 1  # Avoid division by zero

    # --- 3. Plot Bars and Text ---
    for i, route_dict in enumerate(sorted_solution):
        route = route_dict.get('route', [])
        truck_id = route_dict.get('truck_id', i)
        y_labels.append(f"Route (Truck {truck_id})")

        if not route or len(route) < 2:
            continue

        # --- NEW: Add initial Node 0 label ---
        # We place it at time 0, aligned to the right, so it appears *before* the bar starts
        plt.text(x=0, y=i, s=str(route[0]['node']),
                 va='center', ha='right', fontsize=9, fontweight='bold', color=color_black,
                 bbox=dict(facecolor='white', alpha=0.5, pad=0.1, boxstyle='round,pad=0.2'))

        # Iterate through segments (from stop i to stop i+1)
        for j in range(len(route) - 1):
            current_stop = route[j]
            next_stop = route[j + 1]

            # 1. Plot Travel Time (Green)
            travel_start = current_stop['departure']
            travel_duration = next_stop['arrival'] - travel_start

            if travel_duration > 1e-6:  # Avoid plotting zero-duration bars
                plt.barh(i, travel_duration, left=travel_start, color=colors['Travel'],
                         edgecolor=color_yellow, height=0.7)

            # 2. Plot Service Time (Yellow)
            if next_stop['node'] != 0:
                service_start = next_stop['arrival']
                service_duration = next_stop['service']

                if service_duration > 1e-6:
                    plt.barh(i, service_duration, left=service_start, color=colors['Service'],
                             edgecolor=color_yellow, height=0.7)

            # --- NEW: Add node label at the end of this segment ---
            # We plot the label for 'next_stop' at its 'departure' time.
            # ha='right' places the text *before* this x-coordinate,
            # so it sits at the end of the bar segment.
            node_id = next_stop['node']
            text_x = next_stop['departure']

            plt.text(x=text_x, y=i, s=str(node_id),
                     va='center', ha='right', fontsize=9, fontweight='bold', color=color_black,
                     bbox=dict(facecolor='white', alpha=0.5, pad=0.1, boxstyle='round,pad=0.2'))

    # --- 4. Formatting ---
    plt.xlabel("Time")
    plt.ylabel("Route")
    plt.title(title, fontsize=14, fontweight='bold')

    # Set y-axis labels
    plt.yticks(range(num_routes), y_labels)
    plt.gca().invert_yaxis()  # Put Route 0 at the top

    # Adjust x-limit slightly to give labels at start/end space
    plt.xlim(-max_time * 0.02, max_time * 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # --- 5. Create Custom Legend ---
    legend_patches = [
        mpatches.Patch(color=colors['Travel'], label='Travel Time'),
        mpatches.Patch(color=colors['Service'], label='Service Time')
    ]
    plt.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    plt.show()
def show_statistics(scores, algorithm_name):
    """
    Calculates and displays statistical metrics for a list of scores.

    Args:
        scores: List of numerical scores (e.g., final scores from multi-start runs).
        algorithm_name: Name of the algorithm (for context).
    """
    if not scores:
        print(f"No data available for {algorithm_name} statistics.")
        return

    print(f"\n📈 Statistics for {algorithm_name}:")
    print("=" * 40)

    arr = np.array(scores)

    print(f"  Total Runs: {len(arr)}")
    print(f"  Best Score (Min): {np.min(arr):.2f}")
    print(f"  Worst Score (Max): {np.max(arr):.2f}")
    print(f"  Mean Score: {np.mean(arr):.2f}")
    print(f"  Median Score: {np.median(arr):.2f}")
    print(f"  Standard Deviation (SD): {np.std(arr):.2f}")
    print(f"  Range: {np.ptp(arr):.2f}")
    print("=" * 40)



def show_benchmark(instance_name, benchmark_results):
    """
    Displays a comparison table of results from multiple algorithms,
    benchmarked against the known optimum cost for that instance.

    Args:
        instance_name: The name of the instance (e.g., 'A-n32-k5').
        benchmark_results: Dictionary mapping algorithm names to (best_score, time).
            Example: {'Hill Climbing': (812, 45.2), 'Multi-Start TS': (790, 182.7)}
    """

    # Try to get the optimum cost from our dictionary
    optimum = OPTIMUM_COSTS.get(instance_name)

    print("\n\n🏆 Algorithm Benchmark Table")
    print(f"Instance: {instance_name}")

    # --- Print Header ---
    if optimum is not None:
        print(f"Known Optimum Cost: {optimum}")
        print("=" * 80)
        print(f"{'Algorithm':<25} | {'Optimum Cost':<12} | {'Your Cost':<12} | {'Gap (%)':<10} | {'Time (s)':<10}")
        print("-" * 80)
    else:
        # Fallback if we don't know the optimum
        print(f"Known Optimum Cost: N/A")
        print("=" * 60)
        print(f"{'Algorithm':<25} | {'Your Cost':<15} | {'Time (s)':<15}")
        print("-" * 60)

    # --- Print Results for Each Algorithm ---
    total_gap = 0
    valid_gaps = 0

    for algo, (score, time_sec) in benchmark_results.items():
        if optimum is not None:
            # Calculate Gap: ((Your Cost / Optimum) - 1) * 100
            gap = ((score / optimum) - 1) * 100
            total_gap += gap
            valid_gaps += 1
            print(f"{algo:<25} | {optimum:<12} | {score:<12.2f} | {gap:<10.2f}% | {time_sec:<10.2f}")
        else:
            # Fallback if optimum is not known
            print(f"{algo:<25} | {score:<15.2f} | {time_sec:<15.2f}")

    # --- Print Footer & Success Criteria ---
    if optimum is not None:
        print("=" * 80)
        if valid_gaps > 0:
            avg_gap = total_gap / valid_gaps
            print(f"Average Gap across your algorithms: {avg_gap:.2f}%")

            # Check success criteria from your image
            if avg_gap < 7:
                print("Success Criteria: Met (Average Gap < 7%) ✅")
            else:
                print("Success Criteria: Not Met (Average Gap >= 7%) ⚠️")
    else:
        print("=" * 60)


def create_route_dict(truck_id, node_sequence, graph, truck, service_time=0.5):
    """
    Create a complete route dictionary with all timing information.
    Compatible with Graph class structure.

    Args:
        truck_id: ID of the truck
        node_sequence: List of node IDs [0, 3, 7, 5, 0]
        graph: Graph object (with graph.graph and graph.nodes)
        truck: Truck object
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
    """
    Evaluate solution quality (Objective: minimize TOTAL cost/distance).
    This function is now aligned with standard VRP benchmarks.
    """
    if not solution:
        return float('inf')

    total_cost = 0

    for route_dict in solution:
        truck = trucks[route_dict['truck_id']]
        route = route_dict['route']

        if len(route) < 2:
            continue

        # Sum the cost for this single route
        route_cost = 0
        for i in range(len(route) - 1):
            current_stop = route[i]
            next_stop = route[i + 1]

            current_node = current_stop['node']
            next_node = next_stop['node']

            # We still need the departure time in case the edge function
            # is time-dependent (TDVRP)
            departure_time = current_stop['departure']

            edge_func = graph.graph[current_node][next_node]
            if edge_func is None:
                raise ValueError(f"No edge {current_node} -> {next_node}")

            # --- Key Change ---
            # Calculate travel cost (distance) ONLY.
            # We DO NOT add service_time here, as benchmarks are pure distance/cost.
            travel_cost = edge_func(departure_time) * truck.modifier

            route_cost += travel_cost

        # Add this route's cost to the grand total
        total_cost += route_cost

    return total_cost


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

    # Track deliveries per node
    deliveries = [0] * n_nodes

    for truck_idx, route_dict in enumerate(solution):
        # FIX: route_dict is the dict, route is the list inside it
        if not route_dict or 'route' not in route_dict:
            continue

        route = route_dict['route']  # <-- This is the actual route list
        truck_id = route_dict['truck_id']
        truck = trucks[truck_id]

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
        total_load = sum(delivers[1:-1])  # Exclude depot visits
        if total_load > truck.max_capacity:
            return False, f"Truck {truck_idx}: Capacity exceeded ({total_load}/{truck.max_capacity})"

        # Track deliveries
        for node, qty in zip(nodes, delivers):
            deliveries[node] += qty

    # Check 5: All demands satisfied
    for node_idx in range(1, n_nodes):  # Skip depot
        node = graph.nodes[node_idx]

        # Handle both int and dict demand
        if isinstance(node.demand, int):
            demand = node.demand
        elif isinstance(node.demand, dict):
            demand = sum(node.demand.values())
        else:
            demand = 0

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
            # --- Find FIRST valid customer from shuffled list ---
            customer_list = list(customers)
            r.shuffle(customer_list)

            customer_to_add = None

            # Find the FIRST customer in the random list that fits
            for customer in customer_list:
                node = graph.nodes[customer]
                demand = node.demand if isinstance(node.demand, int) else sum(node.demand.values())

                # Check capacity
                if current_load + demand <= truck.max_capacity:
                    customer_to_add = customer  # Found one!
                    break  # Stop searching

            if customer_to_add is None:
                break  # No customer in the random list fits
            # --- END NEW FIX ---

            # Add to route
            node = graph.nodes[customer_to_add]  # Use the new variable
            demand = node.demand if isinstance(node.demand, int) else sum(node.demand.values())

            # Calculate times
            travel_time = graph.graph[current_node][customer_to_add](current_time) * truck.modifier
            current_time += travel_time + service_time
            current_load += demand

            route.append(customer_to_add)
            customers.remove(customer_to_add)
            current_node = customer_to_add


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
    dest_truck = trucks[new_solution[dest_idx]['truck_id']]
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
    source_truck = trucks[new_solution[source_idx]['truck_id']]
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
    node_sequence[i:j + 1] = reversed(node_sequence[i:j + 1])

    # Recalculate route using create_route_dict
    truck = trucks[new_solution[route_idx]['truck_id']]
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


# hillclimbing
def hill_climbing(graph, trucks, max_iterations=1000):
    """
    Hill Climbing for VRP.

    Returns:
        best_solution, best_score, score_history
    """
    current_solution = generate_feasible_initial_solution(graph, trucks)
    best_solution = current_solution
    best_score = evaluation(graph, trucks, best_solution)

    score_history = [best_score]  # Start history with initial score

    for iteration in range(max_iterations):
        # Step 2: Generate a neighbor using a random mutation
        neighbor_solution = apply_random_mutation(current_solution, graph, trucks)

        # Step 3: Evaluate the neighbor
        neighbor_score = evaluation(graph, trucks, neighbor_solution)

        # Step 4: If neighbor is better, move to neighbor
        if neighbor_score < best_score:
            best_solution = neighbor_solution
            best_score = neighbor_score
            current_solution = neighbor_solution
            score_history.append(best_score)  # Record improvement
            # print(f"Iteration {iteration+1}: Improved score = {best_score:.2f}") # Keep for console output
        else:
            current_solution = current_solution

    return best_solution, best_score, score_history  # Return history


"""
Multi-Start Tabu Search for VRP
Combines multiple tabu search runs with different starting solutions
"""


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
                    node_sequence[pos_i:pos_j + 1] = reversed(node_sequence[pos_i:pos_j + 1])

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
        (best_solution, best_score, iterations_used, score_history)
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

    # --- MODIFICATIONS ---
    # 1. Initialize for score history plotting
    score_history = [best_score]
    # 2. Initialize to fix UnboundLocalError
    iterations_since_improvement = 0
    # ---

    # Statistics
    improvements = 0
    tabu_overrides = 0

    # Tabu search loop
    for iteration in range(max_iterations):
        # Clean up old tabu entries
        tabu_list = {k: v for k, v in tabu_list.items() if iteration - v < tabu_tenure}

        # Generate neighbors (limit for speed)
        neighbors = get_all_neighbors(current_solution, graph, trucks, service_time, max_neighbors=100)

        if not neighbors:
            if verbose:
                print(f"  [TS] Iteration {iteration + 1}: No neighbors found, stopping")
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
                print(f"  [TS] Iteration {iteration + 1}: All neighbors tabu, stopping")
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

            # Add to history for plotting
            score_history.append(best_score)

            if verbose:
                print(f"  [TS] Iteration {iteration + 1}: New best = {best_score:.2f}")
        else:
            # This line is now safe
            iterations_since_improvement += 1

        # # Early stopping if no improvement
        # if iterations_since_improvement > max_iterations // 4:
        #     if verbose:
        #         print(f"  [TS] Stopping early: no improvement for {iterations_since_improvement} iterations")
        #     break

    # Return history for plotting
    return best_solution, best_score, iteration + 1, score_history


# ============================================================================
# MULTI-START TABU SEARCH
# ============================================================================

def multi_start_tabu_search(graph, trucks, num_starts=5, iterations_per_start=200,
                            tabu_tenure=20, service_time=0.5, verbose=True,
                            time_limit=None):
    """
    Multi-start tabu search: Run tabu search multiple times with different
    initial solutions and return the best result.

    Returns:
        (best_solution, best_score, statistics_dict, best_history)
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
    global_best_history = []  #  To store history of the best run

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
        if start_idx == 0:
            initial_solution = generate_feasible_initial_solution(graph, trucks, service_time)
        else:
            initial_solution = generate_feasible_initial_solution(graph, trucks, service_time)
            for _ in range(r.randint(5, 15)):
                initial_solution = apply_random_mutation(initial_solution, graph, trucks, service_time)

        initial_score = evaluation(graph, trucks, initial_solution, service_time)

        if verbose:
            print(f"  Initial score: {initial_score:.2f}")

        # --- MODIFIED CALL: Now expecting 4 return values ---
        solution, score, iterations_used, score_history = tabu_search(
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
            global_best_history = score_history  # <-- ADDED: Save the history of this new best run

            if verbose:
                print(f"  ⭐ NEW GLOBAL BEST: {global_best_score:.2f}")

    elapsed_time = time.time() - start_time

    # Compile statistics
    statistics = {
        'total_starts': len(all_scores),
        'best_score': global_best_score,
        'worst_score': max(all_scores) if all_scores else 0,
        'average_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'std_score': (sum((s - sum(all_scores) / len(all_scores)) ** 2 for s in all_scores) / len(
            all_scores)) ** 0.5 if all_scores else 0,
        'total_iterations': sum(all_iterations),
        'average_iterations': sum(all_iterations) / len(all_iterations) if all_iterations else 0,
        'total_time': elapsed_time,
        'improvements': improvements_per_start,
        'all_scores': all_scores  # <-- ADDED: To ensure stats printing works
    }

    # ... (Final verbose report remains the same) ...

    # --- MODIFIED RETURN: Now returning 4 values ---
    return global_best_solution, global_best_score, statistics, global_best_history


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
    global_best_history = []

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
        solution, score, _, score_history = tabu_search(
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
            global_best_history = score_history
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

    return global_best_solution, global_best_score, statistics, global_best_history


#########################################################################################################
#########################################################################################################
#########################################################################################################

if __name__ == "__main__":
    # random.seed(42)
    # --- Setup ---
    # Define the instance file path here
    instance_path = '../media/instances/X-n101-k25.vrp'
    # instance_path = '../media/instances/A-n32-k5.vrp'

    # Extract instance name from path (e.g., "A-n32-k5")
    instance_name = instance_path.split('/')[-1].split('.')[0]

    # Load instance
    g = Graph()
    info = g.load_from_vrplib(instance_path)

    # Create trucks (use capacity from file if available)
    truck_capacity = info.get('vehicle_capacity', 200)
    num_nodes = info.get('num_nodes')
    trucks = [
        Truck(truck_id=i, max_capacity=truck_capacity, modifier=1.0)
        for i in range(num_nodes - 1)  # Over-provisioned fleet
    ]

    # Storage for benchmark results: {Algorithm: (Best Score, Time)}
    benchmark_results = {}

    # ----------------------------------------------------------------
    # 1. Hill Climbing (HC) Run
    print("\n--- 1. Running Hill Climbing ---")
    hc_start_time = time.time()
    best_sol_hc, best_score_hc, hist_hc = hill_climbing(g, trucks, max_iterations=10000)
    hc_time = time.time() - hc_start_time

    print(f"\n✅ HC Best Score: {best_score_hc:.2f} (Time: {hc_time:.2f}s)")
    benchmark_results['Hill Climbing (25k)'] = (best_score_hc, hc_time)

    # Plotting and Analysis for HC
    plot_vrp_solution(g, best_sol_hc, title=f"Hill Climbing Solution (Paths)")
    plot_cost_improvement(hist_hc, title="Hill Climbing: Cost Improvement History")
    plot_solution_gantt(best_sol_hc, title=f"Hill Climbing Schedule Time (Schedule)")

    # ----------------------------------------------------------------
    # 2. Multi-Start Tabu Search (MSTS) Run
    print("\n--- 2. Running Multi-Start Tabu Search ---")
    msts_start_time = time.time()

    # --- MODIFIED CALL: Now expecting 4 return values ---
    global_best_sol, global_best_score, stats, hist_msts = multi_start_tabu_search(
        g, trucks,
        num_starts=10,
        iterations_per_start=2000,
        verbose=False
    )
    msts_time = time.time() - msts_start_time

    print(f"\n✅ MSTS Global Best Score: {global_best_score:.2f} (Time: {msts_time:.2f}s)")
    benchmark_results['Multi-Start TS (10x2k)'] = (global_best_score, msts_time)

    # --- ADDED: Plotting for Tabu Search ---
    plot_vrp_solution(g, global_best_sol, title=f"Multi-Start TS Solution (Paths)")
    plot_cost_improvement(hist_msts, title="Multi-Start TS: Best Run Cost Improvement")
    plot_solution_gantt(global_best_sol, title=f"Multi-Start TS Solution Time (Schedule)")

    # Display Statistics
    if stats['total_starts'] > 0:
        print("\n📈 MSTS Statistical Summary:")
        print(f"  Best Score: {stats['best_score']:.2f}")
        print(f"  Average Score: {stats['average_score']:.2f}")
        print(f"  Standard Deviation: {stats['std_score']:.2f}")

    # 3. Adaptive Multi-Start Tabu Search (AMSTS) Run
    print("\n--- 3. Running Adaptive Multi-Start Tabu Search ---")
    print(f"    (This will run for 180 seconds to match the benchmark...)")
    amsts_start_time = time.time()

    # Note: To plot for Adaptive, you'd need to modify it the same way as MSTS
    best_sol_amsts, best_score_amsts, stats_amsts, hist_amsts = adaptive_multi_start_tabu_search(
        g, trucks,
        time_budget=180,  # Run for 180 seconds (3 minutes)
        verbose=False
    )
    amsts_time = time.time() - amsts_start_time
    plot_vrp_solution(g, best_sol_amsts, title=f"Adaptive Multi-Start TS Solution (Paths)")
    plot_cost_improvement(hist_amsts, title="Adaptive Multi-Start TS: Best Run Cost Improvement")
    plot_solution_gantt(best_sol_amsts, title=f"Adaptive Multi-Start TS Solution Time (Schedule)")

    print(f"\n✅ AMSTS Global Best Score: {best_score_amsts:.2f} (Time: {amsts_time:.2f}s)")
    benchmark_results['Adaptive TS (180s)'] = (best_score_amsts, amsts_time)

    # ----------------------------------------------------------------
    # 4. Final Benchmark
    # ----------------------------------------------------------------
    show_benchmark(instance_name, benchmark_results)