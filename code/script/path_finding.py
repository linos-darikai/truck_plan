import structure
import random as r
import copy

#time printer
def print_hour(hour):
    """
    Convert a fractional hour into hours and minutes and print it.
    """
    print(f"{int(hour//1)}h {int(hour % 1* 60)}min")
    return


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


def calculate_path_time(graph, truck, products, path):
    """
    Calculate the total time (travel + delivery) for a single truck's path.

    Args:
        graph (list): The graph matrix with edge functions.
        truck (Truck): The truck object assigned to this path.
        products (dict): The dictionary of all products.
        path (list): A list of tuples:
            (node_index, products_delivered_dict, leaving_time)
    
    Returns:
        float: The total time for the path in hours.
    """
    total_path_time = 0
    # No travel time if path is empty or just has the starting node
    if not path or len(path) < 2:
        return 0 

    current_node = path[0][0]

    # Iterate from the first stop onwards
    for stop in path[1:]:
        next_node, delivered, leaving_time = stop
        
        # Check if edge exists
        edge_function = graph[current_node][next_node]
        if edge_function is None:
            print(f"Warning: No edge from {current_node} to {next_node}. Skipping segment.")
            continue # Skip this segment

        # 1. Calculate Travel time
        travel_value = edge_function(leaving_time)
        
        # 2. Calculate Delivery time
        delivery_value = sum(qty * products[prod].delivery_time for prod, qty in delivered.items())
        
        # 3. Apply truck modifier and add to total time
        total_path_time += truck.modifier * travel_value + delivery_value
        
        # Update current node for next iteration
        current_node = next_node

    return total_path_time


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
        truck = trucks[i]
        # Use the function to calculate total time (travel + delivery)
        path_time = calculate_path_time(graph, truck, products, path)
        time_each_path.append(path_time)

    # Return 0 if there are no paths, otherwise return the max time
    return max(time_each_path) if time_each_path else 0

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
    for node in range(len(graph.nodes)):
        for pname, needed_qty in node.demand:
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
def cycle_mutation(graph, truckId, products, solution):
    """
    Transform the cycle of a path of a truck...
    1. Transforming to a possi
    """
    return
#change the number of delivery object of 1 node
def _delivery_mutation(graph, trucks, products, current_solution):
    """
    Generates a "neighbor" solution by randomly changing the delivery quantity
    of one product at one stop for one truck.
    """
    # 1. Create a deep copy to avoid changing the original
    # Checks 
    # - Truck exists
    # - Stop exists (not depot)
    # - Product is allowed for truck
    # - Quantity does not go negative
    # If any check fails, return the copy unchanged
    # Otherwise, apply the mutation and return the new solution
    # Randomly +1 or -1 to the quantity
    # If -1 would go negative, set to 0 instead
    # Return the new solution
    # If any error occurs, return the copy unchanged
    
    new_solution = copy.deepcopy(current_solution)

    try:
        # 2. Pick a random truck
        num_trucks = len(new_solution)
        if num_trucks == 0:
            return new_solution # Nothing to mutate
        
        truck_idx = r.randint(0, num_trucks - 1)
        path = new_solution[truck_idx]
        truck = trucks[truck_idx]

        # 3. Pick a random stop (must have at least one stop after depot)
        if len(path) < 2:
            return new_solution # Can't mutate an empty or depot-only path
        
        stop_idx = r.randint(1, len(path) - 1) # Start from 1 to skip depot

        # 4. Pick a random product the truck is allowed to carry
        if not truck.allowed_products:
             # If no restrictions, pick any product
             if not products: return new_solution # No products to deliver
             product_to_mutate = r.choice(list(products.keys()))
        else:
             product_to_mutate = r.choice(list(truck.allowed_products))

        # 5. Get the stop data
        (node, delivered_dict, time) = path[stop_idx]

        # 6. Pick a change direction (+1 or -1)
        change = r.choice([-1, 1])
        
        # 7. Apply the change
        current_qty = delivered_dict.get(product_to_mutate, 0)
        new_qty = current_qty + change

        # 8. Ensure quantity is not negative
        if new_qty < 0:
            new_qty = 0
            
        # 9. Update the delivery dictionary
        delivered_dict[product_to_mutate] = new_qty
        
        # 10. Put the mutated stop back into the path
        path[stop_idx] = (node, delivered_dict, time)
        
    except IndexError:
        # Catch potential errors from empty lists and just return the copy
        pass
    except Exception as e:
        print(f"Error in delivery_mutation: {e}")
        
    return new_solution

def _transfer_delivery_mutation(graph, trucks, products, current_solution):
    """
    Generates a "neighbor" solution by transferring a single delivery 
    (one product at one stop) from one truck to another.
    
    This only works if the second truck already visits that same node
    and is allowed to carry the product.
    """
    # 1. Create a deep copy to avoid changing the original
    # Things to check:
    # - At least 2 trucks exist
    # - Truck A has at least one stop with deliveries
    # - Truck B visits that same node
    # - Truck B is allowed to carry the product
    # If any check fails, return the copy unchanged
    # Otherwise, perform the transfer and return the new solution
    # If any error occurs, return the copy unchanged
    
    new_solution = copy.deepcopy(current_solution)
    
    try:
        num_trucks = len(trucks)
        # 2. Need at least 2 trucks to transfer
        if num_trucks < 2:
            return new_solution 

        # 3. Select two different trucks
        truck_A_idx, truck_B_idx = r.sample(range(num_trucks), 2)
        truck_B = trucks[truck_B_idx]
        path_A = new_solution[truck_A_idx]
        path_B = new_solution[truck_B_idx]

        # 4. Find a delivery to move from Truck A
        if len(path_A) < 2:
            return new_solution # Truck A has no stops
        
        stop_A_idx = r.randint(1, len(path_A) - 1)
        (node_to_move_from, delivered_A, time_A) = path_A[stop_A_idx]

        if not delivered_A:
            return new_solution # This stop has no deliveries to move
            
        # 5. Pick a product from that delivery
        product_to_move = r.choice(list(delivered_A.keys()))
        qty_to_move = delivered_A[product_to_move]

        # 6. Check if Truck B can even carry this product
        if truck_B.allowed_products and product_to_move not in truck_B.allowed_products:
            return new_solution # "Fails" mutation, Truck B can't carry this

        # 7. Find if Truck B already visits this node
        target_stop_B_idx = -1
        for i in range(1, len(path_B)):
            # path_B[i][0] is the node_index for the i-th stop
            if path_B[i][0] == node_to_move_from:
                target_stop_B_idx = i
                break

        # 8. If not, this mutation "fails" (we don't add the stop)
        if target_stop_B_idx == -1:
            return new_solution # Truck B doesn't visit this node

        # 9. Perform the transfer!
        
        # Remove from A
        del delivered_A[product_to_move]
        # Update stop A in the path
        path_A[stop_A_idx] = (node_to_move_from, delivered_A, time_A)

        # Add to B
        (node_B, delivered_B, time_B) = path_B[target_stop_B_idx]
        delivered_B[product_to_move] = delivered_B.get(product_to_move, 0) + qty_to_move
        # Update stop B in the path
        path_B[target_stop_B_idx] = (node_B, delivered_B, time_B)

    except IndexError:
        # Catch potential errors from empty lists
        pass
    except Exception as e:
        print(f"Error in transfer_delivery_mutation: {e}")
        
    return new_solution
    
#change the leaving time of 1 node
def leaving_time_mutation(graph, trucks, products, solution):
    return

#global mutation
def random_possible_mutation(graph, trucks, products, current_solution):
    """
    Selects a random mutation type and applies it to create a new
    "neighbor" solution.
    """
    
    # List of all available mutation functions
    # Add more mutations to this list as they are created
    available_mutations = [
        _delivery_mutation,
        _transfer_delivery_mutation
        # cycle_mutation, # <-- Add this once it's implemented
        # leaving_time_mutation, # <-- Add this once it's implemented
    ]
    
    # 1. Choose a random mutation
    chosen_mutation = r.choice(available_mutations)
    
    # 2. Apply the chosen mutation
    new_solution = chosen_mutation(graph, trucks, products, current_solution)
    
    return new_solution
#hillpath   

#tabou