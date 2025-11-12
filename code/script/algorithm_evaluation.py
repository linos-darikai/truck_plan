import random
import re
import time
import copy
import statistics
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Import from your existing project files
# (This assumes structure.py and path_finding.py are in the same directory
# or your Python path)
try:
    from structure import Graph, Truck
    from path_finding import (
        evaluation,
        hill_climbing,
        tabu_search,
        multi_start_tabu_search,
        generate_feasible_initial_solution,
        OPTIMUM_COSTS
    )
except ImportError:
    print("Error: Could not import from structure.py or path_finding.py.")
    print("Please ensure these files are in the same directory.")


    # Define dummy functions to allow the script to be read,
    # but it will fail on run.
    class Graph:
        pass


    class Truck:
        pass


    def evaluation(g, t, s, **kwargs):
        return 0


    def hill_climbing(g, t, **kwargs):
        return [], 0, []


    def tabu_search(g, t, **kwargs):
        return [], 0, 0, []


    def multi_start_tabu_search(g, t, **kwargs):
        return [], 0, {}, []


    def generate_feasible_initial_solution(g, t, **kwargs):
        return []


    OPTIMUM_COSTS = {}


# --- Helper Functions ---

def load_instance(instance_path):
    """
    Loads a VRPLIB instance file and prepares the Graph, Trucks, and Optimum.
    """
    try:
        instance_name = instance_path.split('/')[-1].split('.')[0]
    except Exception:
        instance_name = "unknown"

    g = Graph()
    info = g.load_from_vrplib(instance_path)

    truck_capacity = info.get('vehicle_capacity', 200)
    num_nodes = info.get('num_nodes')

    # Try to get the number of vehicles from the instance info
    num_trucks = info.get('num_vehicles')

    if num_trucks is None:
        # Fallback: extract from name like 'A-n32-k5'
        k_match = re.search(r'-k(\d+)', instance_name)
        if k_match:
            num_trucks = int(k_match.group(1))
        else:
            # Fallback if not in name
            num_trucks = 25

            # --- FIX ---
    # The simple greedy initial solution generator isn't smart enough
    # to meet the 'k' (num_trucks) constraint on hard instances.
    # We will over-provision the fleet, just like in the original
    # path_finding.py __main__ block, to ensure it *always*
    # generates a feasible starting solution.
    num_trucks_to_provide = num_nodes - 1
    trucks = [Truck(truck_id=i, max_capacity=truck_capacity, modifier=1.0) for i in range(num_trucks_to_provide)]
    # --- END FIX ---

    optimum = OPTIMUM_COSTS.get(instance_name)

    if optimum is None:
        print(f"Warning: Optimum cost for {instance_name} not found in OPTIMUM_COSTS.")

    return g, trucks, optimum, instance_name


def extract_n_from_name(instance_name):
    """
    Utility to get the number of nodes 'n' from an instance name like 'X-n101-k25'.
    """
    match = re.search(r'-n(\d+)', instance_name)
    if match:
        return int(match.group(1))
    return 0


# --- Analysis 1: Statistical Quality (like Notebook 3.1) ---

def run_statistical_analysis(algorithm_func, instance_file, nb_tests=100, **algo_params):
    """
    Runs an algorithm many times on a single instance to check its stability.
    Plots a histogram of the solution quality (gap %)
    """
    print(f"\nRunning Statistical Analysis for '{algorithm_func.__name__}' on '{instance_file}'...")

    g, trucks, optimum, instance_name = load_instance(instance_file)
    if optimum is None:
        print("Cannot run statistical analysis without a known optimum. Skipping.")
        return

    gaps = []
    scores = []
    start_time = time.time()

    for i in range(nb_tests):
        # The algorithm functions (HC, TS) generate their own initial solution
        if algorithm_func.__name__ == 'tabu_search':
            # tabu_search expects an initial solution
            initial_sol = generate_feasible_initial_solution(g, trucks)
            sol, score, _, _ = algorithm_func(g, trucks, initial_solution=initial_sol, **algo_params)
        else:
            sol, score, _ = algorithm_func(g, trucks, **algo_params)

        gap = (score / optimum - 1) * 100
        gaps.append(gap)
        scores.append(score)

    elapsed = time.time() - start_time

    print(f"Completed {nb_tests} runs in {elapsed:.2f}s")
    print(f"  Best Score: {np.min(scores):.2f} (Gap: {np.min(gaps):.2f}%)")
    print(f"  Avg Score:  {np.mean(scores):.2f} (Gap: {np.mean(gaps):.2f}%)")
    print(f"  Median Gap: {np.median(gaps):.2f}%")
    print(f"  Std Dev (Gap): {np.std(gaps):.2f}")

    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(gaps, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Solution Quality Distribution ({algorithm_func.__name__} on {instance_name})")
    plt.xlabel("Gap from Optimum (%)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    #  Plot 2: Box Plot ---
    plot_box_comparison(
        data_lists=[gaps],
        labels=[f"{algorithm_func.__name__}"],
        title=f"Solution Quality Distribution ({algorithm_func.__name__} on {instance_name})",
        xlabel="Algorithm Run",
        ylabel="Gap from Optimum (%)",
        show_optimum=0  # Optimum gap is 0
    )


# --- Analysis 2: Parameter Tuning (like Notebook 3.2) ---

def run_parameter_tuning(algorithm_func, instance_file, param_to_tune, param_values, nb_tests=30, **base_algo_params):
    """
    Tests the impact of a single parameter on solution quality.
    Plots mean score vs. parameter value with a standard deviation band.
    """
    print(f"\nRunning Parameter Tuning for '{param_to_tune}' on '{instance_file}'...")

    g, trucks, optimum, instance_name = load_instance(instance_file)

    means = []
    deviations = []
    all_scores_lists = []

    total_runs = len(param_values) * nb_tests
    print(f"Total runs to perform: {total_runs}")
    run_count = 0
    start_time = time.time()

    for value in param_values:
        scores = []
        current_params = copy.deepcopy(base_algo_params)
        current_params[param_to_tune] = value

        for _ in range(nb_tests):
            if algorithm_func.__name__ == 'tabu_search':
                initial_sol = generate_feasible_initial_solution(g, trucks)
                sol, score, _, _ = algorithm_func(g, trucks, initial_solution=initial_sol, **current_params)
            else:
                sol, score, _ = algorithm_func(g, trucks, **current_params)
            scores.append(score)
            run_count += 1

        means.append(np.mean(scores))
        deviations.append(np.std(scores))
        all_scores_lists.append(scores)  # <-- ADD THIS LINE
        print(f"  {param_to_tune} = {value}: Avg Score = {means[-1]:.2f} (StdDev: {deviations[-1]:.2f})")

    elapsed = time.time() - start_time
    print(f"Parameter tuning completed in {elapsed:.2f}s")

    # Plot Mean and Standard Deviation
    means = np.array(means)
    deviations = np.array(deviations)

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, means, 'o-', label='Mean Score')
    plt.fill_between(
        param_values,
        means - deviations,
        means + deviations,
        alpha=0.2,
        label='Std. Deviation Band'
    )
    plt.title(f"Impact of '{param_to_tune}' on Score ({instance_name})")
    plt.xlabel(param_to_tune)
    plt.ylabel("Final Score (Lower is Better)")
    if optimum:
        plt.axhline(y=optimum, color='r', linestyle='--', label=f'Optimum ({optimum})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    print(f"Generating box plot for {param_to_tune} tuning...")
    plot_box_comparison(
        data_lists=all_scores_lists,
        labels=[str(v) for v in param_values],
        title=f"Score Distribution vs. '{param_to_tune}' ({instance_name})",
        xlabel=param_to_tune,  # This will be the Y-axis label
        ylabel="Final Score",  # This will be the X-axis label
        show_optimum=optimum
    )


# --- Analysis 3: Instance Comparison (like Notebook 3.3) ---

def run_instance_comparison(algorithm_func, instance_files, nb_tests=30, **algo_params):
    """
    Compares algorithm performance (gap %) across different instances.
    Plots mean gap % vs. problem size (n).
    """
    print(f"\nRunning Instance Comparison for '{algorithm_func.__name__}'...")

    results = []  # Store tuples of (n, mean_gap, std_gap, instance_name)

    start_time = time.time()

    for instance_file in instance_files:
        g, trucks, optimum, instance_name = load_instance(instance_file)
        if optimum is None:
            print(f"Skipping {instance_name}: No optimum defined.")
            continue

        n = extract_n_from_name(instance_name)
        gaps = []
        print(f"  Testing instance {instance_name} (n={n})...")

        for _ in range(nb_tests):
            if algorithm_func.__name__ == 'tabu_search':
                initial_sol = generate_feasible_initial_solution(g, trucks)
                sol, score, _, _ = algorithm_func(g, trucks, initial_solution=initial_sol, **algo_params)
            else:
                sol, score, _ = algorithm_func(g, trucks, **algo_params)

            gaps.append((score / optimum - 1) * 100)

        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        results.append((n, mean_gap, std_gap, instance_name))
        print(f"    -> Avg Gap: {mean_gap:.2f}% (StdDev: {std_gap:.2f})")

    elapsed = time.time() - start_time
    print(f"Instance comparison completed in {elapsed:.2f}s")

    if not results:
        print("No results to plot.")
        return

    # Sort results by n (number of nodes)
    results.sort(key=lambda x: x[0])

    ns = [r[0] for r in results]
    mean_gaps = [r[1] for r in results]
    std_gaps = [r[2] for r in results]
    names = [r[3] for r in results]

    # Plot Mean and Standard Deviation
    plt.figure(figsize=(12, 7))
    plt.errorbar(ns, mean_gaps, yerr=std_gaps, fmt='o-', capsize=5, label='Mean Gap (%)')
    plt.title(f"Performance vs. Problem Size (n) ({algorithm_func.__name__})")
    plt.xlabel("Number of Nodes (n)")
    plt.ylabel("Average Gap from Optimum (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # Use instance names as x-ticks if not too many
    if len(ns) <= 10:
        plt.xticks(ns, names, rotation=25, ha='right')
    else:
        plt.xticks(ns)
    plt.tight_layout()
    plt.show()


# --- NEW: Box Plot Function (Horizontal) ---

def plot_box_comparison(data_lists, labels, title, xlabel, ylabel, show_optimum=None):
    """
    Creates a horizontal box plot to compare the distributions of multiple data sets.

    Args:
        data_lists: A list of data lists (e.g., [[run1_scores], [run2_scores], ...])
        labels: The labels for the y-axis (e.g., ['1000 iter', '4000 iter', ...])
        title: The chart title.
        xlabel: The x-axis label.
        ylabel: The y-axis label.
        show_optimum: (Optional) X-value to draw a vertical optimum line.
    """
    plt.figure(figsize=(12, 7))

    # --- MODIFIED ---
    # Set vert=False for a horizontal plot
    bp = plt.boxplot(data_lists, patch_artist=True, tick_labels=labels, vert=False)
    # --- END MODIFIED ---

    # Customize colors (optional, but looks nice)
    colors = ['#A8DADC', '#457B9D', '#1D3557', '#E63946', '#F1FAEE']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_edgecolor('black')
        box.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)

    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)

    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)

    for flier in bp['fliers']:
        flier.set(marker='o', color='#e76f51', alpha=0.5)

    # --- Formatting ---
    plt.title(title, fontsize=14, fontweight='bold')

    # --- MODIFIED: Swapped labels for horizontal plot ---
    plt.xlabel(ylabel, fontsize=12)  # Now the X-axis
    plt.ylabel(xlabel, fontsize=12)  # Now the Y-axis

    # Add optimum line if provided
    if show_optimum is not None:
        # --- MODIFIED: Use axvline (vertical line) for horizontal plot ---
        plt.axvline(x=show_optimum, color='r', linestyle='--', label=f'Optimum ({show_optimum})')
        plt.legend()

    # --- MODIFIED: Grid axis is now 'x' ---
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# --- Main execution block ---

if __name__ == "__main__":
    # Define a list of instances to test
    # Make sure these paths match your file structure
    INSTANCE_PATH_PREFIX = '../media/instances/'

    instance_files_to_test = [
        INSTANCE_PATH_PREFIX + 'A-n32-k5.vrp',
        INSTANCE_PATH_PREFIX + 'X-n101-k25.vrp',
        INSTANCE_PATH_PREFIX + 'X-n153-k22.vrp',
        INSTANCE_PATH_PREFIX + 'X-n513-k21.vrp',
        # INSTANCE_PATH_PREFIX + 'X-n979-k58.vrp',
        # INSTANCE_PATH_PREFIX + 'X-n1001-k43.vrp'

    ]



    # --- Task 1: Statistical Analysis ---
    # Run Hill Climbing 100 times on A-n32-k5 to see its stability
    print("--- 1. Running Statistical Analysis ---")
    run_statistical_analysis(
        algorithm_func=hill_climbing,
        instance_file=instance_files_to_test[0],
        nb_tests=100,
        max_iterations=10000  # Fixed parameter
    )

    # --- Task 2: Parameter Tuning ---
    # Tune 'max_iterations' for Hill Climbing on A-n32-k5
    print("\n--- 2. Running Parameter Tuning ---")
    run_parameter_tuning(
        algorithm_func=hill_climbing,
        instance_file=instance_files_to_test[0],
        param_to_tune='max_iterations',
        param_values=range(1000, 20001, 3000),  # Test from 1k to 20k
        nb_tests=20  # 20 runs per data point
    )

    # Tune 'tabu_tenure' for the single Tabu Search
    print("\n--- 3. Running Parameter Tuning (Tabu Search) ---")
    run_parameter_tuning(
        algorithm_func=tabu_search,
        instance_file=instance_files_to_test[0],
        param_to_tune='tabu_tenure',
        param_values=range(5, 51, 5),  # Test tenure from 5 to 50
        nb_tests=20,
        max_iterations=20000
    )

    # --- Task 3: Instance Comparison ---
    # Compare Hill Climbing performance across instances of different sizes
    print("\n--- 4. Running Instance Comparison")
    run_instance_comparison(
        algorithm_func=hill_climbing,
        instance_files=instance_files_to_test,
        nb_tests=20,  # 20 runs per instance
        max_iterations=10000
    )

    run_parameter_tuning(
        algorithm_func=tabu_search,
        instance_files=instance_files_to_test,
        param_to_tune='tabu_tenure',
        param_values=range(5, 51, 5),  # Test tenure from 5 to 50
        nb_tests=20,
        max_iterations=5000  # Fixed parameter
    )

    print("\n\n--- Evaluation complete ---")