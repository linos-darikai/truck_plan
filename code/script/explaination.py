import random
import math

# -------------------------
# Random Search
# -------------------------
def test_random_search():
    print("Random Search Example:")
    best = None
    for _ in range(100):
        x = random.uniform(-10, 10)
        fx = x**2 - 4*x + 5  # simple function to minimize
        if best is None or fx < best[1]:
            best = (x, fx)
    print(f"Best solution: x = {best[0]:.4f}, f(x) = {best[1]:.4f}\n")


# -------------------------
# Hill Climbing
# -------------------------
def test_hill_climbing():
    print("Hill Climbing Example:")
    x = random.uniform(-10, 10)
    for _ in range(100):
        step = random.uniform(-0.5, 0.5)
        new_x = x + step
        if new_x**2 - 4*new_x + 5 < x**2 - 4*x + 5:
            x = new_x
    print(f"Local minimum: x = {x:.4f}, f(x) = {x**2 - 4*x + 5:.4f}\n")


# -------------------------
# Simulated Annealing
# -------------------------
def test_simulated_annealing():
    print("Simulated Annealing Example:")
    x = random.uniform(-10, 10)
    T = 10.0
    alpha = 0.9
    for _ in range(100):
        step = random.uniform(-1, 1)
        new_x = x + step
        delta = (new_x**2 - 4*new_x + 5) - (x**2 - 4*x + 5)
        if delta < 0 or random.random() < math.exp(-delta / T):
            x = new_x
        T *= alpha
    print(f"Approx. minimum: x = {x:.4f}, f(x) = {x**2 - 4*x + 5:.4f}\n")


# -------------------------
# Genetic Algorithm
# -------------------------
def test_genetic_algorithm():
    print("Genetic Algorithm Example:")
    # Initialize population
    population = [random.uniform(-10, 10) for _ in range(5)]
    for generation in range(10):
        # Evaluate fitness (lower is better)
        population.sort(key=lambda x: x**2 - 4*x + 5)
        parents = population[:3]  # keep top 3
        # Crossover + mutation
        new_population = parents[:]
        while len(new_population) < 5:
            p1, p2 = random.sample(parents, 2)
            child = (p1 + p2)/2 + random.uniform(-0.5, 0.5)
            new_population.append(child)
        population = new_population
    best = min(population, key=lambda x: x**2 - 4*x + 5)
    print(f"Best GA solution: x = {best:.4f}, f(x) = {best**2 - 4*best + 5:.4f}\n")


# -------------------------
# Main: run all tests
# -------------------------
if __name__ == "__main__":
    #test_random_search()
    #test_hill_climbing()
    test_simulated_annealing()
    #test_genetic_algorithm()
