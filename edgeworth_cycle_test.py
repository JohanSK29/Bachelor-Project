from simulation_newest_main import simulation_q_learning
from simulation_newest_main import price_array, profit, initialize_Q, seq_q_step
from simulation_newest_main import simulate_klein_edgeworth_cycle_compt_benchmark, detect_price_cycle
from numba import jit
from numba import config
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True, linewidth=100)

#print("Q1:\n", Q1)
#print("Q2:\n", Q2)

#print the last 20 prices
#print("Last 20 prices for player 1:\n", prices_1[-20:])


def evaluate_end_converge_cycle (num_runs, T, k):

    N = 1000

    # Define a list to store cycle lengths and patterns
    cycle_len_and_pattern = []

    #run the simulation for num_runs
    for run in range(num_runs):
        print(f"Running Q-learning simulation {run + 1}/{num_runs}...")
        # Simulate Q-learning players' profits
        Q1, Q2, profit_1, profit_2, price_1, price_2 = simulation_q_learning(T, k)

        # save the last N prices
        prices_1 = price_1[-N:]
        prices_2 = price_2[-N:]

        #Detect cycles
        cycle_len, pattern_combined = detect_price_cycle(prices_1, prices_2)

        # Convert pattern_combined to native Python floats
        if pattern_combined is not None:
            pattern_combined = tuple((float(p1), float(p2)) for p1, p2 in pattern_combined)

        # Append the cycle length and pattern as a tuple to the list
        cycle_len_and_pattern.append((cycle_len, pattern_combined))

    # Count the occurrences of each cycle length and pattern
    cycle_length_counts = {}
    for cycle_len, pattern in cycle_len_and_pattern:
        if cycle_len not in cycle_length_counts:
            cycle_length_counts[cycle_len] = 0
        cycle_length_counts[cycle_len] += 1


    # Print the cycle lengths and their counts
    print("Cycle lengths and their counts:")
    for cycle_len, count in cycle_length_counts.items():
        print(f"Cycle length: {cycle_len}, Count: {count}")

    # print the number of time a pattern occurs:
    pattern_counts = {}
    focal_pricing_count = 0
    non_focal_pricing_count = 0
    focal_pricing_patterns_counts = {}  # Dictionary to store focal pricing patterns and their counts

    for cycle_len, pattern in cycle_len_and_pattern:
        if pattern is not None:
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
            pattern_counts[pattern] += 1

            # Check if the pattern represents focal pricing
            is_focal = all(p1 == p2 for p1, p2 in pattern)
            if is_focal:
                focal_pricing_count += 1
                if pattern not in focal_pricing_patterns_counts:
                    focal_pricing_patterns_counts[pattern] = 0
                focal_pricing_patterns_counts[pattern] += 1
            else:
                non_focal_pricing_count += 1

    # Print the patterns and their counts
    print("Patterns and their counts:")
    for pattern, count in pattern_counts.items():
        print(f"Pattern: {pattern}, Count: {count}")

    # Print focal pricing evaluation
    print("\nFocal Pricing Evaluation:")
    print(f"Focal Pricing Count: {focal_pricing_count}")
    print(f"Non-Focal Pricing Count: {non_focal_pricing_count}")

    # Print the focal pricing patterns and their counts
    print("\nFocal Pricing Patterns and their counts:")
    for pattern, count in focal_pricing_patterns_counts.items():
        print(f"Pattern: {pattern}, Count: {count}")

    return None

# Set simulation parameters
T = 500_000
k = 101
num_runs = 1000

# Run simulation
Q1, Q2, profit_1, profit_2, prices_1, prices_2 = simulation_q_learning(T, k)

# Optionally trim to the last N steps for clarity
N = 1000
# Trim the prices to the first N steps
prices_1_trim = prices_1[-N:]
prices_2_trim = prices_2[-N:]

avg1, avg2, avg_common, hist = simulate_klein_edgeworth_cycle_compt_benchmark(k,cycles = 1)


cycle_len, pattern_combined = detect_price_cycle(prices_1_trim, prices_2_trim)


lorteliste = evaluate_end_converge_cycle(num_runs, T, k)

print(lorteliste)

#convert hist from np.float64 to float
hist = np.array(hist).astype(float)
#print("Last 20 prices edgeworth:\n", hist[-20:])


prices_1_trim = np.array(prices_1_trim).astype(float)
prices_2_trim = np.array(prices_2_trim).astype(float)
#print("Last 20 prices for player 1:\n", prices_1_trim[-10:])
#print("Last 20 prices for player 2:\n", prices_2_trim[-10:])

#print edgeworth cycle length
#print("Edgeworth cycle length:", len(hist))

#print("Sim cycle length:", cycle_len)
#pattern_combined = np.array(pattern_combined).astype(float)
#print("P1 cycle pattern:\n", pattern_combined)




#print(prices_1_trim)


#print("Edgeworth cycle prices hist:\n", hist)



'''
edgeworth_prices = hist  # list of (p1, p2) tuples
p1_edge = [p[0] for p in edgeworth_prices]
p2_edge = [p[1] for p in edgeworth_prices]

plt.plot(p1_edge, label="P1", marker='o', markersize=5, linestyle='--') 
plt.plot(p2_edge, label="P2", marker='o', markersize=5, linestyle='--')
plt.title("Edgeworth Price Cycle k = 24")
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.legend()
plt.show()
'''


'''
plt.figure(figsize=(12, 6))
plt.plot(prices_1_trim, label="Player 1 Price", alpha=0.8, marker='o', markersize=3,)
plt.plot(prices_2_trim, label="Player 2 Price", alpha=0.8, marker='o', markersize=3,)
plt.xlabel(f"Time step (last {N} steps)")
plt.ylabel("Price")
plt.title("Price Dynamics of Q-Learning Agents")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''