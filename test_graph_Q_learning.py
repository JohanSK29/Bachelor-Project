import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from simulation_newest_main import simulation_q_learning, simulate_klein_edgeworth_cycle_compt_benchmark

@jit(nopython=True)
def simulate_q_learning_common_profit(num_runs, T, window_size, k):
    # Preallocate accumulators for the moving averages
    cumulative_avg_common_profit = np.zeros(T - window_size - 1)

    for run in range(num_runs):
        print(f"Running Q-learning simulation {run + 1}/{num_runs}...")
        # Simulate Q-learning players' profits
        Q1, Q2, profit_1, profit_2, price_1, price_2 = simulation_q_learning(T, k)

        # Preallocate array for common profit
        common_profit = np.zeros(T)
        for i in range(T):
            common_profit[i] = (profit_1[i] + profit_2[i]) / 2

        # Preallocate array for moving average of common profit
        moving_avg_common_profit = np.zeros(T - window_size - 1)

        # Calculate moving average for this run
        for i in range(1, T - window_size):
            moving_avg_common_profit[i - 1] = np.sum(common_profit[i:i + window_size]) / window_size

        # Accumulate the moving averages
        cumulative_avg_common_profit += moving_avg_common_profit

    # Compute the average moving average across all runs
    avg_moving_avg_common_profit = cumulative_avg_common_profit / num_runs

    return avg_moving_avg_common_profit

# Parameters
num_runs = 1000
T = 500_000
window_size = 1000
k = 25

# Calculate k-1 for the title
k_minus_1 = k - 1

# Get the competitive benchmark dynamically
_, _, competitive_benchmark, _ = simulate_klein_edgeworth_cycle_compt_benchmark(k=k, cycles=1)

# Simulate and calculate the average moving average common profit
avg_moving_avg_common_profit = simulate_q_learning_common_profit(num_runs, T, window_size, k)

# Calculate the average common profit
average_common_profit = np.mean(avg_moving_avg_common_profit)

# Plot the average moving average of the common profit
plt.figure(figsize=(10, 6))
plt.plot(avg_moving_avg_common_profit, label="Average Common Profit (Q-Learning Players)", color="blue")
plt.xlabel("Time Steps")
plt.ylabel("Profit (Moving Average)")
plt.title(f"Average Moving Average of Common Profit (Q-Learning Players, k = {k_minus_1})")
plt.axhline(y=0.125, color='r', linestyle='--', label="Joint profit maximizing benchmark")
plt.axhline(y=competitive_benchmark, color='g', linestyle='--', label=f"Competitive benchmark (Edgeworth cycle: {competitive_benchmark:.4f})")
plt.ylim(bottom=0)

# Add a text box for the average common profit
textstr = f"Average Common Profit: {average_common_profit:.4f}"
plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()