from simulation_newest_main import simulate_klein_edgeworth_cycle_compt_benchmark
from simulation_newest_main import simulation_random_players
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit, config, prange

@jit(nopython=True)
def simulate_random_common_profit(num_runs, T, window_size, k):
    # Preallocate accumulators for the moving averages
    cumulative_avg_common_profit = np.zeros(T - window_size - 1)

    for run in range(num_runs):
        print(f"Running simulation {run + 1}/{num_runs}...")
        # Simulate random players' profits
        profit_1, profit_2 = simulation_random_players(T, k)

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
k = 13

# Calculate k-1 for the title
k_minus_1 = k - 1

# Start timing
start_time = time.time()

# Calculate the competitive benchmark
avg1, avg2, avg_common, hist = simulate_klein_edgeworth_cycle_compt_benchmark(k,cycles = 1)
competitive_benchmark = avg_common

# Simulate and calculate the average moving average common profit
avg_moving_avg_common_profit = simulate_random_common_profit(num_runs, T, window_size, k)

# End timing
end_time = time.time()

# Calculate the average common profit
average_common_profit = np.mean(avg_moving_avg_common_profit)


# Plot the average moving average of the common profit
plt.figure(figsize=(10, 6))
plt.plot(avg_moving_avg_common_profit, label="Average Common Profit (Random Players)", color="purple")
plt.xlabel("Time Steps")
plt.ylabel("Profit (Moving Average)")
plt.title(f"Average Moving Average of Common Profit (Random Players, k = {k_minus_1})")
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



# Print execution time
print(f"Execution time: {end_time - start_time:.2f} seconds")