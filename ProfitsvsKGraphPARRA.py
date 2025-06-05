# test_graph_Q_learning_profit_vs_k.py

import numpy as np
import matplotlib.pyplot as plt
from simulation_newest_main import simulation_q_learning
from simulation_newest_main import simulation_random_players
from numba import jit

# ---------------------------------------------------------------------------------------------------
# 1) Parameters
# ---------------------------------------------------------------------------------------------------

np.random.seed(123)

T             = 500_000     # total timesteps per Q‐learning run
last_window   = 1_000       # how many of the final timesteps to average over
num_runs      =  1000         # number of independent repetitions per k‐value
                             # (you can increase this, but be aware each run is 500k steps!)
k_values      = list(range(7, 101, 2))  # e.g. try k = 2,3,…,50.  You can extend to 100 if you like.

# ---------------------------------------------------------------------------------------------------
# 2) Utility function: run Q‐learning, grab “common profit” in last window
# ---------------------------------------------------------------------------------------------------

@jit(nopython=True)
def average_common_profit_last_window(k, T, last_window, num_runs):
    """
    For a given k, run 'num_runs' independent simulation_q_learning(T,k),
    extract (profit_1, profit_2), compute common profit = (profit_1+profit_2)/2,
    and average the final 'last_window' steps.  Then average across all runs.
    """
    avg_profits = np.zeros(num_runs)

    for run_idx in range(num_runs):
        print(f"[k = {k}]  run {run_idx+1}/{num_runs} …")
        profit_1, profit_2= simulation_q_learning(T, k)

        # Convert to numpy arrays for easy slicing:
        profit_1 = np.array(profit_1)
        profit_2 = np.array(profit_2)

        # “Common profit” at each timestep = average of the two players
        common_profit = 0.5 * (profit_1 + profit_2)

        # Average over the very last 'last_window' periods:
        avg_profits[run_idx] = np.mean(common_profit[-last_window:])

    # Return the grand mean across all independent runs:
    return np.mean(avg_profits)


# ---------------------------------------------------------------------------------------------------
# 3) Loop over k_values, collect the “avg last‐1000‐period” profits
# ---------------------------------------------------------------------------------------------------

mean_profits_per_k = []
for k in k_values:
    mean_profit_k = average_common_profit_last_window(k=k,
                                                      T=T,
                                                      last_window=last_window,
                                                      num_runs=num_runs)
    mean_profits_per_k.append(mean_profit_k)

# Convert to numpy arrays for plotting:
k_arr      = np.array(k_values)
profit_arr = np.array(mean_profits_per_k)


# ---------------------------------------------------------------------------------------------------
# 4) Plot “Profit vs. k”
# ---------------------------------------------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(k_arr, profit_arr, marker='o', linestyle='-')
plt.xlabel("k", fontsize=12)
plt.ylabel("Average Profit (last 1000 periods)", fontsize=12)
plt.title("Q Learner Common Profit vs. Number of Price Points k", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
