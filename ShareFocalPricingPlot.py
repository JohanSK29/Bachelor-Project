import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from simulation_newest_main import simulation_q_learning

# Numba-compatible simplified cycle + focal pattern detector
@jit(nopython=True)
def detect_focal_cycle(prices_1, prices_2, max_cycle_len=50, min_repeats=2, tolerance=1e-5):
    n = len(prices_1)
    for cycle_len in range(2, max_cycle_len + 1):
        num_possible_repeats = n // cycle_len
        if num_possible_repeats < min_repeats:
            continue

        start = n - cycle_len
        pattern_1 = prices_1[start:]
        pattern_2 = prices_2[start:]

        match = True
        for i in range(2, min_repeats + 1):
            s = n - i * cycle_len
            e = s + cycle_len
            if e > n:
                match = False
                break
            if not np.all(np.abs(prices_1[s:e] - pattern_1) < tolerance):
                match = False
                break
            if not np.all(np.abs(prices_2[s:e] - pattern_2) < tolerance):
                match = False
                break

        if match:
            # Check if focal: all price_1 == price_2
            is_focal = np.all(np.abs(pattern_1 - pattern_2) < tolerance)
            return is_focal
    return False

# Numba can't call high-level functions like simulation_q_learning directly,
# so we use a Python wrapper that runs numba-optimized cycle detection.
def evaluate_share_focal_fast(num_runs, T, k):
    focal_count = 0
    for _ in range(num_runs):
        _, _, _, _, price_1, price_2 = simulation_q_learning(T, k)
        p1 = np.array(price_1[-1000:], dtype=np.float64)
        p2 = np.array(price_2[-1000:], dtype=np.float64)
        if detect_focal_cycle(p1, p2):
            focal_count += 1
    return focal_count / num_runs


ks = list(range(7, 102, 2)) 
T = 500_000
num_runs = 1000  # You can increase this for better accuracy


focal_shares = []

for k in ks:
    print(f"Running simulations for k = {k-1}...")
    share = evaluate_share_focal_fast(num_runs, T, k)
    focal_shares.append(share)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(ks, focal_shares, 'x-', markersize=10)
plt.xlabel("k (Number of price points)")
plt.ylabel("Share focal pricing end cycle")
plt.title("Share of Focal Pricing Outcomes vs k")
plt.grid(True)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
