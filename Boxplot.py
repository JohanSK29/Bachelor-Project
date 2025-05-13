import matplotlib.pyplot as plt
from simulation_newest_main import simulation_q_learning, detect_price_cycle
import numpy as np
import time

np.random.seed(999)

start_time = time.time()

def collect_cycle_lengths_for_k(k_values, num_runs=100, T=500_000, N=1000):
    result_dict = {k: [] for k in k_values}

    for k in k_values:
        print(f"\nKører simulationer for k = {k}")
        for run in range(num_runs):
            Q1, Q2, profit_1, profit_2, prices_1, prices_2 = simulation_q_learning(T, k)
            prices_1_trim = prices_1[-N:]
            prices_2_trim = prices_2[-N:]

            cycle_len, _ = detect_price_cycle(prices_1_trim, prices_2_trim)
            result_dict[k].append(cycle_len if cycle_len is not None else 0)  # 0 = ingen cyklus fundet
    return result_dict

def plot_cycle_length_boxplots(cycle_lengths_by_k):
    k_values = sorted(cycle_lengths_by_k.keys())
    data = [cycle_lengths_by_k[k] for k in k_values]

    # Juster labels til k-1
    k_minus_1_labels = [k - 1 for k in k_values]

    plt.figure(figsize=(10, 6))
    # Boxplot
    bp = plt.boxplot(data, labels=k_minus_1_labels, vert=True, showmeans=True,
                     meanprops=dict(marker= 'x', color='red', markersize=8))

    plt.xlabel("k (Number of Price Points)")
    plt.ylabel("Cycle Length (Last 1000 Periods)")
    plt.title("Cycle Length Distribution by k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Parametre
k_values = [7, 13, 25, 49, 101]
num_runs = 1000
T = 500_000


# Saml data og tegn graf
cycle_data = collect_cycle_lengths_for_k(k_values, num_runs=num_runs, T=T)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Koden tog {elapsed_time:.2f} sekunder at køre.")
plot_cycle_length_boxplots(cycle_data)

