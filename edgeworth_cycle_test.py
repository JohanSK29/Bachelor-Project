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


# Set simulation parameters
T = 500_000
k = 7

# Run simulation
Q1, Q2, profit_1, profit_2, prices_1, prices_2 = simulation_q_learning(T, k)

# Optionally trim to the last N steps for clarity
N = 1000
prices_1_trim = prices_1[-N:]
prices_2_trim = prices_2[-N:]

avg1, avg2, avg_common, hist = simulate_klein_edgeworth_cycle_compt_benchmark(k,cycles = 1)


cycle_len_p1, pattern_p1 = detect_price_cycle(prices_1_trim)
cycle_len_p2, pattern_p2 = detect_price_cycle(prices_2_trim)


print("Last 20 prices for player 1:\n", prices_1_trim[-20:])
print("Last 20 prices for player 2:\n", prices_2_trim[-20:])

print("P1 cycle length:", cycle_len_p1)
print("P1 cycle pattern:", pattern_p1)

print("P2 cycle length:", cycle_len_p2)
print("P2 cycle pattern:", pattern_p2)



#print(prices_1_trim)




#print the last 50 entries of hist
#convert hist from np.float64 to float
#hist = hist = np.array(hist).astype(float)
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