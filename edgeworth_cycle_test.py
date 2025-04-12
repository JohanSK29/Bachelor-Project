from simulation_newest_main import simulation_q_learning
from simulation_newest_main import price_array, profit, initialize_Q, seq_q_step
from simulation_newest_main import simulate_klein_edgeworth_cycle_compt_benchmark
from numba import jit
from numba import config
import numpy as np

@jit(nopython=True)
def simulation_q_learning(T,k):
    #initialise the q matrices (AXA)
    Q1,Q2 = initialize_Q(k)

    # intiltoze the price vector / action vector (1+k)x1
    action_vector = price_array(k)

    #number of actions :
    number_of_actions = len(action_vector)

    #initialize the starting prices
    p1_current_index = np.random.randint(number_of_actions)
    p1_old_index = np.random.randint(number_of_actions)

    p2_old_index = np.random.randint(number_of_actions)
    p2_current_index = np.random.randint(number_of_actions)

    # save the profit:
    profit_1 = []
    profit_2 = []

    # save the prices:
    prices_1 = []
    prices_2 = []



    for t in range(T):
        if (t%2) ==0:
            Q1,p2_current_index,p1_old_index,p1_current_index =seq_q_step(Q1,
            p1_current_index,
            p2_old_index,
            p2_current_index,
            t,
            action_vector)


            profit_1.append( profit(action_vector[p1_current_index], action_vector[p2_current_index]))
            profit_2.append( profit(action_vector[p2_current_index], action_vector[p1_current_index]))

            prices_1.append(action_vector[p1_current_index])
            prices_2.append(action_vector[p2_current_index])

        else: 
            Q2,p1_current_index,p2_old_index,p2_current_index =seq_q_step(Q2,
            p2_current_index,
            p1_old_index,
            p1_current_index,
            t,
            action_vector)

            profit_2.append( profit(action_vector[p2_current_index], action_vector[p1_current_index]))            
            profit_1.append( profit(action_vector[p1_current_index], action_vector[p2_current_index]))

            prices_1.append(action_vector[p1_current_index])
            prices_2.append(action_vector[p2_current_index])


    return Q1,Q2, profit_1, profit_2, prices_1, prices_2


import numpy as np

def detect_price_cycle(prices, max_cycle_len=50, min_repeats=2, tolerance=1e-5):
    """
    Detects repeated price cycles in a 1D list or array of prices.
    
    Args:
        prices: List or array of recent prices (e.g. last 200)
        max_cycle_len: Maximum cycle length to search for
        min_repeats: Minimum number of repetitions to count as a cycle
        tolerance: Allowed absolute difference for matching due to float rounding

    Returns:
        (cycle_length, pattern) if found, otherwise (None, None)
    """
    prices = np.array(prices)
    n = len(prices)

    for cycle_len in range(2, max_cycle_len + 1):
        num_possible_repeats = n // cycle_len
        if num_possible_repeats < min_repeats:
            continue

        pattern = prices[-cycle_len:]  # Take last cycle_len points as candidate pattern

        match = True
        for i in range(2, min_repeats + 1):
            start = -i * cycle_len
            end = start + cycle_len
            if end < -n:
                match = False
                break
            window = prices[start:end]
            if not np.allclose(window, pattern, atol=tolerance):
                match = False
                break

        if match:
            return cycle_len, pattern

    return None, None


np.set_printoptions(precision=3, suppress=True, linewidth=100)

#print("Q1:\n", Q1)
#print("Q2:\n", Q2)

#print the last 20 prices
#print("Last 20 prices for player 1:\n", prices_1[-20:])

import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

# Set simulation parameters
T = 500_000
k = 25

# Run simulation
Q1, Q2, profit_1, profit_2, prices_1, prices_2 = simulation_q_learning(T, k)

# Optionally trim to the last N steps for clarity
N = 1000
prices_1_trim = prices_1[-N:]
prices_2_trim = prices_2[-N:]

avg1, avg2, avg_common, hist = simulate_klein_edgeworth_cycle_compt_benchmark(k,cycles = 1)


cycle_len_p1, pattern_p1 = detect_price_cycle(prices_1_trim)
cycle_len_p2, pattern_p2 = detect_price_cycle(prices_2_trim)

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