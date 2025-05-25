from simulation_newest_main import simulation_q_learning
from simulation_newest_main import detect_price_cycle
import numpy as np
N=1000
T=500_000
y="no"
k=7
np.random.seed(7)
x=True

pattern1 = ((0.5, 0.8333333333333333), (0.5, 0.3333333333333333), (0.16666666666666666, 0.3333333333333333), (0.16666666666666666, 0.8333333333333333))
# pattern2 = ((0.340,0.420), (0.340,0.280), (0.270,0.280), (
#     0.270,0.140), (0.910,0.590), (0.330,0.590), (0.330,0.240), (
#         0.120,0.240), (0.120,0.980), (0.600,0.980), (0.600,0.420))
pattern2 = ((0.67, 0.98), (0.67, 0.56), (0.36, 0.56), (0.36, 0.27), (0.26, 0.27), (0.26, 0.22), (0.84, 0.22), (0.84, 0.61), (0.46, 0.61), (0.46, 0.32), (0.25, 0.32), (0.25, 0.98))

pattern2round = tuple(tuple(round(x, 2) for x in inner) for inner in pattern2)

while x:
    Q1, Q2, profit_1, profit_2, price_1, price_2 = simulation_q_learning(T, k)
            # save the last N prices
    prices_1 = price_1[-N:]
    prices_2 = price_2[-N:]

        #Detect cycles
    cycle_len, pattern_combined = detect_price_cycle(prices_1, prices_2)
    if y == "yes":
        pattern_combinedround =  tuple(tuple(round(x, 2) for x in inner) for inner in pattern_combined)
        print(cycle_len, pattern_combinedround)
    if pattern_combined== pattern1:
        x = 0
        print(pattern_combined)
        print(Q1,Q2)
