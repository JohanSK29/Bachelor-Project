import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_prices = 10  # Number of discrete price levels
alpha = 0.1  # Learning rate
delta = 0.95  # Discount factor
epsilon = 0.1  # Exploration probability
num_episodes = 30000  # Number of simulation rounds
burn_in = 1000  # Ignore first 1000 episodes

discrete_price_space = np.linspace(0, 1, num_prices)

# Initialize Q-tables for both firms
Q1 = np.zeros((num_prices, num_prices))
Q2 = np.zeros((num_prices, num_prices))

def demand(p_i, p_j):
    if p_i < p_j:
        return 1 - p_i
    elif p_i == p_j:
        return (1 - p_i) / 2
    else:
        return 0

def profit(p_i, p_j):
    return p_i * demand(p_i, p_j)

# Tracking profitability
profits_1 = []
profits_2 = []

for episode in range(num_episodes):
    index_1, index_2 = np.random.randint(0, num_prices, size=2)
    total_profit_1, total_profit_2 = 0, 0
    
    for t in range(100):
        if np.random.rand() < epsilon:
            new_index_1 = np.random.randint(0, num_prices)
        else:
            new_index_1 = np.argmax(Q1[:, index_2])
        
        reward_1 = profit(discrete_price_space[new_index_1], discrete_price_space[index_2])
        total_profit_1 += reward_1
        
        Q1[index_1, index_2] = (1 - alpha) * Q1[index_1, index_2] + alpha * (
            reward_1 + delta * np.max(Q1[new_index_1, :])
        )
        
        index_1 = new_index_1
        
        if np.random.rand() < epsilon:
            new_index_2 = np.random.randint(0, num_prices)
        else:
            new_index_2 = np.argmax(Q2[:, index_1])

