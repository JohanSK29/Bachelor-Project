import time
import numpy as np
from numba import jit, config


config.DISABLE_JIT = False

#Parameters:
#time steps, 500_000
T = 500_000
#Discount factor
gamma_discount_factor = 0.95
#Learning parameter
alpha = 0.3
#Decay parameter for Epsilon
theta = 0.0000275

#define the price array which will also be the actions vector 
@jit(nopython=True)
def price_array(k):
    return np.linspace(0,1,k)

#Episilon exploration parameter function. 
@jit(nopython=True)
def current_epsilon_value(t,theta=theta):
    epsilon = (1-theta)**t
    return epsilon

#NOTES: A Q table is Action X States matrix, where a action is a price, (Klein 2021), 
#which differs from Julius but not the other BA project (Morten and Johanne)
#Notes current_state corresponds to the index of the current state
#Under the markow assumption a players state is just the other players price and therefore actions
#Since the Players are only two and have the same number of actions, this will be a Q = AxA  matrix or 
#Q = PriceXPrice matrix
#This note is important for explaining in the BA.
#Should our Q-learning go back in time more than just the previous state, maybe 2 or 3, 
#it would break markow assumption but maybe there is somehting here
#Also we could explore if the player does not have the same price room as the other and see what then will happen

#Exploration selection

#def action_choice(Q, current_state_index, action_vector,t):
@jit(nopython=True)
def action_choice(Q, current_state_index,t):
    "Epsilon greedy selection"
    current_epsilon= current_epsilon_value(t)
    num_actions = Q.shape[0]
    if np.random.uniform(0,1) < current_epsilon:
        return np.random.randint(num_actions)

        # return np.random.choice(action_vector)
    else: 
        return np.argmax(Q[:,current_state_index])
        # max_index = np.argmax(Q[:,current_state_index])
        # return action_vector(max_index)

@jit(nopython=True)
def demand(pi,pj):
    if pi < pj:
        d = 1-pi
    elif pi > pj:
        d = 0
    elif pi == pj:
        d = 0.5 * (1-pi)
    return d

@jit(nopython=True)
def profit(pi,pj):
    return pi * demand(pi,pj)

# Initializes the 2 Q matrices v


@jit(nopython=True)
def seq_q_step(Q,current_action_index, old_state_index, current_state_index , t, action_vector , 
            gamma = gamma_discount_factor, alpha = alpha):

    current_price = action_vector[current_action_index]

    old_opponent_price = action_vector[old_state_index]

    old_profit = profit(current_price,old_opponent_price)

    current_opponent_price = action_vector[current_state_index]

    new_profit = profit(current_price,current_opponent_price)

    previous_estimate = Q[current_action_index,old_state_index]

    new_estimate = old_profit + gamma*new_profit+ (gamma**2) *np.max(Q[:,current_state_index])

    Q[current_action_index,old_state_index] = (1-alpha)* previous_estimate + alpha*new_estimate

    new_action_index = action_choice(Q,current_state_index,t)


    return Q, current_state_index, current_action_index, new_action_index


# define two Q matrices, one for each player
@jit(nopython=True)
def initialize_Q(k):
    Q1 = np.zeros((k,k))
    Q2 = np.zeros((k,k))
    return Q1,Q2

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

    # save the average running profit for each player. The average is taken over 100 time steps
    #running_average_profit_1 = []
    #running_average_profit_2 = []

    cumulative_profit_1 = 0
    cumulative_profit_2 = 0 

    profit_1 = []
    profit_2 = []

    for t in range(T):
        if (t%2) ==0:
            Q1,p2_current_index,p1_old_index,p1_current_index =seq_q_step(Q1,
            p1_current_index,
            p2_old_index,
            p2_current_index,
            t,
            action_vector)
            #average profit over 100 time steps, this is not the same as the average profit over the whole simulation
            

            #Store profit for player 1 and 2
            profit_1.append(profit(action_vector[p1_current_index], action_vector[p2_current_index]))
            profit_2.append(profit(action_vector[p2_current_index], action_vector[p1_current_index]))

        else: 
            Q2,p1_current_index,p2_old_index,p2_current_index =seq_q_step(Q2,
            p2_current_index,
            p1_old_index,
            p1_current_index,
            t,
            action_vector)


            #Store profit for player 1 and 2
            profit_1.append(profit(action_vector[p1_current_index], action_vector[p2_current_index]))
            profit_2.append(profit(action_vector[p2_current_index], action_vector[p1_current_index]))
            

    return Q1,Q2, profit_1, profit_2


# Test the simulation with running averages
@jit(nopython=True)
def test_simulation_q_learning_with_running_average():
    T = 500_000  # Use a smaller T for testing
    k = 100
    window_size = 1000
    Q1, Q2, profit_1, profit_2 = simulation_q_learning(T, k)

    # Convert profit lists to NumPy arrays for compatibility with Numba
    profit_1 = np.array(profit_1)
    profit_2 = np.array(profit_2)

    moving_avg_player1 = []
    moving_avg_player2 = []

    for i in range(1,T-window_size):
        moving_avg_player1.append(np.sum(profit_1[i:i+window_size]) / window_size)
        moving_avg_player2.append(np.sum(profit_2[i:i+window_size]) / window_size)



    # Print the final running averages for visual inspection
    #print("Final running average profit for Player 1:", moving_avg_player1[0:20])
    #print("Final running average profit for Player 2:", profit_2[T-20:T])
    return moving_avg_player1, moving_avg_player2

@jit(nopython=True)
def simulate_multiple_runs(num_runs=1000, T=500_000, k=6, window_size=1000):
    # Preallocate accumulators for the moving averages
    cumulative_avg_player1 = np.zeros(T - window_size - 1)
    cumulative_avg_player2 = np.zeros(T - window_size - 1)

    for run in range(num_runs):
        print(f"Running simulation {run + 1}/{num_runs}...")
        # Run a single simulation
        Q1, Q2, profit_1, profit_2 = simulation_q_learning(T, k)

        # Convert profit lists to NumPy arrays
        profit_1 = np.array(profit_1)
        profit_2 = np.array(profit_2)

        # Preallocate arrays for moving averages for this run
        moving_avg_player1 = np.zeros(T - window_size - 1)
        moving_avg_player2 = np.zeros(T - window_size - 1)

        # Calculate moving averages for this run
        for i in range(1, T - window_size):
            moving_avg_player1[i - 1] = np.sum(profit_1[i:i + window_size]) / window_size
            moving_avg_player2[i - 1] = np.sum(profit_2[i:i + window_size]) / window_size

        # Accumulate the moving averages
        cumulative_avg_player1 += moving_avg_player1
        cumulative_avg_player2 += moving_avg_player2

    # Compute the average moving averages across all runs
    avg_moving_avg_player1 = cumulative_avg_player1 / num_runs
    avg_moving_avg_player2 = cumulative_avg_player2 / num_runs

    return avg_moving_avg_player1, avg_moving_avg_player2

# Simulate 1000 runs
num_runs = 1000
T = 500_000
k = 100
window_size = 1000

avg_moving_avg_player1, avg_moving_avg_player2 = simulate_multiple_runs(num_runs, T, k, window_size)



# Plot the moving average of the profit for player 1
import matplotlib.pyplot as plt



start_time = time.time()

# Plot the results
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Plot Player 1's average moving average
axes[0].plot(avg_moving_avg_player1, label="Player 1 Average Moving Average Profit", color="blue")
axes[0].set_xlabel("Time Steps")
axes[0].set_ylabel("Profit (Moving Average)")
axes[0].set_title("Player 1: Average Moving Average Profit Across 1000 Runs")
axes[0].axhline(y=0.125, color='r', linestyle='--', label="Collusive benchmark")
axes[0].axhline(y=0.0611, color='g', linestyle='--', label="Market benchmark")
axes[0].set_ylim(bottom=0)
axes[0].legend()
axes[0].grid(True)

# Plot Player 2's average moving average
axes[1].plot(avg_moving_avg_player2, label="Player 2 Average Moving Average Profit", color="red")
axes[1].set_xlabel("Time Steps")
axes[1].set_ylabel("Profit (Moving Average)")
axes[1].set_title("Player 2: Average Moving Average Profit Across 1000 Runs")
axes[1].axhline(y=0.125, color='r', linestyle='--', label="Collusive benchmark")
axes[1].axhline(y=0.0611, color='g', linestyle='--', label="Market benchmark")
axes[1].set_ylim(bottom=0)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

'''
# Run the test
moving_avg_player1, moving_avg_player2 = test_simulation_q_learning_with_running_average()
# Plot the moving average of the profit for Player 1
plt.figure(figsize=(10, 6))
plt.plot(moving_avg_player1, label="Player 1 Moving Average Profit", color="blue")
plt.xlabel("Time Steps")
plt.ylabel("Profit (Moving Average)")
plt.title("Moving Average Profit for Player 1")
#Add a dotted line at x = 0.125
plt.axhline(y=0.125, color='r', linestyle='--', label="Collusive benchmark")
#Add a dotted line at x = 0.0611
plt.axhline(y=0.0611, color='g', linestyle='--', label="Market benchmark")
# Set the y-axis to start from 0
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.show()

# Plot the moving average of the profit for player 2
plt.figure(figsize=(10, 6))
plt.plot(moving_avg_player2, label="Player 2 Moving Average Profit", color="red")
plt.xlabel("Time Steps")
plt.ylabel("Profit (Moving Average)")
plt.title("Moving Average Profit for Player 2")
#Add a dotted line at x = 0.125
plt.axhline(y=0.125, color='r', linestyle='--', label="Collusive benchmark")
#Add a dotted line at x = 0.0611
plt.axhline(y=0.0611, color='g', linestyle='--', label="Market benchmark")
# Set the y-axis to start from 0
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.show()

'''

end_time = time.time()

elapsed = end_time - start_time
print(f"Execution took {elapsed:.3f} seconds.")
