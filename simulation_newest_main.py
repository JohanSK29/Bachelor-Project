import time
import numpy as np
import matplotlib as plt
from numba import jit, config, prange


np.random.seed(123)  # For reproducibility


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
def seq_q_step(Q, current_action_index, old_state_index, current_state_index, t, action_vector , 
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
            
            #average profit over 100 time steps, this is not the same as the average profit over the whole simulation
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



@jit(nopython=True)
def simulation_random_players(T,k):
    action_vector = price_array(k)

    #number of actions :
    number_of_actions = len(action_vector)
    
    profit_1 = []
    profit_2 = []
    

    p1_current_index = np.random.randint(number_of_actions)

    p2_current_index = np.random.randint(number_of_actions)

    for t in range(T):
        if (t % 2) == 0:
            # Spiller 1 vælger ny tilfældig pris
            p1_current_index = np.random.randint(number_of_actions)
        else:
            # Spiller 2 vælger ny tilfældig pris
            p2_current_index = np.random.randint(number_of_actions)

        p1_price = action_vector[p1_current_index]
        p2_price = action_vector[p2_current_index]

        # Beregn og gem profit for begge spillere
        profit_1.append(profit(p1_price, p2_price))
        profit_2.append(profit(p2_price, p1_price))

    return profit_1, profit_2


#the most competitive Edgeworth price cycle MPE identified by Maskin and Tirole (1988):
#  Firms undercut each other by one increment until prices reach their lower bound, after which one firm resets prices to one increment above monopoly price and the cycle restarts.
#  It is taken that the first firm that observes the lower-bound price resets the price cycle. 
# This provides in this case an average per-period profit of approximately 0.0611 for  k=6 (which increases in the limit of k= inf to approximately 0.0833).

def simulate_klein_edgeworth_cycle_compt_benchmark(k,cycles):
    prices = np.linspace(0, 1, k)
    step = prices[1] - prices[0]

    # Monopoly price and reset target
    monopoly_index = np.argmax([profit(p, p) for p in prices])
    reset_price = min(prices[monopoly_index] + step, 1.0)
    reset_index = np.argmin(np.abs(prices - reset_price))

    # Start at reset price adn the first step, which is for firm 2 to undercut firm 1 
    p1_index = reset_index
    p2_index = monopoly_index

    history = []
    turn = 0
    cycles_counter = 0
    #wihile loops starts with firm 1, turn 
    while True:
        if turn>1 and p1_index == reset_index and p2_index == monopoly_index:
            cycles_counter +=1

        if cycles_counter == cycles:
            break

    
        p1_price = prices[p1_index]
        p2_price = prices[p2_index]
        history.append((p1_price, p2_price))
        # Start state detection for exit


        # Alternate moves
        if turn % 2 == 0:
            # Player 1 moves
            if p2_index == 0:
                p1_index = reset_index 

            elif p2_index > 0:
                p1_index = p2_index-1
        
        else:
            # Player 2 moves
            if p1_index == 0:
                p2_index = reset_index 


            elif p1_index > 0:
                p2_index = p1_index-1

        turn += 1

        if turn > 2000000:
            raise RuntimeError("Cycle didn't converge — something's off.")

    # Compute profits
    profits_1 = [profit(p1, p2) for p1, p2 in history]
    profits_2 = [profit(p2, p1) for p1, p2 in history]

    return np.mean(profits_1), np.mean(profits_2),np.mean(profits_1+profits_2), history


'''
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
'''

def detect_price_cycle(prices_1, prices_2, max_cycle_len=50, min_repeats=2, tolerance=1e-5):
    """
    Detects repeated price cycles in two 1D lists or arrays of prices.

    Args:
        prices_1: List or array of recent prices for player 1 (e.g., last 200)
        prices_2: List or array of recent prices for player 2 (e.g., last 200)
        max_cycle_len: Maximum cycle length to search for
        min_repeats: Minimum number of repetitions to count as a cycle
        tolerance: Allowed absolute difference for matching due to float rounding

    Returns:
        (cycle_length, combined_pattern) if found, otherwise (None, None)
        combined_pattern is a tuple of (player_1_price, player_2_price) for each step in the cycle.
    """
    prices_1 = np.array(prices_1)
    prices_2 = np.array(prices_2)
    n = len(prices_1)

    # Ensure both price lists have the same length
    if len(prices_1) != len(prices_2):
        raise ValueError("prices_1 and prices_2 must have the same length")

    for cycle_len in range(2, max_cycle_len + 1):
        num_possible_repeats = n // cycle_len
        if num_possible_repeats < min_repeats:
            continue

        # Take the last cycle_len points as candidate patterns for both players
        pattern_1 = prices_1[-cycle_len:]
        pattern_2 = prices_2[-cycle_len:]

        match = True
        for i in range(2, min_repeats + 1):
            start = -i * cycle_len
            end = start + cycle_len
            if end < -n:
                match = False
                break

            # Check if the windows match the patterns for both players
            window_1 = prices_1[start:end]
            window_2 = prices_2[start:end]
            if not (np.allclose(window_1, pattern_1, atol=tolerance) and
                    np.allclose(window_2, pattern_2, atol=tolerance)):
                match = False
                break

        if match:
            # Combine the patterns into a single tuple for each step
            combined_pattern = tuple((p1, p2) for p1, p2 in zip(pattern_1, pattern_2))
            return cycle_len, combined_pattern

    return None, None

#avg1, avg2, avg_common, hist = simulate_klein_edgeworth_cycle_compt_benchmark(k=3001,cycles = 1)
#print(f"P1 avg profit: {avg1:.4f}, P2 avg: {avg2:.4f}, common: {avg_common:.4f}")
#print(f"Cycle length: {len(hist)}")
#print(hist)
