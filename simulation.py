
import numpy as np

#Parameters:
#time steps, 500_000
T = 500_000
#Discount factor
gamma_discount_factor = 0.95
#Learning parameter
alpha = 0.3
#Decay parameter for Epsilon
theta = 0.00072

#define the price array which will also be the actions vector 
def price_array(k):
    return np.linspace(0,1,k+1)

#Episilon exploration parameter function. 
def current_epsilon_value(t,theta=theta):
    epsilon = (1-theta)**t
    return epsilon

#NOTES: A Q table is Action X States matrix, where a action is a price, (Klein 2021), 
#which differs from Julius but not the other BA project (Morten and Johanne)
#Notes current_state corresponds to the index of the current state
# Under the markow assumption a players state is just the other players price and therefore actions
# Since the Players are only two and have the same number of actions, this will be a Q = AxA  matrix or 
# Q = PriceXPrice matrix
# This note is important for explaining in the BA.
# Should our Q-learning go back in time more than just the previous state, maybe 2 or 3, 
# it would break markow assumption but maybe there is somehting here
# Also we could explore if the player does not have the same price room as the other and see what then will happen

#Exploration selection

def action_choice(Q, current_state_index, action_vector,t):
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

#Exploration selection

def demand(pi,pj):
    if pi < pj:
        d = 1-pi
    elif pi > pj:
        d = 0
    elif pi == pj:
        d = 0.5 * (1-pi)
    return d

def profit(pi,pj):
    return pi * demand(pi,pj)


def seq_q_step(Q, current_state, action_vector, t, Price_current_agent, Price_opponent, 
gamma = gamma_discount_factor, alpha = alpha):



    new_estimate =

    previous_estimate = 

    Q_new = 

    action = action_choice(Q, current_state, action_vector, t)


