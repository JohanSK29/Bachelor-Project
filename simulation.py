
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

#Exploration selection
def action_choice(Q, current_state, action_vector,t):
    "Epsilon greedy selection"
    current_epsilon= current_epsilon_value(t)
    if np.random.uniform(0,1) < current_epsilon:
        return np.random.choice(action_vector)
    else: 
        return print("Lortepik")

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


