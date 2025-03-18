
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
    return np.linspace(0,1,k)

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

#def action_choice(Q, current_state_index, action_vector,t):
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

# Initializes the 2 Q matrices v



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
def initialize_Q(k):
    Q1 = np.zeros((k,k))
    Q2 = np.zeros((k,k))
    return Q1,Q2



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
    running_profit_1 = []
    running_profit_2 = []
    for t in range(T):
        if (t%2) ==0:
            Q1,p2_current_index,p1_old_index,p1_current_index =seq_q_step(Q1,
            p1_current_index,
            p2_old_index,
            p2_current_index,
            t,
            action_vector)
            if  t>9950 and t<10000:
                running_profit_1.append(profit(action_vector[p1_current_index],action_vector[p2_current_index]))
        else: 
            Q2,p1_current_index,p2_old_index,p2_current_index =seq_q_step(Q2,
            p2_current_index,
            p1_old_index,
            p1_current_index,
            t,
            action_vector)
        

            #if we want to store action do it hea

    return Q1,Q2, running_profit_1

#Run a test simulation
Q1,Q2,r1 = simulation_q_learning(T,7)
print(Q1)
print(Q2)
print(r1)


