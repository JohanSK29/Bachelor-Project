import matplotlib.pyplot as plt

# Define the best response functions
def QBR_i(pj):
    if pj in [0, 0.167]:
        return 0
    elif pj == 0.333:
        return 0.167
    elif pj == 0.5:
        return 0.333
    elif pj in [0.667, 0.833, 1]:
        return 0.5
    else:
        return None

def QBR_j(pi):
    if pi in [0, 0.167, 0.667]:
        return 0.833
    elif pi == 0.333:
        return 0.167
    elif pi == 0.5:
        return 0.333
    elif pi in [0.833, 1]:
        return 0.667
    else:
        return None

# Price grid
grid = [0, 0.167, 0.333, 0.5, 0.667, 0.833, 1]


# Evaluate BR functions
br_i_x = [QBR_j(pi) for pi in grid]  
br_i_y = grid                        

br_j_x = grid                        
br_j_y = [QBR_i(pj) for pj in grid]  

# Create the plot
plt.figure(figsize=(6, 6))
plt.scatter(br_i_x, br_i_y, label=r'$QBR_j(p_i)$', color='blue', marker='o', s=80)
plt.scatter(br_j_x, br_j_y, label=r'$QBR_i(p_j)$', color='red', marker='x', s=80)

# Style
plt.xlabel(r'$p_j$', fontsize=14)
plt.ylabel(r'$p_i$', fontsize=14)
plt.xticks(grid)
plt.yticks(grid)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Q Best Response Functions (Discrete)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()
