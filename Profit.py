from simulation_newest_main import profit
print(profit(0.5,0.83) , profit(0.83,0.5))
print(profit(0.5,0.33),profit (0.33,0.5))
print(profit(0.167,0.33),profit(0.33,0.167))
print(profit(0.167,0.83),profit(0.83,0.167))

# List of price pairs
price_pairs = [
    (0.340, 0.420), (0.340, 0.280), (0.270, 0.280), (0.270, 0.140),
    (0.910, 0.140), (0.910, 0.590), (0.330, 0.590), (0.330, 0.240),
    (0.120, 0.240), (0.120, 0.980), (0.600, 0.980), (0.600, 0.420)
]

# Loop through each pair and calculate profit for both orderings
for pi, pj in price_pairs:
    profit_ij = profit(pi, pj)
    profit_ji = profit(pj, pi)
    print(f"profit({pi}, {pj}) = {profit_ij:.4f}, profit({pj}, {pi}) = {profit_ji:.4f}")