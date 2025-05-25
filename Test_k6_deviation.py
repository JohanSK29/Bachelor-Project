from simulation_newest_main import profit

start_positions_i_deviate_1= [(0,0.333),(0.333,0.333),(1,0.333),(0.833,0.333),(0.667,0.333),(0.5,0.333)]

start_positions_i_deviate_2 = [(0,0.833),(1,0.833),(0.167,0.833),(0.833,0.833),(0.667,0.833),(0.333,0.833)]

start_positions_j_deviate_1 = [(0.5,0.833),(0.5,1),(0.5,0),(0.5,0.167),(0.5,0.5),(0.5,0.667)]

start_positions_j_deviate_2 = [(0.167,0.333),(0.167,1),(0.167,0),(0.167,0.5),(0.167,0.667),(0.167,0.167)]

def QBRi(pj):
    if pj == 0.167:
        return 0
    if pj == 0:
        return 0
    
    if pj == 0.333:
        return 0.167

    if pj == 0.5:
        return 0.333

    if pj == 0.667:
        return 0.5
    if pj == 0.833:
        return 0.5
    if pj == 1:
        return 0.5
    else: 
        return "Fail"


def QBRj(pi):
    if pi == 0.333:
        return 0.167
    if pi == 0.5:
        return 0.333 
    if pi == 0.833:
        return 0.667
    if pi == 1:
        return 0.667
    if pi == 0:
        return 0.833
    if pi == 0.167:
        return 0.833
    if pi == 0.667:
        return 0.833
    else: 
        return "Fail"


def test_i_deviate(pi,pj):
    profits_i_j=[]
    pi_pj = []
    pi_pj.append((pi,pj))
    a=True
    profit_i = profit(pi,pj)
    profit_j = profit(pj,pi)
    profits_i_j.append((profit_i,profit_j))
    #print(profits_i_j)
    #print(pi_pj)
    x=0
    while a:
        if x%2:
            pi=QBRi(pj)
        else:
            pj=QBRj(pi)
            
        x+=1
        if (pi,pj) ==(0.5, 0.833) or (pi,pj)==(0.5,0.333) or (pi,pj) == (0.167,0.333) or (pi,pj) == (0.167,0.833):
            print("back in cycle")
            break

        pi_pj.append((pi,pj))
        profit_i = profit(pi,pj)
        profit_j = profit(pj,pi)
        profits_i_j.append((profit_i,profit_j))
        #print(profits_i_j)
        #print(pi_pj)
    avg_profit_i = sum(p[0] for p in profits_i_j) / len(profits_i_j)
    return avg_profit_i

#print(test_i_deviate(0.5,0.333))

def test_j_deviate(pi,pj):
    profits_i_j=[]
    pi_pj = []
    pi_pj.append((pi,pj))
    a=True
    profit_i = profit(pi,pj)
    profit_j = profit(pj,pi)
    profits_i_j.append((profit_i,profit_j))
    #print(profits_i_j)
    #print(pi_pj)
    x=0
    while a:
        if x%2:
            pj=QBRj(pi)
        else:
            pi=QBRi(pj)
        x+=1
        if (pi,pj) ==(0.5, 0.833) or (pi,pj)==(0.5,0.333) or (pi,pj) == (0.167,0.333) or (pi,pj) == (0.167,0.833):
            print("back in cycle")
            break

        pi_pj.append((pi,pj))
        profit_i = profit(pi,pj)
        profit_j = profit(pj,pi)
        profits_i_j.append((profit_i,profit_j))
        #print(profits_i_j)
        #print(pi_pj)
    avg_profit_j = sum(p[1] for p in profits_i_j) / len(profits_i_j)

    return avg_profit_j

avg_i_1 = []
avg_i_2 = []
for i in range (len(start_positions_i_deviate_1)):
    avg_i_1.append((start_positions_i_deviate_1[i],test_i_deviate(start_positions_i_deviate_1[i][0],start_positions_i_deviate_1[i][1])))

for i in range (len(start_positions_i_deviate_2)):
    avg_i_2.append((start_positions_i_deviate_2[i],test_i_deviate(start_positions_i_deviate_2[i][0],start_positions_i_deviate_2[i][1])))

print(avg_i_1)
print(avg_i_2)


avg_j_1 = []
avg_j_2 = []
for i in range (len(start_positions_j_deviate_1)):
    avg_j_1.append((start_positions_j_deviate_1[i],test_j_deviate(start_positions_j_deviate_1[i][0],start_positions_j_deviate_1[i][1])))

for i in range (len(start_positions_j_deviate_2)):
    avg_j_2.append((start_positions_j_deviate_2[i],test_j_deviate(start_positions_j_deviate_2[i][0],start_positions_j_deviate_2[i][1])))

print(avg_j_1)
print(avg_j_2)
