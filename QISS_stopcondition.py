#  Copyright 2023 Anna Maria Krol

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import mstats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from count_valid_states import all_valid_states_and_costs
# Validating the stopcondition of grovers adaptive search with plot
# And some additional functions that were used in the process of deciding on a stop condition for GAS that might be of interest.


# Calculate wether a run of Grover's lands on a marked state (success) based on the chance of that happening
def grovers(N, t, j):
    theta = np.arcsin(np.sqrt(t/N))
    kj = 1/np.sqrt(t)*np.sin((2*j+1)*theta)
    lj = 1/np.sqrt(N-t)*np.cos((2*j+1)*theta)
    P_success = t*kj**2
    P_fail = (N-t)*lj**2
    return P_success

# Calculate how many iterations it costs to land on a marked state (success) when we don't know the number of solutions
def G_unknown_number_sols(N,t):
    total_iterations = 0
    success = False
    m = 1
    lam = 6/5
    while(not success):
        j = random.randint(0, np.ceil(m-1))
        total_iterations += j
        P_success = grovers(N, t, j)
        success = (random.random() < P_success)
        m = np.min([lam*m, np.sqrt(N)])
        if total_iterations > 10:
            print("Total iterations > 10, exiting loop")
            success = True
    return total_iterations

# Count the number of iterations before the minimum cost has been achieved
def GAS_iters_before_mincost(N, costs, valid, max_iter, C_init, C_min):
    total_iterations = 0
    iterations = []
    m = 1
    lam = 6/5
    C_max = C_init
    while total_iterations < max_iter and C_max > C_min:
        j = random.randint(0, np.ceil(m-1))
        total_iterations += j
        m = np.min([lam*m, np.sqrt(N)])
        possibilities = []
        for i in range(len(costs)):
            if costs[i] < C_max and valid[i]==1:
                possibilities.append(costs[i])
        if len(possibilities) > 0:
            P_success = grovers(N, len(possibilities), j)
            if(random.random() < P_success):
                C_max = random.choice(possibilities)
                m = 1
        iterations.append(j)    
    if C_max != C_min:
        return [-1]
    return iterations

# Run Grover's adaptive search for 10 iterations, see what the cost is at the end
def GAS(costs, valid):
    total_iterations = 0
    m = 1
    lam = 6/5
    C_max = 20
    for _ in range(10):
        j = random.randint(0, np.ceil(m-1))
        total_iterations += j
        m = np.min([lam*m, np.sqrt(N)])
        possibilities = []
        for i in range(len(costs)):
            if costs[i] < C_max and valid[i]==1:
                possibilities.append(costs[i])
        if len(possibilities) > 0:
            P_success = grovers(N, len(possibilities), j)
            if(random.random() < P_success):
                C_max = random.choice(possibilities)
                m = 1
        else:
            P_success = 0
    return C_max

# Run Grover's adaptive search for a set amount of iterations, see what the cost is that it finds
def GAS_max_iter(N, costs, valid, max_iter, C_init):
    total_iterations = 0
    m = 1
    lam = 6/5
    C_max = C_init
    cost_lst = []
    while total_iterations < max_iter:
        j = random.randint(0, np.ceil(m-1))
        if (total_iterations+j <= max_iter):
            total_iterations += j
            m = np.min([lam*m, np.sqrt(N)])
            possibilities = []
            for i in range(len(costs)):
                if costs[i] < C_max and valid[i]==1:
                    possibilities.append(costs[i])
            if len(possibilities) > 0:
                P_success = grovers(N, len(possibilities), j)
                if(random.random() < P_success):
                    C_max = random.choice(possibilities)
                    m = 1
            
            cost_lst.append([total_iterations, C_max])
    if(total_iterations != max_iter):
        print("Error: ")
        print("Total iterations: ", total_iterations)
    return cost_lst



if __name__ == "__main__":

    numberofdays = 3

    N = 2**(4*numberofdays)
    sqrt_N = 2**(2*numberofdays)
    C_init = 19*numberofdays + 1

    valid_lst, valid_lst_cst, valid_lst_sols = all_valid_states_and_costs(numberofdays)
    costs = [i[1] for i in valid_lst]
    C_min = valid_lst_cst[-1]
    valid = [i[0] for i in valid_lst]

    total_runs = 10000
    data_for_average_number_of_iters = []
    max_iters = 2*sqrt_N

    data = []
    for _ in range(total_runs):
        data.append(GAS_max_iter(N, costs, valid, max_iters, C_init))

    data_new = [[] for i in range(max_iters+1)]
    for i in range(len(data)):
        lst = data[i]
        for item in lst:
            data_new[item[0]].append(item[1])

    min_list = np.zeros(len(data_new)+1)
    ave_list = np.zeros(len(data_new)+1)
    max_list = np.zeros(len(data_new))
    quantiles = np.zeros((len(data_new),7))

    # so both lines in the plot start at the minimum initial cost (because the way the algorithm works there are many runs that have found a lower cost at 0 iterations)
    min_list[0] = C_init
    ave_list[0] = C_init
    for i in range(len(data_new)):
        min_list[i+1] = min(data_new[i])
        ave_list[i+1] = sum(data_new[i])/len(data_new[i])
        max_list[i] = max(data_new[i])
        quantiles[i] = mstats.mquantiles(data_new[i], prob=[0.01, 0.05,0.25, 0.5,0.75, 0.95, 0.99])

    # set the maximum to the max of all the values with more rotations (later in list)
    for i in range(len(max_list)): 
        max_list[i] = max(max_list[i:])

    # Same for the quantiles
    for i in range(len((quantiles.T)[-1])):
        (quantiles.T)[-1][i] = max((quantiles.T)[-1][i:])
        (quantiles.T)[-2][i] = max((quantiles.T)[-2][i:])
        (quantiles.T)[-3][i] = max((quantiles.T)[-3][i:])

    # list with extra zero for plotting
    x_extra0 = [0]
    x_extra0.extend(list(range(len(data_new))))
    
    # colors for plotting, starts with dark blue and ends with very light blue
    color_lst = ['#1f77b4', '#568cc1', '#7ba2cd', '#9eb8da', '#becfe6', '#dfe7f3', '#e5ebf6', '#e1e7f1']
    
    # Two subplots to make it look like a broken y-axis
    fig, (ax,ax2) = plt.subplots(2,1, figsize=(4,3), sharex=True, gridspec_kw={'height_ratios': [12,1]})

    ax.plot(x_extra0, min_list, label='Minimum', color=color_lst[6])
    ax.plot(range(len(data_new)), max_list, label='Maximum', color=color_lst[6])

    ax.plot(x_extra0, ave_list, color=color_lst[0])
    ax.plot(range(len(data_new)), quantiles.T[3], color=color_lst[1])

    ax.fill_between(range(len(data_new)), y1=(quantiles.T)[0], y2=(quantiles.T)[-1], color=color_lst[7])
    ax.fill_between(range(len(data_new)), y1=(quantiles.T)[1], y2=(quantiles.T)[-2], color=color_lst[4])
    ax.fill_between(range(len(data_new)), y1=(quantiles.T)[2], y2=(quantiles.T)[-3], color=color_lst[3])

    legend_elements = [Patch(facecolor='#ffffff', edgecolor=color_lst[6], label="0%-100%"),
                    Patch(facecolor=color_lst[7], edgecolor=color_lst[7], label="1%-99%"),
                    Patch(facecolor=color_lst[4], edgecolor=color_lst[4], label="5%-95%"),
                    Patch(facecolor=color_lst[3], edgecolor=color_lst[3], label="25%-75%"),
                    Line2D([0], [0], color=color_lst[2], lw=2, label='Median'),
                    Line2D([0], [0], color=color_lst[0], lw=2, label='Average')]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax.plot([sqrt_N,sqrt_N],[min_list[-1],C_init+0.5], lw=1, color='gray', clip_on=False)
    ax2.plot([sqrt_N,sqrt_N],[0,4], color='gray', lw=1, clip_on=False)

    plt.xticks(range(0,2*N,20))

    # Making the plot look nice
    ax.set_ylim(min_list[-1]-0.5,C_init + 0.5)
    ax2.set_ylim(0,1.5)

    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False, top=False)

    ax.set_yticks(range(int(min_list[-1]),C_init + 1))
    ax.set_yticklabels(range(int(min_list[-1]),C_init + 1), fontsize=6)
    ax2.set_yticks(range(2))
    ax2.set_yticklabels(range(2), fontsize=6)

    d = .015  # how big to make the diagonal lines in axes coordinates
    ax.plot((-d, +d), (-d, +d), transform=ax.transAxes, color='k', clip_on=False, lw=1)        # top-left diagonal
    ax2.plot((-d, +d), (1 - 12*d, 1 + 12*d), transform=ax2.transAxes, color='k', clip_on=False, lw=1)  # bottom-left diagonal
    
    plt.text(sqrt_N + 0.2, 0.3, '√N', fontsize=10)

    fig.subplots_adjust(hspace=0.075, bottom=0.15, left=0.15)
    plt.xlabel('Number of rotations', size=10)
    ax.set_ylabel('Cost ($)', size=10)
    ax.set_title("Minimum cost found by GAS\nvs total # of rotations", size=10)
    plt.show()
