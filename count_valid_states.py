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
import random as random
# Counting the number of valid states for the simplified model for a varying number of days.
# Functions are also used when plotting measurement results in the ipynb files
# to automatically color the bars that correspond to valid states.

# returns the boolean value of whether the input state_string (such as (simulated) measurement results of a quantum circuit)
def validate_state(state_string, number_of_days, C_max=-1, B=5, B_max=10, V_target=-1, delta=-1, shifts_s1=[0,5,8,10], shifts_s2=[0,4,7,9], costs_s1=[0,5,8,10], costs_s2=[0,4,7,9], num_qubits_per_shop_per_day=2):
    # values from the simplified model, function can also be used with different input numbers.

    # Default values for these dependent on # of days 
    if V_target < 0: 
        V_target = 8*number_of_days
    if delta < 0:
        delta = round(0.05*V_target) 
    if delta == 0:
        delta = 1 # It can only be zero for running this for one day, but then there are no valid solutions so we set it to 1
    if C_max < 0:
        C_max = (max(costs_s1) + max(costs_s2) + 1)*number_of_days

    conditions = []
    V_out = B
    C_out = 0

    # We split the state string to get the bits per shop per day.
    for day in range(number_of_days):
        s1_index = int('0b'+state_string[day*num_qubits_per_shop_per_day:(day+1)*num_qubits_per_shop_per_day], 2)
        s2_index = int('0b'+state_string[(number_of_days+day)*num_qubits_per_shop_per_day:(number_of_days+day+1)*num_qubits_per_shop_per_day], 2)
        shift_s1 = shifts_s1[s1_index]
        shift_s2 = shifts_s2[s2_index]
        B += (shift_s1-shift_s2)
        B = max([0,B])
        V_out += shift_s1
        C_out += (costs_s1[s1_index] + costs_s2[s2_index])
        conditions.append(B<=B_max)
    V_out -= B
    conditions.append(V_out >= V_target-delta)
    conditions.append(V_out <= V_target+delta)
    conditions.append(C_out < C_max)
    return(all(conditions))

# Returns:
#   valid_list: for all possible input states (shift schedules): [validity of state (True/False), cost of the schedule]
#   valid_lst_cst: Range(20*ndays, C_min, -1)   (used for plotting)
#   valid_lst_sols: List with the number of valid solutions that have a cost below the corresponing item in valid_lst_cst.
#   So if valid_lst_sols[i] = 63, then there are 63 solutions with a lower cost than the number in valid_lst_cst[i]. 
def all_valid_states_and_costs(ndays):
    valid_lst = []
    for i in range(16**ndays):
        state_string = format(i, '#0' + str(4*ndays+2) + 'b')[2:]
        conditions = []
        B = 5
        B_max = 10
        V_target = 8*ndays
        delta = round(0.05*V_target) 
        if delta == 0:
            # It can only be zero for running this for 1 day, but then there are no valid solutions so let's set it to 1
            delta = 1 
        shifts_s1 = [0,5,8,10]
        shifts_s2 = [0,4,7,9]
        costs_s1 = shifts_s1
        costs_s2 = shifts_s2
        num_qubits_per_shop_per_day = 2
        V_out = B
        C_out = 0

        for day in range(ndays):
            s1_index = int('0b'+state_string[day*num_qubits_per_shop_per_day:(day+1)*num_qubits_per_shop_per_day], 2)
            s2_index = int('0b'+state_string[(ndays+day)*num_qubits_per_shop_per_day:(ndays+day+1)*num_qubits_per_shop_per_day], 2)
            shift_s1 = shifts_s1[s1_index]
            shift_s2 = shifts_s2[s2_index]
            B += (shift_s1-shift_s2)
            B = max([0,B])
            V_out += shift_s1
            C_out += (costs_s1[s1_index] + costs_s2[s2_index])
            conditions.append(B<=B_max)
        
        V_out -= B
        conditions.append(V_out >= V_target-delta)
        conditions.append(V_out <= V_target+delta)
        valid_lst.append([all(conditions), C_out])

    C_max = 20*ndays

    valid_lst_cst = []
    valid_lst_sols = []
    nsols = 1
    while(nsols > 0):
        nsols = 0
        for item in valid_lst:
            if item[0] and item[1] < C_max:
                nsols += 1
        valid_lst_cst.append(C_max)
        valid_lst_sols.append(nsols)
        C_max -= 1
    return valid_lst, valid_lst_cst, valid_lst_sols

if __name__ == "__main__":
    ndays = 2

    valid_lst, valid_lst_cst, valid_lst_sols = all_valid_states_and_costs(ndays)

    print("Number of valid solutions: ", valid_lst_sols[0], ", minimum cost: ", valid_lst_cst[-1])
    plt.bar(range(len(valid_lst_sols)), valid_lst_sols, tick_label=[str(i) for i in valid_lst_cst])
    plt.xlabel("Maximum cost ($)")
    plt.ylabel("Number of valid solutions")
    numberdays = ["zero days", "one day", "two days", "three days", "four days", "five days", "six days", "seven days", "eight days", "nine days", "ten days"]

    plt.title("Number of valid solutions\nfor " + numberdays[ndays] + ", with decreasing maximum cost")
    plt.show()
