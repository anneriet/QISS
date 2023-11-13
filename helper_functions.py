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

import numpy as np
import random as random
from qiskit import QuantumRegister

# To use separately defined Qiskit registers as a single set of qubits (for example as the target for subcircuits)
def combine_registers(reg_a, reg_b):
    combinelist = []
    if((type(reg_a)) == QuantumRegister or (type(reg_a) == list)):
        for qubit in reg_a:
            combinelist.append(qubit)
    else:
        combinelist.append(reg_a)
    if((type(reg_b) == QuantumRegister) or (type(reg_b) == list)):
        for qubit in reg_b:
            combinelist.append(qubit)
    else:
        combinelist.append(reg_b)
    return combinelist

# https://www.geeksforgeeks.org/decimal-equivalent-gray-code-inverse/
def grayCode(n):
    # Right Shift the number by 1 taking xor with original number
    return n ^ (n >> 1)

# reorderding the 0s and 1s in the measurement results from the order of the qubits in the register or from custom index_lst. 
def counts_select_from_reg(counts, circuit, register):
    index_lst = []
    for qubit in register:
        index_lst.append(circuit.find_bit(qubit).index)
    newdict = {}
    for key, value in counts.items():
        newkey = ''.join([key[x] for x in index_lst])
        newdict[newkey] = value
    return newdict

def counts_select_from_ind(counts, circuit, index_lst):
    newdict = {}
    for key, value in counts.items():
        newkey = ''.join([key[x] for x in index_lst])
        newdict[newkey] = value
    return newdict

# calculate the cost of a solution
def calculatecost(state_string, costs_s1, costs_s2, number_of_days, num_qubits_per_shop_per_day):
    C_out = 0
    for day in range(number_of_days):
        s1_index = int('0b'+state_string[day*num_qubits_per_shop_per_day:(day+1)*num_qubits_per_shop_per_day], 2)
        s2_index = int('0b'+state_string[(number_of_days+day)*num_qubits_per_shop_per_day:(number_of_days+day+1)*num_qubits_per_shop_per_day], 2)
        C_out += (costs_s1[s1_index] + costs_s2[s2_index])
    return(C_out)