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
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import PhaseGate, QFT
from helper_functions import combine_registers, grayCode
from mcphase_cnx_noancillas import cnx
# See the Python notebook files for explanations about the functions
# Operators are also defined here for importing into the jupyter notebooks

def U(circuit, shiftlength, data_qubits):
    name = "add" 
    sub_circ = QuantumCircuit(data_qubits, name=name)
    factor = 1
    for target_qbit in data_qubits:
        sub_circ.append(PhaseGate(factor*np.pi*shiftlength),[target_qbit])
        factor /= 2
    circuit.append(sub_circ,data_qubits)

def U_controlled(circuit, shiftlength, control_qubits, data_qubits, ctrl_state):
    name = "add"
    newctrlstate = format(ctrl_state, '#0' + str(len(control_qubits)+2) + 'b')[:1:-1]
    sub_circ = QuantumCircuit(control_qubits, data_qubits, name=name)
    factor = 1
    for target_qbit in data_qubits:
        cphasegate = PhaseGate(factor*np.pi*shiftlength).control(len(control_qubits), ctrl_state=newctrlstate)
        sub_circ.append(cphasegate, combine_registers(control_qubits, target_qbit))
        factor /= 2
    circuit.append(sub_circ, combine_registers(control_qubits, data_qubits))

def add(circuit, shifts, data_qubits, control_qubits=None):
    # If no control qubits, then do an uncontrolled U(shifts) operation (shifts should just be a single number)
    name = "add"
    if(control_qubits == None):
        sub_circ = QuantumCircuit(data_qubits, name=name)
        U(sub_circ, shifts, data_qubits)
        circuit.append(sub_circ, data_qubits)
    else:
        sub_circ = QuantumCircuit(control_qubits, data_qubits, name=name)
        for i in range(len(shifts)):
            U_controlled(sub_circ, shifts[len(shifts) -grayCode(i)-1], control_qubits, data_qubits, len(shifts) -grayCode(i)-1)
        circuit.append(sub_circ, combine_registers(control_qubits, data_qubits))

def subtract(circuit, shifts, data_qubits, control_qubits=None):
    # If no control qubits, do an uncontrolled U(-shifts) operation
    name = "sub" 
    if(control_qubits==None):
        sub_circ = QuantumCircuit(data_qubits, name=name)
        negativeshift = (2**data_qubits.size-shifts)% 2**data_qubits.size
        U(sub_circ, negativeshift, data_qubits)
        circuit.append(sub_circ, data_qubits)
    else:
        sub_circ = QuantumCircuit(control_qubits, data_qubits, name=name)
        for i in range(len(shifts)):
            negativeshift = (2**data_qubits.size-shifts[ len(shifts) -grayCode(i)-1])% 2**data_qubits.size
            U_controlled(sub_circ, negativeshift, control_qubits, data_qubits, len(shifts) -grayCode(i)-1)
        circuit.append(sub_circ, combine_registers(control_qubits, data_qubits))

def MAX0_B(circuit, data_qubits, ancilla):
    name = "MAX(0,B)"
    maxBcircuit = QuantumCircuit(data_qubits, ancilla, name=name)
    # Apply CCXs to first qubit (MSB) and each of the next ones, to target ancilla q[nqb]
    for i in range(len(data_qubits)-1):
        maxBcircuit.ccx(data_qubits[-1], data_qubits[i], ancilla[i])
        maxBcircuit.cx(ancilla[i], data_qubits[i])
    maxBcircuit.cx(data_qubits[-1],ancilla[-1])
    maxBcircuit.cx(ancilla[-1], data_qubits[-1])
    circuit.append(maxBcircuit, combine_registers(data_qubits, ancilla))

# Diffuser for Grover's search
def diffuser(circuit, qubits):
    circuit.h(qubits)
    circuit.x(qubits)
    circuit.h(qubits[-1])
    circuit.mct(qubits[:-1], qubits[-1])
    circuit.h(qubits[-1])
    circuit.x(qubits)
    circuit.h(qubits)

# Calculate the value of the buffer after each day, also check the buffer condition
def bufferfor1day(circuit, buffer_qubits, s1_qubits, s1_shifts, s2_qubits, s2_shifts, ancilla_qubits, condition_qubit, B_max, min_prec):
    # B_out = B_init + S1 - S2
    add(circuit, s1_shifts, buffer_qubits, s1_qubits)
    subtract(circuit, s2_shifts, buffer_qubits, s2_qubits)

    # MAX(0, B_out)
    circuit.append(QFT(buffer_qubits.size, do_swaps=False, inverse=True), buffer_qubits)
    MAX0_B(circuit, buffer_qubits, ancilla_qubits)
    circuit.append(QFT(buffer_qubits.size, do_swaps=False, inverse=False), buffer_qubits)

    # Condition c1: B_out <= B_max
    subtract(circuit, B_max+min_prec, buffer_qubits)

    circuit.append(QFT(buffer_qubits.size, do_swaps=False, inverse=True), buffer_qubits)
    circuit.cnot(buffer_qubits[-1], condition_qubit)
    circuit.append(QFT(buffer_qubits.size, do_swaps=False, inverse=False), buffer_qubits)

    # uncompute
    add(circuit, B_max+min_prec, buffer_qubits)

# Most complete definition of the oracle
# See QISS_with_cost.ipynb for how to use.
# Included for completeness, and to streamline importing QISS into another project
def oracle(shop1_qubits, shop2_qubits, cost_qubits, buffer_qubits, max_ancilla, condition_qubits, output_qubit, number_of_days, B_init, B_max, shifts_s1, shifts_s2, costs_s1, costs_s2, C_up, V_target, delta, minimum_precision, num_shop_qubits, num_buffer_qubits):
    shftsched_oracle = QuantumCircuit(shop1_qubits, shop2_qubits, cost_qubits, buffer_qubits, max_ancilla, condition_qubits, output_qubit, name="Oracle")
    shftsched_oracle.h(cost_qubits)
    shftsched_oracle.h(buffer_qubits)

    # Initial buffer content
    add(shftsched_oracle, B_init,buffer_qubits)

    # Calculate the buffer content (and condition) for each day
    for day in range(number_of_days):
            bufferfor1day(shftsched_oracle, buffer_qubits, shop1_qubits[day*num_shop_qubits:(day+1)*num_shop_qubits], shifts_s1, shop2_qubits[day*num_shop_qubits:(day+1)*num_shop_qubits], shifts_s2, max_ancilla[day*num_buffer_qubits:(day+1)*num_buffer_qubits], condition_qubits[day], B_max, minimum_precision)

    # Calculate (-)output volume: -V_out = B_final - S1 - B_init
    for day in range(number_of_days):
        subtract(shftsched_oracle, shifts_s1, buffer_qubits, shop1_qubits[day*num_shop_qubits:(day+1)*num_shop_qubits])
    subtract(shftsched_oracle, B_init, buffer_qubits)

    # condition c2: V_out >= V_low
    add(shftsched_oracle, V_target-delta-minimum_precision, buffer_qubits)
    shftsched_oracle.append(QFT(buffer_qubits.size, do_swaps=False, inverse=True), buffer_qubits)
    shftsched_oracle.cnot(buffer_qubits[-1], condition_qubits[-3])
    shftsched_oracle.append(QFT(buffer_qubits.size, do_swaps=False, inverse=False), buffer_qubits)

    # condition c3: V_out <= V_up 
    add(shftsched_oracle, 2*delta + minimum_precision, buffer_qubits) 
    shftsched_oracle.append(QFT(buffer_qubits.size, do_swaps=False, inverse=True), buffer_qubits)
    shftsched_oracle.x(condition_qubits[-2])
    shftsched_oracle.cnot(buffer_qubits[-1], condition_qubits[-2])

    # Add the cost for both shops for each day 
    for day in range(number_of_days):
        add(shftsched_oracle, costs_s1, cost_qubits, shop1_qubits[day*num_shop_qubits:(day+1)*num_shop_qubits])
        add(shftsched_oracle, costs_s2, cost_qubits, shop2_qubits[day*num_shop_qubits:(day+1)*num_shop_qubits])
    
    # condition c4: C_out < C_up
    subtract(shftsched_oracle, C_up, cost_qubits)
    shftsched_oracle.append(QFT(cost_qubits.size, do_swaps=False, inverse=True), cost_qubits)
    shftsched_oracle.cnot(cost_qubits[-1], condition_qubits[-1])

    return shftsched_oracle


# The full QISS algorithm 
# See QISS_with_cost.ipynb for how to use.
# Included for completeness, and to streamline importing QISS into another project
def QISS(ndays, B_init, B_max, V_target, delta, shifts_s1, shifts_s2, costs_s1, costs_s2, C_max, n_iter):
    minimum_precision = 1
    num_buffer_qubits = int(np.ceil(np.log2(B_max + max(shifts_s1)))) + 1 # Max possible value: B_max + maximum added in a day + 1 bit for 2s complement
    num_qubits_per_shop_per_day = int(np.ceil(np.log2(len(shifts_s1))))
    num_cost_qubits = int(np.ceil(np.log2(ndays*(max(shifts_s1)+max(shifts_s2))))) + 1 # Extra qubit for 2s complement negative number condition checking
    num_ancillas = ndays*num_buffer_qubits
    num_condition_qubits = 3+ndays

    shop1_qubits = QuantumRegister(num_qubits_per_shop_per_day*ndays, name='qs1')
    shop2_qubits = QuantumRegister(num_qubits_per_shop_per_day*ndays, name='qs2')
    cost_qubits = QuantumRegister(num_cost_qubits, name='qcost')
    buffer_qubits = QuantumRegister(num_buffer_qubits, name='qb')

    max_ancilla = QuantumRegister(num_ancillas, name='a')
    condition_qubits = QuantumRegister(num_condition_qubits, name='c')
    output_qubit = QuantumRegister(1, name='out')
    cbits = ClassicalRegister(2*num_qubits_per_shop_per_day*ndays, name='cbits')

    qc = QuantumCircuit(shop1_qubits, shop2_qubits, cost_qubits, buffer_qubits, max_ancilla, condition_qubits, output_qubit, cbits)

    # Initialize output qubit in state |->
    qc.x(output_qubit)
    qc.h(output_qubit)

    # Initialize qubits in state |s>
    qc.h(shop1_qubits)
    qc.h(shop2_qubits)

    shftsched_oracle = oracle(shop1_qubits, shop2_qubits, cost_qubits, buffer_qubits, max_ancilla, condition_qubits, output_qubit, ndays, B_init, B_max, shifts_s1, shifts_s2, costs_s1, costs_s2, C_max, V_target, delta, minimum_precision, num_qubits_per_shop_per_day, num_buffer_qubits)

    for _ in range(n_iter):
        qc.append(shftsched_oracle, qc.qubits)

        # CCCX gate on all conditionals to the |-> qubit
        
        # qc.append(MCXGate(len(condition_qubits)), combine_registers(condition_qubits, output_qubit)) # alternative MCX gate
        qc.append(cnx(len(condition_qubits)+1), combine_registers(condition_qubits, output_qubit))
        
        # uncompute
        qc.append(shftsched_oracle.inverse(), qc.qubits)

        diffuser(qc, combine_registers(shop1_qubits, shop2_qubits))
    return qc