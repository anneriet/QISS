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
from qiskit import QuantumCircuit, QuantumRegister
import random as random
from helper_functions import combine_registers, grayCode
# Custom multicontrolled phase and x-gates that do not use ancillas

def newcphase(theta):
    subcirc = QuantumCircuit(2)
    subcirc.p(theta/2, [1])
    subcirc.cnot([0],[1])
    subcirc.p(-theta/2, [1])
    subcirc.cnot([0],[1])
    subcirc.p(theta/2, [0])
    return subcirc

def newcrz(theta):
    subcirc = QuantumCircuit(2)
    subcirc.p(theta/2, [1])
    subcirc.cnot([0],[1])
    subcirc.p(-theta/2, [1])
    subcirc.cnot([0],[1])
    subcirc.p(theta/2, [0])
    return subcirc

def ccphase(theta):
    subcirc = QuantumCircuit(3)
    subcirc.append(newcphase(theta/2), [1,2])
    subcirc.cnot([0], [1])
    subcirc.append(newcphase(-theta/2), [1,2])
    subcirc.cnot([0], [1])
    subcirc.append(newcphase(theta/2), [0,2])
    return subcirc


def mcphase(theta, nqubits):
    subcirc = QuantumCircuit(nqubits)
    if(nqubits == 3):
        subcirc.append(ccphase(theta), list(range(nqubits)))
    elif(nqubits == 2):
        subcirc.append(newcphase(theta), list(range(nqubits)))
    elif(nqubits == 1):
        subcirc.p(theta, [0])
    else:
        subcirc.append(newcphase(theta/2), [nqubits-2, nqubits-1])
        subcirc.append(cnx(nqubits-1), list(range(nqubits-1)))
        subcirc.append(newcphase(-theta/2), [nqubits-2, nqubits-1])
        subcirc.append(cnx(nqubits-1), list(range(nqubits-1)))
        subcirc.append(mcphase(theta/2, nqubits-1), list(range(nqubits-2)) + [nqubits-1])
    return subcirc

def cnx(nqubits):
    qubits = QuantumRegister(nqubits)
    qc = QuantumCircuit(qubits)
    if len(qubits) >= 3:
        qc.h(qubits[-1])
        qc.append(newcphase(np.pi/2), [qubits[-2], qubits[-1]])
        qc.append(cnx(nqubits-1), (qubits[:-1]))
        qc.append(newcphase(-np.pi/2), [qubits[-2], qubits[-1]])
        qc.append(cnx(nqubits-1), (qubits[:-1]))
        qc.append(mcphase(np.pi/2, nqubits-1), combine_registers(qubits[:-2],qubits[-1]))
        qc.h(qubits[-1])

    elif len(qubits)==3:
        qc.ccx(*qubits)
    elif len(qubits)==2:
        qc.cx(*qubits)
    return qc

if __name__ == "__main__":
    print(cnx(3))       