import pennylane as qml
from itertools import product
import numpy as np
import torch
TSub = qml.adjoint(qml.TAdd)

def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
    """For TAdd"""
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0, num_qubits, 0]
        i += 1
    """For TSub"""
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [num_qubits, 0, c, x, num_qubits, 0]
        i += 1
    """s denotes subspace of TRZ 1, 2 -->  TRZ[0,1], TRZ[0,2]"""
    for r, s in product(range(num_qubits),
                           range(1,3)):
        dictionary[i] = [num_qubits, 0, num_qubits, 0, r, s]
        i += 1
    return dictionary

num_qubits, num_layers = 2,6
@qml.qnode(qml.device("default.qutrit", wires=num_qubits))
def make_circuit(state):
    state = state.clone()

    for i in range(num_layers):
        
        cnot_pos1 = np.where(state[i][0:num_qubits] == 1)
        cnot_pos2 = np.where(state[i][num_qubits:2*num_qubits] == 1)
        # print(cnot_pos2)
        targ1 = cnot_pos1[0]
        ctrl1 = cnot_pos1[1]        
        if len(ctrl1) != 0:
            for r in range(len(ctrl1)):
                qml.TAdd(wires=[ctrl1[r], targ1[r]])
        targ2 = cnot_pos2[0]
        ctrl2 = cnot_pos2[1]        
        if len(ctrl2) != 0:
            for r in range(len(ctrl2)):
                TSub(wires=[ctrl2[r], targ2[r]])
        
        rot_pos = np.where(state[i][2*num_qubits: 2*num_qubits+2] == 1)
        rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
        
        if len(rot_qubit_list) != 0:
            for pos, r in enumerate(rot_direction_list):
                rot_qubit = rot_qubit_list[pos]
                if r == 0:
                    qml.TRZ(0, wires=rot_qubit, subspace=[0, 1])
                elif r == 1:
                    qml.TRZ(0, wires=rot_qubit, subspace=[0, 2])

                else:
                    print(f'rot-axis = {r} is in invalid')
                    assert r >1
    return qml.expval(qml.PauliZ(1))

moments = [0]*num_qubits
actions = [0,1,2,3,4,5]
act_dict = dictionary_of_actions(2)
state = torch.zeros((num_layers, 2*num_qubits+2+2, num_qubits))

for act in actions:
    action = act_dict[act]
    # print(action)
    ctrl1,ctrl2=action[0], action[2]
    targ1,targ2=(action[0]+action[1])%num_qubits,\
                (action[2]+action[3])%num_qubits
    rot_qubit = action[4]
    rot_axis = action[5]
    if rot_qubit < num_qubits:
        gate_tensor = moments[ rot_qubit ]
    elif ctrl1 < num_qubits:
        gate_tensor = max( moments[ctrl1], moments[targ1] )
    elif ctrl2 < num_qubits:
        gate_tensor = max( moments[ctrl2], moments[targ2] )
    
    if ctrl1 < num_qubits:
        state[gate_tensor][targ1][ctrl1] = 1
    elif ctrl2 < num_qubits:
        state[gate_tensor][targ2+num_qubits][ctrl2] = 1
    elif rot_qubit < num_qubits:
        state[gate_tensor][2*num_qubits+rot_axis-1][rot_qubit] = 1

    if rot_qubit < num_qubits:
        moments[ rot_qubit ] += 1
    elif ctrl1 < num_qubits:
        max_of_two_moments = max( moments[ctrl1], moments[targ1] )
        moments[ctrl1] = max_of_two_moments +1
        moments[targ1] = max_of_two_moments +1
    elif ctrl2 < num_qubits:
        max_of_two_moments = max( moments[ctrl2], moments[targ2] )
        moments[ctrl2] = max_of_two_moments +1
        moments[targ2] = max_of_two_moments +1
    drawer = qml.draw(make_circuit)
    print(drawer(state))
    print(state)