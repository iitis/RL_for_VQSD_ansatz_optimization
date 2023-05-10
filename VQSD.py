from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit import *
from qiskit import QuantumCircuit
from qiskit import BasicAer
import numpy as np


# -----------------------------------------------------------------------------

class Parametric_Circuit:
    def __init__(self,n_qubits):
        self.n_qubits = n_qubits
        self.ansatz = QuantumCircuit(n_qubits)

    def construct_ansatz(self, state):
        # print('--------------')
        # print(state[:3])
        
        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.cx([ctrl[r].item()], [targ[r].item()])
            
            rot_direction_list = rot_pos[0]
            rot_qubit_list = rot_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        self.ansatz.rx(thetas[0][rot_qubit].item(), rot_qubit.item())
                    elif r == 1:
                        self.ansatz.ry(thetas[1][rot_qubit].item(), rot_qubit.item())
                    elif r == 2:
                        self.ansatz.rz(thetas[2][rot_qubit].item(), rot_qubit.item())
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2                       
        return self.ansatz

        

def cost_function_qiskit(angles, circuit, dephasing_circuit, n_qubits, shots, prob_typ, state, purity_before_diag,  which_angles = []):
    
    """"
    Function for Qiskit cost function minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    
    Output:
    expval [float] : cost function 
    
    """
    
    no = 0
    for i in circuit:
        gate_detail = list(i)[0]
        if gate_detail.name in ['rx', 'ry', 'rz']:
            list(i)[0].params = [angles[no]]
            no+=1

    if prob_typ == 'pure_state':
        qc_cost = QuantumCircuit(2*n_qubits, 2*n_qubits)
        for qu in range(n_qubits):
            qc.cx([qu+n_qubits], qu)
        qc = qc_cost.compose(circuit, range(n_qubits))   
        qc = qc.compose(circuit, range(n_qubits, 2*n_qubits))
        purity_before_diag = 1
        for cu in range(n_qubits):
            qc.measure([cu], [cu])

        backend = BasicAer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=shots).result()
        counts = result.get_counts()
        for co in counts.keys():
            if co == '0'*2*n_qubits:
                purity_after_diag = counts[co]/shots
        else:
            purity_after_diag = 0

    elif prob_typ in ['mixed_state', 'reduce_heisen_model']:
        dm_evo = state.evolve(circuit)
        dm_evo = np.kron(dm_evo.data, dm_evo.data)
        dm_evo = DensityMatrix(dm_evo).evolve(dephasing_circuit)
        
        par_trace = partial_trace(dm_evo, list(range(n_qubits)))
        purity_after_diag = np.trace(par_trace.data @ par_trace.data).real
    return purity_before_diag - purity_after_diag

def get_cost(n_qubits, circuit, dephasing_circuit, prob_typ, state, purity_before_diag, shots):
    
    if prob_typ == 'pure_state':
        qc_cost = QuantumCircuit(2*n_qubits, 2*n_qubits)
        qc_cost = qc_cost.compose(circuit, range(n_qubits))
        qc_cost = qc_cost.compose(circuit, range(n_qubits, 2*n_qubits))
        for qu in range(n_qubits):
            qc_cost.cx([qu+n_qubits], qu)
        for cu in range(n_qubits):
            qc_cost.measure(cu, cu)

        backend = BasicAer.get_backend('qasm_simulator')
        result = execute(qc_cost, backend, shots=shots).result()
        counts = result.get_counts()

        for co in counts.keys():
            if co == '0'*(2*n_qubits):
                purity_after_diag = counts[co]/shots
        else:
            purity_after_diag = 0
    
    elif prob_typ in ['mixed_state', 'reduce_heisen_model']:
        dm_evo = state.evolve(circuit)
        dm_evo = np.kron(dm_evo.data, dm_evo.data)
        dm_evo = DensityMatrix(dm_evo).evolve(dephasing_circuit)
        
        par_trace = partial_trace(dm_evo, list(range(n_qubits)))
        purity_after_diag = np.trace(par_trace.data @ par_trace.data).real
    return purity_before_diag - purity_after_diag
    


if __name__ == "__main__":
    pass