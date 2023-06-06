import configparser
import numpy as np
import json
import pickle
import argparse, sys
from qiskit.providers.basicaer import BasicAer
from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)
from itertools import product
from qiskit.aqua.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.aqua.operators import X, Z, Y, I
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.quantum_info import partial_trace
from qiskit.quantum_info.states.random import random_density_matrix, random_statevector
from collections import deque

# https://gist.github.com/thearn/5424219
def low_rank_approx(SVD=None, A=None, r=1):
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar = np.add(Ar, s[i] * np.outer(u.T[i], v[i]))
    return Ar

def random_state_gen(n, rank, type_state, seed):
    state_dict = {}
    if type_state == 'mixed':
        rdm = random_density_matrix(2**n, rank, seed=seed)
    elif type_state == 'arb-pure':
        rdm = random_statevector(2**n, seed=seed)
    
    state_dict['state'] = rdm
    state_dict['trace'] = np.trace(np.matmul(rdm.data,rdm.data)).real
    with open(f'state_data/{n}_qubit_{type_state}_rank_{rank}_state_data_seed_{seed}.p', 'wb') as fp:
        pickle.dump(state_dict, fp)

def ground_state_reduced_heisenberg_model(num_qubit):

    # HAMILTONIAN
    if num_qubit == 4:
        J = 0.25
        H = (J * X ^ X ^ I ^ I ^ I ^ I ^ I ^ I) + (J * Y ^ Y ^ I ^ I ^ I ^ I ^ I ^ I) + \
        (J * Z ^ Z ^ I ^ I ^ I ^ I ^ I ^ I) + (J * I ^ X ^ X ^ I ^ I ^ I ^ I ^ I) + \
        (J * I ^ Y ^ Y ^ I ^ I ^ I ^ I ^ I) + (J * I ^ Z ^ Z ^ I ^ I ^ I ^ I ^ I) + \
        (J * I ^ I ^ X ^ X ^ I ^ I ^ I ^ I) + (J * I ^ I ^ Y ^ Y ^ I ^ I ^ I ^ I) + \
        (J * I ^ I ^ Z ^ Z ^ I ^ I ^ I ^ I) + (J * I ^ I ^ I ^ X ^ X ^ I ^ I ^ I) + \
        (J * I ^ I ^ I ^ Y ^ Y ^ I ^ I ^ I) + (J * I ^ I ^ I ^ Z ^ Z ^ I ^ I ^ I) + \
        (J * I ^ I ^ I ^ I ^ X ^ X ^ I ^ I) + (J * I ^ I ^ I ^ I ^ Y ^ Y ^ I ^ I) + \
        (J * I ^ I ^ I ^ I ^ Z ^ Z ^ I ^ I) + (J * I ^ I ^ I ^ I ^ I ^ X ^ X ^ I) + \
        (J * I ^ I ^ I ^ I ^ I ^ Y ^ Y ^ I) + (J * I ^ I ^ I ^ I ^ I ^ Z ^ Z ^ I) + \
        (J * I ^ I ^ I ^ I ^ I ^ I ^ X ^ X) + (J * I ^ I ^ I ^ I ^ I ^ I ^ Y ^ Y) + \
        (J * I ^ I ^ I ^ I ^ I ^ I ^ Z ^ Z) + (J * X ^ I ^ I ^ I ^ I ^ I ^ I ^ X) + \
        (J * Y ^ I ^ I ^ I ^ I ^ I ^ I ^ Y) + (J * Z ^ I ^ I ^ I ^ I ^ I ^ I ^ Z)
    elif num_qubit == 3:
        J = 1/3
        H = (J * X ^ X ^ I ^ I ^ I ^ I) + (J * Y ^ Y ^ I ^ I ^ I ^ I) + \
            (J * Z ^ Z ^ I ^ I ^ I ^ I) + \
            (J * I ^ X ^ X ^ I ^ I ^ I) + (J * I ^ Y ^ Y ^ I ^ I ^ I) + \
            (J * I ^ Z ^ Z ^ I ^ I ^ I) + \
            (J * I ^ I ^ X ^ X ^ I ^ I) + (J * I ^ I ^ Y ^ Y ^ I ^ I) + \
            (J * I ^ I ^ Z ^ Z ^ I ^ I) + \
            (J * I ^ I ^ I ^ X ^ X ^ I) + (J * I ^ I ^ I ^ Y ^ Y ^ I) + \
            (J * I ^ I ^ I ^ Z ^ Z ^ I) + \
            (J * I ^ I ^ I ^ I ^ X ^ X) + (J * I ^ I ^ I ^ I ^ Y ^ Y) + \
            (J * I ^ I ^ I ^ I ^ Z ^ Z) + \
            (J * X ^ I ^ I ^ I ^ I ^ X) + (J * Y ^ I ^ I ^ I ^ I ^ Y) + \
            (J * Z ^ I ^ I ^ I ^ I ^ Z)
    elif num_qubit == 2:
        J = 1/2
        H = (J * X ^ X ^ I ^ I) + (J * Y ^ Y ^ I ^ I) + \
            (J * Z ^ Z ^ I ^ I) + \
            (J * I ^ X ^ X ^ I) + (J * I ^ Y ^ Y ^ I) + \
            (J * I ^ Z ^ Z ^ I) + \
            (J * I ^ I ^ X ^ X) + (J * I ^ I ^ Y ^ Y) + \
            (J * I ^ I ^ Z ^ Z) + \
            (J * X ^ I ^ I ^ X) + (J * Y ^ I ^ I ^ Y) + \
            (J * Z ^ I ^ I ^ Z)

    state_dict = {}
    aqua_globals.random_seed = 50
    var_form = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')    
    vqe = VQE(H, var_form, SLSQP(maxiter = 500),
            quantum_instance=QuantumInstance(backend=BasicAer.get_backend('statevector_simulator')))
    result = vqe.compute_minimum_eigenvalue(operator=H)
    gs = result['eigenstate']
    # print(gs)
    rdm = partial_trace(gs, list(range(num_qubit)))
    state_dict['state'] = rdm #np.kron(rdm.data,rdm.data)
    state_dict['trace'] = np.trace(np.matmul(rdm.data,rdm.data)).real
    with open(f'state_data/{num_qubit}_qubit_reduced_heisenberg_model_state_data.p', 'wb') as fp:
        pickle.dump(state_dict, fp)

def get_config(config_name,experiment_name, path='configuration_files',
               verbose=True):
    config_dict = {}
    Config = configparser.ConfigParser()
    Config.read('{}/{}{}'.format(path,config_name,experiment_name))
    for sections in Config:
        config_dict[sections] = {}
        for key, val in Config.items(sections):
            # config_dict[sections].update({key: json.loads(val)})
            try:
                config_dict[sections].update({key: int(val)})
            except ValueError:
                config_dict[sections].update({key: val})
            floats = ['learning_rate',  'dropout', 'alpha', 
                      'beta', 'beta_incr', 
                      "shift_threshold_ball","succes_switch","tolearance_to_thresh","memory_reset_threshold",
                      "fake_min_energy","_true_en"]
            strings = ['ham_type', 'fn_type', 'geometry','method','agent_type',
                       "agent_class","init_seed","init_path","init_thresh","method",
                       "mapping","optim_alg", "curriculum_type"]
            lists = ['episodes','neurons', 'accept_err','epsilon_decay',"epsilon_min",
                     "epsilon_decay",'final_gamma','memory_clean',
                     'update_target_net', 'epsilon_restart', "thresholds", "switch_episodes"]
            if key in floats:
                config_dict[sections].update({key: float(val)})
            elif key in strings:
                config_dict[sections].update({key: str(val)})
            elif key in lists:
                config_dict[sections].update({key: json.loads(val)})
    del config_dict['DEFAULT']
    return config_dict

def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary
        
def dict_of_actions_revert_q(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits-1,-1,-1),
                        range(num_qubits-1,0,-1)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits-1,-1,-1),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary
        

def average(buffer:deque):
    x = list(buffer)
    return sum(x)/len(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_type', type=str, default='mixed', help='State generation for diagonalization.')
    parser.add_argument('--max_dim', type=int, default=2, help='Maximum dimension of the quantum state to generate.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the state.')
    args = parser.parse_args(sys.argv[1:])
    
    if args.state_type == 'reduced-heisenberg':
        ground_state_reduced_heisenberg_model(args.max_dim)
        print(f'{args.max_dim}-qubit reduced Heisenberg state and eigenvalued are saved.')
    
    if args.state_type == 'mixed':
        seed = args.seed
        n_max = args.max_dim
        for dim in range(1, n_max+1):
            for rank in range(dim, 2**dim+1):
                random_state_gen(dim, rank, args.state_type, seed)
                print(f'[Info] Size: {dim}, rank: {rank}, seed: {seed} done!')
    print('[Info] Check `state_data` folder!')