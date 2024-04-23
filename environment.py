import time
import torch
from utils import dictionary_of_actions, low_rank_approx
from sys import stdout
import scipy
import VQSD as vc
import os
import numpy as np
import copy
import curricula
import pickle
import cudaq

import copy
# from qulacsvis import circuit_drawer

class CircuitEnv():

    def __init__(self, conf, device):
        
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        self.state_type = conf['problem']['type']
        self.seed = conf['problem']['seed']
        self.shots = conf['problem']['shots_for_diag']
        self.rank = conf['problem']['rank']
            
        self.fake_min_energy = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else None
        self.fn_type = conf['env']['fn_type']
        self.optim_method = conf['non_local_opt']['method']
        self.optim_alg = conf['non_local_opt']['optim_alg']
        self.global_iters = conf['non_local_opt']['global_iters']

        self.dephasing_circuit = QuantumCircuit(2*self.num_qubits)
        for i in range(self.num_qubits):
            self.dephasing_circuit.cx([i+self.num_qubits], [i])
        
        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.
        # LOADING MIXED STATE DATA
        if self.rank != 'full':
            # print(self.rank)
            file = f"state_data/{self.num_qubits}_qubit_{self.state_type.split('_')[0]}_rank_{self.rank}_state_data_seed_{self.seed}.p"
            print(file)
            state_data = []
            with (open(f"state_data/{self.num_qubits}_qubit_{self.state_type.split('_')[0]}_rank_{self.rank}_state_data_seed_{self.seed}.p", "rb")) as openfile:
                while True:
                    try:
                        # print(pickle.load(openfile))
                        state_data.append(pickle.load(openfile))
                    except EOFError:
                        break
        
        # LOADING REDUCED HEISENBERG STATE DATA
        else:
            self.reduce_rank = conf['problem']['reduce_rank']
            state_data = []
            with (open(f"state_data/{self.num_qubits}_qubit_{self.state_type}_state_data.p", "rb")) as openfile:
                while True:
                    try:
                        state_data.append(pickle.load(openfile))
                    except EOFError:
                        break
        
        self.state_to_diag = state_data[0]['state']
        self.purity_before_diag = state_data[0]['trace']

        # print(self.state_to_diag)

        if self.rank == 'full' and self.reduce_rank != 'full':
            u, s, v = np.linalg.svd(self.state_to_diag.data)
            self.state_to_diag = low_rank_approx((u,s,v), np.asmatrix(self.state_to_diag.data), r=self.reduce_rank)
            self.purity_before_diag = np.trace(self.state_to_diag @ self.state_to_diag).real
            self.state_to_diag = DensityMatrix(self.state_to_diag)
            # print(scipy.linalg.eig(self.state_to_diag)[0])
        # print(self.state_to_diag)
        

        # print(self.state_to_diag, self.purity_before_diag)

        # exit()
        #This is to feed RL agent with noiseless expectation values found by spsa if turned False. Default should be True
        self.noise_flag = True
        self.state_with_angles = conf['agent']['angles']
        self.current_number_of_cnots = 0
        
        # If you want to run agent from scratch without *any* curriculum just use the setting with
        self.curriculum_dict = {}
        self.device = device
        self.done_threshold = conf['env']['accept_err']

        self.curriculum_dict[self.state_type] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=0)
        

        stdout.flush()
        self.state_size = self.num_layers*self.num_qubits*(self.num_qubits+3+3)
        self.step_counter = -1
        self.prev_energy = None
        self.moments = [0]*self.num_qubits
        self.illegal_actions = [[]]*self.num_qubits
        self.energy = 0

        self.action_size = (self.num_qubits*(self.num_qubits+2))
        self.previous_action = [0, 0, 0, 0]


    def step(self, action, train_flag = True) :

        """
        Action is performed on the first empty layer.
        ##Variable 'actual_layer' points last non-empty layer.
        
        Variable 'step_counter' points last non-empty layer.
        """  
        
        next_state = self.state.clone()
        #self.actual_layer += 1
        self.step_counter += 1

        """
        First two elements of the 'action' vector describes position of the CNOT gate.
        Position of rotation gate and its axis are described by action[2] and action[3].
        When action[0] == num_qubits, then there is no CNOT gate.
        When action[2] == num_qubits, then there is no Rotation gate.
        """
        ctrl = action[0]
        targ = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis = action[3]
        self.action = action
        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[ rot_qubit ]
        elif ctrl < self.num_qubits:
            gate_tensor = max( self.moments[ctrl], self.moments[targ] )

        
        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+rot_axis-1][rot_qubit] = 1

        if rot_qubit < self.num_qubits:
            self.moments[ rot_qubit ] += 1
        elif ctrl < self.num_qubits:
            max_of_two_moments = max( self.moments[ctrl], self.moments[targ] )
            self.moments[ctrl] = max_of_two_moments +1
            self.moments[targ] = max_of_two_moments +1
            
        ## Repeat action penalty
        self.current_action = action
        self.update_illegal_actions()
        # print(self.state[:3])
        # print('-x-x-x-x-x-x-')
        # print(self.make_circuit())
        # print('-x-x-x-x-x-x-')


        # print(next_state[:3])
        if self.optim_method in ["scipy_each_step"]:
            thetas, nfev, opt_ang = self.scipy_optim(self.optim_alg)
            for i in range(self.num_layers):
                for j in range(3):
                    next_state[i][self.num_qubits+3+j,:] = thetas[i][j,:]

        self.state = next_state.clone()
        self.opt_ang = opt_ang
        energy,energy_noiseless = self.get_cost_func()
        # print(self.opt_ang)
        if self.noise_flag == False:
            energy = energy_noiseless

        self.energy = energy
    
        self.error = float(abs(energy))
        
        # print(self.error)
        # self.error_noiseless = float(abs(self.min_eig-energy_noiseless))
        self.nfev = nfev
        rwd = self.reward_fn(energy)
        self.prev_energy = np.copy(energy)
        self.save_circ = self.make_circuit()
        energy_done = int(self.error < self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(energy_done or layers_done)
        self.previous_action = copy.deepcopy(action)
        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)
        
        if done:
            # print(self.make_circuit())
            # print(self.error)
            # print(self.step_counter)
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_prob)] = copy.deepcopy(self.curriculum)
        
        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        else:
            next_state = next_state[:, :self.num_qubits+3]
            return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """
        ## state_per_layer: (Control_qubit, NOT_qubit, R_qubit, R_axis, R_angle)
        state = torch.zeros((self.num_layers, self.num_qubits+3+3, self.num_qubits))
        self.state = state
        
        self.reset_env_variables()

        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
        else:
            state = state[:, :self.num_qubits+3]
            return state.reshape(-1).to(self.device)
        
    def reset_env_variables(self):
        # statistics_generated = np.clip(np.random.negative_binomial(n=40,p=0.6, size=100),0,40)
        # print(statistics_generated)
        # c = Counter(statistics_generated)
        # self.halting_step = 40  #c.most_common(1)[0][0]
        self.current_prob = self.state_type
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_prob)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits]*4
        self.illegal_actions = [[]]*self.num_qubits
        self.make_circuit(self.state)
        # print(self.make_circuit())
        self.step_counter = -1

        # initiate moments
        self.moments = [0]*self.num_qubits
        self.prev_energy = self.get_cost_func(self.state)[0]

    def make_circuit(self, thetas=None):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
        if thetas is None:
            thetas = state[:, self.num_qubits+3:]
        
        circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_layers):
            
            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]
            
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.cx([ctrl[r]], [targ[r]])
                  
            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits+3] == 1)
            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
            
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    # print(rot_qubit)
                    if r == 0:
                        circuit.rx(thetas[i][0][rot_qubit].item(), rot_qubit)
                    elif r == 1:
                        circuit.ry(thetas[i][1][rot_qubit].item(), rot_qubit)
                    elif r == 2:
                        circuit.rz(thetas[i][2][rot_qubit].item(), rot_qubit)
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2
        return circuit


    def get_cost_func(self, thetas=None):
        circ = self.make_circuit(thetas)
        qc = QuantumCircuit(self.num_qubits)
        # for q in range(self.num_qubits):
        #     qc.h(q)
        qiskit_inst = vc.Parametric_Circuit(n_qubits = self.num_qubits)
        circ = qiskit_inst.construct_ansatz(self.state)
        circ = qc.compose(circ)
        expval = vc.get_cost(self.num_qubits, circ, self.dephasing_circuit, self.state_type, self.state_to_diag, self.purity_before_diag, self.shots)
        energy = expval
        # print(energy)
        return energy, 0
        
    def scipy_optim(self, method, which_angles = [] ):
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:,self.num_qubits: self.num_qubits+3] == 1).nonzero( as_tuple = True )
        angles = thetas[rot_pos]

        # THE STATE THAT NEED TO BE DIAGONALIZED
        qc = QuantumCircuit(self.num_qubits)
        
        # for q in range(self.num_qubits):
        #     qc.h(q)
        
        qiskit_inst = vc.Parametric_Circuit(n_qubits=self.num_qubits)
        qiskit_circuit = qiskit_inst.construct_ansatz(state)
        x0 = np.asarray(angles.cpu().detach())
        qiskit_circuit = qc.compose(qiskit_circuit)

        def cost(x):
            return vc.cost_function_qiskit( x, circuit = qiskit_circuit, dephasing_circuit = self.dephasing_circuit, 
                                            n_qubits = self.num_qubits, shots = self.shots, 
                                            prob_typ = self.state_type, state = self.state_to_diag, 
                                            purity_before_diag = self.purity_before_diag, 
                                            which_angles = [])

        if list(which_angles):
            result_min_qiskit = scipy.optimize.minimize(cost, x0 = x0[which_angles], method = method, options = {'maxiter':self.global_iters})
            x0[which_angles] = result_min_qiskit['x']
            thetas = state[:, self.num_qubits+3:]
            thetas[rot_pos] = torch.tensor(x0, dtype=torch.float)
        else:
            result_min_qiskit = scipy.optimize.minimize(cost, x0 = x0, method = method, options = {'maxiter':self.global_iters})
            thetas = state[:, self.num_qubits+3:]
            thetas[rot_pos] = torch.tensor(result_min_qiskit['x'], dtype=torch.float)
        return thetas, result_min_qiskit['nfev'], result_min_qiskit['x']

    def reward_fn(self, energy):
        # sv = Saver().get_new_episode()
        # print(sv)
        if self.fn_type == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
        elif self.fn_type == "two_step_end":
#             max_depth = self.actual_layer == (self.num_layers - 1)
            max_depth = self.step_counter == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
            else:
                return 0.0
        elif self.fn_type == "naive":
            return 0. + 1.*(self.error < self.done_threshold)
        elif self.fn_type == "incremental":
            return (self.prev_energy - energy)/abs(self.prev_energy - self.min_eig)
        elif self.fn_type == "incremental_clipped":
            return np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
        elif self.fn_type == "nive_fives":
#             max_depth = self.actual_layer == (self.num_layers-1)
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = 0.
            return rwd
        
        elif self.fn_type == "incremental_with_fixed_ends":
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - 1),-1,1)
            return rwd
        
        elif self.fn_type == "log":
            return -np.log(1-(energy/self.purity_before_diag))
        
        elif self.fn_type == "log_to_ground":
            # return -np.log(abs(energy - self.min_eig))
            return -np.log(self.error)
        
        elif self.fn_type == "log_to_threshold":
            if self.error < self.done_threshold + 1e-5:
                rwd = 11
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 50
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_500":
            if self.error < self.done_threshold + 1e-5:
                rwd = 500
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_non_repeat_energy":
            if self.error < self.done_threshold + 1e-5:
                rwd = 30
            elif np.abs(self.energy-self.prev_energy) <= 1e-3:
                rwd = -30
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_no_repeat_actions":
            if self.current_action == self.previous_action:
                return -1 # IS -10 HIGH NEGATIVE PENALTY?
            elif self.error < self.done_threshold + 1e-5:
                rwd = 20
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_neg_punish":
            return -np.log(1-(energy/self.min_eig)) - 5
        
        elif self.fn_type == "end_energy":
#             max_depth = self.actual_layer == (self.num_layers - 1)
            max_depth = self.step_counter == (self.num_layers - 1)
            
            if ((self.error < self.done_threshold) or max_depth):
                rwd = (self.max_eig - energy) / (abs(self.min_eig) + abs(self.max_eig))
            else:
                rwd = 0.0

        elif self.fn_type == "hybrid_reward":
            path = 'threshold_crossed.npy'
            if os.path.exists(path):
                
                threshold_pass_info = np.load(path)
                if threshold_pass_info > 8:
#                     max_depth = self.actual_layer == (self.num_layers-1)
                    max_depth = self.step_counter == (self.num_layers - 1)
                    if (self.error < self.done_threshold):
                        rwd = 5.
                    elif max_depth:
                        rwd = -5.
                    else:
                        rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
                    return rwd
                else:
                    if self.error < self.done_threshold + 1e-5:
                        rwd = 11
                    else:
                        rwd = -np.log(abs(self.error - self.done_threshold))
                    return rwd
            else:
                np.save('threshold_crossed.npy', 0)
        
        elif self.fn_type == 'negative_above_chem_acc':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 1000*(self.done_threshold/self.error)
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_non_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = self.done_threshold/self.error
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_slight_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 100*(self.done_threshold/self.error)
            return rwd


        elif self.fn_type == "cnot_reduce":
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = self.num_layers - self.cnot_rwd_weight*self.current_number_of_cnots
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return 

        
    def update_illegal_actions(self):
        action = self.current_action
        illegal_action = self.illegal_actions
        
        # if self.num_qubits > 2:
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        rot_qubit, rot_axis = action[2], action[3]
        # else:
        #     ctrl, targ = self.num_qubits, self.num_qubits
        #     rot_qubit, rot_axis = action[0], action[1]

        if ctrl < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[2] == self.num_qubits:
                        
                            if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[0] or targ == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break                          
            else:
                illegal_action[0] = action

                            
        if rot_qubit < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[0] == self.num_qubits:
                            
                            if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            elif rot_qubit != ill_ac[2]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                                        
                            elif rot_qubit == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break 
            else:
                illegal_action[0] = action
        
        for indx in range(self.num_qubits):
            for jndx in range(indx+1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx +1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        
        for indx in range(self.num_qubits-1):
            if len(illegal_action[indx])==0:
                illegal_action[indx] = illegal_action[indx+1]
                illegal_action[indx+1] = []
        
        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        # print(self.illegal_actions)
        return illegal_action_decode




if __name__ == "__main__":
    pass