import torch
import numpy as np
import random
from collections import namedtuple, Counter
import copy
import torch.nn.functional as F
import torch.nn as nn
from utils import dictionary_of_actions, dict_of_actions_revert_q

from .DeepQ import DQN, ReplayMemory

class BootstrappedDQN(DQN):
    def __init__(self, conf, action_size, state_size, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        
        self.final_gamma = conf['agent']['final_gamma']
        self.epsilon = conf['agent']['epsilon_init']
        self.epsilon_min = conf['agent']['epsilon_min']
        self.epsilon_decay = conf['agent']['epsilon_decay']
        learning_rate = conf['agent']['learning_rate']
        self.update_target_net = conf['agent']['update_target_net']
        neuron_list = conf['agent']['neurons']
        drop_prob = conf['agent']['dropout']
        self.with_angles = conf['agent']['angles']

        if "memory_reset_switch" in conf['agent'].keys():
            self.memory_reset_switch =  conf['agent']["memory_reset_switch"]
            self.memory_reset_threshold = conf['agent']["memory_reset_threshold"]
            self.memory_reset_counter = 0
        else:
            self.memory_reset_switch =  False
            self.memory_reset_threshold = False
            self.memory_reset_counter = False

        self.action_size = action_size
        self.state_size = state_size if self.with_angles else state_size - self.num_layers*self.num_qubits*3
    
        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]) else self.state_size
  
        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)

        self.head_count = conf['agent']['head_count']
        # print('---------x--------')
        # print(type(self.head_count))
        # print('---------x--------')
        heads = nn.ModuleList([self.unpack_network(neuron_list, drop_prob).to(device) for _ in range(self.head_count)])
        self.policy_net = EnsembleNet(heads)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.memory = BootstrapReplayMemory(conf['agent']['memory_size'], self.head_count, conf['agent']['memory_dropout'])

        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma,1/self.num_layers),2)]).to(device)   # discount rate
        self.epsilon = 1.0  # exploration rate

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.device = device
        self.step_counter = 0

        self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward',
                                    'next_state','done','mask'))

    def act(self, state, ill_actions, head_idx:int=None):
        state = state.unsqueeze(0)
        is_random_explore = False

        # Exploration via random selection
        if torch.rand(1).item() <= self.epsilon:
            rand_ac = torch.randint(self.action_size, (1,)).item()
            while rand_ac in ill_actions:
                rand_ac = torch.randint(self.action_size, (1,)).item()
            is_random_explore = True
            return (rand_ac, is_random_explore)
        
        if head_idx is not None:
            ac = self.policy_net(state, head_idx).cpu().argmax().item()
            return (ac, is_random_explore)
            
        else:
            # Ensemble vote
            actions = self.policy_net(state)

            for action in actions:
                for ill_ac in ill_actions:
                    action[:,ill_ac]=-np.inf
            actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
            actions = Counter(actions)
            ac = actions.most_common(1)[0][0]
            return (ac, is_random_explore)
            

    def replay(self, batch_size):
        if self.step_counter %self.update_target_net ==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1


        transitions = self.memory.sample(batch_size)
        batch = self.Transition(*zip(*transitions))

        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)#, device=self.device)
        reward_batch = torch.stack(batch.reward)#.to(device=self.device)
        done_batch = torch.stack(batch.done)#.to(device=self.device)
        mask_batch = batch.mask
        
        total_loss = []
        for k in range(self.head_count):
            state_action_values = self.policy_net.forward(state_batch,k).gather(1, action_batch.unsqueeze(1))
        
            
            """ Double DQN """        
            next_state_values = self.target_net.forward(next_state_batch,k)
            next_state_actions = self.policy_net.forward(next_state_batch,k).max(1)[1].detach()
            next_state_values = next_state_values.gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
            
        
        
            """ Compute the expected Q values """
            expected_state_action_values = (next_state_values * self.gamma) * (1-done_batch) + reward_batch
            expected_state_action_values = expected_state_action_values.view(-1, 1)
            assert state_action_values.shape == expected_state_action_values.shape, "Wrong shapes in loss"

            head_mask = torch.tensor([True if mask_sample[k]==1 else False for mask_sample in mask_batch])
            expected_state_action_values = expected_state_action_values[head_mask]
            state_action_values = state_action_values[head_mask]       

            cost = self.fit(state_action_values, expected_state_action_values)
            
            total_loss.append(cost)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon,self.epsilon_min)
        assert self.epsilon >= self.epsilon_min, "Problem with epsilons"
        return sum(total_loss)/self.head_count


## REPLACE WITH QCNN??? ##
class EnsembleNet(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads_list = heads

    def _heads(self, x):
        return [head(x) for head in self.heads_list]
    
    def forward(self, x, k=None):
        if k is not None:
            return self.heads_list[k](x)
        else:
            return self._heads(x)
    
    def eval(self):
        for head in self.heads_list:
            head.eval()
    
    def train(self):
        for head in self.heads_list:
            head.train()

   


class BootstrapReplayMemory(ReplayMemory):
    def __init__(self, capacity: int, head_count, bernoulli_prob = 0.9):
        super().__init__(capacity)
        
        self.head_count = head_count
        self.bernoulli_prob = bernoulli_prob
        self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward',
                                    'next_state','done','mask'))
    

    def push(self, *args):
        # print(type(self.bernoulli_prob))
        mask = torch.tensor(np.random.binomial(1, float(self.bernoulli_prob), self.head_count))
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(mask=mask,*args)
        self.position = (self.position + 1) % self.capacity

    
    