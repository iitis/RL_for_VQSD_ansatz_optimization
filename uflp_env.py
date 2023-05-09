from environment import CircuitEnv
from collections import deque
import numpy as np
import random
import math

class UflpEnv(CircuitEnv):
    def __init__(self, conf, device):
        super().__init__(conf, device)

        self.p_init = float(conf["env"]["p_init"])
        self.hb_length = int(conf["env"]["hb_length"])
        self.hb_batch_size = int(conf["env"]["hb_batch_size"])
        self.history_buffer = deque(maxlen=self.hb_length)

        uncertainty_module = conf['env']['uncertainty_module']

        if uncertainty_module == "approximate_counts":
            ac_size = int(conf["env"]["ac_size"])
            ac_lambda = float(conf["env"]["ac_lambda"])
            self.uncertainty_module = ApproximateCounts(ac_size, ac_lambda)
        elif uncertainty_module == "bootstrap_uncertainty":
            self.uncertainty_module = BootstrappedUncertainty()
        
      
    def update_history_buffer(self, state, action):
        """
        state - visited stated to be added to the buffer
        action - new action to be added to the buffer
        """
        if len(self.history_buffer) == self.history_buffer.maxlen:
            self.history_buffer.pop()
        self.history_buffer.appendleft((state,tuple(action)))

    def get_uncertainty(self, state_action):

        if self.uncertainty_module == "approximate_counts":
            visits_no = self.approximate_counts.get_visit_count(state_action)
            return math.pow((visits_no + self.ac_lambda), -0.5)

    def get_init_state_action(self):
        """
        Sample state action pairs from the history buffer 
        and return the one with highest uncertainty
        """
        candidates = random.sample(list(self.history_buffer),self.hb_batch_size)
        uncertainty_ranking = [[sa, self.uncertainty_module.get_uncertainty(sa)] for sa in candidates]
        
        _,init_sa = max(enumerate(uncertainty_ranking), key=lambda x: x[1][1])
        return init_sa[0]
        
    def step(self, action, train_flag=True):
        self.update_history_buffer(self.state,action)
        self.uncertainty_module.record_state_action((self.state,action))
        res = super().step(action, train_flag)
        
        return res

    def reset(self):
        x = np.random.uniform()

        if x < self.p_init or len(self.history_buffer) == 0:
            # Reset env per usual
            return super().reset()
    
        # Else reset the env to one of the states sampled from the history buffer
        
        init_state, init_action = self.get_init_state_action()
        self.state = init_state
        super().reset_env_variables()

        # Step with initial action
        (state,_,_) = self.step(init_action)
        
        return state

class UncertaintyModule(object):
    def __init__(self):
        pass

    def get_uncertainty(self):
        return 0.0
    
    def record_state_action(self, state_action):
        pass

class BootstrappedUncertainty(UncertaintyModule):
    def __init__(self):
        pass

class ApproximateCounts(UncertaintyModule):
    def __init__(self, size, reg_lambda):
        self.size = size
        self.reg_lambda = reg_lambda
        self.register = self.create_buckets()

    def get_uncertainty(self, state_action):
        visits_no = self.get_visit_count(state_action)
        return math.pow((visits_no + self.reg_lambda), -0.5)
 
    def create_buckets(self):
        return [0 for _ in range(self.size)]
    
    def get_hash(self, state_action):
        state = state_action[0]
        action = tuple(state_action[1])
        # print(state, type(action))
        # print(type(action[0]))
        # exit()
        return hash((state, action))
     
    def record_state_action(self, state_action):   
        hashed_key = self.get_hash(state_action) % self.size        
        self.register[hashed_key] += 1
 
    # Return searched value with specific key
    def get_visit_count(self, state_action):
        hashed_key = hash(state_action) % self.size 
        return self.register[hashed_key]

   
if __name__ == '__main__':
    pass
 

            
        
        
        

    