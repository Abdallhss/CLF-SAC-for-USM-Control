
"""## Soft Actor Critic

#### Spinningup Implementation
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from copy import deepcopy
from torch.optim import Adam
import matplotlib.pyplot as plt
import pickle
#plt.style.use(['science','ieee'])

# Return appropiate Shape for the replay buffer arrays
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class X2hNetwork(nn.Module):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, obs_dim, hidden_dim, LSTM_out_dim, activation=F.relu):
        super().__init__()

        self.activation = activation

        self.in_layer = nn.Linear(obs_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_hc_layer = nn.Linear(hidden_dim, LSTM_out_dim)

        # weights initialization
        #self.linear4.apply(linear_weights_init)
        
    def forward(self, state):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        
        x = torch.cat([state], -1) 
        x = self.activation(self.in_layer(x))  # linear layer for 3d input only applied on the last dim
        #x = self.activation(self.hidden_layer(x))
        y1 = self.out_hc_layer(x)
        y2 = self.out_hc_layer(x)

        return y1, y2
        
class LSTM_Network(nn.Module):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, obs_dim, act_dim, hidden_dim, LSTM_out_dim, activation=F.relu):
        super().__init__()

        self.activation = activation

        self.linear1 = nn.Linear(obs_dim+act_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, LSTM_out_dim,batch_first=True)
        # weights initialization
        #self.linear4.apply(linear_weights_init)
        
    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        
        x = torch.cat([state, last_action], -1) 
        x = self.activation(self.linear1(x))  # linear layer for 3d input only applied on the last dim
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)


class QNetwork(nn.Module):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, obs_dim, act_dim, hidden_dim, LSTM_out_dim, activation=F.relu):
        super().__init__()

        self.activation = activation

        self.in_layer = nn.Linear(obs_dim+act_dim+LSTM_out_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 10)
        # weights initialization
        #self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action, lstm_output):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        
        x = torch.cat([state, action, lstm_output], -1) 
        x = self.activation(self.in_layer(x))  # linear layer for 3d input only applied on the last dim
        #x = self.activation(self.hidden_layer(x))
        x = self.activation(self.hidden_layer(x))
        x = self.out_layer(x)
        #x = x.permute(1,0,2)  # back to same axes as input    
        return x    # lstm_hidden is actually tuple: (hidden, cell)

class SPDNetwork(nn.Module):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, obs_dim, hidden_dim, LSTM_out_dim, activation=F.relu):
        super().__init__()

        self.activation = activation
        self.in_layer = nn.Linear(obs_dim + LSTM_out_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)
        # weights initialization
        #self.linear4.apply(linear_weights_init)
        
    def forward(self, state, lstm_output):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        
        x = torch.cat([state,lstm_output], -1) 
        x = self.activation(self.in_layer(x))  # linear layer for 3d input only applied on the last dim
        x = self.activation(self.hidden_layer(x))
        x = self.out_layer(x)
        #x = x.permute(1,0,2)  # back to same axes as input    
        return x    # lstm_hidden is actually tuple: (hidden, cell)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, LSTM_out_dim, act_limit=2.0, log_std_min=-20, log_std_max=2, activation=F.relu):
        super().__init__()

        self.activation = activation

        self.act_limit = act_limit
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.in_layer = nn.Linear(obs_dim+LSTM_out_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, act_dim)        
        self.log_std_linear = nn.Linear(hidden_dim, act_dim)


    def forward(self, state, lstm_output ,deterministic=False,with_logprob=True):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
     
        x = torch.cat([state, lstm_output], -1)
        x = self.activation(self.in_layer(x))
        x = self.activation(self.hidden_layer(x))
        mean    = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        pi_distribution = Normal(mean, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mean
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit[0] * pi_action
        
        return pi_action, logp_pi

class LSTMActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim , act_limit, hidden_dim=64, LSTM_out_dim = 5, activation=F.relu):
        super().__init__()

        # build policy and value functions
        self.lstm = LSTM_Network(obs_dim, act_dim, hidden_dim,LSTM_out_dim,activation=activation)
        self.q = QNetwork(obs_dim, act_dim, hidden_dim,LSTM_out_dim, activation=activation)
        self.spd = SPDNetwork(obs_dim, hidden_dim,LSTM_out_dim, activation=activation)
        self.pi = PolicyNetwork(obs_dim, act_dim, hidden_dim, LSTM_out_dim, act_limit = act_limit, activation=activation)
        self.x2hc = X2hNetwork(obs_dim, hidden_dim, LSTM_out_dim, activation=activation)

    def get_action(self, state, last_action, hidden_in, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).unsqueeze(0)  # increase 2 dims to match with training data
            last_action = torch.FloatTensor(np.array(last_action)).unsqueeze(0).unsqueeze(0)

            lstm_output, hidden_out = self.lstm(state, last_action, hidden_in)
            pi_action, _ =   self.pi(state, lstm_output , deterministic, False)

            return pi_action[0][0].numpy(), hidden_out

    def get_hidden_in(self, state):
        with torch.no_grad():
          state = torch.FloatTensor(np.array(state)).unsqueeze(0).unsqueeze(0)  # increase 2 dims to match with training data
          h,c = self.x2hc(state)
          return h, c

'''
Replay Buffer Class

A simple FIFO experience replay buffer for SAC agents.
Stores the experiences collected to be later used for actor and critic optimization.
'''
class ReplayBufferLSTM2:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        #self.weights = np.zeros(int(capacity))
        #self.max_weight = 10**-2
        #self.delta = 10**-4

    def push(self, state, action, last_action, reward, next_state, spd):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, spd)
        #self.weights[self.position] = self.max_weight  # new sample has max weights
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
        
        if np.random.random()<0.1:
          self.joinEpisodes()

    def sample_batch(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, spd_lst =[], [], [],[],[],[]

        #set_weights = self.weights[:self.position] + self.delta
        #probabilities = set_weights / sum(set_weights)
        #self.indices = np.random.choice(range(self.position), batch_size, p=probabilities, replace=False)
        self.indices = np.random.choice(range(self.position), batch_size, replace=False)

        for i in self.indices:
            state, action, last_action, reward, next_state, spd = self.buffer[i]
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            spd_lst.append(spd)


        batch = dict(state = torch.FloatTensor(np.array(s_lst)),
                     action = torch.FloatTensor(np.array(a_lst)),
                     last_action = torch.FloatTensor(np.array(la_lst)),
                     reward = torch.FloatTensor(np.array(r_lst)),
                     next_state = torch.FloatTensor(np.array(ns_lst)),
                     spd = torch.FloatTensor(np.array(spd_lst)))
        return batch

    def joinEpisodes(self):
      i,j = np.random.randint(self.position,size=2)
      state_i, action_i, last_action_i, reward_i, next_state_i, spd_i = self.buffer[i]
      state_j, action_j, last_action_j, reward_j, next_state_j, spd_j = self.buffer[j]

      k = np.random.randint(10)
      #k = 5
      state_ij = state_i[:k] + state_j[k:]
      action_ij = action_i[:k] + action_j[k:]
      last_action_ij = last_action_i[:k] + last_action_j[k:]
      reward_ij = reward_i[:k] + reward_j[k:]
      next_state_ij = next_state_i[:k] + next_state_j[k:]
      spd_ij = spd_i[:k] + spd_j[k:]

      self.push(state_ij, action_ij, last_action_ij, reward_ij, next_state_ij, spd_ij)


    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def __len__(self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

    def save_buffer(self,env_type='SIM'):
        with open("logs/buffer_{}".format(env_type), "wb") as fp:
            pickle.dump(self.buffer, fp)

    def load_buffer(self,env_type='SIM'):
        with open("logs/buffer_{}".format(env_type), "rb") as fp:
            self.buffer = pickle.load(fp)
        self.position = len(self.buffer)

            
'''
Soft Actor-Critic (SAC)
Main Algorithm including initialization, training, and evaluation.
'''                
class SAC:
    """
    Soft Actor-Critic (SAC)

    Args:
        env_type : to initalize environment as simulation or experiment.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q`` module, and a ``spd`` module.
            The ``act`` method, ``pi`` module. and ``spd`` module should accept batches of 
            observations as inputs, and ``q`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q``, and ``spd`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.

            ``q``         (batch,10)       | Tensor containing 10 current estimate
                                           | of Q* for the provided observations
                                           | and actions. 

            ``spd``       (batch,)         | Tensor containing the current 
                                           | estimate of speed for the provided observations.
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.

            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        obs_dim (int): observation space dimension.

        act_dim (int): action space dimension.

        act_dim (int): action limit bo.

        buffer_size (int): Maximum length of replay buffer.

        hidden_sizes (tuple=>list=>int): The size of hiddem layers of the neural networks

        activation (Pytorch Method): The nonlinear activation for the neural networkds

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. 

        lr (float): Learning rate (used for both policy and value learning).

        target_entropy (float): Entropy regularization coefficient. 
                  (lower target entropy encourages policy exploitation.)

        seed (int): Seed for random number generators (Pytorch/Numpy).

    """
    def __init__(self, env_type = "SIM",  obs_dim = 5, act_dim = 1, LSTM_out_dim = 5, act_limit = 1,
                 buffer_size = int(1e6), hidden_dim = 64, activation=F.relu,
                 gamma=0.5, polyak=0.995, lr=1e-3, target_entropy = -2, seed=1, load_buffer = None):
        
        
    
        # Initialize Replay Buffers
        self.replay_buffer = ReplayBufferLSTM2(buffer_size)

        if load_buffer:
            self.replay_buffer.load_buffer(load_buffer)
                    
        
        
        # Initialize Simulation Environment -- time step == 1 s
        # time step determines the estimated temperature rise in simulation
        self.env_type = env_type
        if env_type == 'SIM':
            from USM_SIM import USM_SIM
            self.env = USM_SIM(dt=1)
        
        elif env_type == 'EXP':
            #from USM_EXP import USM_EXP
            #self.env = USM_EXP()
            from USM_SIM import USM_SIM
            self.env = USM_SIM(dt=1)
        self.scale_obs = self.env.scale_obs
        
        # Set random seed -- for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Set variables
        self.obs_dim = obs_dim
        self.act_dim = act_dim  
        self.hidden_dim = hidden_dim
        self.LSTM_out_dim = LSTM_out_dim  
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.target_entropy = target_entropy
        self.lr = lr

        # Create actor-critic module and target networks
        self.ac = LSTMActorCritic(obs_dim, act_dim, act_limit = act_limit, hidden_dim= hidden_dim, LSTM_out_dim = LSTM_out_dim,  activation=activation)
        self.q_targ = deepcopy(self.ac.q)
        self.lstm_targ = deepcopy(self.ac.lstm)

        # Initialize trainable weights alpha/beta -- 0.01
        # alpha/beta are exponent of trainable weights log_alpha/log_beta
        # this bounds alpha/beta to be positive (>0)
        self.log_alpha = torch.tensor(1, dtype=torch.float32, requires_grad=True)
        self.alpha = self.log_alpha.exp()

        self.log_beta = torch.tensor(-1, dtype=torch.float32, requires_grad=True)
        self.beta = self.log_beta.exp()
   
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_targ.parameters():
            p.requires_grad = False
          
        # List of parameters for Q-networks (For freezing weights when neccessary)
        self.q_params       = self.ac.q.parameters()
        self.spd_params     = self.ac.spd.parameters()
        self.pi_params      = self.ac.pi.parameters()
        self.lstm_params    = self.ac.lstm.parameters()
        self.x2hc_params    = self.ac.x2hc.parameters()


        # Set up optimizers for policy, q-function, and speed estimator
        self.pi_optimizer = Adam(self.pi_params, lr=lr/2)
        self.q_optimizer = Adam(self.q_params, lr=lr)
        self.lstm_optimizer = Adam(self.lstm_params, lr=lr)
        self.x2hc_optimizer = Adam(self.x2hc_params, lr=lr)
        self.spd_optimizer = Adam(self.spd_params , lr=lr)

        # Set up optimizers for trainable weights log_alpha/log_beta
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr/2)
        self.beta_optimizer = Adam([self.log_beta], lr=lr/5)

    
    # Set up function for computing SAC Q-loss for a batch of experiences
    def compute_loss_q(self,data, hidden_in):
        state = data['state']
        action = data['action']
        last_action = data['last_action']
        reward = data['reward']
        next_state = data['next_state'] 

        lstm_output_q, hidden_out = self.ac.lstm(state, last_action, hidden_in)
        q = self.ac.q(state, action, lstm_output_q)
        
        with torch.no_grad():
            # Target actions come from *current* policy
            lstm_output_a2, _ = self.ac.lstm(next_state, action, hidden_out)
            next_action, logp_a2 = self.ac.pi(next_state, lstm_output_a2)

            # Target Q-values
            q_targ = self.q_targ(next_state, next_action, lstm_output_a2)
            q_pi_targ = torch.min(q_targ,dim=-1).values - self.alpha * logp_a2
            #print(q.shape,q_pi_targ.shape,logp_a2.shape,reward.shape)
            backup = reward  + self.gamma * (q_pi_targ) - self.beta*torch.abs(2*action.squeeze())
            backup = backup.unsqueeze(-1)  
        
        loss_q = torch.square(q - backup)
        loss_q = loss_q.mean()

        return loss_q, loss_q.detach().numpy()

    # Set up function for computing speed estimation loss
    def compute_loss_spd(self,data,hidden_in):
        state = data['state']
        last_action = data['last_action']
        spd = data['spd'].unsqueeze(-1) 
        #print("SPD",spd.mean(dim=1))
        lstm_output_q, _ = self.ac.lstm(state, last_action, hidden_in)
        spd_est = self.ac.spd(state,lstm_output_q)
        #print("SPD_est",spd_est.mean(dim=1))

        loss_spd = torch.square(spd - spd_est)
        loss_spd = loss_spd.mean()

        return loss_spd, loss_spd.detach().numpy()

    # Set up function for computing speed estimation loss
    def compute_loss_pi(self,data,hidden_in):
        #hidden_in = data['hidden_in']
        state = data['state']
        last_action = data['last_action']
        
        lstm_output_pi, _ = self.ac.lstm(state, last_action, hidden_in)
        pi, logp_pi = self.ac.pi(state, lstm_output_pi)

        q = self.ac.q(state, pi, lstm_output_pi)

        q_pi = torch.min(q,dim=-1,keepdim=True).values 
        # Entropy-regularized policy loss

        loss_pi = -(q_pi + self.alpha * (-logp_pi.unsqueeze(-1) - self.target_entropy) + 2*self.beta * pi).mean()
        loss_alpha = self.log_alpha * (-logp_pi.detach() - self.target_entropy).mean()
        loss_beta = self.log_beta * 2*pi.detach().mean()

        return loss_pi, loss_pi.detach().numpy(), loss_alpha, loss_beta


   # Update networks given a batch of experiences
    def update(self,data):

        for p in self.q_params:
            p.requires_grad = True
        for p in self.lstm_params:
            p.requires_grad = False
        for p in self.x2hc_params:
            p.requires_grad = False
        

        state = data['state']
        state_hidden = state[:,0,:].unsqueeze(dim=0) 
        hidden_in = self.ac.x2hc(state_hidden)

        # Training Q Function
        loss_q, self.q_info = self.compute_loss_q(data,hidden_in)
        self.q_optimizer.zero_grad()

        loss_q.backward()
        self.q_optimizer.step()        


        for p in self.q_params:
            p.requires_grad = False

        hidden_in = self.ac.x2hc(state_hidden)

        # Training Policy Function
        loss_pi, self.pi_info, loss_alpha, loss_beta = self.compute_loss_pi(data,hidden_in)

        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.beta_optimizer.zero_grad()
        loss_beta.backward()
        self.beta_optimizer.step()
        self.beta = self.log_beta.exp()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Training Policy Function
        
        for p in self.lstm_params:
            p.requires_grad = True
        for p in self.x2hc_params:
            p.requires_grad = True

        hidden_in = self.ac.x2hc(state_hidden)

        loss_spd, self.spd_info = self.compute_loss_spd(data,hidden_in)

        self.spd_optimizer.zero_grad()
        self.lstm_optimizer.zero_grad()
        self.x2hc_optimizer.zero_grad()

        loss_spd.backward()

        self.spd_optimizer.step()
        self.lstm_optimizer.step()
        self.x2hc_optimizer.step()


        #self.lstm_optimizer.zero_grad()
        #self.x2hc_optimizer.zero_grad()
        #self.lstm_optimizer.step()
        #self.x2hc_optimizer.step()
    
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q.parameters(), self.q_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update targetRuntimeError: Tensors must have same number of dimensions: got 3 and 2
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

            #for p, p_targ in zip(self.ac.lstm.parameters(), self.lstm_targ.parameters()):
            #    p_targ.data.mul_(self.polyak)
            #    p_targ.data.add_((1 - self.polyak) * p.data)
                
                
    # Estimate Q-value during traing -- for logging
    def get_q(self,o, a, l_a,hin):
      o = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
      a = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
      l_a = torch.as_tensor(l_a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

      lstm, _ = self.ac.lstm(o,l_a,hin)
      q = self.ac.q(o, a, lstm)
      q_ = torch.min(q)
      return q_.detach().numpy()

    # Estimate speed given observation -- for logging
    def get_speed_est(self,o,l_a,hin):
      spd = o[5]
      o = o[:self.obs_dim]
      o = self.scale_obs(o)
      o = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
      l_a = torch.as_tensor(l_a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
      lstm, _ = self.ac.lstm(o,l_a,hin)
      spd_est = self.ac.spd(o, lstm)
      return [spd_est.detach().numpy().squeeze(), spd]
    
    
    # Agent Training loop
    def train_agent(self, batch_size=64, max_ep_len=10, num_eps=3000,
                    update_every=10, eval_every = 250, pretrain = None):
        '''
        batch_size (int): Minibatch size for SGD.

        max_ep_len (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        num_eps (int): Number of episodes to run and train agent.
        
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
        
        eval_envery (int): Number of episodes that should elapse between policy
            evaluations 

        '''
    
       # Initilaize Logs and losses arrays
        LOGs = []
        losses = []
        ##Pretraining -- OffLine
        if pretrain:
          self.replay_buffer.load_buffer(pretrain)
          for i in range(10000):
            batch = self.replay_buffer.sample_batch(batch_size)
            self.update(batch)
            losses.append([self.q_info,-self.pi_info, self.spd_info])
            if i %200:
              print("{} % done".format(i//200))
          self.save_network()
        
       # Main loop: collect experience in env and update/log each epoch
        for ep in range(1,num_eps):

            # Randomize environement variables
            #params, vars = self.env.vary_param()
            
            # Reset the environment (random intialization)
            o,ep_ret = self.env.reset(), 0, 
            spd = 2*(o[5]-150)/300

           # Scale observation
            o = o[:self.obs_dim]
            o = self.scale_obs(o)

            # Initialize episodic q_value
            q = 0
            
            # Initialize array for LSTM buffer
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_spd = []

            l_a = np.ones(self.act_dim)
            hidden_in = self.ac.get_hidden_in(o)

            # loop over for one episode
            for t in range(max_ep_len):
                
                #Get action from current policy
                a, hidden_out = self.ac.get_action(o, l_a, hidden_in)
                # update episodic q
                q += self.get_q(o, a, l_a, hidden_in)


                 # Step the env and collect reward
                o2, r = self.env.step_frequency(a)
                
                # assign speed for speed estimation
                spd2 = 2*(o2[5]-150)/300
                
                # Scale next observation
                o2 = o2[:self.obs_dim]
                o2 = self.scale_obs(o2)

                # Update Episodic reward
                ep_ret += r
              
                
                # Append expereince ofr LSTM buffer arrays
                episode_state.append(o)
                episode_action.append(a)
                episode_last_action.append(l_a)
                episode_reward.append(r)
                episode_next_state.append(o2)
                episode_spd.append(spd)

                
                # Super critical, easy to overlook step: make sure to update 
                # most recent observation!
                o = o2
                spd = spd2
                l_a = a 
                hidden_in = hidden_out

                # Update handling
                if len(self.replay_buffer) > batch_size:
                    if (ep*max_ep_len + t) % update_every == 0:
                        for j in range(update_every):
                            # Sample a batch and update networks
                            batch = self.replay_buffer.sample_batch(batch_size)
                            self.update(data=batch)
                            # Append current losses (pi_loss is maximized)
                            losses.append([self.q_info,-self.pi_info,self.spd_info])

            # Store LSTM full episode experience
            self.replay_buffer.push(episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_spd)
            # Print Logging message
            if ep % 10 == 0:
                print("Episode: {} --> Temp: {} --> TargetSpeed/Speed: {}/{} --> Torque: {}--> reward: {} --> Q0: {} --> alpha: {} --> beta: {}".format(ep,self.env.get_temp(),self.env.get_targetSpeed(), self.env.get_speed(), self.env.get_torque(),ep_ret,q*(1-self.gamma),self.alpha.detach().numpy(),self.beta.detach().numpy()))
            
            # Save some logs
            LOGs.append(np.array([ep,self.env.get_temp(),self.env.get_torque(),self.env.get_targetSpeed(),ep_ret,self.env.get_speed(),q*(1-self.gamma),self.alpha.detach().numpy(),self.beta.detach().numpy()]))

            # Evaluate agent amid training
            if ep % eval_every == 0:
                self.run_evals_ep(ep)
            
            # Introduce random Noise
            #self.env.set_noise(np.abs(np.random.normal())*50,np.abs(np.random.normal())*0.05)

        # Save Normal buffer
        self.replay_buffer.save_buffer() 
        
        # Mean behaviour
        #self.env.vary_param(np.zeros(6))     

        return np.array(LOGs),np.array(losses)


        return np.array(LOGs),np.array(losses)
    
    # Load trained networks
    def load_network(self, env_type='sim'):
        self.ac.pi.load_state_dict(torch.load('pi_' + env_type))
        self.ac.q.load_state_dict(torch.load('q_' + env_type))
        self.ac.spd.load_state_dict(torch.load('spd_' + env_type))
        self.ac.lstm.load_state_dict(torch.load('lstm_' + env_type))
        self.ac.x2hc.load_state_dict(torch.load('x2hc_' + env_type))


        self.log_alpha = torch.load('log_alpha_{}.pt'.format(env_type)).requires_grad_(True)
        self.log_beta = torch.load('log_beta_{}.pt'.format(env_type)).requires_grad_(True)
        self.alpha = self.log_alpha.exp()
        self.beta = self.log_beta.exp()

        # Set up optimizers for trainable weights log_alpha/log_beta
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr/2)
        self.beta_optimizer = Adam([self.log_beta], lr=self.lr/5)
    
    # Save trained networks
    def save_network(self, env_type='sim'):
        torch.save(self.ac.pi.state_dict(),'pi_'+env_type)
        torch.save(self.ac.q.state_dict(),'q_'+env_type)
        torch.save(self.ac.spd.state_dict(),'spd_'+env_type)
        torch.save(self.ac.lstm.state_dict(),'lstm_'+env_type)
        torch.save(self.ac.x2hc.state_dict(),'x2hc_'+env_type)

        torch.save(self.log_alpha, 'log_alpha_{}.pt'.format(env_type))
        torch.save(self.log_beta, 'log_beta_{}.pt'.format(env_type))
        
    # General evaluation procedure given a set of target speeds, load torques,
    # and initial conditions
    def eval_agent(self, targetSpeeds, torques = None, freq = None,temp = None):
        # Reset the environment to initial driving frequency and temperature
        o = self.env.reset(freq=freq,temp=temp,T=torques[0], targetSpeed = targetSpeeds[0])
        ocopy = o.copy()
        # Store initial state
        states = [o]
        actions = [np.zeros(self.act_dim)]
        ocopy = o.copy()
        o = o[:self.obs_dim]
        o = self.scale_obs(o)

        hidden_in = self.ac.get_hidden_in(o)
        l_a = np.ones(self.act_dim)
        speeds = [self.get_speed_est(ocopy,l_a,hidden_in)]

        #Start Evaluation
        for i in range(len(targetSpeeds)):
          # Get system state under current load torque or target speed
          o = self.env.set_state(targetSpeed = targetSpeeds[i],torque = torques[i])
          o = o[:self.obs_dim]
          o = self.scale_obs(o)
           # Find a deterministic action (only mean)
          a, hidden_in = self.ac.get_action(o, l_a, hidden_in, deterministic=True)
          # Step the env
          o, _ = self.env.step_frequency(a)
          # Append states, action, and estimated speed
          actions.append(a*2)
          states.append(o)
          speed_estimation = self.get_speed_est(o,a,hidden_in)
          speeds.append(speed_estimation)
          l_a = a
          #Update temp experiment
          #if (self.env_type == 'EXP') and (i % 50 == 0):
          #    self.env.update_temp()

        states = np.array(states)
        speeds = np.array(speeds)
        #self.plot_eval(states,speeds,actions)
        actions = np.array(actions).reshape(-1,1)
        return np.concatenate((states, speeds, actions),axis=-1)

    def run_evals_ep(self,ep,n = 20,freq=None,temp=None,T=0):
        #Constant Speed
        Speeds = np.arange(n+1).reshape(-1,1)
        torques = [T]*n
        for targetSpeed in [300,200,100,0]:
            targetSpeeds = [targetSpeed]*n
            speeds = self.eval_agent(targetSpeeds,torques,freq,temp)
            plt.suptitle('Speed Tracking -- Episode: ' + str(ep), y=0.95, fontsize=20,fontweight='bold')
            Speeds = np.concatenate((Speeds,speeds),axis=-1)
            #self.env.stop()
            #time.sleep(300)
            #self.env.start(amp=3,freq=42)
        np.savetxt('logs/ep_{}_conSpeed_noload_{}rpm_{}.txt'.format(ep,targetSpeed,self.env_type), Speeds, delimiter=',')

    def run_evals_full(self):
        
        #Step Speed
        targetlevels = np.linspace(0,300, 7)
        targetlevels = np.concatenate((targetlevels,targetlevels[::-1][1:]))
        targetSpeeds = []
        for level in targetlevels:
            targetSpeeds += [level]*20 
            
        for torque in [0,0.2,0.5]:
            speeds = self.eval_agent(targetSpeeds,torques=[torque]*len(targetSpeeds),freq=39,temp=20)
            np.savetxt('logs/stepSpeed_conTorque_{}Nm_{}.txt'.format(torque,self.env_type), speeds, delimiter=',')
        
        #Sinusoidal Speed -- Constant Torque
        T = 250    #total steps
        n = 5      #num cycles
        for torque in [0,0.2,0.5]:
            for targSpeed in [175,150,100,50,25]:
                targetSpeeds = targSpeed-targSpeed*np.cos(np.linspace(0,n,T)*2*np.pi)
                speeds = self.eval_agent(targetSpeeds,torques=[torque]*len(targetSpeeds),freq=39,temp=20)
                np.savetxt('logs/sinSpeed_{}rpm_conTorque_{}Nm_{}.txt'.format(targSpeed*2,torque,self.env_type), speeds, delimiter=',')
        
        #Step Torque -- [0-1] N.m
        targetlevels = np.linspace(0,1,11)
        targetlevels = np.concatenate((targetlevels,targetlevels[::-1][1:]))
        torques = []
        for level in targetlevels:
            torques += [level]*20 
        for targSpeed in [100,200,300]:    
            speeds = self.eval_agent([targSpeed]*len(torques),torques,freq=39,temp=20)
            np.savetxt('logs/conSpeed_{}rpm_stepTorque_1Nm_{}.txt'.format(targSpeed,self.env_type), speeds, delimiter=',')
        
       #Step Torque -- [0-0.5] N.m
        targetlevels = np.linspace(0,0.5,6)
        targetlevels = np.concatenate((targetlevels,targetlevels[::-1][1:]))
        torques = []
        for level in targetlevels:
            torques += [level]*20 
        for targSpeed in [100,200,300]:    
            speeds = self.eval_agent([targSpeed]*len(torques),torques,freq=39,temp=20)
            np.savetxt('logs/conSpeed_{}rpm_stepTorque_0.5Nm_{}.txt'.format(targSpeed,self.env_type), speeds, delimiter=',')
        
        #Sinusoidal Torque -- Constant Speed
        for targspeed in [100,200,300]:
            for torque in [0.25,0.5]:
                T = 250    #total steps
                n = 5      #num cycles   
                torques = torque-torque*np.cos(np.linspace(0,n,T)*2*np.pi)
                speeds = self.eval_agent([targspeed]*len(torques),torques,freq=39,temp=20)
                np.savetxt('logs/conSpeed_{}rpm_sinTorque_{}Nm_{}.txt'.format(torque,targspeed,self.env_type), speeds, delimiter=',')
            
        
    # Plot main logs
    def plot_training_logs(self,LOGS,losses):
        # save logs
        np.savetxt('logs/logs_{}.txt'.format(self.env_type), LOGS, delimiter=',')
        np.savetxt('logs/losses_{}.txt'.format(self.env_type), losses, delimiter=',')

        # Moving average of reward and errors
        Ep_ret = LOGS[:,4]
        avg_ret = np.convolve(Ep_ret, np.ones(5), 'valid') / 5
        Err = np.abs(LOGS[:,5] - LOGS[:,3])
        Avg_err = np.convolve(Err, np.ones(5), 'valid') / 5
        
        
        plt.figure(figsize=(16,12))
        
        plt.subplot(2,2,1)
        plt.plot(Ep_ret,label="Episode Reward");
        plt.plot(avg_ret, label="Average Reward");
        plt.plot(LOGS[:,6], label="Expected Reward");
        plt.xlabel('Episode');
        plt.ylabel('Reward');
        plt.title('Learning Curve');
        plt.legend()
        
        plt.subplot(2,2,2)
        plt.plot(LOGS[:,3],label='Target Speed');
        plt.plot(LOGS[:,5],label='Actual Speed');
        plt.plot(Avg_err,label='Moving Average Err');
        plt.xlabel('Episode');
        plt.ylabel('Speed [rpm]');
        plt.legend()
        
        plt.subplot(2,2,3)
        plt.plot(LOGS[:,1]);
        plt.xlabel('Episode');
        plt.ylabel('USM Temperature [°C]');
        
        plt.subplot(2,2,4)
        plt.plot(LOGS[:,2]);
        plt.xlabel('Episode');
        plt.ylabel('Load Torque [N.m]');
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.plot(losses[:,0])
        plt.ylabel("Q-loss")

        plt.subplot(2,1,2)
        plt.plot(losses[:,1])
        plt.ylabel("Pi-loss")
        
        # Plot trainable parameters alpha/beta
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.plot(LOGS[:,-1])
        plt.ylabel("Beta")


        plt.subplot(2,1,2)
        plt.plot(LOGS[:,-2])
        plt.ylabel("Alpha")
        
    # Plot the evaluation results
    def plot_eval(self,states,speeds, actions):
      
        plt.figure(figsize=(16,12))
    
        #Plot 1
        ax1 = plt.subplot(2,2,1)
        ax1.plot(states[:,5],'r',label='Speed')
        ax1.plot(states[:,3],'b--',label='TargetSpeed')
        ax1.plot(abs(states[:,5]-states[:,3]), 'k',label = 'Speed Error')
        ax1.set_xlabel('Step',fontweight='bold',fontsize=16)
        ax1.set_ylabel('Speed [rpm]',fontweight='bold',fontsize=16)
        #plt.title('Constant Speed Tracking',fontweight='bold')
        
        ax11 = ax1.twinx()
        ax11.plot((states[:,2]),'g--',label='Feedback Voltage')
        ax11.set_ylabel('Feedback Voltage [V]',fontweight='bold',fontsize=16);
        ax1.legend();
        ax11.legend();


        ax2 = plt.subplot(2,2,2)
        ax2.plot(states[:,0],label='Driving Frequency')
        ax2.set_xlabel('Step',fontweight='bold',fontsize=16);
        ax2.set_ylabel('Driving Frequency [kHz]',fontweight='bold',fontsize=16);

        ax22 = ax2.twinx()
        ax22.plot((actions),'r--',label='Freq Action')
        ax22.set_ylabel('Frequency Action [kHz]',fontweight='bold',fontsize=16);
        ax2.legend();
        ax22.legend();
        #Plot 2
        ax3 = plt.subplot(2,2,3)
        ax3.plot(states[:,4], label='Temperature')
        ax3.set_xlabel('Step',fontweight='bold',fontsize=16);
        ax3.set_ylabel('Temperature [°C]',fontweight='bold',fontsize=16);
        
        ax33 = ax3.twinx()
        ax33.plot(states[:,1],'r--', label='Torque')
        ax33.set_ylabel('Torque [N.m]',fontweight='bold',fontsize=16);
        ax3.legend();
        ax33.legend();
        #Plot 4
        plt.subplot(2,2,4)
        plt.plot(speeds[:,1],label='Actual Speed')
        plt.plot((speeds[:,0]),label='Estimated Speed')
        plt.xlabel('Step',fontweight='bold',fontsize=16)
        plt.ylabel('Speed [rpm]',fontweight='bold',fontsize=16)
        #plt.title('Constant Speed Tracking',fontweight='bold')
        
        plt.legend();
        