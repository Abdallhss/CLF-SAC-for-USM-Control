
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

# A general multi-layer-perceptron network structure
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

'''
Soft Policy (Actor) Class
Given a state, the network outputs the mean and standard deviation used to sample
the action from a gaussian distribution. The action is squashed using tanh function.
For a sampled action, the log probiblity is calculated under the squashed gaussian.  
'''
class SquashedGaussianMLPActor(nn.Module):
    
    # Initialize Class using Pytorch nn.Module super class
    # The mu_layer and log_std_layer share the same starting layer net
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = torch.as_tensor(act_limit, dtype=torch.float32)

        # Limits for clipping the log std of the action
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    # Define the forward function 
    # In case of policy evaluation (deterministic) -- Only the mean action is outputed.
    # During training, the action is normally sampled for better exploration
    # The log probability is calculated from the gaussian distribution taking into account
    # the tanh squashing
    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        else:
            logp_pi = None

        # Squash and scale the action
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

'''
Critic Class
Given a state and action, the network outputs the Q-value (expected discounted future total reward)
'''
class MLPQFunction(nn.Module):
    # Initialize Class using Pytorch nn.Module super class
    # Use multi-head output to regularize the output Q
    # Faster training compared to using multiple Q-networks.
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,n_outputs=10):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [n_outputs], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

'''
Speed Estimation Class
Given a state, the network outputs a speed estimation at the given state.
'''
class MLPVFunction(nn.Module):
    # Initialize Class using Pytorch nn.Module super class
    # Use multi-head output to regularize the output Q
    # Faster training compared to using multiple Q-networks.
    def __init__(self, obs_dim, hidden_sizes, activation,n_outputs=1):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [n_outputs], activation)

    def forward(self, obs):
        v = self.v(torch.cat([obs], dim=-1))
        return torch.squeeze(v, -1) # Critical to ensure q has right shape.

'''
Actor-Critic Class
Combines the actor network, the critic network, and a network for speed estimation. 
'''
class MLPActorCritic(nn.Module):


    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=((64,64),(64,64)),
                 activation=(nn.ReLU,nn.ReLU)):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes[0], activation[0], act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes[1], activation[1])
        self.spd = MLPVFunction(obs_dim, hidden_sizes[1], activation[1])

    # Take actions to collect experiences (no gradient is calculated)
    def act(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

'''
Replay Buffer Class

A simple FIFO experience replay buffer for SAC agents.
Stores the experiences collected to be later used for actor and critic optimization.
'''
class ReplayBuffer:


    def __init__(self, obs_dim, act_dim, size, env_type):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.spd_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        
        self.env_type = env_type
    def store(self, obs, act, rew, next_obs, spd):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.spd_buf[self.ptr] = spd
        
        # update pointer position
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    # Sample randomly a batch of expereinces for training.
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     spd=self.spd_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def save_buffer(self):
        
        np.save('logs/obs_buf_' + self.env_type, self.obs_buf)
        np.save('logs/obs2_buf_' + self.env_type, self.obs2_buf)
        np.save('logs/act_buf_' + self.env_type, self.act_buf)
        np.save('logs/rew_buf_' + self.env_type, self.rew_buf)
        np.save('logs/spd_buf_' + self.env_type, self.spd_buf)

        np.save('logs/size' + self.env_type, self.size)
    
    def load_buffer(self,env):
        self.obs_buf = np.load('logs/obs_buf_{}.npy'.format(env))
        self.obs2_buf = np.load('logs/obs2_buf_{}.npy'.format(env))
        self.act_buf = np.load('logs/act_buf_{}.npy'.format(env))
        self.rew_buf = np.load('logs/rew_buf_{}.npy'.format(env))
        self.spd_buf = np.load('logs/spd_buf_{}.npy'.format(env))

        self.size = np.load('logs/size{}.npy'.format(env))
        self.ptr = self.size
        #self.size = 4000 * 10


'''
Replay Buffer Class for LSTM SAC training.

A simple FIFO experience replay buffer for SAC agents.
To save time training LSTM online -- same experiences can batched by episode and latter used for LSTM pretraining
'''

class ReplayBufferLSTM2:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    And each sample contains the whole episode instead of a single step.
    """
    def __init__(self, capacity, env_type):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        self.env_type = env_type
    def push(self, state, action, last_action, reward, next_state,spd):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, spd)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def __len__(self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)
    
    # Save buffer for later pre-training 
    def save_buffer(self):
        with open("logs/buffer_{}".format(self.env_type), "wb") as fp:
            pickle.dump(self.buffer, fp)
    
    # Load buffer
    def load_buffer(self):
        with open("logs/buffer_{}".format(self.env_type), "rb") as fp:
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
    def __init__(self, env_type = "SIM", obs_dim = 4, act_dim = 1, act_limit = [2],
                 buffer_size = int(1e6), hidden_sizes = ([64],[64]*2), activation=(nn.ReLU, nn.ReLU),
                 gamma=0.5, polyak=0.995, lr=1e-3, target_entropy = -2, seed=1, load_buffer = None):
        
        
    
        # Initialize Replay Buffers
        self.replay_buffer = ReplayBuffer(obs_dim = obs_dim, act_dim = act_dim,
                                          size= buffer_size, env_type=env_type)
        if load_buffer:
            self.replay_buffer.load_buffer(load_buffer)
            
        self.replay_buffer_LSTM = ReplayBufferLSTM2(buffer_size, env_type=env_type)
        
        
        
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
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.target_entropy = target_entropy
        self.lr = lr

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(obs_dim, act_dim, act_limit, hidden_sizes = hidden_sizes, activation = activation)
        self.targ_q = deepcopy(self.ac.q)
        
        # Initialize trainable weights alpha/beta -- 0.01
        # alpha/beta are exponent of trainable weights log_alpha/log_beta
        # this bounds alpha/beta to be positive (>0)
        self.log_alpha = torch.tensor(1, dtype=torch.float32, requires_grad=True)
        self.alpha = self.log_alpha.exp()

        self.log_beta = torch.tensor(-1, dtype=torch.float32, requires_grad=True)
        self.beta = self.log_beta.exp()
   
        
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.targ_q.parameters():
            p.requires_grad = False
          
        # List of parameters for Q-networks (For freezing weights when neccessary)
        self.q_params = self.ac.q.parameters()

        # Set up optimizers for policy, q-function, and speed estimator
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr/2)
        self.q_optimizer = Adam(self.q_params, lr=lr)
        self.spd_optimizer = Adam(self.ac.spd.parameters(), lr=lr)

        # Set up optimizers for trainable weights log_alpha/log_beta
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr/2)
        self.beta_optimizer = Adam([self.log_beta], lr=lr/5)

    
    # Set up function for computing SAC Q-loss for a batch of experiences
    def compute_loss_q(self,data):
        o, a, r, o2 = data['obs'], data['act'], data['rew'], data['obs2']

        # Estimated Q
        q = self.ac.q(o,a)

        # Bellman backup for target Q value
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)
            # Target Q-values
            q_pi_targ = self.targ_q(o2, a2)
            
            # add weighted entropy tern to target q
            q_pi_targ = torch.min(q_pi_targ,dim=1).values - self.alpha * logp_a2
            # add current reward for state/action as well as action penalization
            backup = r  + self.gamma * (q_pi_targ ) - self.beta*torch.abs(2*a.squeeze())
            # reshape for proper broadcasting
            backup = torch.reshape(backup,(-1,1))

        # MSE loss against Bellman backup
        loss_q = torch.square(q - backup).mean()
        loss_q_info = loss_q.detach().numpy()
        # Return loss_q for training -- loss_q_info for logging
        return loss_q, loss_q_info 

    # Set up function for computing speed estimation loss
    def compute_loss_spd(self,data):
        o, spd = data['obs2'], data['spd']

        # Estimated Speed
        spd_est = self.ac.spd(o)

        # MSE loss
        loss_spd = torch.square(spd_est - spd).mean()
        loss_spd_info = loss_spd.detach().numpy()
        return loss_spd, loss_spd_info 

    # Set up function for computing speed estimation loss
    def compute_loss_pi(self,data):
        o = data['obs']

        # Compute action and its log probablity (gradient information neccessary)
        pi, logp_pi = self.ac.pi(o)
        
        # Estimate Q-value for the action
        q_pi = self.ac.q(o, pi)
        # Take min for a conservative Q-estimation
        q_pi = torch.min(q_pi,dim=1).values

        # Policy loss is mean to maximize Q-value
        # Additionaly two constraint are introducted with two lagrange multipliers for action and entropy
        # alpha/beta are exponents of log_alpha/log_beta 
        loss_pi = -(q_pi + self.beta*2*pi + self.alpha * (-logp_pi - self.target_entropy)).mean()  
        loss_pi_info = loss_pi.detach().numpy()
        
        # penalize deviation from target entorpy
        loss_alpha = self.log_alpha * (-logp_pi.detach() - self.target_entropy).mean()
        # penalize deviation from zero to minimize action
        loss_beta = self.log_beta * (2*pi.detach()).mean()      
         
        return loss_pi, loss_pi_info, loss_alpha, loss_beta


   # Update networks given a batch of experiences
    def update(self,data):

        # First run one gradient descent step for Q
        loss_q, self.q_info = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        loss_pi, self.pi_info,loss_alpha, loss_beta  = self.compute_loss_pi(data)

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # One gradient descent step for log_alpha
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # One gradient descent step for log_beta
        self.beta_optimizer.zero_grad()
        loss_beta.backward()
        self.beta_optimizer.step()
        self.beta = self.log_beta.exp()

        
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Next run one gradient descent step for speed estimator.
        loss_spd, self.spd_info = self.compute_loss_spd(data)
        self.spd_optimizer.zero_grad()
        loss_spd.backward()
        self.spd_optimizer.step()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q.parameters(), self.targ_q.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
                
                
    # Estimate Q-value during traing -- for logging
    def get_q(self,o,a):
      o = torch.as_tensor(o, dtype=torch.float32)
      a = torch.as_tensor(a, dtype=torch.float32)
      q1_ = self.ac.q(o, a)
      q_ = torch.min(q1_)
      return q_.detach().numpy()

    # Estimate speed given observation -- for logging
    def get_speed_est(self,o):
      spd = o[5]
      o = o[:self.obs_dim]
      o = self.scale_obs(o)
      o = torch.as_tensor(o, dtype=torch.float32)
      spd_est = self.ac.spd(o)*150+150
      
      return [spd_est.detach().numpy(), spd]
    
    
    # Agent Training loop
    def train_agent(self, batch_size=64, max_ep_len=10, num_eps=3000,
                    update_every=10, eval_every = 250):
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
        
       # Main loop: collect experience in env and update/log each epoch
        for ep in range(1,num_eps):

            # Randomize environement variables
            #params, vars = self.env.vary_param()
            
            # Reset the environment (random intialization)
            o, ep_ret = self.env.reset(ep=ep), 0

           # Scale observation
            o = o[:self.obs_dim]
            o = self.scale_obs(o)

            # Initialize episodic q_value
            q = 0
            
            # Initialize array for LSTM buffer
            l_a = np.zeros(self.act_dim)
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_spd = []

            # loop over for one episode
            for t in range(max_ep_len):
                
                #Get action from current policy
                a = self.ac.act(o)
                # update episodic q
                q += self.get_q(o,a)


                 # Step the env and collect reward
                o2, r = self.env.step_frequency(a)
                
                # assign speed for speed estimation
                spd = 2*(o2[5]-150)/300
                
                # Scale next observation
                o2 = o2[:self.obs_dim]
                o2 = self.scale_obs(o2)

                # Update Episodic reward
                ep_ret += r
              
                # Store experience to replay buffer
                self.replay_buffer.store(o, a, r, o2, spd)
                
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
                l_a = a 

                # Update handling
                if (ep*max_ep_len + t) % update_every == 0:
                    for j in range(update_every):
                        # Sample a batch and update networks
                        batch = self.replay_buffer.sample_batch(batch_size)
                        self.update(data=batch)
                        # Append current losses (pi_loss is maximized)
                        losses.append([self.q_info,-self.pi_info,self.spd_info])

            # Store LSTM full episode experience
            self.replay_buffer_LSTM.push(episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_spd)

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
        
        # Save LSTM buffer for later pre-training
        self.replay_buffer_LSTM.save_buffer()

        # Mean behaviour
        #self.env.vary_param(np.zeros(6))     

        return np.array(LOGs),np.array(losses)


        return np.array(LOGs),np.array(losses)
    
    # Load trained networks
    def load_network(self, env_type='sim'):
        self.ac.pi.load_state_dict(torch.load('pi_' + env_type))
        self.ac.q.load_state_dict(torch.load('q_' + env_type))
        self.ac.spd.load_state_dict(torch.load('spd_' + env_type))

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
        torch.save(self.log_alpha, 'log_alpha_{}.pt'.format(env_type))
        torch.save(self.log_beta, 'log_beta_{}.pt'.format(env_type))
        
    # General evaluation procedure given a set of target speeds, load torques,
    # and initial conditions
    def eval_agent(self, targetSpeeds, torques = None, freq = None,temp = None):
        # Reset the environment to initial driving frequency and temperature
        o = self.env.reset(freq=freq,temp=temp,T=torques[0], targetSpeed = targetSpeeds[0])
        # Store initial state
        states = [o]
        actions = [np.zeros(1)]
        speeds = [self.get_speed_est(o)]

        #Start Evaluation
        for i in range(len(targetSpeeds)):
          # Get system state under current load torque or target speed
          o = self.env.set_state(targetSpeed = targetSpeeds[i],torque = torques[i])
          o = o[:self.obs_dim]
          o = self.scale_obs(o)
           # Find a deterministic action (only mean)
          a = self.ac.act(o, deterministic=True)
          # Step the env
          o, _ = self.env.step_frequency(a)
          # Append states, action, and estimated speed
          actions.append(a*2)
          states.append(o)
          speeds.append(self.get_speed_est(o))
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
        