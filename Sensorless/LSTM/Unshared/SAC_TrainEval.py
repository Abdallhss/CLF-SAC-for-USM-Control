# -*- coding: utf-8 -*-



from turtle import hideturtle
from SAC_Agent import SAC
import torch.nn.functional as F
import time

# Set parameters for agent
obs_dim = 4
act_dim = 1
LSTM_out_dim = 4
hidden_dim = 64

act_limit= [1]

buffer_size = int(1e6)
activation=F.relu

polyak=0.99
gamma=0.5
lr = 5e-3
seed = 5

for choice in [0,1]:
    #choice =   #0/1
    env_type = ['SIM','EXP'][choice]
    load_buffer = [None,'SIM'][choice]
    # Setup agent
    agent = SAC(env_type=env_type,obs_dim = obs_dim, act_dim = act_dim, act_limit= act_limit,
                 LSTM_out_dim = LSTM_out_dim, hidden_dim= hidden_dim, polyak = polyak, gamma = gamma,
                lr=lr, buffer_size= buffer_size,  activation=activation,seed=seed, load_buffer=load_buffer)

    # Setup paths for loading and saving agent
    agent_save = ['sim','exp'][choice]
    agent_load = [None,'sim'][choice]
    agent_train = True

    # Load agent
    if agent_load != None:
        agent.load_network(agent_load)

    # Start Motor
    #if env_type == 'EXP':
    #    agent.env.start(amp=3.5,freq=42)
        
    if agent_train:
        start_time = time.time()
        

        
        # Set parameters
        num_eps     = [4000,2000][choice]
        max_ep_len  = 10
        batch_size  = 64
        update_every= 10
        eval_every  = [1000,500][choice]
        
        # Starting training
        LOGS,losses = agent.train_agent(num_eps=num_eps, max_ep_len = max_ep_len,
                            batch_size = batch_size,update_every=update_every,eval_every=eval_every)
        
        print("training time: {}".format(time.time()-start_time))
        agent.plot_training_logs(LOGS,losses)
        
    # Save agent
    if agent_save != None:
        agent.save_network(agent_save)
            

    # if env_type == 'EXP':
    #     agent.env.stop()
    #     time.sleep(60)
    #     agent.env.start(amp=3.5,freq=42)
    #     time.sleep(2)

    # Evaluation
    agent.run_evals_ep('trained_39khz_200',n=200,freq=39,temp=30)
    agent.run_evals_ep('trained_45khz_200',n=200,freq=45,temp=30)
    agent.run_evals_full()

    #if env_type == 'EXP':
    #    agent.env.stop()


