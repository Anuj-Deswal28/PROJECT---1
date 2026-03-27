import flappy_bird_gymnasium
import gymnasium as gym
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from Deep_Q_network import DQN
from memory import Replay_Memory
import random
import os


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
    
RUNS_DIR ='runs'
os.makedirs(RUNS_DIR , exist_ok=True)    

class agent():
    
    def __init__(self, params_set):
        self.params_set = params_set
        with open("variables.yaml") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[params_set]
            
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        self.LOG_FILE = os.path.join(RUNS_DIR,f"{self.params_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR,f"{self.params_set}.pt")
        
    
    def run(self, is_traning = True , render = False):
        env = gym.make("FlappyBird-v0", render_mode="human"if render else None)   # setting up environment

        num_states = env.observation_space.shape[0]   # input dim  for neural network
        num_action = env.action_space.n               # output dim for neural network

        policy_dqn = DQN(num_states , num_action).to(device)   # creating our neural-network object which we will further train for desired action
        
        if is_traning:
            memory = Replay_Memory(self.replay_memory_size)
            epsilone = self.epsilon_init
            
            # BUILDING TARGET NETWORK
            target_nt = DQN(num_states , num_action).to(device)
            # COPYING WEIGHT AND BAISE VALUES FROM POLICY NETWORK TO TARGET NETWORK
            target_nt.load_state_dict(policy_dqn.state_dict())
            
            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            
            best_reward = float('-inf')
            
        else:
            # best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            
        for episode in itertools.count(): 
            
            state , _ = env.reset()   # calculating current state
            state = torch.tensor( state , dtype = torch.float32 , device = device)
            
            episode_rewards = 0
            terminated = False
            
            while (not terminated and episode_rewards < self.reward_threshold):
                # Next action: EPSILON GREEDY POLICY
                #  (feed the observation to your agent here)
                if is_traning and random.random() < epsilone:
                    action = env.action_space.sample()    # explore
                    action = torch.tensor( action , dtype = torch.long , device = device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()   # exploit
    
                # Processing:
                new_state , reward, terminated, _, _ = env.step(action.item())
                
                episode_rewards += reward
                
                # creating tensors
                reward = torch.tensor(reward , dtype=torch.float32 , device=device)
                new_state = torch.tensor(new_state , dtype=torch.float32 , device=device)

                if is_traning:
                    memory.add((state, action, new_state, reward, terminated))
                    steps += 1
            
                state = new_state     
                
                
            print(f"EPISODE :-> {episode+1} \n TOTAL REWARD :-> {episode_rewards} || EPSILON :-> {epsilone} ") 
            
            # EPSILONE DECAY
            if is_traning:
                epsilone = max(epsilone * self.epsilon_decay , self.epsilon_min)
                
                if episode_rewards > best_reward:
                    log_msg = f"best reward = {episode_rewards} for episode = {episode+1}"
                    with open(self.LOG_FILE , 'a') as f:
                        f.write(log_msg + "\n")
                        
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_rewards
                
            if is_traning and len(memory) > self.mini_batch_size:
                # get samples for training
                mini_batch = memory.sample(self.mini_batch_size)
                
                self.optimize(mini_batch , policy_dqn, target_nt)
                
                # SYNC THE TRAINING NETWORK 
                if steps > self.network_sync_rate :
                    target_nt.load_state_dict(policy_dqn.state_dict())
                    steps = 0
                    
                    
    def optimize(self, mini_batch , policy_dqn, target_nt):
        # get batches
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)
        
        # calculating target Q value 
        with torch.no_grad():
            target_q = rewards + (1-terminations)*self.gamma*target_nt(next_states).max(dim=1)[0]
            
            
        # calculating current Q value
        current_q = policy_dqn(states).gather(dim=1, index = actions.unsqueeze(dim=1)).squeeze()
        
        # computing LOSS
        loss = self.loss_fn(current_q,target_q)
        
        # optimizing model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'train or test model')
    parser.add_argument('hyperparameters', help="")
    parser.add_argument('--train', help='Training mode', action = 'store_true')
    args = parser.parse_args()
    
    dql = agent(params_set = args.hyperparameters)
    
    if args.train:
        dql.run(is_traning = True)
    else:
        dql.run(is_traning = False , render = True)
        
        

                
