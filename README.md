# PROJECT---1
This is a Reinforcement-learning Flappybird game project, in which we train our model/bird to favorably pass the given obstacles using Deep-Q-networks.

# STEPS TO RUN 
STEP-1 :-> download the reposatory in your local machine
STEP-2 :-> make sure you have all packages installed for use.[torch , flappy_bird_gymnasium , yaml etc]
STEP-3 :-> change your directory to the folder in which you loaded the repo.
STEP-4 :-> train the model by hitting this command in terminal [ **python model.py flappybirdv0 --train** ]
STEP-5 :-> stop the training by pressing [**ctr c**]
STEP-6 :-> test the model by hitting this command in terminal [ **python model.py flappybirdv0 --train** ]

# IMPORTANT DETAILS 
@ The model is set to train infinitly until you manually stop the training or the reward count in single experience hit'e value 1000.
@ We only used single hidden layer in POLICY-network to keep things simple.
@ Hyper-parameters are mentioned in seprate .yaml file , if you want to change hyper-parameters just change the values in .yaml file.
@ Network sync rate is kept for optimal performance in low-performace devices.
@ Eplsilon greedy poicy is used for optimal performace.

[**TO GET OPTIMAL PERFORMACE IN TESTING PHASE RECOMMENDED TRAINING EPISODES ARE :- 30,00,000 ; although difference could be seen after every 25,000 episodes**]
[**RECOMMENDED TO TRAIN ON GPU**]

# DESCRIPTION 
This project implements a Deep Q-Network (DQN) agent to learn and play Flappy Bird using reinforcement learning. The agent interacts with the environment, learns from experience via replay memory, and improves its policy over time to maximize rewards.

# FEATURES
@ Deep Q-Learning implementation from scratch using PyTorch
@ Integration with Gymnasium-based Flappy Bird environment
@ Experience Replay for stable learning
@ Target Network for improved convergence
@ Epsilon-Greedy strategy for exploration vs exploitation
@ Model saving and logging of best performance


# TECH-STACK
Python
PyTorch
Gymnasium
NumPy
other modules

# PROJECT STRUCTURE
@ model.py → Main training and evaluation loop
@ Deep_Q_network.py → Neural network model (Q-network)
@ memory.py → Replay memory implementation
@ variables.yaml → Hyperparameter configurations
@ runs/ → Saved models and logs
@ _pycache_ → cache memory handling

# WORKING
The agent observes the game state from the environment.Selects actions using an epsilon-greedy policy.Stores experiences in replay memory.Samples mini-batches to train the Q-network.Uses a target network to stabilize learning.Continuously improves performance over episodes.

# FUTURE ADDON SCOPES
* Double DQN implementation
* Dueling Networks
* Prioritized Experience Replay
* Hyperparameter tuning automation
* Visualization of training metrics

# GOAL
The goal is to maximize cumulative reward by learning optimal actions (flap or no-flap) to keep the bird alive as long as possible.

**AUTHOR :- ANUJ**

