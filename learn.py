import time
import torch
from torch import nn
import os
import random
from gridworld import GridWorld
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class Q_Net(nn.Module):
    def __init__(self, input_features, output_features, neuron_count = 16):
        super().__init__();
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=neuron_count),
            nn.ReLU(),
            nn.Linear(in_features=neuron_count, out_features=output_features)   
        );
    
    def forward(self, x):
        return self.layer_stack(x);

class Replay_Buf():
    def __init__(self, cap):
        self.memory = deque([], maxlen = cap);
    
    def append(self, transition):
        self.memory.append(transition);
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size);
    
    def __len__(self):
        return len(self.memory);
    
def evaluate_agent(env, policy_net, episodes=100, render_every = 30):
    rewards = [];

    for _ in range(episodes):
        state = env.reset();
        done = False;
        total_reward = 0;

        steps = 0;
        while not done and steps < 200:
            if render_every is not None and (_ % render_every == 0):
                env.render()
                time.sleep(0.1);
                
            steps += 1
            x_tens = torch.from_numpy(state).float().unsqueeze(0).to(device);
            q_vals = policy_net(x_tens);
            action = torch.argmax(q_vals).item();

            next_state, reward, done = env.step(action);
            total_reward += reward;
            state = next_state;

        rewards.append(total_reward);

    return rewards;
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu");
env = GridWorld(size=20);

pol_dqn = Q_Net(input_features=4, output_features=4, neuron_count=16).to(device);
targ_dqn = Q_Net(input_features=4, output_features=4, neuron_count=16).to(device);
targ_dqn.load_state_dict(pol_dqn.state_dict());

buf = Replay_Buf(5000);

epsilon = 1;
epsilon_min = 0.05;
epsilon_decay = 0.999;
gamma = 0.95;
batch_size = 32;
sync_rate = 20;
learning_rate = 0.001;

loss_fn = nn.MSELoss();
optimizer = torch.optim.Adam(pol_dqn.parameters(), lr= learning_rate);
global_step = 0;

episodes = 2000;

for episode in range(episodes):
    state = env.reset();
    episode_reward = 0;
    
    done = False;
    steps = 0 ;
    
    while not done and steps < 200:
        steps += 1;
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else: 
            x_tens = torch.from_numpy(state).float().unsqueeze(0).to(device);
            q_vals = pol_dqn(x_tens);
            action = torch.argmax(q_vals).item();
    
        next_state, reward, done = env.step(action);
        episode_reward += reward;
        
        transition = (np.array(state, dtype=np.float32).flatten(),
                    action,
                    reward,
                    np.array(next_state, dtype=np.float32).flatten(),
                    done)
        buf.append(transition)
        
        state = next_state;
        global_step += 1;
        
        if len(buf) > batch_size:
            batch = buf.sample(batch_size)
            
            batch_states_np, batch_actions_np, batch_rewards_np, batch_next_states_np, batch_dones_np = zip(*batch);
            
            batch_states = torch.from_numpy(np.stack([np.array(s, dtype=np.float32).flatten() for s in batch_states_np])).to(device);
            batch_next_states = torch.from_numpy(np.stack([np.array(s, dtype=np.float32).flatten() for s in batch_next_states_np])).to(device);
            
            batch_actions = torch.from_numpy(np.array(batch_actions_np, dtype=np.int64)).to(device);
            batch_rewards = torch.from_numpy(np.array(batch_rewards_np, dtype=np.float32)).to(device);
            batch_dones = torch.from_numpy(np.array(batch_dones_np, dtype=np.float32)).to(device);
            
            q_vals_policy = pol_dqn(batch_states);
            actions = batch_actions.unsqueeze(1);
            chosen_action_qs = q_vals_policy.gather(1, actions).squeeze(1);
            
            next_q_values = targ_dqn(batch_next_states);
            max_next_q_values, _ = next_q_values.max(dim=1);
            target_qs = batch_rewards + gamma * max_next_q_values * (1 - batch_dones);
            
            loss = loss_fn(chosen_action_qs, target_qs);
            
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        
        if global_step % sync_rate == 0:
            targ_dqn.load_state_dict(pol_dqn.state_dict());
        
    epsilon = max(epsilon_min, epsilon * epsilon_decay);
    if episode % 100 == 0:
        print(f"Episode {episode} | epsilon={epsilon:.3f} | last reward={episode_reward}")
            
eval_rewards = evaluate_agent(env, pol_dqn, episodes=100);

plt.figure(figsize=(8,5));
plt.plot(eval_rewards, label="Episode reward");
plt.xlabel("Episode");
plt.ylabel("Total Reward");
plt.title("Evaluation of trained agent");
plt.legend();
plt.show();

print(f"Mean reward: {np.mean(eval_rewards):.2f}");
print(f"Max reward: {np.max(eval_rewards)}");
print(f"Min reward: {np.min(eval_rewards)}");
            
        