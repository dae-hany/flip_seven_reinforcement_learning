import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random
from typing import Dict, Tuple, Any, Optional
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import the environment
from flip_seven_env import FlipSevenCoreEnv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================================
# HYPERPARAMETERS (Fixed for fair comparison)
# ============================================================================
NUM_TOTAL_GAMES_TO_TRAIN = 1000
TARGET_UPDATE_FREQUENCY = 10
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MIN_REPLAY_SIZE = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# Q-NETWORK (With Dueling Support)
# ============================================================================
class QNetwork(nn.Module):
    def __init__(self, hand_numbers_dim=13, hand_modifiers_dim=6, deck_composition_dim=19, score_dim=1, hidden_dim=128, action_space_size=2, use_dueling=False):
        super(QNetwork, self).__init__()
        self.use_dueling = use_dueling
        
        self.hand_numbers_net = nn.Sequential(nn.Linear(hand_numbers_dim, 32), nn.ReLU())
        self.hand_modifiers_net = nn.Sequential(nn.Linear(hand_modifiers_dim, 16), nn.ReLU())
        self.deck_composition_net = nn.Sequential(nn.Linear(deck_composition_dim, 64), nn.ReLU())
        self.score_net = nn.Sequential(nn.Linear(score_dim, 8), nn.ReLU())
        concat_dim = 32 + 16 + 64 + 8
        
        self.shared_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        
        if self.use_dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_space_size)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, action_space_size)
            )
    
    def forward(self, obs_dict):
        hand_numbers_feat = self.hand_numbers_net(obs_dict["current_hand_numbers"])
        hand_modifiers_feat = self.hand_modifiers_net(obs_dict["current_hand_modifiers"])
        deck_composition_feat = self.deck_composition_net(obs_dict["deck_composition"])
        score_feat = self.score_net(obs_dict["total_game_score"])
        combined_feat = torch.cat([hand_numbers_feat, hand_modifiers_feat, deck_composition_feat, score_feat], dim=1)
        
        shared_features = self.shared_net(combined_feat)
        
        if self.use_dueling:
            value = self.value_stream(shared_features)
            advantage = self.advantage_stream(shared_features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(shared_features)
            
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        for obs, action, reward, next_obs, done in batch:
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, use_double_dqn=False, use_dueling=False, action_space_size=2, learning_rate=LEARNING_RATE, gamma=GAMMA, device=DEVICE):
        self.use_double_dqn = use_double_dqn
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.device = device
        
        self.q_network = QNetwork(use_dueling=use_dueling).to(device)
        self.target_network = QNetwork(use_dueling=use_dueling).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def _dict_to_tensor(self, obs_dict):
        return {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs_dict.items()}
    
    def _batch_dict_to_tensor(self, obs_batch):
        batched_dict = {}
        keys = obs_batch[0].keys()
        for key in keys:
            batched_dict[key] = torch.FloatTensor(np.array([obs[key] for obs in obs_batch])).to(self.device)
        return batched_dict

    def select_action(self, obs, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        with torch.no_grad():
            obs_tensor = self._dict_to_tensor(obs)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def learn(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE or len(self.replay_buffer) < BATCH_SIZE:
            return None
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.replay_buffer.sample(BATCH_SIZE)
        obs_tensor = self._batch_dict_to_tensor(obs_batch)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_tensor = self._batch_dict_to_tensor(next_obs_batch)
        done_tensor = torch.FloatTensor(done_batch).to(self.device)

        current_q_values = self.q_network(obs_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN Logic
                next_q_values_local = self.q_network(next_obs_tensor)
                best_actions = next_q_values_local.argmax(dim=1)
                next_q_values_target = self.target_network(next_obs_tensor)
                max_next_q_values = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN Logic
                next_q_values = self.target_network(next_obs_tensor)
                max_next_q_values = next_q_values.max(dim=1)[0]
            
            target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_agent(use_double: bool, use_dueling: bool, label: str):
    print(f"\nStarting training: {label} (Double={use_double}, Dueling={use_dueling})...")
    
    # Use Linear Reward for stability as found in previous experiment
    env = FlipSevenCoreEnv(reward_type='linear')
    agent = DQNAgent(use_double_dqn=use_double, use_dueling=use_dueling)
    
    all_game_rounds = []
    
    for game in range(NUM_TOTAL_GAMES_TO_TRAIN):
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        obs, info = env.reset()
        
        game_total_rounds = 0
        
        while info.get("total_game_score", 0) < 200:
            game_total_rounds += 1
            terminated = False
            while not terminated:
                action = agent.select_action(obs)
                next_obs, reward, terminated, _, info = env.step(action)
                agent.store_transition(obs, action, reward, next_obs, terminated)
                agent.learn()
                obs = next_obs
            
            if info.get("total_game_score", 0) < 200:
                obs, info = env.reset()
        
        all_game_rounds.append(game_total_rounds)
        agent.decay_epsilon()
        if (game + 1) % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
            
        if (game + 1) % 100 == 0:
            print(f"  Game {game+1}/{NUM_TOTAL_GAMES_TO_TRAIN} | Avg Rounds (last 100): {np.mean(all_game_rounds[-100:]):.2f}")

    return all_game_rounds

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # 1. Standard DQN
    rounds_standard = train_agent(use_double=False, use_dueling=False, label="Standard DQN")
    
    # 2. Double DQN
    rounds_double = train_agent(use_double=True, use_dueling=False, label="Double DQN")
    
    # 3. Dueling DQN
    rounds_dueling = train_agent(use_double=False, use_dueling=True, label="Dueling DQN")
    
    # 4. Double Dueling DQN
    rounds_double_dueling = train_agent(use_double=True, use_dueling=True, label="Double Dueling DQN")
    
    # Visualization
    print("\nGenerating Comparison Plot...")
    
    data = {
        'Standard': rounds_standard,
        'Double': rounds_double,
        'Dueling': rounds_dueling,
        'Double+Dueling': rounds_double_dueling
    }
    
    plt.figure(figsize=(14, 8))
    
    for label, rounds in data.items():
        df = pd.DataFrame({'Rounds': rounds})
        ma = df['Rounds'].rolling(window=50, min_periods=1).mean()
        plt.plot(df.index, ma, label=label, linewidth=2)
    
    plt.title('Advanced DQN Features Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Game Number', fontsize=12)
    plt.ylabel('Rounds to Reach 200 Points (Lower is Better)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./runs/advanced_dqn_comparison_plot.png', dpi=150)
    print("Comparison plot saved to ./runs/advanced_dqn_comparison_plot.png")
