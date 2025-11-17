import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random
from typing import Dict, Tuple, Any
import gymnasium as gym

from flip_seven_env import FlipSevenCoreEnv


# ============================================================================
# HYPERPARAMETERS
# ============================================================================
NUM_TOTAL_GAMES_TO_TRAIN = 1000
TARGET_UPDATE_FREQUENCY = 10  # Update target network every N games
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Decay epsilon after each game
MIN_REPLAY_SIZE = 1000  # Start learning after this many transitions

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================================
# Q-NETWORK ARCHITECTURE (Handles Dict Observation Space)
# ============================================================================
class QNetwork(nn.Module):
    """
    Q-Network that processes the Dict observation space from FlipSevenCoreEnv.
    
    Architecture:
    1. Process each of the 4 observation components separately
    2. Concatenate all processed features
    3. Pass through shared MLP
    4. Output 2 Q-values (for 'Stay' and 'Hit' actions)
    """
    
    def __init__(
        self,
        hand_numbers_dim: int = 13,
        hand_modifiers_dim: int = 6,
        deck_composition_dim: int = 19,
        score_dim: int = 1,
        hidden_dim: int = 128
    ):
        super(QNetwork, self).__init__()
        
        # Separate processing layers for each observation component
        self.hand_numbers_net = nn.Sequential(
            nn.Linear(hand_numbers_dim, 32),
            nn.ReLU()
        )
        
        self.hand_modifiers_net = nn.Sequential(
            nn.Linear(hand_modifiers_dim, 16),
            nn.ReLU()
        )
        
        self.deck_composition_net = nn.Sequential(
            nn.Linear(deck_composition_dim, 64),
            nn.ReLU()
        )
        
        self.score_net = nn.Sequential(
            nn.Linear(score_dim, 8),
            nn.ReLU()
        )
        
        # Calculate total concatenated feature dimension
        concat_dim = 32 + 16 + 64 + 8  # = 120
        
        # Shared MLP layers
        self.shared_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: Q(s, Stay), Q(s, Hit)
        )
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            obs_dict: Dictionary containing batched observations
                - "current_hand_numbers": (batch_size, 13)
                - "current_hand_modifiers": (batch_size, 6)
                - "deck_composition": (batch_size, 19)
                - "total_game_score": (batch_size, 1)
        
        Returns:
            Q-values: (batch_size, 2)
        """
        # Process each component separately
        hand_numbers_feat = self.hand_numbers_net(obs_dict["current_hand_numbers"])
        hand_modifiers_feat = self.hand_modifiers_net(obs_dict["current_hand_modifiers"])
        deck_composition_feat = self.deck_composition_net(obs_dict["deck_composition"])
        score_feat = self.score_net(obs_dict["total_game_score"])
        
        # Concatenate all features
        combined_feat = torch.cat([
            hand_numbers_feat,
            hand_modifiers_feat,
            deck_composition_feat,
            score_feat
        ], dim=1)
        
        # Pass through shared MLP
        q_values = self.shared_net(combined_feat)
        
        return q_values


# ============================================================================
# REPLAY BUFFER
# ============================================================================
class ReplayBuffer:
    """
    Experience Replay Buffer for storing transitions.
    """
    
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool
    ):
        """Store a transition in the buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for obs, action, reward, next_obs, done in batch:
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent:
    """
    DQN Agent that learns to play Flip 7.
    """
    
    def __init__(
        self,
        action_space_size: int = 2,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        device: torch.device = DEVICE
    ):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.device = device
        
        # Initialize Q-networks
        self.q_network = QNetwork().to(device)
        self.target_network = QNetwork().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Epsilon for epsilon-greedy policy
        self.epsilon = EPSILON_START
    
    def _dict_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation dictionary to tensor dictionary."""
        return {
            key: torch.FloatTensor(value).unsqueeze(0).to(self.device)
            for key, value in obs_dict.items()
        }
    
    def _batch_dict_to_tensor(self, obs_batch: list) -> Dict[str, torch.Tensor]:
        """Convert batch of observation dictionaries to batched tensor dictionary."""
        batched_dict = {}
        
        # Get all keys from first observation
        keys = obs_batch[0].keys()
        
        for key in keys:
            # Stack all observations for this key
            batched_dict[key] = torch.FloatTensor(
                np.array([obs[key] for obs in obs_batch])
            ).to(self.device)
        
        return batched_dict
    
    def select_action(self, obs: Dict[str, np.ndarray], eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: Observation dictionary
            eval_mode: If True, always select greedy action (no exploration)
        
        Returns:
            Selected action (0 = Stay, 1 = Hit)
        """
        # Exploration: random action
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        # Exploitation: greedy action
        with torch.no_grad():
            obs_tensor = self._dict_to_tensor(obs)
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool
    ):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)
    
    def learn(self):
        """Perform one learning step (sample batch and update Q-network)."""
        # Don't learn until we have enough samples
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return
        
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        
        # Sample a batch from replay buffer
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(BATCH_SIZE)
        
        # Convert to tensors
        obs_tensor = self._batch_dict_to_tensor(obs_batch)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_tensor = self._batch_dict_to_tensor(next_obs_batch)
        done_tensor = torch.FloatTensor(done_batch).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(obs_tensor)
        current_q_values = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy policy."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save(self, filepath: str):
        """Save the Q-network."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the Q-network."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train():
    """
    Main training loop that handles Game/Round structure correctly.
    """
    # Initialize environment and agent
    env = FlipSevenCoreEnv()
    agent = DQNAgent()
    
    # Training statistics
    total_rounds_per_game = []
    total_scores_per_game = []
    
    print("=" * 70)
    print("Starting DQN Training on FlipSevenCoreEnv")
    print("=" * 70)
    print(f"Total games to train: {NUM_TOTAL_GAMES_TO_TRAIN}")
    print(f"Replay buffer size: {REPLAY_BUFFER_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gamma: {GAMMA}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epsilon: {EPSILON_START} -> {EPSILON_END} (decay: {EPSILON_DECAY})")
    print(f"Target update frequency: every {TARGET_UPDATE_FREQUENCY} games")
    print("=" * 70)
    
    # Main training loop: iterate over GAMES
    for game in range(NUM_TOTAL_GAMES_TO_TRAIN):
        
        # ====================================================================
        # 1. MANUALLY RESET THE ENTIRE GAME
        # ====================================================================
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()  # Reinitialize all 85 cards
        
        # Prepare the FIRST round
        obs, info = env.reset()
        
        game_total_rounds = 0
        game_total_reward = 0.0
        
        # ====================================================================
        # 2. THE GAME LOOP (continues until total_score >= 200)
        # ====================================================================
        while info.get("total_game_score", 0) < 200:
            game_total_rounds += 1
            terminated = False
            round_reward = 0.0
            
            # ================================================================
            # 3. THE ROUND (EPISODE) LOOP
            # ================================================================
            while not terminated:
                
                # Select action using epsilon-greedy policy
                action = agent.select_action(obs)
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store transition in replay buffer
                agent.store_transition(obs, action, reward, next_obs, terminated)
                
                # Perform one learning step
                agent.learn()
                
                # Update observation
                obs = next_obs
                round_reward += reward
            
            game_total_reward += round_reward
            
            # ================================================================
            # 4. END OF ROUND (terminated=True)
            # ================================================================
            # Prepare the NEXT round by calling reset()
            # This clears the hand but does NOT reset total_score
            if info.get("total_game_score", 0) < 200:
                obs, info = env.reset()
        
        # ====================================================================
        # 5. END OF GAME
        # ====================================================================
        final_score = info.get("total_game_score", 0)
        total_rounds_per_game.append(game_total_rounds)
        total_scores_per_game.append(final_score)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update target network periodically
        if (game + 1) % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
        
        # Logging
        if (game + 1) % 10 == 0:
            avg_rounds = np.mean(total_rounds_per_game[-10:])
            avg_score = np.mean(total_scores_per_game[-10:])
            print(f"Game {game + 1}/{NUM_TOTAL_GAMES_TO_TRAIN} | "
                  f"Rounds: {game_total_rounds} | "
                  f"Score: {final_score} | "
                  f"Avg Rounds (last 10): {avg_rounds:.2f} | "
                  f"Avg Score (last 10): {avg_score:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
        
        # Save model periodically
        if (game + 1) % 100 == 0:
            agent.save(f"dqn_flip7_game_{game + 1}.pth")
    
    # ========================================================================
    # TRAINING COMPLETED
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)
    print(f"Average rounds per game: {np.mean(total_rounds_per_game):.2f}")
    print(f"Min rounds in a game: {np.min(total_rounds_per_game)}")
    print(f"Max rounds in a game: {np.max(total_rounds_per_game)}")
    print(f"Average final score: {np.mean(total_scores_per_game):.2f}")
    print("=" * 70)
    
    # Save final model
    agent.save("dqn_flip7_final.pth")
    
    # Return agent for evaluation
    return agent, env


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate(agent: DQNAgent, env: FlipSevenCoreEnv, num_games: int = 10):
    """
    Evaluate the trained agent.
    
    Args:
        agent: Trained DQN agent
        env: Environment
        num_games: Number of games to evaluate
    """
    print("\n" + "=" * 70)
    print(f"Evaluating agent for {num_games} games...")
    print("=" * 70)
    
    eval_rounds_per_game = []
    eval_scores_per_game = []
    
    for game in range(num_games):
        # Reset game
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        
        obs, info = env.reset()
        game_total_rounds = 0
        
        # Play until 200 points
        while info.get("total_game_score", 0) < 200:
            game_total_rounds += 1
            terminated = False
            
            while not terminated:
                # Always select greedy action (eval_mode=True)
                action = agent.select_action(obs, eval_mode=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
            
            if info.get("total_game_score", 0) < 200:
                obs, info = env.reset()
        
        final_score = info.get("total_game_score", 0)
        eval_rounds_per_game.append(game_total_rounds)
        eval_scores_per_game.append(final_score)
        
        print(f"Eval Game {game + 1}: {game_total_rounds} rounds, Final Score: {final_score}")
    
    print("=" * 70)
    print(f"Evaluation Results (over {num_games} games):")
    print(f"  Average rounds: {np.mean(eval_rounds_per_game):.2f}")
    print(f"  Min rounds: {np.min(eval_rounds_per_game)}")
    print(f"  Max rounds: {np.max(eval_rounds_per_game)}")
    print(f"  Average final score: {np.mean(eval_scores_per_game):.2f}")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Train the agent
    trained_agent, trained_env = train()
    
    # Evaluate the trained agent
    evaluate(trained_agent, trained_env, num_games=10)
