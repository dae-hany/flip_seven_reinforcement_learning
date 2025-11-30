# simulate_6players.py
#
# Flip Seven 6-Player Simulation
# Players:
# 1. DQN Agent
# 2. Daehan Player (Rational)
# 3. Only Hit Player (Always Hit)
# 4. One Hit Player (Hit 1 time then Stay)
# 5. Two Hit Player (Hit 2 times then Stay)
# 6. Three Hit Player (Hit 3 times then Stay)
#

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple

from flip_seven_env import FlipSevenCoreEnv
from agent import DQNAgent
from simulate_duel import DaehanPlayer

# OpenMP duplicate library error fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# SIMPLE PLAYERS
# ============================================================================
class SimplePlayer:
    def __init__(self, name, strategy_type, hit_limit=None):
        self.name = name
        self.strategy_type = strategy_type # 'always_hit', 'fixed_hit'
        self.hit_limit = hit_limit
        self.hits_this_round = 0
        
    def reset_round(self):
        self.hits_this_round = 0
        
    def select_action(self, obs, info):
        # Action 0: Stay, 1: Hit
        
        if self.strategy_type == 'always_hit':
            return 1 # Always Hit
            
        elif self.strategy_type == 'fixed_hit':
            if self.hits_this_round < self.hit_limit:
                self.hits_this_round += 1
                return 1 # Hit
            else:
                return 0 # Stay
        
        return 0 # Default Stay

# ============================================================================
# 6-PLAYER SIMULATION
# ============================================================================
def simulate_6players(num_games=1000, goal_score=200):
    print(f"\nStarting 6-Player Simulation")
    print(f"Games: {num_games} | Goal Score: {goal_score}")
    print("=" * 70)
    
    # Initialize Environment
    env = FlipSevenCoreEnv()
    
    # Initialize Players
    # 1. DQN Agent
    agent = DQNAgent(device=DEVICE)
    model_path = './runs/dqn_flip7_final.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Using random agent.")
    
    # 2. Daehan Player
    daehan = DaehanPlayer("Daehan Player")
    
    # 3-6. Simple Players
    p3 = SimplePlayer("Only Hit", 'always_hit')
    p4 = SimplePlayer("One Hit", 'fixed_hit', hit_limit=1)
    p5 = SimplePlayer("Two Hit", 'fixed_hit', hit_limit=2)
    p6 = SimplePlayer("Three Hit", 'fixed_hit', hit_limit=3)
    
    players = [
        {'type': 'agent', 'obj': agent, 'name': 'DQN Agent'},
        {'type': 'daehan', 'obj': daehan, 'name': 'Daehan Player'},
        {'type': 'simple', 'obj': p3, 'name': 'Only Hit'},
        {'type': 'simple', 'obj': p4, 'name': 'One Hit'},
        {'type': 'simple', 'obj': p5, 'name': 'Two Hit'},
        {'type': 'simple', 'obj': p6, 'name': 'Three Hit'}
    ]
    
    # Statistics
    wins = {p['name']: 0 for p in players}
    total_rounds = {p['name']: [] for p in players}
    
    for game_idx in range(num_games):
        # Reset Game
        scores = {p['name']: 0 for p in players}
        
        # Reset Deck
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        game_rounds = 0
        winner = None
        
        # Game Loop
        while winner is None:
            game_rounds += 1
            
            for p in players:
                # Check if game already ended by previous player in this round loop?
                # Usually turns are sequential. If P1 wins, game over immediately? 
                # Or finish round? Standard board games usually finish immediately or equal turns.
                # Let's assume immediate win for simplicity and speed.
                if winner: break
                
                # Sync Score
                env.total_score = scores[p['name']]
                obs, info = env.reset()
                
                # Reset internal state for simple players
                if p['type'] == 'simple':
                    p['obj'].reset_round()
                
                terminated = False
                while not terminated:
                    if p['type'] == 'agent':
                        action = p['obj'].select_action(obs, eval_mode=True)
                    elif p['type'] == 'daehan':
                        action = p['obj'].select_action(obs, info)
                    elif p['type'] == 'simple':
                        action = p['obj'].select_action(obs, info)
                    
                    next_obs, reward, terminated, _, info = env.step(action)
                    obs = next_obs
                
                # Update Score
                scores[p['name']] = info["total_game_score"]
                
                if scores[p['name']] >= goal_score:
                    winner = p['name']
                    wins[winner] += 1
                    total_rounds[winner].append(game_rounds)
                    break
        
        if (game_idx + 1) % 100 == 0:
            print(f"Game {game_idx + 1}/{num_games} | Winner: {winner}")

    # ========================================================================
    # ANALYSIS & VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("6-Player Simulation Results")
    print("=" * 70)
    
    # Sort by wins
    sorted_results = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    
    for name, win_count in sorted_results:
        win_rate = (win_count / num_games) * 100
        avg_rnd = np.mean(total_rounds[name]) if total_rounds[name] else 0
        print(f"{name}: {win_count} Wins ({win_rate:.1f}%) | Avg Rounds: {avg_rnd:.2f}")
        
    # Plotting
    names = [x[0] for x in sorted_results]
    counts = [x[1] for x in sorted_results]
    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#4ECDC4', '#FF6B6B', '#95E1D3'] # Gold, Silver, Bronze, etc.
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(names, counts, color=colors[:len(names)])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height} ({height/num_games*100:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title(f'Flip Seven 6-Player Battle ({num_games} Games)', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Wins', fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    
    save_path = './runs/6player_simulation_results.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nResult plot saved to {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    simulate_6players(num_games=1000)
