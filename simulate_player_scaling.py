# simulate_player_scaling.py
#
# Analyzes DQN Agent's Win Rate vs Number of Players (2 to 6)
#

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, Any, List

from flip_seven_env import FlipSevenCoreEnv
from agent import DQNAgent
from simulate_duel import DaehanPlayer
from simulate_6players import SimplePlayer

# OpenMP duplicate library error fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation(num_players, num_games=1000, goal_score=200):
    """
    Runs a simulation for a specific number of players.
    Returns the Agent's win rate.
    """
    # Initialize Environment
    env = FlipSevenCoreEnv()
    
    # Initialize Players
    agent = DQNAgent(device=DEVICE)
    model_path = './runs/dqn_flip7_final.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
    
    daehan = DaehanPlayer("Daehan Player")
    
    # Pool of extra players to add
    extra_players_pool = [
        SimplePlayer("Three Hit", 'fixed_hit', hit_limit=3), # Strongest simple
        SimplePlayer("Only Hit", 'always_hit'),              # Aggressive
        SimplePlayer("Two Hit", 'fixed_hit', hit_limit=2),   # Passive
        SimplePlayer("One Hit", 'fixed_hit', hit_limit=1)    # Very Passive
    ]
    
    # Construct Player List
    players = [
        {'type': 'agent', 'obj': agent, 'name': 'DQN Agent'},
        {'type': 'daehan', 'obj': daehan, 'name': 'Daehan Player'}
    ]
    
    # Add extra players based on num_players
    # N=2: Agent, Daehan
    # N=3: + Three Hit
    # N=4: + Only Hit
    # ...
    if num_players > 2:
        players.extend([
            {'type': 'simple', 'obj': p, 'name': p.name} 
            for p in extra_players_pool[:num_players-2]
        ])
        
    print(f"Simulating {num_players} Players: {[p['name'] for p in players]}")
    
    agent_wins = 0
    
    for game_idx in range(num_games):
        scores = {p['name']: 0 for p in players}
        
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        winner = None
        
        while winner is None:
            for p in players:
                if winner: break
                
                env.total_score = scores[p['name']]
                obs, info = env.reset()
                
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
                
                scores[p['name']] = info["total_game_score"]
                
                if scores[p['name']] >= goal_score:
                    winner = p['name']
                    if winner == 'DQN Agent':
                        agent_wins += 1
                    break
    
    win_rate = (agent_wins / num_games) * 100
    print(f"  -> Agent Win Rate: {win_rate:.1f}%")
    return win_rate

if __name__ == "__main__":
    print("=" * 70)
    print("Player Scaling Analysis (2 to 6 Players)")
    print("=" * 70)
    
    player_counts = [2, 3, 4, 5, 6]
    win_rates = []
    
    for n in player_counts:
        rate = run_simulation(n, num_games=1000)
        win_rates.append(rate)
        
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(player_counts, win_rates, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
    
    for i, rate in enumerate(win_rates):
        plt.text(player_counts[i], rate + 1, f'{rate:.1f}%', ha='center', fontweight='bold')
        
    plt.title('DQN Agent Win Rate vs Number of Players', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Players', fontsize=12)
    plt.ylabel('Agent Win Rate (%)', fontsize=12)
    plt.xticks(player_counts)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(win_rates) * 1.2)
    
    save_path = './runs/player_scaling_analysis.png'
    plt.savefig(save_path, dpi=150)
    print("\n" + "=" * 70)
    print(f"Analysis Complete. Plot saved to {save_path}")
    print("=" * 70)
