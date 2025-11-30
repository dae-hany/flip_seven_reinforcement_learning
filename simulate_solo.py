# simulate_solo.py
#
# Analyzes Solo Performance: Average rounds to reach 200 points
#

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any, List

from flip_seven_env import FlipSevenCoreEnv
from agent import DQNAgent
from simulate_duel import DaehanPlayer

# OpenMP duplicate library error fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_solo_simulation(player_obj, player_type, num_games=1000, goal_score=200):
    """
    Runs solo simulation for a single player.
    Returns average rounds to reach goal.
    """
    env = FlipSevenCoreEnv()
    total_rounds_list = []
    
    # Handle name attribute safely
    p_name = getattr(player_obj, 'name', 'DQN Agent' if player_type == 'agent' else 'Unknown')
    print(f"Simulating Solo: {p_name} ({num_games} games)")
    
    for _ in range(num_games):
        score = 0
        rounds = 0
        
        # Reset Deck per game
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        while score < goal_score:
            rounds += 1
            env.total_score = score
            obs, info = env.reset()
            
            terminated = False
            while not terminated:
                if player_type == 'agent':
                    action = player_obj.select_action(obs, eval_mode=True)
                elif player_type == 'daehan':
                    action = player_obj.select_action(obs, info)
                
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
            
            score = info["total_game_score"]
            
        total_rounds_list.append(rounds)
        
    avg_rounds = np.mean(total_rounds_list)
    print(f"  -> Average Rounds: {avg_rounds:.2f}")
    return avg_rounds, total_rounds_list

if __name__ == "__main__":
    print("=" * 70)
    print("Solo Performance Analysis (Goal: 200 Points)")
    print("=" * 70)
    
    # 1. Daehan Player
    daehan = DaehanPlayer("Daehan Player")
    avg_daehan, rounds_daehan = run_solo_simulation(daehan, 'daehan', num_games=1000)
    
    # 2. DQN Agent
    agent = DQNAgent(device=DEVICE)
    model_path = './runs/dqn_flip7_final.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
        avg_agent, rounds_agent = run_solo_simulation(agent, 'agent', num_games=1000)
    else:
        print("Agent model not found, skipping agent simulation.")
        avg_agent = 0
        rounds_agent = []

    # Visualization (Histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(rounds_daehan, bins=range(min(rounds_daehan), max(rounds_daehan) + 2), 
             alpha=0.5, label=f'Daehan (Avg: {avg_daehan:.2f})', color='#FF6B6B', density=True)
    
    if rounds_agent:
        plt.hist(rounds_agent, bins=range(min(rounds_agent), max(rounds_agent) + 2), 
                 alpha=0.5, label=f'Agent (Avg: {avg_agent:.2f})', color='#4ECDC4', density=True)
    
    plt.title('Distribution of Rounds to Reach 200 Points (Solo)', fontsize=16, fontweight='bold')
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Frequency (Density)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = './runs/solo_performance_analysis.png'
    plt.savefig(save_path, dpi=150)
    print("\n" + "=" * 70)
    print(f"Analysis Complete. Plot saved to {save_path}")
    print("=" * 70)
