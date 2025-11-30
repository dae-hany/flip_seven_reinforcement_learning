# simulate_daehan.py
#
# Analyzes Daehan Player's Solo Performance
# Goal: Calculate average rounds to reach 200 points.
#

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from flip_seven_env import FlipSevenCoreEnv
from simulate_duel import DaehanPlayer

# OpenMP duplicate library error fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def simulate_daehan_solo(num_games=1000, goal_score=200):
    print(f"\nStarting Solo Simulation for Daehan Player")
    print(f"Games: {num_games} | Goal Score: {goal_score}")
    print("=" * 70)
    
    env = FlipSevenCoreEnv()
    daehan = DaehanPlayer("Daehan Player")
    
    total_rounds_list = []
    
    for i in range(num_games):
        score = 0
        rounds = 0
        
        # Reset Deck per game (Solo Game)
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
                action = daehan.select_action(obs, info)
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
            
            score = info["total_game_score"]
            
        total_rounds_list.append(rounds)
        
        if (i + 1) % 100 == 0:
            print(f"Game {i + 1}/{num_games} finished. Rounds: {rounds}")
            
    avg_rounds = np.mean(total_rounds_list)
    min_rounds = np.min(total_rounds_list)
    max_rounds = np.max(total_rounds_list)
    
    print("\n" + "=" * 70)
    print("Simulation Results")
    print("=" * 70)
    print(f"Player: {daehan.name}")
    print(f"Total Games: {num_games}")
    print(f"Goal Score: {goal_score}")
    print(f"Average Rounds: {avg_rounds:.2f}")
    print(f"Best Game (Min Rounds): {min_rounds}")
    print(f"Worst Game (Max Rounds): {max_rounds}")
    print("=" * 70)

if __name__ == "__main__":
    simulate_daehan_solo(num_games=1000)
