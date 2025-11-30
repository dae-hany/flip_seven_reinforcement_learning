# simulate_duel.py
#
# Flip Seven 2-Player Simulation
# Player 1: Trained DQN Agent
# Player 2: Daehan Player (Rational Card Counter)
#

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple

from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX, MODIFIER_TO_IDX, NUMBER_CARD_TYPES, MODIFIER_CARD_TYPES
from agent import DQNAgent

# OpenMP duplicate library error fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DAEHAN PLAYER (Rational Card Counter)
# ============================================================================
class DaehanPlayer:
    """
    A rational player who uses card counting and expected value calculations.
    """
    def __init__(self, name="Daehan Player"):
        self.name = name
    
    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Selects action based on Expected Value (EV).
        
        Action 0: Stay
        Action 1: Hit
        """
        # 1. Get current state info
        current_round_score = info.get("current_round_score_if_stay", 0)
        
        # If score is 0 (start of round), always Hit
        if current_round_score == 0:
            return 1
            
        # 2. Analyze Deck Composition (Card Counting)
        deck_composition = obs["deck_composition"] # shape (19,)
        total_cards_in_deck = np.sum(deck_composition)
        
        if total_cards_in_deck == 0:
            # Should not happen if env reshuffles, but safe fallback
            return 0 
            
        # 3. Identify Bust Cards vs Safe Cards
        # Bust Cards: Number cards that are already in hand
        # Safe Cards: Number cards NOT in hand + Modifier cards
        
        current_hand_numbers = obs["current_hand_numbers"] # shape (13,) binary
        
        bust_card_count = 0
        safe_card_count = 0
        expected_gain_sum = 0.0
        
        # Iterate through all card types
        for card_idx, count in enumerate(deck_composition):
            if count == 0:
                continue
                
            # Determine card type and value
            if card_idx < 13: # Number Card (0-12)
                card_val = card_idx
                if current_hand_numbers[card_val] == 1:
                    # This is a Bust Card
                    bust_card_count += count
                else:
                    # This is a Safe Number Card
                    safe_card_count += count
                    expected_gain_sum += (card_val * count)
            else: # Modifier Card
                # Modifiers are always safe from Bust (though x2 needs care)
                safe_card_count += count
                
                # Estimate value for modifiers
                # +2, +4, +6, +8, +10, x2
                # Indices: 13, 14, 15, 16, 17, 18
                mod_idx = card_idx - 13
                if mod_idx == 0: val = 2
                elif mod_idx == 1: val = 4
                elif mod_idx == 2: val = 6
                elif mod_idx == 3: val = 8
                elif mod_idx == 4: val = 10
                elif mod_idx == 5: val = current_round_score # x2 doubles current score
                else: val = 5 # Fallback
                
                expected_gain_sum += (val * count)
        
        # 4. Calculate Probabilities
        p_bust = bust_card_count / total_cards_in_deck
        p_safe = safe_card_count / total_cards_in_deck
        
        # 5. Calculate Expected Value
        # E[Gain] = Average value of a safe card
        avg_gain = expected_gain_sum / safe_card_count if safe_card_count > 0 else 0
        
        # EV(Stay) = Current Score
        ev_stay = current_round_score
        
        # EV(Hit) = P(Safe) * (Current Score + Avg Gain) + P(Bust) * 0
        # (Assuming Bust results in 0 round score)
        ev_hit = p_safe * (current_round_score + avg_gain)
        
        # 6. Decision
        if ev_hit > ev_stay:
            return 1 # Hit
        else:
            return 0 # Stay

# ============================================================================
# DUEL SIMULATION
# ============================================================================
def simulate_duel(num_games=100, goal_score=200):
    print(f"\nStarting Duel Simulation: DQN Agent vs {DaehanPlayer().name}")
    print(f"Games: {num_games} | Goal Score: {goal_score}")
    print("=" * 70)
    
    # Initialize Environment and Players
    env = FlipSevenCoreEnv() # Shared environment
    
    # Load Agent
    agent = DQNAgent(device=DEVICE)
    model_path = './runs/dqn_flip7_final.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Using random agent.")
    
    daehan = DaehanPlayer()
    
    # Statistics
    results = {
        "Agent_Wins": 0,
        "Daehan_Wins": 0,
        "Agent_Total_Rounds": [],
        "Daehan_Total_Rounds": [],
        "Agent_Final_Scores": [],
        "Daehan_Final_Scores": []
    }
    
    for game_idx in range(num_games):
        # Reset Game Scores
        agent_score = 0
        daehan_score = 0
        
        # Reset Deck for the Game
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        game_rounds = 0
        winner = None
        
        # Game Loop
        while agent_score < goal_score and daehan_score < goal_score:
            game_rounds += 1
            
            # --- Turn 1: DQN Agent ---
            # Sync Env Score for Agent
            env.total_score = agent_score
            obs, info = env.reset() # Hand reset, Deck kept
            
            terminated = False
            while not terminated:
                action = agent.select_action(obs, eval_mode=True)
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
            
            # Update Agent Score
            agent_score = info["total_game_score"]
            
            if agent_score >= goal_score:
                winner = "Agent"
                break
                
            # --- Turn 2: Daehan Player ---
            # Sync Env Score for Daehan
            env.total_score = daehan_score
            obs, info = env.reset() # Hand reset, Deck kept
            
            terminated = False
            while not terminated:
                action = daehan.select_action(obs, info)
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
                
            # Update Daehan Score
            daehan_score = info["total_game_score"]
            
            if daehan_score >= goal_score:
                winner = "Daehan"
                break
        
        # Record Result
        if winner == "Agent":
            results["Agent_Wins"] += 1
            results["Agent_Total_Rounds"].append(game_rounds)
        else:
            results["Daehan_Wins"] += 1
            results["Daehan_Total_Rounds"].append(game_rounds)
            
        results["Agent_Final_Scores"].append(agent_score)
        results["Daehan_Final_Scores"].append(daehan_score)
        
        if (game_idx + 1) % 10 == 0:
            print(f"Game {game_idx + 1}/{num_games} | Winner: {winner} | Score: {agent_score} vs {daehan_score}")

    # ========================================================================
    # ANALYSIS & VISUALIZATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("Duel Results Summary")
    print("=" * 70)
    
    agent_win_rate = (results["Agent_Wins"] / num_games) * 100
    daehan_win_rate = (results["Daehan_Wins"] / num_games) * 100
    
    print(f"Total Games: {num_games}")
    print(f"DQN Agent Wins: {results['Agent_Wins']} ({agent_win_rate:.1f}%)")
    print(f"Daehan Player Wins: {results['Daehan_Wins']} ({daehan_win_rate:.1f}%)")
    
    avg_rounds_agent = np.mean(results["Agent_Total_Rounds"]) if results["Agent_Total_Rounds"] else 0
    avg_rounds_daehan = np.mean(results["Daehan_Total_Rounds"]) if results["Daehan_Total_Rounds"] else 0
    
    print(f"Avg Rounds to Win (Agent): {avg_rounds_agent:.2f}")
    print(f"Avg Rounds to Win (Daehan): {avg_rounds_daehan:.2f}")
    
    # Plotting
    labels = ['DQN Agent', 'Daehan Player']
    wins = [results['Agent_Wins'], results['Daehan_Wins']]
    colors = ['#4ECDC4', '#FF6B6B']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, wins, color=colors)
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height} ({height/num_games*100:.1f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title(f'Flip Seven Duel Results ({num_games} Games)', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Wins', fontsize=12)
    plt.ylim(0, num_games * 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    save_path = './runs/duel_simulation_results.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nResult plot saved to {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    simulate_duel(num_games=10000)
