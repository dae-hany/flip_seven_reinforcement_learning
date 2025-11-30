# analyze_superiority.py
#
# Detailed Analysis of DQN Agent vs Daehan Player
# Metrics:
# 1. Flip 7 Success Rate
# 2. Bust Rate
# 3. Average Score per Round (excluding 0)
# 4. Risk Management (Hit Rate at High Risk)
#

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, Any, List

from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX
from agent import DQNAgent
from simulate_duel import DaehanPlayer

# OpenMP duplicate library error fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_bust_prob(obs):
    """Calculates the probability of busting given the current observation."""
    deck_composition = obs["deck_composition"]
    current_hand_numbers = obs["current_hand_numbers"]
    
    total_cards = np.sum(deck_composition)
    if total_cards == 0: return 0.0
    
    bust_cards = 0
    for i in range(13):
        if current_hand_numbers[i] == 1:
            bust_cards += deck_composition[i]
            
    return bust_cards / total_cards

def run_analysis(num_games=10000):
    print(f"\nStarting Superiority Analysis ({num_games} games)")
    print("=" * 70)
    
    env = FlipSevenCoreEnv()
    
    # Players
    agent = DQNAgent(device=DEVICE)
    model_path = './runs/dqn_flip7_final.pth'
    if os.path.exists(model_path):
        agent.load(model_path)
    
    daehan = DaehanPlayer("Daehan Player")
    
    players = [
        {'type': 'agent', 'obj': agent, 'name': 'DQN Agent'},
        {'type': 'daehan', 'obj': daehan, 'name': 'Daehan Player'}
    ]
    
    # Metrics Storage
    stats = {p['name']: {
        'rounds_played': 0,
        'flip_7_count': 0,
        'bust_count': 0,
        'total_score_sum': 0, # Sum of scores from non-bust rounds
        'non_bust_rounds': 0,
        'high_risk_situations': 0, # Bust prob > 30%
        'high_risk_hits': 0
    } for p in players}
    
    for game_idx in range(num_games):
        # Reset Deck per game (Independent games for fair comparison statistics)
        # We run them in parallel universes (same starting deck for each) to reduce variance?
        # Or just run many games. Let's run alternating turns in same game like duel, 
        # but track individual round stats.
        
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
                
                terminated = False
                bust_prob_at_decision = 0.0
                
                while not terminated:
                    # Calculate risk before action
                    bust_prob = calculate_bust_prob(obs)
                    
                    if p['type'] == 'agent':
                        action = p['obj'].select_action(obs, eval_mode=True)
                    elif p['type'] == 'daehan':
                        action = p['obj'].select_action(obs, info)
                    
                    # Track Risk Taking
                    if bust_prob > 0.30: # High Risk Threshold
                        stats[p['name']]['high_risk_situations'] += 1
                        if action == 1: # Hit
                            stats[p['name']]['high_risk_hits'] += 1
                    
                    next_obs, reward, terminated, _, info = env.step(action)
                    obs = next_obs
                
                # Round Ended
                stats[p['name']]['rounds_played'] += 1
                round_score = info.get("current_round_score_if_stay", 0) # This might be 0 if bust?
                # Wait, env.step returns info. If bust, calculate_score(bust=True) returns 0.
                # But we need to know if it was a Flip 7 or Bust.
                
                # Check Flip 7
                # If score > 0 and cards count == 7? 
                # Or check if last reward was high?
                # Let's infer from score and hand.
                # Actually, env doesn't explicitly say "Flip 7" in info.
                # But we can check hand size in obs? No, obs is reset.
                # We can check score. If score > 0, it wasn't a bust.
                
                # Re-calculate score logic to identify Flip 7 / Bust
                # If round_score == 0, it's a Bust (mostly, unless hand was empty? No, empty hand is score 0 but not terminated)
                # Terminated with score 0 -> Bust.
                
                if round_score == 0:
                    stats[p['name']]['bust_count'] += 1
                else:
                    stats[p['name']]['non_bust_rounds'] += 1
                    stats[p['name']]['total_score_sum'] += round_score
                    
                    # Check Flip 7 (Bonus 15 points implies Flip 7)
                    # It's hard to know for sure without internal state, but usually high score.
                    # Let's modify Env or just assume high score?
                    # Actually, we can't easily know Flip 7 count without modifying env.
                    # Let's skip explicit Flip 7 count for now, or infer from score?
                    # If we really want Flip 7 count, we should have returned it in info.
                    # Let's use "Score >= 40" as a proxy for "Big Win" (Flip 7 usually gives > 40).
                    if round_score >= 40:
                         stats[p['name']]['flip_7_count'] += 1 # Proxy for Big Win
                
                scores[p['name']] = info["total_game_score"]
                if scores[p['name']] >= 200:
                    winner = p['name']
                    
        if (game_idx + 1) % 1000 == 0:
            print(f"Game {game_idx + 1}/{num_games} completed.")

    # ========================================================================
    # REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("Superiority Analysis Results")
    print("=" * 70)
    
    metrics = ['Bust Rate', 'Big Win Rate (Score >= 40)', 'Avg Score (Safe Rounds)', 'High Risk Hit Rate']
    results_data = {m: [] for m in metrics}
    
    for p_name in [p['name'] for p in players]:
        s = stats[p_name]
        rounds = s['rounds_played']
        
        bust_rate = (s['bust_count'] / rounds) * 100
        big_win_rate = (s['flip_7_count'] / rounds) * 100
        avg_score = s['total_score_sum'] / s['non_bust_rounds'] if s['non_bust_rounds'] > 0 else 0
        risk_hit_rate = (s['high_risk_hits'] / s['high_risk_situations']) * 100 if s['high_risk_situations'] > 0 else 0
        
        results_data['Bust Rate'].append(bust_rate)
        results_data['Big Win Rate (Score >= 40)'].append(big_win_rate)
        results_data['Avg Score (Safe Rounds)'].append(avg_score)
        results_data['High Risk Hit Rate'].append(risk_hit_rate)
        
        print(f"[{p_name}]")
        print(f"  - Bust Rate: {bust_rate:.2f}%")
        print(f"  - Big Win Rate: {big_win_rate:.2f}%")
        print(f"  - Avg Score (Safe): {avg_score:.2f}")
        print(f"  - High Risk Hit Rate: {risk_hit_rate:.2f}%")
        print("-" * 30)

    # Plotting
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    agent_vals = [results_data[m][0] for m in metrics]
    daehan_vals = [results_data[m][1] for m in metrics]
    
    rects1 = ax.bar(x - width/2, agent_vals, width, label='DQN Agent', color='#4ECDC4')
    rects2 = ax.bar(x + width/2, daehan_vals, width, label='Daehan Player', color='#FF6B6B')
    
    ax.set_ylabel('Percentage / Score')
    ax.set_title('Agent vs Daehan Superiority Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('./runs/superiority_analysis.png')
    print(f"\nPlot saved to ./runs/superiority_analysis.png")

if __name__ == "__main__":
    run_analysis(num_games=10000)
