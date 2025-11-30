# test_policy_with_12_11_10.py
#
# 특정 시나리오 테스트: 손패에 {12, 11, 10}이 있을 때
# Case A (Safe): 덱에 12, 11, 10이 하나도 없음
# Case B (Risky): 덱에 나머지 12, 11, 10이 모두 있음
#

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
from typing import Dict, List, Set
import matplotlib.pyplot as plt

from flip_seven_env import CARD_TO_IDX, MODIFIER_TO_IDX, NUMBER_CARD_TYPES, MODIFIER_CARD_TYPES

# OpenMP 중복 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Q-NETWORK ARCHITECTURE (Must match training)
# ============================================================================
class QNetwork(nn.Module):
    """
    FlipSevenCoreEnv의 관측 공간을 처리하는 큐 네트워크
    """
    def __init__(
        self,
        hand_numbers_dim: int = 13,
        hand_modifiers_dim: int = 6,
        deck_composition_dim: int = 19,
        score_dim: int = 1,
        hidden_dim: int = 128,
        action_space_size: int = 2,
        use_dueling: bool = False
    ):
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
            self.output_layer = nn.Linear(hidden_dim, action_space_size)
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        hand_numbers_feat = self.hand_numbers_net(obs_dict["current_hand_numbers"])
        hand_modifiers_feat = self.hand_modifiers_net(obs_dict["current_hand_modifiers"])
        deck_composition_feat = self.deck_composition_net(obs_dict["deck_composition"])
        score_feat = self.score_net(obs_dict["total_game_score"])
        
        combined_feat = torch.cat([
            hand_numbers_feat,
            hand_modifiers_feat,
            deck_composition_feat,
            score_feat
        ], dim=1)
        
        shared_features = self.shared_net(combined_feat)
        
        if self.use_dueling:
            value = self.value_stream(shared_features)
            advantage = self.advantage_stream(shared_features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(shared_features)
        
        return q_values

# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent:
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.q_network = QNetwork(use_dueling=False).to(device) # Config matches training
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.q_network.eval()
        print(f"Model loaded from {filepath}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_obs(hand_nums: Set[int], hand_mods: List[str], deck_list: List[str], total_score: int) -> Dict[str, np.ndarray]:
    hand_num_obs = np.zeros(13, dtype=np.int32)
    for num in hand_nums:
        hand_num_obs[num] = 1
    
    hand_mod_obs = np.zeros(6, dtype=np.int32)
    for mod in hand_mods:
        hand_mod_obs[MODIFIER_TO_IDX[mod]] = 1
    
    deck_comp_obs = np.zeros(19, dtype=np.int32)
    for card in deck_list:
        deck_comp_obs[CARD_TO_IDX[card]] += 1
    
    total_score_obs = np.array([total_score], dtype=np.int32)
    
    return {
        "current_hand_numbers": hand_num_obs,
        "current_hand_modifiers": hand_mod_obs,
        "deck_composition": deck_comp_obs,
        "total_game_score": total_score_obs
    }

def create_full_deck() -> List[str]:
    deck = []
    for i in range(1, 13):
        deck.extend([str(i)] * i)
    deck.append("0")
    deck.extend(MODIFIER_CARD_TYPES)
    return deck

def get_q_values(agent: DQNAgent, env_state: Dict[str, np.ndarray]) -> tuple:
    obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(agent.device) for k, v in env_state.items()}
    with torch.no_grad():
        q_values = agent.q_network(obs_tensor)
        q_stay = q_values[0, 0].item()
        q_hit = q_values[0, 1].item()
    return q_stay, q_hit

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Scenario Test: Hand {12, 11, 10}")
    print("=" * 70)
    
    # 1. Load Agent
    agent = DQNAgent(device=DEVICE)
    model_path = './runs/dqn_flip7_final.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)
    agent.load(model_path)
    
    # 2. Setup Hand and Score
    hand_nums = {10, 11, 12}
    hand_mods = []
    current_score = 10 + 11 + 12 # 33
    
    print(f"\nHand: {hand_nums}")
    print(f"Current Score: {current_score}")
    
    # 3. Create Decks
    full_deck = create_full_deck()
    
    # Remove cards currently in hand from the "base" deck
    # We have one 10, one 11, one 12 in hand.
    base_deck = full_deck.copy()
    for card in ['10', '11', '12']:
        base_deck.remove(card)
        
    # Case A: Safe (No 10, 11, 12 in deck)
    deck_safe = [c for c in base_deck if c not in ['10', '11', '12']]
    
    # Case B: Risky (All remaining 10, 11, 12 in deck)
    deck_risky = base_deck.copy() # Contains all remaining 10s, 11s, 12s
    
    print(f"\n[Case A: Safe Deck]")
    print(f"  - Deck Count: {len(deck_safe)}")
    print(f"  - Contains '10': {'10' in deck_safe}")
    print(f"  - Contains '11': {'11' in deck_safe}")
    print(f"  - Contains '12': {'12' in deck_safe}")
    
    print(f"\n[Case B: Risky Deck]")
    print(f"  - Deck Count: {len(deck_risky)}")
    print(f"  - Count of '10': {deck_risky.count('10')}")
    print(f"  - Count of '11': {deck_risky.count('11')}")
    print(f"  - Count of '12': {deck_risky.count('12')}")

    # 4. Get Q-Values
    obs_safe = create_obs(hand_nums, hand_mods, deck_safe, current_score)
    q_stay_safe, q_hit_safe = get_q_values(agent, obs_safe)
    
    obs_risky = create_obs(hand_nums, hand_mods, deck_risky, current_score)
    q_stay_risky, q_hit_risky = get_q_values(agent, obs_risky)
    
    # 5. Report
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    print(f"\nCase A (Safe - No Bust Cards in Deck):")
    print(f"  Q(Stay): {q_stay_safe:.4f}")
    print(f"  Q(Hit) : {q_hit_safe:.4f}")
    print(f"  Action : {'HIT' if q_hit_safe > q_stay_safe else 'STAY'}")
    
    print(f"\nCase B (Risky - Many Bust Cards in Deck):")
    print(f"  Q(Stay): {q_stay_risky:.4f}")
    print(f"  Q(Hit) : {q_hit_risky:.4f}")
    print(f"  Action : {'HIT' if q_hit_risky > q_stay_risky else 'STAY'}")
    
    print("\n" + "-" * 70)
    diff = q_hit_safe - q_hit_risky
    print(f"Q(Hit) Difference (Safe - Risky): {diff:.4f}")
    
    if diff > 0:
        print("✓ SUCCESS: Agent values Hit higher when dangerous cards are gone.")
    else:
        print("✗ FAILURE: Agent does not value Hit higher in safe condition.")
        
    if q_hit_safe > q_stay_safe and q_hit_risky < q_stay_risky:
        print("✓ PERFECT: Agent switches from Stay to Hit based on deck composition!")
    elif q_hit_safe > q_stay_safe and q_hit_risky > q_stay_risky:
         print("! NOTE: Agent Hits in both cases, but check if Safe Hit is stronger.")
    elif q_hit_safe < q_stay_safe and q_hit_risky < q_stay_risky:
         print("! NOTE: Agent Stays in both cases, but check if Risky Hit is much lower.")

    # Visualization
    labels = ['Safe (No 10,11,12)', 'Risky (Has 10,11,12)']
    q_stays = [q_stay_safe, q_stay_risky]
    q_hits = [q_hit_safe, q_hit_risky]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, q_stays, width, label='Q(Stay)', color='lightcoral')
    rects2 = ax.bar(x + width/2, q_hits, width, label='Q(Hit)', color='skyblue')
    
    ax.set_ylabel('Q-Value')
    ax.set_title('Agent Policy: Hand {10, 11, 12} (Score 33)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('./runs/policy_analysis_12_11_10.png')
    print(f"\nPlot saved to ./runs/policy_analysis_12_11_10.png")
