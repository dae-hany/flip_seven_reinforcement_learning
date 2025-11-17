# test_policy_with_card_counting_test.py
#
# 모든 숫자 카드(0~12)에 대한 카드 카운팅 학습 여부를 테스트합니다.
# Case A (Bust 위험): 덱에 해당 카드가 남아있음
# Case B (Bust 안전): 덱에 해당 카드가 전혀 없음
#
import os
# OpenMP 중복 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
from typing import Dict, Any
import matplotlib.pyplot as plt

from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX, MODIFIER_TO_IDX, NUMBER_CARD_TYPES, MODIFIER_CARD_TYPES

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Q-NETWORK ARCHITECTURE
# ============================================================================
class QNetwork(nn.Module):
    """
    Q-Network that processes the Dict observation space from FlipSevenCoreEnv.
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
        
        q_values = self.shared_net(combined_feat)
        return q_values


# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent:
    """
    DQN 에이전트 (모델 로드 및 Q-values 조회만 사용)
    """
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.q_network = QNetwork().to(device)
        self.target_network = QNetwork().to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.epsilon = 0.0
    
    def load(self, filepath: str):
        """저장된 Q-network 가중치를 불러옵니다."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.q_network.eval()
        self.target_network.eval()
        print(f"모델을 {filepath} 에서 성공적으로 불러왔습니다.\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_obs(hand_nums: set, hand_mods: list, deck_list: list, total_score: int) -> Dict[str, np.ndarray]:
    """
    수동으로 관측 딕셔너리를 생성합니다.
    
    Args:
        hand_nums: 손에 있는 숫자 카드 집합 (예: {8, 12})
        hand_mods: 손에 있는 수정자 카드 리스트 (예: ['+4', 'x2'])
        deck_list: 덱에 남아있는 카드 리스트 (예: ['12', '11', ..., '+2'])
        total_score: 현재 게임 총점
    
    Returns:
        observation dictionary
    """
    # 1. current_hand_numbers
    hand_num_obs = np.zeros(13, dtype=np.int32)
    for num in hand_nums:
        hand_num_obs[num] = 1
    
    # 2. current_hand_modifiers
    hand_mod_obs = np.zeros(6, dtype=np.int32)
    for mod in hand_mods:
        hand_mod_obs[MODIFIER_TO_IDX[mod]] = 1
    
    # 3. deck_composition
    deck_comp_obs = np.zeros(19, dtype=np.int32)
    for card in deck_list:
        deck_comp_obs[CARD_TO_IDX[card]] += 1
    
    # 4. total_game_score
    total_score_obs = np.array([total_score], dtype=np.int32)
    
    return {
        "current_hand_numbers": hand_num_obs,
        "current_hand_modifiers": hand_mod_obs,
        "deck_composition": deck_comp_obs,
        "total_game_score": total_score_obs
    }


def create_full_deck() -> list:
    """
    전체 85장 카드 덱을 생성합니다 (환경의 _initialize_deck_to_discard와 동일).
    """
    deck = []
    # Number Cards (79 total)
    for i in range(1, 13):
        deck.extend([str(i)] * i)
    deck.append("0")  # 1x "0" card
    
    # Modifier Cards (6 total)
    deck.extend(MODIFIER_CARD_TYPES)
    
    return deck


def get_q_values(agent: DQNAgent, env_state: Dict[str, np.ndarray]) -> tuple:
    """
    에이전트의 Q-values를 계산합니다.
    
    Args:
        agent: DQN 에이전트
        env_state: 관측 딕셔너리
    
    Returns:
        (q_stay, q_hit) 튜플
    """
    # Convert to tensor (batch size = 1)
    obs_tensor = {
        key: torch.FloatTensor(value).unsqueeze(0).to(agent.device)
        for key, value in env_state.items()
    }
    
    # Get Q-values
    with torch.no_grad():
        q_values = agent.q_network(obs_tensor)
        q_stay = q_values[0, 0].item()
        q_hit = q_values[0, 1].item()
    
    return q_stay, q_hit


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    print("=" * 70)
    print("Card Counting Test: Analyzing Q-Values for All Number Cards")
    print("=" * 70)
    print()
    
    # 1. 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    agent.load('./runs/dqn_flip7_final.pth')
    
    # 2. 전체 덱 생성
    full_deck = create_full_deck()
    print(f"전체 덱 생성 완료: {len(full_deck)}장\n")
    
    # 3. 데이터 수집을 위한 리스트 초기화
    card_numbers = []
    q_stay_case_a = []  # Bust 위험 (덱에 카드 있음)
    q_hit_case_a = []
    q_stay_case_b = []  # Bust 안전 (덱에 카드 없음)
    q_hit_case_b = []
    
    # 4. 모든 숫자 카드에 대해 테스트
    print("=" * 70)
    print("Testing Card Counting for All Number Cards (0-12)")
    print("=" * 70)
    print()
    
    for card_str in NUMBER_CARD_TYPES:
        card_val = int(card_str)
        hand = {card_val}
        
        print(f"--- Testing Card '{card_str}' ---")
        
        # Case A: 덱에 해당 카드가 남아있음 (Bust 위험)
        deck_with_card = full_deck  # 전체 덱 사용
        obs_case_a = create_obs(hand, [], deck_with_card, 50)
        q_stay_a, q_hit_a = get_q_values(agent, obs_case_a)
        
        print(f"  Case A (Bust 위험): Q(Stay)={q_stay_a:.2f}, Q(Hit)={q_hit_a:.2f}")
        
        # Case B: 덱에 해당 카드가 전혀 없음 (Bust 불가능)
        deck_without_card = [c for c in full_deck if c != card_str]
        obs_case_b = create_obs(hand, [], deck_without_card, 50)
        q_stay_b, q_hit_b = get_q_values(agent, obs_case_b)
        
        print(f"  Case B (Bust 안전): Q(Stay)={q_stay_b:.2f}, Q(Hit)={q_hit_b:.2f}")
        
        # Q(Hit) 차이 계산
        hit_diff = q_hit_b - q_hit_a
        print(f"  → Q(Hit) 차이 (B - A): {hit_diff:.2f}")
        
        if hit_diff > 0:
            print(f"  ✓ 에이전트가 카드 '{card_str}'에 대한 카운팅을 학습함 (Hit가 안전할 때 더 선호)")
        else:
            print(f"  ✗ 카드 '{card_str}'에 대한 명확한 카운팅 신호 없음")
        
        print()
        
        # 데이터 저장
        card_numbers.append(card_str)
        q_stay_case_a.append(q_stay_a)
        q_hit_case_a.append(q_hit_a)
        q_stay_case_b.append(q_stay_b)
        q_hit_case_b.append(q_hit_b)
    
    # ========================================================================
    # 5. 시각화: 그룹형 막대 그래프
    # ========================================================================
    print("=" * 70)
    print("Generating Visualization...")
    print("=" * 70)
    
    x = np.arange(len(card_numbers))  # 카드 위치
    width = 0.2  # 막대 너비
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 4개의 막대 그룹 생성
    bars1 = ax.bar(x - 1.5*width, q_stay_case_a, width, label='Case A: Q(Stay) - Bust Risk', 
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, q_hit_case_a, width, label='Case A: Q(Hit) - Bust Risk', 
                   color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, q_stay_case_b, width, label='Case B: Q(Stay) - Bust Safe', 
                   color='#FFB6B6', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, q_hit_case_b, width, label='Case B: Q(Hit) - Bust Safe', 
                   color='#95E1D3', alpha=0.8)
    
    # 그래프 설정
    ax.set_xlabel('Card Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Q-Value', fontsize=14, fontweight='bold')
    ax.set_title('Card Counting Analysis: Q-Values for All Number Cards (0-12)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(card_numbers, fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    import os
    os.makedirs('./runs', exist_ok=True)
    save_path = './runs/policy_analysis_card_counting.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"그래프 저장 완료: {save_path}")
    print("=" * 70)
    
    # ========================================================================
    # 6. 분석 요약
    # ========================================================================
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    
    # Q(Hit) 차이 계산
    hit_diffs = [q_hit_case_b[i] - q_hit_case_a[i] for i in range(len(card_numbers))]
    avg_hit_diff = np.mean(hit_diffs)
    positive_count = sum(1 for d in hit_diffs if d > 0)
    
    print(f"평균 Q(Hit) 차이 (Case B - Case A): {avg_hit_diff:.2f}")
    print(f"긍정적 차이를 보인 카드 수: {positive_count}/{len(card_numbers)}")
    
    if avg_hit_diff > 0 and positive_count >= len(card_numbers) * 0.7:
        print("\n✓ 결론: 에이전트가 카드 카운팅을 성공적으로 학습했습니다!")
        print("  덱에 중복 카드가 없을 때 Hit를 더 선호하는 경향을 보입니다.")
    else:
        print("\n✗ 결론: 카드 카운팅 학습이 불충분합니다.")
        print("  추가 훈련이나 네트워크 구조 개선이 필요할 수 있습니다.")
    
    print("=" * 70)
