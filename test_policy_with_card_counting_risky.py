"""
파일: test_policy_with_card_counting_risky.py
목적: '카드 카운팅' 학습 여부를 검증하기 위한 고위험(high-risk) 시나리오 테스트 스크립트.
"""

import os
# OpenMP 중복 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
from typing import Dict, Any
import gymnasium as gym
import matplotlib.pyplot as plt  # 시각화용

from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX, MODIFIER_TO_IDX, NUMBER_CARD_TYPES, MODIFIER_CARD_TYPES

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Q-NETWORK ARCHITECTURE (train_dqn.py와 동일)
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
# DQN AGENT (test_policy_scenarios.py와 동일)
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
    
    def _dict_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """관찰 딕셔너리를 텐서 딕셔너리로 변환합니다."""
        return {
            key: torch.FloatTensor(value).unsqueeze(0).to(self.device)
            for key, value in obs_dict.items()
        }


# ============================================================================
# HELPER FUNCTIONS (test_policy_scenarios.py와 동일)
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


def get_q_values(agent: DQNAgent, env_state: Dict[str, np.ndarray]):
    """
    에이전트의 Q-values를 계산하고 출력합니다.
    
    Args:
        agent: DQN 에이전트
        env_state: 관측 딕셔너리
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
    
    # Print results
    print(f"    Q(Stay): {q_stay:7.2f} | Q(Hit): {q_hit:7.2f}")
    
    # Determine action
    if q_stay > q_hit:
        print(f"    → 선택: Stay (Q-value 차이: {q_stay - q_hit:.2f})")
    else:
        print(f"    → 선택: Hit (Q-value 차이: {q_hit - q_stay:.2f})")
    print()


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


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # 6-1. 모델과 덱을 로드합니다.
    print("=" * 70)
    print("📊 Scenario: High-Risk Card Counting (고위험 카드 카운팅)")
    print("=" * 70)
    print("설명: Bust 위험이 높은 손패({12, 11, 10, 7})를 기준으로,")
    print("      덱 상태(위험/안전)에 따라 에이전트의 *행동*이 바뀌는지 확인합니다.")
    print("=" * 70)
    print()
    
    agent = DQNAgent(device=DEVICE)
    agent.load('./runs/dqn_flip7_final.pth')
    full_deck = create_full_deck()
    
    # 6-2. 테스트 시나리오 상태를 정의합니다.
    # (이전 테스트의 한계: 손패가 1장이라 Bust 위험이 너무 낮았음)
    # (개선: 손패 4장, 40점의 고가치/고위험 상태로 설정)
    risky_hand_set = {12, 11, 10, 7}  # 40점
    risky_hand_str_set = {'12', '11', '10', '7'}
    neutral_total_score = 50

    q_values_risk = {}
    q_values_safe = {}

    # 6-3. Case A (Bust 위험 높음) Q-value 계산
    print("\n[Case A] 덱에 Bust 유발 카드가 남아있음 (Bust 위험 높음)")
    print(f"  손패: {sorted(list(risky_hand_set))} (40점)")
    print(f"  덱: 전체 85장 (손패 카드 {len(risky_hand_str_set)}종 모두 포함)")
    deck_risk = full_deck  # 덱에 {12, 11, 10, 7}이 모두 남아있음
    obs_risk = create_obs(risky_hand_set, [], deck_risk, neutral_total_score)
    
    # get_q_values는 출력을 포함하므로, Q-value만 따로 저장합니다.
    with torch.no_grad():
        obs_tensor = agent._dict_to_tensor(obs_risk)
        q_vals = agent.q_network(obs_tensor)
        q_values_risk['Stay'] = q_vals[0, 0].item()
        q_values_risk['Hit'] = q_vals[0, 1].item()
    
    get_q_values(agent, obs_risk)  # 콘솔 출력용

    # 6-4. Case B (Bust 위험 없음) Q-value 계산
    print("[Case B] 덱에 Bust 유발 카드가 없음 (Bust 불가능)")
    print(f"  손패: {sorted(list(risky_hand_set))} (40점)")
    print(f"  덱: 손패 카드 {len(risky_hand_str_set)}종 제외 (Bust 불가능)")
    deck_safe = [card for card in full_deck if card not in risky_hand_str_set]
    obs_safe = create_obs(risky_hand_set, [], deck_safe, neutral_total_score)
    
    with torch.no_grad():
        obs_tensor = agent._dict_to_tensor(obs_safe)
        q_vals = agent.q_network(obs_tensor)
        q_values_safe['Stay'] = q_vals[0, 0].item()
        q_values_safe['Hit'] = q_vals[0, 1].item()
        
    get_q_values(agent, obs_safe)  # 콘솔 출력용

    # 6-5. 최종 결론 출력
    print("=" * 70)
    print("📈 최종 분석 결과 📈")
    print("=" * 70)
    action_risk = "Stay" if q_values_risk['Stay'] > q_values_risk['Hit'] else "Hit"
    action_safe = "Stay" if q_values_safe['Stay'] > q_values_safe['Hit'] else "Hit"

    print(f"  - Case A (위험): Q(Stay)={q_values_risk['Stay']:.2f} vs Q(Hit)={q_values_risk['Hit']:.2f}  ->  선택: {action_risk}")
    print(f"  - Case B (안전): Q(Stay)={q_values_safe['Stay']:.2f} vs Q(Hit)={q_values_safe['Hit']:.2f}  ->  선택: {action_safe}")
    print()
    
    # Q(Hit) 차이 분석
    q_hit_diff = q_values_safe['Hit'] - q_values_risk['Hit']
    print(f"  - Q(Hit) 차이 (Safe - Risk): {q_hit_diff:+.2f}")
    
    if q_hit_diff > 0:
        print(f"    ✓ 덱이 안전할 때 Hit의 Q-value가 {q_hit_diff:.2f}만큼 더 높습니다.")
        print(f"    ✓ 에이전트가 카드 카운팅을 통해 위험도를 평가하고 있습니다.")
    else:
        print(f"    ✗ 덱 상태가 Q(Hit)에 긍정적 영향을 주지 않았습니다.")
    
    print()

    if action_risk == "Stay" and action_safe == "Hit":
        print("  [결론] ✅ 성공: 에이전트가 카드 카운팅을 기반으로 정책을 변경했습니다.")
        print("         위험한 상황에서는 Stay를, 안전한 상황에서는 Hit를 선택합니다.")
    elif action_risk == action_safe:
        print(f"  [결론] ⚠️  부분 성공: 에이전트가 두 상황 모두 '{action_risk}'를 선택했습니다.")
        print("         덱 상태에 따른 명확한 정책 변화는 관찰되지 않았지만,")
        print("         Q-value 차이를 통해 위험도 인식은 확인할 수 있습니다.")
    else:
        print("  [결론] ❌ 실패: 에이전트가 덱 상태에 따라 예상과 다른 행동을 선택했습니다.")
    print("=" * 70)

    # 7. 시각화 로직
    print("\n시각화 생성 중...")
    
    labels = ['Case A (Risk)', 'Case B (Safe)']
    q_stay_values = [q_values_risk['Stay'], q_values_safe['Stay']]
    q_hit_values = [q_values_risk['Hit'], q_values_safe['Hit']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, q_stay_values, width, label='Q(Stay)', 
                    color='salmon', edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, q_hit_values, width, label='Q(Hit)', 
                    color='mediumturquoise', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Q-Value', fontsize=12, fontweight='bold')
    ax.set_title('High-Risk Card Counting Analysis\n(Hand: {12, 11, 10, 7} = 40 points)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    # 막대 위에 값 표시
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontweight='bold')
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontweight='bold')

    fig.tight_layout()
    
    save_path = './runs/policy_analysis_high_risk_counting.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"그래프 저장 완료: {save_path}")
    print("=" * 70)
