# test_policy_with_goal_awareness_test.py
#
# 200점 목표 점수 인식 여부를 테스트합니다.
# total_game_score를 0부터 200까지 변화시키며 Q-value의 변화를 관찰합니다.
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
    print("Goal Awareness Test: Analyzing 200-Point Target Recognition")
    print("=" * 70)
    print()
    
    # 1. 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    agent.load('./runs/dqn_flip7_final.pth')
    
    # 2. 전체 덱 생성
    full_deck = create_full_deck()
    print(f"전체 덱 생성 완료: {len(full_deck)}장\n")
    
    # 3. 테스트 설정
    # 고정된 손패: {12, 10} = 22점
    test_hand = {12, 10}
    round_score = sum(test_hand)  # 22점
    
    print("=" * 70)
    print(f"Test Configuration:")
    print(f"  Hand: {test_hand} (Round Score: {round_score} points)")
    print(f"  Total Game Score: 0 ~ 200 (step=5)")
    print(f"  Expected: Q(Stay) should surpass Q(Hit) around score ~{200 - round_score}")
    print("=" * 70)
    print()
    
    # 4. 데이터 수집을 위한 리스트 초기화
    total_scores = []
    q_stay_values = []
    q_hit_values = []
    
    # 5. total_game_score를 0부터 200까지 변화시키며 Q-values 계산
    print("Computing Q-values for varying total_game_score...")
    for total_score in range(0, 201, 5):
        obs = create_obs(test_hand, [], full_deck, total_score)
        q_stay, q_hit = get_q_values(agent, obs)
        
        total_scores.append(total_score)
        q_stay_values.append(q_stay)
        q_hit_values.append(q_hit)
        
        # 주요 지점에서 출력
        if total_score % 20 == 0:
            action_pref = "Stay" if q_stay > q_hit else "Hit"
            print(f"  Score={total_score:3d}: Q(Stay)={q_stay:7.2f}, Q(Hit)={q_hit:7.2f} → {action_pref}")
    
    print("\nQ-value 계산 완료!\n")
    
    # ========================================================================
    # 6. 시각화: 선 그래프
    # ========================================================================
    print("=" * 70)
    print("Generating Visualization...")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Q(Stay)와 Q(Hit) 선 그래프
    ax.plot(total_scores, q_stay_values, linewidth=2.5, label='Q(Stay)', 
            color='#FF6B6B', marker='o', markersize=5, markevery=4)
    ax.plot(total_scores, q_hit_values, linewidth=2.5, label='Q(Hit)', 
            color='#4ECDC4', marker='s', markersize=5, markevery=4)
    
    # 목표 점수 (200점) 및 임계점 표시
    critical_score = 200 - round_score
    ax.axvline(x=200, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Goal (200 points)')
    ax.axvline(x=critical_score, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Critical Point (~{critical_score})')
    
    # 0선 표시
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # 그래프 설정
    ax.set_xlabel('Total Game Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Q-Value', fontsize=14, fontweight='bold')
    ax.set_title(f'Goal Awareness Analysis: Q-Values vs Total Game Score\n(Hand: {test_hand}, Round Score: {round_score} points)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # x축 범위 설정
    ax.set_xlim(-5, 205)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    import os
    os.makedirs('./runs', exist_ok=True)
    save_path = './runs/policy_analysis_goal_awareness.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"그래프 저장 완료: {save_path}")
    print("=" * 70)
    
    # ========================================================================
    # 7. 분석 요약
    # ========================================================================
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    
    # Q(Stay) - Q(Hit) 차이 계산
    q_diff = [q_stay_values[i] - q_hit_values[i] for i in range(len(total_scores))]
    
    # 역전 지점 찾기 (Q(Stay)가 Q(Hit)를 처음 넘어서는 지점)
    crossover_idx = None
    for i in range(len(q_diff)):
        if q_diff[i] > 0:
            crossover_idx = i
            break
    
    if crossover_idx is not None:
        crossover_score = total_scores[crossover_idx]
        print(f"✓ Q(Stay)가 Q(Hit)를 처음 넘어서는 지점: Total Score = {crossover_score}")
        print(f"  (예상 임계점: ~{critical_score}, 실제: {crossover_score})")
        
        # 임계점 근처에서 역전이 일어났는지 확인
        if abs(crossover_score - critical_score) <= 20:
            print("\n✓ 결론: 에이전트가 200점 목표를 성공적으로 인식하고 있습니다!")
            print("  목표 달성에 필요한 점수 근처에서 Stay를 선호하기 시작합니다.")
        else:
            print("\n⚠ 주의: Q(Stay) 역전 지점이 예상과 다소 차이가 있습니다.")
            print("  에이전트의 목표 인식이 부정확하거나 다른 요인이 영향을 줄 수 있습니다.")
    else:
        print("✗ Q(Stay)가 Q(Hit)를 넘어서는 지점을 찾지 못했습니다.")
        print("\n✗ 결론: 에이전트가 200점 목표를 명확히 인식하지 못하고 있습니다.")
        print("  추가 훈련이나 보상 구조 개선이 필요할 수 있습니다.")
    
    # 최종 구간(180~200)에서의 Q-value 경향 분석
    final_region_start = 180
    final_indices = [i for i, s in enumerate(total_scores) if s >= final_region_start]
    
    if final_indices:
        avg_q_stay_final = np.mean([q_stay_values[i] for i in final_indices])
        avg_q_hit_final = np.mean([q_hit_values[i] for i in final_indices])
        
        print(f"\n최종 구간 (Score {final_region_start}~200) 평균 Q-values:")
        print(f"  Q(Stay) 평균: {avg_q_stay_final:.2f}")
        print(f"  Q(Hit) 평균: {avg_q_hit_final:.2f}")
        
        if avg_q_stay_final > avg_q_hit_final:
            print("  ✓ 최종 구간에서 Stay를 일관되게 선호합니다.")
        else:
            print("  ✗ 최종 구간에서도 Hit를 선호하여 목표 인식이 부족합니다.")
    
    print("=" * 70)
