# test_policy_with_risk_vs_reward.py
#
# 현재 라운드 점수에 따른 위험 감수(Risk vs. Reward) 성향을 테스트합니다.
# 점수가 낮을 때는 Hit을, 높을 때는 Stay를 선호해야 합니다.
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
    print("Risk vs. Reward Test: Analyzing Risk Tolerance by Round Score")
    print("=" * 70)
    print()
    
    # 1. 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    agent.load('./runs/dqn_flip7_final.pth')
    
    # 2. 전체 덱 생성
    full_deck = create_full_deck()
    print(f"전체 덱 생성 완료: {len(full_deck)}장\n")
    
    # 3. 테스트 시나리오 정의 (점수가 점진적으로 증가하는 손패들)
    test_scenarios = [
        ({3}, "Low"),
        ({5}, "Low"),
        ({8}, "Medium-Low"),
        ({10}, "Medium-Low"),
        ({10, 5}, "Medium"),
        ({10, 8}, "Medium"),
        ({12, 10}, "Medium-High"),
        ({12, 10, 5}, "High"),
        ({12, 11, 10}, "High"),
        ({12, 11, 10, 5}, "Very High"),
        ({12, 11, 10, 8}, "Very High"),
        ({12, 11, 10, 9, 5}, "Extremely High"),
    ]
    
    print("=" * 70)
    print("Test Configuration:")
    print(f"  Total Game Score: 50 (중립적 값)")
    print(f"  Testing {len(test_scenarios)} different hand scenarios")
    print("=" * 70)
    print()
    
    # 4. 데이터 수집을 위한 리스트 초기화
    round_scores = []
    q_stay_values = []
    q_hit_values = []
    hand_descriptions = []
    
    # 5. 각 시나리오에 대해 Q-values 계산
    print("Computing Q-values for varying round scores...")
    print()
    
    for hand, description in test_scenarios:
        # 라운드 점수 계산
        round_score = sum(hand)
        
        # 관측 생성 및 Q-values 계산
        obs = create_obs(hand, [], full_deck, 50)
        q_stay, q_hit = get_q_values(agent, obs)
        
        # 데이터 저장
        round_scores.append(round_score)
        q_stay_values.append(q_stay)
        q_hit_values.append(q_hit)
        hand_descriptions.append(description)
        
        # 결과 출력
        action_pref = "Stay" if q_stay > q_hit else "Hit"
        q_diff = abs(q_stay - q_hit)
        hand_str = str(sorted(list(hand)))
        print(f"  Hand: {hand_str:20s} | Score: {round_score:2d} | "
              f"Q(Stay)={q_stay:7.2f}, Q(Hit)={q_hit:7.2f} | "
              f"Prefer: {action_pref:4s} (diff={q_diff:.2f})")
    
    print("\nQ-value 계산 완료!\n")
    
    # ========================================================================
    # 6. 시각화: 선 그래프
    # ========================================================================
    print("=" * 70)
    print("Generating Visualization...")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Q(Stay)와 Q(Hit) 선 그래프
    ax.plot(round_scores, q_stay_values, linewidth=2.5, label='Q(Stay)', 
            color='#FF6B6B', marker='o', markersize=7)
    ax.plot(round_scores, q_hit_values, linewidth=2.5, label='Q(Hit)', 
            color='#4ECDC4', marker='s', markersize=7)
    
    # 0선 표시
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # 그래프 설정
    ax.set_xlabel('Current Round Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Q-Value', fontsize=14, fontweight='bold')
    ax.set_title('Risk vs. Reward Analysis: Q-Values vs Current Round Score\n(Total Game Score: 50)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 교차 지점 강조 (Q(Stay)가 Q(Hit)를 넘어서는 지점)
    q_diff = [q_stay_values[i] - q_hit_values[i] for i in range(len(round_scores))]
    for i in range(len(q_diff) - 1):
        if q_diff[i] <= 0 < q_diff[i + 1]:
            # 선형 보간으로 교차점 추정
            x1, x2 = round_scores[i], round_scores[i + 1]
            y1, y2 = q_diff[i], q_diff[i + 1]
            crossover_x = x1 - y1 * (x2 - x1) / (y2 - y1)
            
            ax.axvline(x=crossover_x, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'Crossover Point (~{crossover_x:.1f})')
            ax.plot(crossover_x, 
                   q_stay_values[i] + (q_stay_values[i+1] - q_stay_values[i]) * (crossover_x - x1) / (x2 - x1),
                   'ro', markersize=10, label='Policy Switch')
            break
    
    # 범례 재정렬
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=11)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    import os
    os.makedirs('./runs', exist_ok=True)
    save_path = './runs/policy_analysis_risk_vs_reward.png'
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
    q_diff = [q_stay_values[i] - q_hit_values[i] for i in range(len(round_scores))]
    
    # 역전 지점 찾기 (Q(Stay)가 Q(Hit)를 처음 넘어서는 지점)
    crossover_idx = None
    for i in range(len(q_diff)):
        if q_diff[i] > 0:
            crossover_idx = i
            break
    
    # 저점수/고점수 구간 분석
    low_score_indices = [i for i, s in enumerate(round_scores) if s <= 10]
    high_score_indices = [i for i, s in enumerate(round_scores) if s >= 25]
    
    if low_score_indices:
        avg_q_diff_low = np.mean([q_diff[i] for i in low_score_indices])
        low_prefer_hit = sum(1 for i in low_score_indices if q_diff[i] < 0)
        print(f"저점수 구간 (≤10점):")
        print(f"  평균 Q(Stay) - Q(Hit): {avg_q_diff_low:.2f}")
        print(f"  Hit 선호 비율: {low_prefer_hit}/{len(low_score_indices)} "
              f"({100*low_prefer_hit/len(low_score_indices):.1f}%)")
    
    if high_score_indices:
        avg_q_diff_high = np.mean([q_diff[i] for i in high_score_indices])
        high_prefer_stay = sum(1 for i in high_score_indices if q_diff[i] > 0)
        print(f"\n고점수 구간 (≥25점):")
        print(f"  평균 Q(Stay) - Q(Hit): {avg_q_diff_high:.2f}")
        print(f"  Stay 선호 비율: {high_prefer_stay}/{len(high_score_indices)} "
              f"({100*high_prefer_stay/len(high_score_indices):.1f}%)")
    
    if crossover_idx is not None:
        crossover_score = round_scores[crossover_idx]
        print(f"\n✓ 정책 전환 지점: Round Score = {crossover_score}점")
        print(f"  이 점수부터 Stay를 선호하기 시작합니다.")
        
        # 합리적인 전환인지 평가 (대략 15-30점 사이면 합리적)
        if 15 <= crossover_score <= 35:
            print("\n✓ 결론: 에이전트가 위험 대비 보상을 합리적으로 평가하고 있습니다!")
            print("  낮은 점수에서는 위험을 감수(Hit)하고,")
            print("  충분한 점수를 확보하면 안전하게 Stay를 선택합니다.")
        else:
            print("\n⚠ 주의: 정책 전환 지점이 예상 범위(15-35점)를 벗어났습니다.")
            if crossover_score < 15:
                print("  에이전트가 너무 보수적일 수 있습니다.")
            else:
                print("  에이전트가 지나치게 공격적일 수 있습니다.")
    else:
        # 모든 점수에서 Hit만 선호하거나 Stay만 선호하는 경우
        if all(d < 0 for d in q_diff):
            print("✗ 모든 점수에서 Hit를 선호합니다.")
            print("\n✗ 결론: 에이전트가 과도하게 공격적이며, 위험 관리를 학습하지 못했습니다.")
        elif all(d > 0 for d in q_diff):
            print("✗ 모든 점수에서 Stay를 선호합니다.")
            print("\n✗ 결론: 에이전트가 과도하게 보수적이며, 점수 최대화를 학습하지 못했습니다.")
        else:
            print("✗ 명확한 정책 전환 지점을 찾지 못했습니다.")
            print("\n✗ 결론: 에이전트의 위험 평가 정책이 불명확합니다.")
    
    print("=" * 70)
