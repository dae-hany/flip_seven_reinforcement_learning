# test_policy_scenarios.py
#
# 최종 훈련된 DQN 에이전트의 Q-values를 특정 시나리오에서 분석합니다.
# 이를 통해 에이전트가 카드 카운팅과 목표 인식을 학습했는지 정성적으로 평가합니다.
#
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
from typing import Dict, Any
import gymnasium as gym

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
# DQN AGENT (간소화 버전)
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
    
    print("=" * 70)
    print("Policy Scenario Testing: Analyzing Learned Q-Values")
    print("=" * 70)
    print()
    
    # 1. 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    agent.load('./runs/dqn_flip7_final.pth')
    
    # 2. 전체 덱 생성
    full_deck = create_full_deck()
    print(f"전체 덱 생성 완료: {len(full_deck)}장\n")
    
    # ========================================================================
    # SCENARIO 1: Card Counting Test
    # ========================================================================
    print("=" * 70)
    print("Scenario 1: Card Counting (카드 카운팅 학습 여부)")
    print("=" * 70)
    print("설명: 손에 'n'이 있을 때, 덱에 'n'이 남아있는지 여부에 따라")
    print("      에이전트의 행동이 달라지는지 확인합니다.")
    print("=" * 70)
    print()
    
    # NUMBER_CARD_TYPES (['0', '1', ..., '12']) 리스트를 순회합니다.
    for card_str in NUMBER_CARD_TYPES:
        card_val = int(card_str)
        s1_hand = {card_val}

        print(f"--- [ 테스트 대상: 손패에 '{card_str}'이 있는 경우 ] ---")

        # Case A: 덱에 해당 카드가 여전히 남아있음 (Bust 위험 있음)
        # (논리 오류 수정: full_deck을 그대로 사용하여 위험 상황을 시뮬레이션)
        print(f"  [Case A] 덱에 '{card_str}'이 남아있음 (Bust 위험)")
        s1_deck_with_card = full_deck 
        s1_obs_risk = create_obs(s1_hand, [], s1_deck_with_card, 50)
        get_q_values(agent, s1_obs_risk)
        
        # Case B: 덱에 해당 카드가 전혀 없음 (Bust 불가능)
        print(f"  [Case B] 덱에 '{card_str}'이 전혀 없음 (Bust 불가능)")
        s1_deck_no_card = [card for card in full_deck if card != card_str]
        s1_obs_safe = create_obs(s1_hand, [], s1_deck_no_card, 50)
        get_q_values(agent, s1_obs_safe)
    
    print("  ✓ 예상 결과: Case B에서 Hit의 Q-value가 더 높아야 함")
    print("  ✓ 이는 에이전트가 카드 카운팅을 학습했음을 의미함")
    print()
    
    # ========================================================================
    # SCENARIO 2: Goal Awareness Test
    # ========================================================================
    print("=" * 70)
    print("Scenario 2: Goal Awareness (목표 인식 학습 여부)")
    print("=" * 70)
    print("설명: 동일한 라운드 점수(25점)를 가지고 있을 때,")
    print("      게임 총점에 따라 에이전트의 행동이 달라지는지 확인합니다.")
    print("=" * 70)
    print()
    
    s2_hand = {12, 7, 6}  # 12 + 7 + 6 = 25 points
    s2_deck = full_deck  # 간단히 전체 덱 사용
    
    # Case A: 총점 100 (Stay해도 125점이므로 200점에 못 미침)
    print("  [Case A] 현재 총점: 100 (Stay 시 125점 → 200점 미달)")
    s2_obs_far = create_obs(s2_hand, [], s2_deck, 100)
    get_q_values(agent, s2_obs_far)
    
    # Case B: 총점 180 (Stay하면 205점이므로 게임 승리!)
    print("  [Case B] 현재 총점: 180 (Stay 시 205점 → 게임 승리!)")
    s2_obs_close = create_obs(s2_hand, [], s2_deck, 180)
    get_q_values(agent, s2_obs_close)
    
    print("  ✓ 예상 결과: Case B에서 Stay의 Q-value가 훨씬 높아야 함")
    print("  ✓ 이는 에이전트가 200점 목표를 인식하고 있음을 의미함")
    print()
    
    # ========================================================================
    # SCENARIO 3: Risk vs. Reward (추가 시나리오)
    # ========================================================================
    print("=" * 70)
    print("Scenario 3: Risk vs. Reward (위험 대비 보상 평가)")
    print("=" * 70)
    print("설명: 낮은 점수를 가지고 있을 때와 높은 점수를 가지고 있을 때")
    print("      에이전트의 위험 감수 성향이 달라지는지 확인합니다.")
    print("=" * 70)
    print()
    
    # Case A: 낮은 점수 (5점) - Hit 해야 함
    print("  [Case A] 손패: {5}, 라운드 점수: 5점 (너무 낮음)")
    s3_hand_low = {5}
    s3_obs_low = create_obs(s3_hand_low, [], full_deck, 50)
    get_q_values(agent, s3_obs_low)
    
    # Case B: 높은 점수 (40점 이상) - Stay 고려
    print("  [Case B] 손패: {12, 11, 10, 7}, 라운드 점수: 40점 (높음)")
    s3_hand_high = {12, 11, 10, 7}
    s3_obs_high = create_obs(s3_hand_high, [], full_deck, 50)
    get_q_values(agent, s3_obs_high)
    
    print("  ✓ 예상 결과: Case A에서는 Hit, Case B에서는 Stay가 선호되어야 함")
    print("  ✓ 이는 에이전트가 점수 기반 위험 관리를 학습했음을 의미함")
    print()
    
    # ========================================================================
    # SCENARIO 4: Modifier Card Effect
    # ========================================================================
    print("=" * 70)
    print("Scenario 4: Modifier Card Effect (수정자 카드 영향)")
    print("=" * 70)
    print("설명: 수정자 카드(x2)가 있을 때 에이전트의 평가가 달라지는지 확인")
    print("=" * 70)
    print()
    
    # Case A: 수정자 없음
    print("  [Case A] 손패: {10, 5}, 수정자: 없음 (15점)")
    s4_hand = {10, 5}
    s4_obs_no_mod = create_obs(s4_hand, [], full_deck, 50)
    get_q_values(agent, s4_obs_no_mod)
    
    # Case B: x2 수정자 있음
    print("  [Case B] 손패: {10, 5}, 수정자: x2 (30점)")
    s4_obs_with_x2 = create_obs(s4_hand, ['x2'], full_deck, 50)
    get_q_values(agent, s4_obs_with_x2)
    
    print("  ✓ 예상 결과: Case B에서 Stay의 Q-value가 더 높아야 함")
    print("  ✓ 이는 에이전트가 수정자 카드의 효과를 이해하고 있음을 의미함")
    print()
    
    print("=" * 70)
    print("시나리오 테스트 완료!")
    print("=" * 70)
