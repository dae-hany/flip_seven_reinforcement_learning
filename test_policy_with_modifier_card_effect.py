# test_policy_with_modifier_card_effect.py
#
# 6종류의 수정자 카드(+2, +4, +6, +8, +10, x2)가 Q-value에 미치는 영향을 분석합니다.
# 동일한 기본 손패에 서로 다른 수정자를 추가하여 정책 변화를 관찰합니다.
#

import os
# OpenMP 중복 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from flip_seven_env import CARD_TO_IDX, MODIFIER_TO_IDX, MODIFIER_CARD_TYPES
from agent import DQNAgent
import config

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def calculate_score_with_modifier(hand_nums: set, modifiers: list) -> int:
    """
    손패와 수정자를 고려한 라운드 점수를 계산합니다.
    
    Args:
        hand_nums: 손에 있는 숫자 카드 집합
        modifiers: 손에 있는 수정자 카드 리스트
    
    Returns:
        계산된 라운드 점수
    """
    # 기본 숫자 카드 합
    number_sum = sum(hand_nums)
    
    # x2 배수 적용
    if 'x2' in modifiers:
        number_sum = number_sum * 2
    
    # 추가 보너스 점수
    modifier_sum = 0
    for mod in modifiers:
        if mod.startswith('+'):
            modifier_sum += int(mod[1:])
    
    # Flip 7 보너스 (7장일 경우)
    flip_7_bonus = 15 if len(hand_nums) == 7 else 0
    
    total_score = number_sum + modifier_sum + flip_7_bonus
    return int(total_score)


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
    print("Modifier Card Effect Test: Analyzing Impact of Modifier Cards")
    print("=" * 70)
    print()
    
    # 1. 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    # config.FINAL_MODEL_PATH 사용 권장
    model_path = config.FINAL_MODEL_PATH if os.path.exists(config.FINAL_MODEL_PATH) else './runs/dqn_flip7_final.pth'
    
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    # 2. 전체 덱 생성
    full_deck = create_full_deck()
    print(f"전체 덱 생성 완료: {len(full_deck)}장\n")
    
    # 3. 테스트 설정
    # 기본 손패: {12} = 12점
    base_hand = {12}
    base_score = sum(base_hand)
    
    # 7가지 수정자 시나리오
    modifier_scenarios = [
        ([], "None"),
        (['+2'], "+2"),
        (['+4'], "+4"),
        (['+6'], "+6"),
        (['+8'], "+8"),
        (['+10'], "+10"),
        (['x2'], "x2"),
    ]
    
    print("=" * 70)
    print("Test Configuration:")
    print(f"  Base Hand: {base_hand} (Base Score: {base_score} points)")
    print(f"  Total Game Score: 50 (중립적 값)")
    print(f"  Testing {len(modifier_scenarios)} modifier scenarios")
    print("=" * 70)
    print()
    
    # 4. 데이터 수집을 위한 리스트 초기화
    scenario_labels = []
    effective_scores = []
    q_stay_values = []
    q_hit_values = []
    
    # 5. 각 수정자 시나리오에 대해 Q-values 계산
    print("Computing Q-values for each modifier scenario...")
    print()
    
    for modifiers, label in modifier_scenarios:
        # 실제 라운드 점수 계산
        effective_score = calculate_score_with_modifier(base_hand, modifiers)
        
        # 관측 생성 및 Q-values 계산
        obs = create_obs(base_hand, modifiers, full_deck, 50)
        q_stay, q_hit = get_q_values(agent, obs)
        
        # 데이터 저장
        scenario_labels.append(label)
        effective_scores.append(effective_score)
        q_stay_values.append(q_stay)
        q_hit_values.append(q_hit)
        
        # 결과 출력
        action_pref = "Stay" if q_stay > q_hit else "Hit"
        q_diff = abs(q_stay - q_hit)
        print(f"  Modifier: {label:6s} | Effective Score: {effective_score:2d} | "
              f"Q(Stay)={q_stay:7.2f}, Q(Hit)={q_hit:7.2f} | "
              f"Prefer: {action_pref:4s} (diff={q_diff:.2f})")
    
    print("\nQ-value 계산 완료!\n")
    
    # ========================================================================
    # 6. 시각화: 그룹형 막대 그래프
    # ========================================================================
    print("=" * 70)
    print("Generating Visualization...")
    print("=" * 70)
    
    x = np.arange(len(scenario_labels))  # 시나리오 위치
    width = 0.35  # 막대 너비
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Q(Stay)와 Q(Hit) 막대 그래프
    bars1 = ax.bar(x - width/2, q_stay_values, width, label='Q(Stay)', 
                   color='#FF6B6B', alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, q_hit_values, width, label='Q(Hit)', 
                   color='#4ECDC4', alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # 막대 위에 값 표시
    def autolabel(bars):
        """막대 위에 값을 표시합니다."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    
    # 그래프 설정
    ax.set_xlabel('Modifier Card', fontsize=14, fontweight='bold')
    ax.set_ylabel('Q-Value', fontsize=14, fontweight='bold')
    ax.set_title(f'Modifier Card Effect Analysis: Q-Values by Modifier Type\n(Base Hand: {base_hand}, Total Game Score: 50)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 각 시나리오의 실제 점수를 x축 아래에 표시
    for i, score in enumerate(effective_scores):
        ax.text(i, ax.get_ylim()[0] * 0.95, f'({score}pts)', 
               ha='center', va='top', fontsize=9, color='gray', style='italic')
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    import os
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(config.PLOTS_DIR, f'policy_analysis_modifier_effect_{base_hand}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"그래프 저장 완료: {save_path}")
    print("=" * 70)