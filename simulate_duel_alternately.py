"""
DQN Agent vs Daehan Player 교차 턴(Alternating Turns) 대결 시뮬레이션

이 스크립트는 두 플레이어가 동일한 라운드 내에서 턴을 번갈아가며 진행하는
방식을 시뮬레이션합니다. (예: Agent Hit -> Daehan Hit -> Agent Stay ...)
"""

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any, List, Set

# 기존 모듈 임포트
from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX, MODIFIER_TO_IDX
from agent import DQNAgent
from simulate_duel import DaehanPlayer
import config

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlternatingGameManager:
    """
    교차 턴 방식의 게임을 관리하는 클래스
    FlipSevenCoreEnv의 덱 로직을 재사용하되, 라운드 진행 로직은 재정의합니다.
    """
    def __init__(self):
        self.env = FlipSevenCoreEnv()
        self.reset_game_state()

    def reset_game_state(self):
        """게임 전체 상태 초기화"""
        self.env.draw_deck = collections.deque()
        self.env.discard_pile = []
        self.env._initialize_deck_to_discard()
        self.env._shuffle_discard_into_deck()
        self.agent_score = 0
        self.daehan_score = 0

    def _construct_obs(self, hand_nums: Set[int], hand_mods: List[str], total_score: int) -> Dict[str, np.ndarray]:
        """플레이어 시점의 관측(Observation) 생성"""
        hand_num_obs = np.zeros(13, dtype=np.int32)
        for num in hand_nums:
            hand_num_obs[num] = 1
        
        hand_mod_obs = np.zeros(6, dtype=np.int32)
        for mod in hand_mods:
            hand_mod_obs[MODIFIER_TO_IDX[mod]] = 1
            
        deck_comp_obs = np.zeros(19, dtype=np.int32)
        for card in self.env.draw_deck:
            deck_comp_obs[CARD_TO_IDX[card]] += 1
            
        total_score_obs = np.array([total_score], dtype=np.int32)
        
        return {
            "current_hand_numbers": hand_num_obs,
            "current_hand_modifiers": hand_mod_obs,
            "deck_composition": deck_comp_obs,
            "total_game_score": total_score_obs
        }

    def _calculate_current_points(self, hand_nums: Set[int], hand_mods: List[str]) -> int:
        """현재 손패의 점수 계산"""
        number_sum = sum(hand_nums)
        if 'x2' in hand_mods:
            number_sum *= 2
        modifier_sum = 0
        for mod in hand_mods:
            if mod.startswith('+'):
                modifier_sum += int(mod[1:])
        flip_7_bonus = 15 if len(hand_nums) == 7 else 0
        return number_sum + modifier_sum + flip_7_bonus

    def _draw_card(self) -> str:
        """덱에서 카드를 한 장 뽑습니다."""
        if not self.env.draw_deck:
            self.env._shuffle_discard_into_deck()
        return self.env.draw_deck.popleft()

    def play_round(self, agent: DQNAgent, daehan: DaehanPlayer):
        """한 라운드를 교차 턴 방식으로 진행합니다."""
        p1_hand_nums, p1_hand_mods, p1_done, p1_bust = set(), [], False, False
        p2_hand_nums, p2_hand_mods, p2_done, p2_bust = set(), [], False, False

        # 두 플레이어 모두 끝날 때까지 반복
        while not (p1_done and p2_done):
            # --- DQN Agent Turn ---
            if not p1_done:
                obs = self._construct_obs(p1_hand_nums, p1_hand_mods, self.agent_score)
                action = agent.select_action(obs, eval_mode=True)
                if action == 0: 
                    p1_done = True
                else:
                    card = self._draw_card()
                    if card.isdigit():
                        val = int(card)
                        if val in p1_hand_nums:
                            p1_bust = True
                            p1_done = True
                            self.env.discard_pile.append(card)
                        else:
                            p1_hand_nums.add(val)
                            if len(p1_hand_nums) == 7: p1_done = True
                    else:
                        p1_hand_mods.append(card)

            # --- Daehan Player Turn ---
            if not p2_done:
                current_points = self._calculate_current_points(p2_hand_nums, p2_hand_mods)
                info = {
                    "current_round_score_if_stay": current_points,
                    "cards_in_deck": len(self.env.draw_deck),
                    "cards_in_discard": len(self.env.discard_pile)
                }
                obs = self._construct_obs(p2_hand_nums, p2_hand_mods, self.daehan_score)
                action = daehan.select_action(obs, info)
                
                if action == 0:
                    p2_done = True
                else:
                    card = self._draw_card()
                    if card.isdigit():
                        val = int(card)
                        if val in p2_hand_nums:
                            p2_bust = True
                            p2_done = True
                            self.env.discard_pile.append(card)
                        else:
                            p2_hand_nums.add(val)
                            if len(p2_hand_nums) == 7: p2_done = True
                    else:
                        p2_hand_mods.append(card)

        # 점수 정산 및 카드 처리
        if not p1_bust:
            self.agent_score += self._calculate_current_points(p1_hand_nums, p1_hand_mods)
            self.env.discard_pile.extend([str(n) for n in p1_hand_nums] + p1_hand_mods)
            
        if not p2_bust:
            self.daehan_score += self._calculate_current_points(p2_hand_nums, p2_hand_mods)
            self.env.discard_pile.extend([str(n) for n in p2_hand_nums] + p2_hand_mods)

def run_alternating_simulation(num_games=1000, goal_score=200):
    print(f"\n교차 턴(Alternating Turns) 1:1 대결 시뮬레이션 시작")
    print(f"게임 수: {num_games} | 목표 점수: {goal_score}")
    print("=" * 70)
    
    manager = AlternatingGameManager()
    agent = DQNAgent(device=DEVICE)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print("모델을 찾을 수 없어 랜덤 에이전트로 진행합니다.")
        
    daehan = DaehanPlayer("Daehan Player")
    results = {"Agent": 0, "Daehan": 0}
    total_rounds = []
    
    for i in range(num_games):
        manager.reset_game_state()
        rounds = 0
        while manager.agent_score < goal_score and manager.daehan_score < goal_score:
            rounds += 1
            manager.play_round(agent, daehan)
            
        if manager.agent_score >= goal_score:
            results["Agent"] += 1
        else:
            results["Daehan"] += 1
        total_rounds.append(rounds)
        
        if (i + 1) % 100 == 0:
             print(f"게임 {i+1}/{num_games} 완료.")

    print("\n" + "=" * 70)
    print(f"최종 결과 (평균 {np.mean(total_rounds):.2f} 라운드 소요)")
    print(f"DQN Agent 승리: {results['Agent']} ({results['Agent']/num_games*100:.1f}%)")
    print(f"Daehan Player 승리: {results['Daehan']} ({results['Daehan']/num_games*100:.1f}%)")
    print("=" * 70)
    
    # === 시각화 (수치 표시 추가) ===
    labels = ['DQN Agent', 'Daehan Player']
    wins = [results['Agent'], results['Daehan']]
    colors = ['#4ECDC4', '#FF6B6B']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, wins, color=colors)
    
    # 막대 위에 수치 및 퍼센트 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({height/num_games*100:.1f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    plt.title(f'Alternating Turns Duel Results ({num_games} Games)', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Wins', fontsize=12)
    plt.ylim(0, max(wins) * 1.15) # 텍스트 공간 확보를 위해 y축 여유
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(config.PLOTS_DIR, 'alternating_duel_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    run_alternating_simulation(num_games=10000)