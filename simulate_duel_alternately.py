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
from typing import Dict, Any, List, Tuple, Set

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
        # 환경 인스턴스는 덱(Deck)과 버림 카드(Discard Pile) 관리용으로만 사용
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
        """
        특정 플레이어의 시점에서 관측(Observation) 딕셔너리를 생성합니다.
        (FlipSevenCoreEnv._get_obs 와 동일한 포맷)
        """
        # 1. Hand Numbers
        hand_num_obs = np.zeros(13, dtype=np.int32)
        for num in hand_nums:
            hand_num_obs[num] = 1
        
        # 2. Hand Modifiers
        hand_mod_obs = np.zeros(6, dtype=np.int32)
        for mod in hand_mods:
            hand_mod_obs[MODIFIER_TO_IDX[mod]] = 1
            
        # 3. Deck Composition (공유된 덱 상태 반영)
        deck_comp_obs = np.zeros(19, dtype=np.int32)
        for card in self.env.draw_deck:
            deck_comp_obs[CARD_TO_IDX[card]] += 1
            
        # 4. Total Score
        total_score_obs = np.array([total_score], dtype=np.int32)
        
        return {
            "current_hand_numbers": hand_num_obs,
            "current_hand_modifiers": hand_mod_obs,
            "deck_composition": deck_comp_obs,
            "total_game_score": total_score_obs
        }

    def _calculate_current_points(self, hand_nums: Set[int], hand_mods: List[str]) -> int:
        """현재 손패의 점수를 계산합니다."""
        # FlipSevenCoreEnv._calculate_score 로직 참조
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
        """덱에서 카드를 한 장 뽑습니다. 덱이 비면 리셔플합니다."""
        if not self.env.draw_deck:
            self.env._shuffle_discard_into_deck()
        return self.env.draw_deck.popleft()

    def play_round(self, agent: DQNAgent, daehan: DaehanPlayer):
        """
        한 라운드를 교차 턴 방식으로 진행합니다.
        """
        # 각 플레이어의 라운드 상태 초기화
        p1_hand_nums = set()
        p1_hand_mods = []
        p1_done = False  # Stay, Bust, Flip7 여부
        p1_bust = False
        
        p2_hand_nums = set()
        p2_hand_mods = []
        p2_done = False
        p2_bust = False

        # 두 플레이어 모두 끝날 때까지 반복
        while not (p1_done and p2_done):
            
            # --- DQN Agent Turn ---
            if not p1_done:
                obs = self._construct_obs(p1_hand_nums, p1_hand_mods, self.agent_score)
                action = agent.select_action(obs, eval_mode=True)
                
                if action == 0: # Stay
                    p1_done = True
                else: # Hit
                    card = self._draw_card()
                    
                    # 카드 처리 로직
                    if card.isdigit(): # 숫자 카드
                        val = int(card)
                        if val in p1_hand_nums: # Bust
                            p1_bust = True
                            p1_done = True
                            self.env.discard_pile.append(card) # Bust된 카드는 바로 버림
                        else:
                            p1_hand_nums.add(val)
                            if len(p1_hand_nums) == 7: # Flip 7
                                p1_done = True
                    else: # 수정자 카드
                        p1_hand_mods.append(card)

            # --- Daehan Player Turn ---
            if not p2_done:
                obs = self._construct_obs(p2_hand_nums, p2_hand_mods, self.daehan_score)
                
                # Daehan Player를 위한 info 생성
                current_points = self._calculate_current_points(p2_hand_nums, p2_hand_mods)
                info = {
                    "current_round_score_if_stay": current_points,
                    "cards_in_deck": len(self.env.draw_deck),
                    "cards_in_discard": len(self.env.discard_pile)
                }
                
                action = daehan.select_action(obs, info)
                
                if action == 0: # Stay
                    p2_done = True
                else: # Hit
                    card = self._draw_card()
                    
                    if card.isdigit():
                        val = int(card)
                        if val in p2_hand_nums: # Bust
                            p2_bust = True
                            p2_done = True
                            self.env.discard_pile.append(card)
                        else:
                            p2_hand_nums.add(val)
                            if len(p2_hand_nums) == 7: # Flip 7
                                p2_done = True
                    else:
                        p2_hand_mods.append(card)

        # --- 라운드 종료 및 점수 정산 ---
        
        # Agent 점수
        if not p1_bust:
            score = self._calculate_current_points(p1_hand_nums, p1_hand_mods)
            self.agent_score += score
            
        # Daehan 점수
        if not p2_bust:
            score = self._calculate_current_points(p2_hand_nums, p2_hand_mods)
            self.daehan_score += score
            
        # 사용한 카드 버림 더미로 이동 (Bust된 카드는 이미 위에서 처리됨)
        # Agent Hand -> Discard
        if not p1_bust: # Bust가 아닐 때만 손패가 남아있음
             self.env.discard_pile.extend([str(n) for n in p1_hand_nums])
             self.env.discard_pile.extend(p1_hand_mods)
             
        # Daehan Hand -> Discard
        if not p2_bust:
             self.env.discard_pile.extend([str(n) for n in p2_hand_nums])
             self.env.discard_pile.extend(p2_hand_mods)

def run_alternating_simulation(num_games=1000, goal_score=200):
    print(f"\n교차 턴(Alternating Turns) 1:1 대결 시뮬레이션 시작")
    print(f"게임 수: {num_games} | 목표 점수: {goal_score}")
    print("=" * 70)
    
    manager = AlternatingGameManager()
    
    # 에이전트 로드
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
             print(f"게임 {i+1}/{num_games} 완료. (현재 스코어 - Agent: {manager.agent_score}, Daehan: {manager.daehan_score})")

    # 결과 출력
    print("\n" + "=" * 70)
    print("최종 결과")
    print("=" * 70)
    print(f"DQN Agent 승리: {results['Agent']} ({results['Agent']/num_games*100:.1f}%)")
    print(f"Daehan Player 승리: {results['Daehan']} ({results['Daehan']/num_games*100:.1f}%)")
    print(f"평균 소요 라운드: {np.mean(total_rounds):.2f}")
    
    # 그래프 저장
    labels = ['DQN Agent', 'Daehan Player']
    wins = [results['Agent'], results['Daehan']]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, wins, color=['#4ECDC4', '#FF6B6B'])
    plt.title(f'Alternating Turns Duel Results ({num_games} Games)')
    plt.ylabel('Wins')
    
    save_path = os.path.join(config.PLOTS_DIR, 'alternating_duel_results.png')
    plt.savefig(save_path)
    print(f"그래프 저장됨: {save_path}")

if __name__ == "__main__":
    run_alternating_simulation(num_games=1000)