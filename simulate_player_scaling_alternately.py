"""
플레이어 수에 따른 교차 턴(Alternating Turns) 승률 변화 분석

이 스크립트는 플레이어 수가 2명에서 6명으로 증가함에 따라,
정해진 순서(Daehan -> Aggressive -> Conservative -> Random)로 상대가 추가될 때
DQN 에이전트의 승률이 어떻게 변화하는지 분석합니다.
"""

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any, List, Set

# 모듈 임포트
from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX, MODIFIER_TO_IDX
from agent import DQNAgent
from simulate_duel import DaehanPlayer
import config

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 단순 전략 플레이어 클래스
# ============================================================================
class RandomPlayer:
    """무작위로 행동하는 플레이어"""
    def __init__(self, name="Random"):
        self.name = name
    def select_action(self, obs, info):
        return np.random.randint(0, 2)

class ConservativePlayer:
    """보수적인 플레이어: 점수가 15점 이상이면 무조건 Stay"""
    def __init__(self, name="Conservative"):
        self.name = name
    def select_action(self, obs, info):
        current_score = info.get("current_round_score_if_stay", 0)
        return 0 if current_score >= 15 else 1

class AggressivePlayer:
    """공격적인 플레이어: 점수가 30점 미만이면 무조건 Hit"""
    def __init__(self, name="Aggressive"):
        self.name = name
    def select_action(self, obs, info):
        current_score = info.get("current_round_score_if_stay", 0)
        return 1 if current_score < 30 else 0

# ============================================================================
# 게임 매니저 (교차 턴 로직)
# ============================================================================
class AlternatingGameManager:
    """
    다수 플레이어의 교차 턴 방식 게임을 관리하는 클래스
    """
    def __init__(self):
        self.env = FlipSevenCoreEnv()
        self.players = []
        self.scores = {}

    def reset_game(self, players: List[Dict[str, Any]]):
        """게임을 초기화하고 플레이어 목록을 설정합니다."""
        self.env.draw_deck = collections.deque()
        self.env.discard_pile = []
        self.env._initialize_deck_to_discard()
        self.env._shuffle_discard_into_deck()
        
        self.players = players
        self.scores = {p['name']: 0 for p in players}

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
        modifier_sum = sum([int(m[1:]) for m in hand_mods if m.startswith('+')])
        flip_7_bonus = 15 if len(hand_nums) == 7 else 0
        return number_sum + modifier_sum + flip_7_bonus

    def _draw_card(self) -> str:
        """덱에서 카드 뽑기 (필요 시 리셔플)"""
        if not self.env.draw_deck:
            self.env._shuffle_discard_into_deck()
        return self.env.draw_deck.popleft()

    def play_round(self):
        """
        모든 플레이어가 동시에 참여하는 한 라운드를 진행합니다.
        플레이어들이 순서대로 한 번씩 행동을 결정합니다.
        """
        # 각 플레이어의 라운드 상태 초기화
        player_states = []
        for p in self.players:
            player_states.append({
                'hand_nums': set(),
                'hand_mods': [],
                'done': False, # Stay, Bust, Flip7 등으로 라운드 종료 여부
                'bust': False
            })

        # 모든 플레이어가 라운드를 마칠 때까지 반복
        while not all(state['done'] for state in player_states):
            
            for i, p in enumerate(self.players):
                state = player_states[i]
                
                # 이미 라운드를 끝낸 플레이어는 패스
                if state['done']:
                    continue
                
                # --- 플레이어 행동 결정 ---
                obs = self._construct_obs(state['hand_nums'], state['hand_mods'], self.scores[p['name']])
                
                # Info 생성 (Daehan 및 Simple Player용)
                current_points = self._calculate_current_points(state['hand_nums'], state['hand_mods'])
                info = {
                    "current_round_score_if_stay": current_points,
                    "cards_in_deck": len(self.env.draw_deck),
                    "cards_in_discard": len(self.env.discard_pile),
                    "total_game_score": self.scores[p['name']]
                }
                
                # 플레이어 타입에 따른 행동 선택
                if p['type'] == 'agent':
                    action = p['obj'].select_action(obs, eval_mode=True)
                elif p['type'] == 'daehan':
                    action = p['obj'].select_action(obs, info)
                elif p['type'] == 'simple':
                    action = p['obj'].select_action(obs, info)
                else:
                    action = 0 # Default Stay
                
                # --- 행동 처리 ---
                if action == 0: # Stay
                    state['done'] = True
                else: # Hit
                    card = self._draw_card()
                    
                    if card.isdigit(): # 숫자 카드
                        val = int(card)
                        if val in state['hand_nums']: # Bust
                            state['bust'] = True
                            state['done'] = True
                            self.env.discard_pile.append(card) # Bust 카드 즉시 버림
                        else:
                            state['hand_nums'].add(val)
                            if len(state['hand_nums']) == 7: # Flip 7
                                state['done'] = True
                    else: # 수정자 카드
                        state['hand_mods'].append(card)
        
        # --- 라운드 종료 후 정산 ---
        for i, p in enumerate(self.players):
            state = player_states[i]
            
            # 점수 추가 (Bust 아닐 경우)
            if not state['bust']:
                score = self._calculate_current_points(state['hand_nums'], state['hand_mods'])
                self.scores[p['name']] += score
                
                # 사용한 카드 버림 (Bust된 카드는 이미 버려짐)
                self.env.discard_pile.extend([str(n) for n in state['hand_nums']])
                self.env.discard_pile.extend(state['hand_mods'])

# ============================================================================
# 스케일링 분석 실행
# ============================================================================
def run_scaling_analysis_alternately(num_games_per_setting=500, goal_score=200):
    print(f"\n플레이어 수에 따른 교차 턴(Alternating) 승률 분석 시작")
    print(f"설정당 게임 수: {num_games_per_setting} | 목표 점수: {goal_score}")
    print("=" * 70)
    
    manager = AlternatingGameManager()
    
    # 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print("모델 로드 실패, 랜덤 에이전트로 진행합니다.")
        
    # 상대 플레이어 풀 (순서대로 추가됨)
    # 순서: Daehan 1 -> Daehan 2 -> Aggressive -> Conservative -> Random
    opponent_pool = [
        {'type': 'daehan', 'obj': DaehanPlayer("Daehan 1"), 'name': 'Daehan 1'},
        {'type': 'daehan', 'obj': DaehanPlayer("Daehan 2"), 'name': 'Daehan 2'},
        {'type': 'simple', 'obj': AggressivePlayer("Aggressive"), 'name': 'Aggressive'},
        {'type': 'simple', 'obj': ConservativePlayer("Conservative"), 'name': 'Conservative'},
        {'type': 'simple', 'obj': RandomPlayer("Random"), 'name': 'Random'}
    ]
    
    results = {
        'num_players': [],
        'agent_win_rate': [],
        'baseline_win_rate': [] # 1/N 승률
    }
    
    # 플레이어 수 2명 ~ 6명 테스트
    for n_players in range(2, 7):
        # 현재 설정에 맞는 플레이어 구성
        current_opponents = opponent_pool[:n_players-1]
        players = [{'type': 'agent', 'obj': agent, 'name': 'DQN Agent'}] + current_opponents
        
        opponent_names = ", ".join([p['name'] for p in current_opponents])
        print(f"\n[테스트] 플레이어 수: {n_players}명 (VS: {opponent_names})")
        
        agent_wins = 0
        
        for i in range(num_games_per_setting):
            manager.reset_game(players)
            winner = None
            
            while winner is None:
                manager.play_round()
                
                # 승자 체크 (200점 이상 중 최고점)
                candidates = []
                for name, score in manager.scores.items():
                    if score >= goal_score:
                        candidates.append((name, score))
                
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    winner = candidates[0][0]
                    if winner == 'DQN Agent':
                        agent_wins += 1
            
            if (i + 1) % 100 == 0:
                print(".", end="", flush=True)
                
        win_rate = (agent_wins / num_games_per_setting) * 100
        baseline = (1 / n_players) * 100
        
        results['num_players'].append(n_players)
        results['agent_win_rate'].append(win_rate)
        results['baseline_win_rate'].append(baseline)
        
        print(f"\n  -> Agent 승률: {win_rate:.1f}% (기준: {baseline:.1f}%)")

    # ========================================================================
    # 결과 시각화
    # ========================================================================
    print("\n" + "=" * 70)
    print("분석 완료. 그래프 저장 중...")
    
    plt.figure(figsize=(10, 6))
    
    # Agent 승률 그래프
    plt.plot(results['num_players'], results['agent_win_rate'], 
             marker='o', linewidth=3, color='#4ECDC4', label='DQN Agent Win Rate')
    
    # 기준선(1/N) 그래프
    plt.plot(results['num_players'], results['baseline_win_rate'], 
             linestyle='--', color='gray', label='Baseline (1/N)')
    
    # 수치 표시
    for x, y in zip(results['num_players'], results['agent_win_rate']):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), 
                     ha='center', fontweight='bold', fontsize=11)
        
    plt.title('DQN Agent Win Rate vs Number of Players (Alternating Turns), {} per Game'.format(num_games_per_setting), fontsize=16, fontweight='bold')
    plt.xlabel('Number of Players', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(results['num_players'])
    
    # x축 라벨에 추가된 상대방 정보 표시 (옵션)
    # labels = [f"{n}\n(+{opponent_pool[n-2]['name']})" if n > 1 else str(n) for n in results['num_players']]
    # plt.xticks(results['num_players'], labels)

    save_path = os.path.join(config.PLOTS_DIR, 'player_scaling_alternating_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장 완료: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_scaling_analysis_alternately(num_games_per_setting=500)