"""
6인 플레이어 교차 턴(Alternating Turns) 대결 시뮬레이션

이 스크립트는 DQN 에이전트와 5명의 다른 플레이어가 한 장씩 번갈아 가며
카드를 뽑는 방식(Turn-based)으로 6인 게임을 시뮬레이션합니다.
"""

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any, List, Set

from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX, MODIFIER_TO_IDX
from agent import DQNAgent
from simulate_duel import DaehanPlayer
import config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 단순 전략 플레이어 클래스
# ============================================================================
class RandomPlayer:
    def __init__(self, name="Random"): self.name = name
    def select_action(self, obs, info): return np.random.randint(0, 2)

class ConservativePlayer:
    def __init__(self, name="Conservative"): self.name = name
    def select_action(self, obs, info):
        return 0 if info.get("current_round_score_if_stay", 0) >= 15 else 1

class AggressivePlayer:
    def __init__(self, name="Aggressive"): self.name = name
    def select_action(self, obs, info):
        return 1 if info.get("current_round_score_if_stay", 0) < 30 else 0

# ============================================================================
# 게임 매니저
# ============================================================================
class AlternatingGameManager:
    def __init__(self):
        self.env = FlipSevenCoreEnv()
        self.players = []
        self.scores = {}

    def reset_game(self, players: List[Dict[str, Any]]):
        self.env.draw_deck = collections.deque()
        self.env.discard_pile = []
        self.env._initialize_deck_to_discard()
        self.env._shuffle_discard_into_deck()
        self.players = players
        self.scores = {p['name']: 0 for p in players}

    def _construct_obs(self, hand_nums, hand_mods, total_score):
        hand_num_obs = np.zeros(13, dtype=np.int32)
        for num in hand_nums: hand_num_obs[num] = 1
        hand_mod_obs = np.zeros(6, dtype=np.int32)
        for mod in hand_mods: hand_mod_obs[MODIFIER_TO_IDX[mod]] = 1
        deck_comp_obs = np.zeros(19, dtype=np.int32)
        for card in self.env.draw_deck: deck_comp_obs[CARD_TO_IDX[card]] += 1
        return {
            "current_hand_numbers": hand_num_obs,
            "current_hand_modifiers": hand_mod_obs,
            "deck_composition": deck_comp_obs,
            "total_game_score": np.array([total_score], dtype=np.int32)
        }

    def _calculate_current_points(self, hand_nums, hand_mods):
        number_sum = sum(hand_nums)
        if 'x2' in hand_mods: number_sum *= 2
        modifier_sum = sum([int(m[1:]) for m in hand_mods if m.startswith('+')])
        flip_7_bonus = 15 if len(hand_nums) == 7 else 0
        return number_sum + modifier_sum + flip_7_bonus

    def _draw_card(self):
        if not self.env.draw_deck: self.env._shuffle_discard_into_deck()
        return self.env.draw_deck.popleft()

    def play_round(self):
        player_states = [{'hand_nums': set(), 'hand_mods': [], 'done': False, 'bust': False} for _ in self.players]
        
        while not all(state['done'] for state in player_states):
            for i, p in enumerate(self.players):
                state = player_states[i]
                if state['done']: continue
                
                obs = self._construct_obs(state['hand_nums'], state['hand_mods'], self.scores[p['name']])
                current_points = self._calculate_current_points(state['hand_nums'], state['hand_mods'])
                info = {
                    "current_round_score_if_stay": current_points,
                    "cards_in_deck": len(self.env.draw_deck),
                    "cards_in_discard": len(self.env.discard_pile)
                }
                
                if p['type'] == 'agent':
                    action = p['obj'].select_action(obs, eval_mode=True)
                else:
                    action = p['obj'].select_action(obs, info)
                
                if action == 0:
                    state['done'] = True
                else:
                    card = self._draw_card()
                    if card.isdigit():
                        val = int(card)
                        if val in state['hand_nums']:
                            state['bust'] = True; state['done'] = True
                            self.env.discard_pile.append(card)
                        else:
                            state['hand_nums'].add(val)
                            if len(state['hand_nums']) == 7: state['done'] = True
                    else:
                        state['hand_mods'].append(card)
        
        for i, p in enumerate(self.players):
            state = player_states[i]
            if not state['bust']:
                score = self._calculate_current_points(state['hand_nums'], state['hand_mods'])
                self.scores[p['name']] += score
                self.env.discard_pile.extend([str(n) for n in state['hand_nums']] + state['hand_mods'])

# ============================================================================
# 메인 실행
# ============================================================================
def simulate_6players_alternately(num_games=1000, goal_score=200):
    print(f"\n6인 교차 턴(Alternating) 게임 시뮬레이션 시작 ({num_games} 게임)")
    print("=" * 70)
    
    manager = AlternatingGameManager()
    agent = DQNAgent(device=DEVICE)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print("모델 로드 실패, 랜덤 에이전트 사용")
        
    players = [
        {'type': 'agent', 'obj': agent, 'name': 'DQN Agent'},
        {'type': 'daehan', 'obj': DaehanPlayer("Daehan 1"), 'name': 'Daehan 1'},
        {'type': 'daehan', 'obj': DaehanPlayer("Daehan 2"), 'name': 'Daehan 2'},
        {'type': 'simple', 'obj': ConservativePlayer("Conservative"), 'name': 'Conservative'},
        {'type': 'simple', 'obj': AggressivePlayer("Aggressive"), 'name': 'Aggressive'},
        {'type': 'simple', 'obj': RandomPlayer("Random"), 'name': 'Random'}
    ]
    
    wins = {p['name']: 0 for p in players}
    
    for game_idx in range(num_games):
        manager.reset_game(players)
        winner = None
        while winner is None:
            manager.play_round()
            candidates = [(name, score) for name, score in manager.scores.items() if score >= goal_score]
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                winner = candidates[0][0]
                wins[winner] += 1
        
        if (game_idx + 1) % 100 == 0:
            print(f"게임 {game_idx + 1}/{num_games} 완료.")
            
    print("\n" + "=" * 70)
    print("6인 교차 턴 게임 결과 요약")
    print("=" * 70)
    
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_wins]
    win_counts = [item[1] for item in sorted_wins]
    
    for name, count in sorted_wins:
        print(f"{name}: {count}승 ({(count/num_games)*100:.1f}%)")

    # === 시각화 (수치 표시 추가) ===
    plt.figure(figsize=(12, 6))
    # 색상 매핑 (이름 기반으로 고정 색상 할당)
    color_map = {
        'DQN Agent': '#4ECDC4', 'Daehan 1': '#FF6B6B', 'Daehan 2': '#FF6B6B',
        'Conservative': '#FFE66D', 'Aggressive': '#FF8C42', 'Random': '#A8E6CF'
    }
    bar_colors = [color_map.get(label, 'gray') for label in labels]
    
    bars = plt.bar(labels, win_counts, color=bar_colors)
    
    # 막대 위에 수치 및 퍼센트 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}\n({height/num_games*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    plt.title(f'6-Player Alternating Turns Simulation ({num_games} Games)', fontsize=16, fontweight='bold')
    plt.ylabel('Wins', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, max(win_counts) * 1.15) # 텍스트 공간 확보
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(config.PLOTS_DIR, '6player_alternating_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n결과 그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    simulate_6players_alternately(num_games=10000)