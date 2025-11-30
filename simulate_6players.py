"""
6인 플레이어 시뮬레이션 (DQN Agent vs 5 Others)

이 스크립트는 DQN 에이전트와 5명의 다른 플레이어(Daehan Player 및 다양한 전략) 간의
6인 게임을 시뮬레이션합니다.
"""

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, Any, List

from flip_seven_env import FlipSevenCoreEnv
from agent import DQNAgent
from simulate_duel import DaehanPlayer
import config

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 단순 전략 플레이어 클래스들
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
# 6인 시뮬레이션
# ============================================================================
def simulate_6players(num_games=1000, goal_score=200):
    print(f"\n6인 게임 시뮬레이션 시작 ({num_games} 게임)")
    print("=" * 70)
    
    env = FlipSevenCoreEnv()
    
    # 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print(f"경고: 모델 파일을 찾을 수 없습니다: {config.FINAL_MODEL_PATH}")
        print("랜덤 에이전트로 진행합니다.")
        
    # 플레이어 구성
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
        scores = {p['name']: 0 for p in players}
        
        # 덱 초기화
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        winner = None
        
        while winner is None:
            # 순차적 턴 진행
            for p in players:
                if winner: break
                
                env.total_score = scores[p['name']]
                obs, info = env.reset()
                
                terminated = False
                while not terminated:
                    if p['type'] == 'agent':
                        action = p['obj'].select_action(obs, eval_mode=True)
                    elif p['type'] == 'daehan':
                        action = p['obj'].select_action(obs, info)
                    else: # simple players
                        action = p['obj'].select_action(obs, info)
                        
                    next_obs, reward, terminated, _, info = env.step(action)
                    obs = next_obs
                
                scores[p['name']] = info["total_game_score"]
                
                if scores[p['name']] >= goal_score:
                    winner = p['name']
                    wins[p['name']] += 1
        
        if (game_idx + 1) % 100 == 0:
            print(f"게임 {game_idx + 1}/{num_games} 완료.")
            
    # ========================================================================
    # 결과 분석
    # ========================================================================
    print("\n" + "=" * 70)
    print("6인 게임 결과 요약")
    print("=" * 70)
    
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    
    labels = []
    win_counts = []
    
    for name, count in sorted_wins:
        rate = (count / num_games) * 100
        print(f"{name}: {count}승 ({rate:.1f}%)")
        labels.append(name)
        win_counts.append(count)
        
    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, win_counts, color=['#4ECDC4', '#FF6B6B', '#FF6B6B', '#FFE66D', '#FF8C42', '#A8E6CF'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height} ({height/num_games*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title(f'6-Player Simulation Results ({num_games} Games)', fontsize=16, fontweight='bold')
    plt.ylabel('Wins', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(config.PLOTS_DIR, '6player_simulation_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n결과 그래프 저장 완료: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    simulate_6players(num_games=1000)
