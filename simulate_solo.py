"""
솔로 플레이 성능 분석 (Rounds to 200 Points)

이 스크립트는 DQN 에이전트와 Daehan Player가 각각 혼자 게임을 진행했을 때,
200점에 도달하기까지 평균 몇 라운드가 걸리는지 측정합니다.
이는 순수한 득점 효율성을 비교하는 지표입니다.
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

def run_solo_simulation(player_obj, player_type, num_games=1000):
    """
    단일 플레이어에 대한 솔로 시뮬레이션을 수행합니다.
    """
    env = FlipSevenCoreEnv()
    rounds_history = []
    
    for _ in range(num_games):
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        obs, info = env.reset()
        rounds = 0
        
        while env.total_score < 200:
            rounds += 1
            terminated = False
            while not terminated:
                if player_type == 'agent':
                    action = player_obj.select_action(obs, eval_mode=True)
                else:
                    action = player_obj.select_action(obs, info)
                    
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
            
            if env.total_score < 200:
                obs, info = env.reset()
                
        rounds_history.append(rounds)
        
    return rounds_history

def compare_solo_performance(num_games=1000):
    print(f"\n솔로 성능 비교 분석 시작 ({num_games} 게임)")
    print("=" * 70)
    
    # 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print(f"경고: 모델 파일을 찾을 수 없습니다: {config.FINAL_MODEL_PATH}")
        print("랜덤 에이전트로 진행합니다.")
        
    daehan = DaehanPlayer("Daehan Player")
    
    # 시뮬레이션 실행
    print("DQN Agent 시뮬레이션 중...")
    agent_rounds = run_solo_simulation(agent, 'agent', num_games)
    
    print("Daehan Player 시뮬레이션 중...")
    daehan_rounds = run_solo_simulation(daehan, 'daehan', num_games)
    
    # 결과 분석
    avg_agent = np.mean(agent_rounds)
    avg_daehan = np.mean(daehan_rounds)
    
    print("\n" + "=" * 70)
    print("솔로 성능 결과 (200점 도달 평균 라운드)")
    print("=" * 70)
    print(f"DQN Agent: {avg_agent:.2f} 라운드")
    print(f"Daehan Player: {avg_daehan:.2f} 라운드")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    data = [agent_rounds, daehan_rounds]
    labels = ['DQN Agent', 'Daehan Player']
    colors = ['#4ECDC4', '#FF6B6B']
    
    # 박스 플롯
    bplot = plt.boxplot(data, patch_artist=True, labels=labels)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.title(f'Solo Performance: Rounds to Reach 200 Points ({num_games} Games)', fontsize=14)
    plt.ylabel('Rounds', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 평균값 표시
    for i, avg in enumerate([avg_agent, avg_daehan]):
        plt.text(i+1, avg, f'Avg: {avg:.2f}', ha='center', va='bottom', fontweight='bold', color='black')
        
    save_path = os.path.join(config.PLOTS_DIR, 'solo_performance_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n결과 그래프 저장 완료: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    compare_solo_performance(num_games=1000)
