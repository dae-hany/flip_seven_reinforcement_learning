"""
플레이어 수에 따른 승률 변화 분석 (Scaling Analysis)

이 스크립트는 플레이어 수가 2명에서 6명으로 증가함에 따라
DQN 에이전트의 승률이 어떻게 변화하는지 분석합니다.
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

def simulate_game(env, players, goal_score=200):
    """단일 게임을 시뮬레이션하고 승자를 반환합니다."""
    scores = {p['name']: 0 for p in players}
    
    # 덱 초기화
    env.draw_deck = collections.deque()
    env.discard_pile = []
    env._initialize_deck_to_discard()
    env._shuffle_discard_into_deck()
    
    winner = None
    
    while winner is None:
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
                    
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
            
            scores[p['name']] = info["total_game_score"]
            
            if scores[p['name']] >= goal_score:
                winner = p['name']
                
    return winner

def run_scaling_analysis(num_games_per_setting=500):
    print(f"\n플레이어 수에 따른 승률 분석 시작 (설정당 {num_games_per_setting} 게임)")
    print("=" * 70)
    
    env = FlipSevenCoreEnv()
    
    # 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print(f"경고: 모델 파일을 찾을 수 없습니다: {config.FINAL_MODEL_PATH}")
        print("랜덤 에이전트로 진행합니다.")
        
    results = {
        'num_players': [],
        'agent_win_rate': [],
        'baseline_win_rate': [] # 1/N 승률
    }
    
    # 플레이어 수 2명 ~ 6명 테스트
    for n_players in range(2, 7):
        print(f"\n[테스트] 플레이어 수: {n_players}명")
        
        # 플레이어 구성: Agent 1명 + (N-1)명의 Daehan Players
        players = [{'type': 'agent', 'obj': agent, 'name': 'DQN Agent'}]
        for i in range(n_players - 1):
            players.append({
                'type': 'daehan', 
                'obj': DaehanPlayer(f"Daehan {i+1}"), 
                'name': f"Daehan {i+1}"
            })
            
        agent_wins = 0
        
        for i in range(num_games_per_setting):
            winner = simulate_game(env, players)
            if winner == 'DQN Agent':
                agent_wins += 1
            
            if (i + 1) % 100 == 0:
                print(f".", end="", flush=True)
                
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
    
    plt.plot(results['num_players'], results['agent_win_rate'], 
             marker='o', linewidth=3, color='#4ECDC4', label='DQN Agent Win Rate')
    plt.plot(results['num_players'], results['baseline_win_rate'], 
             linestyle='--', color='gray', label='Baseline (1/N)')
    
    for x, y in zip(results['num_players'], results['agent_win_rate']):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.title('DQN Agent Win Rate vs Number of Players', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Players', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(results['num_players'])
    
    save_path = os.path.join(config.PLOTS_DIR, 'player_scaling_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장 완료: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_scaling_analysis(num_games_per_setting=500)
