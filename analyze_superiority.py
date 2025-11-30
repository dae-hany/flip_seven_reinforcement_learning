"""
DQN 에이전트 vs Daehan Player 우위 분석 스크립트

이 스크립트는 DQN 에이전트와 Daehan Player(합리적 카드 카운터) 간의
상세한 성능 비교를 수행합니다.

주요 분석 지표:
1. Bust Rate (파산 확률)
2. Big Win Rate (대승 확률, 점수 >= 40)
3. Average Score (안전한 라운드 평균 점수)
4. High Risk Hit Rate (위험 상황에서의 Hit 비율)
"""

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, Any, List

from flip_seven_env import FlipSevenCoreEnv, CARD_TO_IDX
from agent import DQNAgent
from simulate_duel import DaehanPlayer
import config

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_bust_prob(obs: Dict[str, np.ndarray]) -> float:
    """
    현재 관측된 덱 상태를 기반으로 Bust 확률을 계산합니다.
    
    Args:
        obs: 현재 관측값
        
    Returns:
        Bust 확률 (0.0 ~ 1.0)
    """
    deck_composition = obs["deck_composition"]
    current_hand_numbers = obs["current_hand_numbers"]
    
    total_cards = np.sum(deck_composition)
    if total_cards == 0: return 0.0
    
    # 손에 이미 있는 숫자가 덱에 얼마나 남아있는지 계산
    bust_cards = 0
    for i in range(13):
        if current_hand_numbers[i] == 1:
            bust_cards += deck_composition[i]
            
    return bust_cards / total_cards

def run_analysis(num_games: int = 10000):
    """
    분석 시뮬레이션을 실행합니다.
    """
    print(f"\n우위 분석 시작 ({num_games} 게임)")
    print("=" * 70)
    
    env = FlipSevenCoreEnv()
    
    # 플레이어 설정
    agent = DQNAgent(device=DEVICE)
    
    # 모델 로드 (config에 정의된 경로 사용)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print(f"경고: 모델 파일을 찾을 수 없습니다: {config.FINAL_MODEL_PATH}")
        print("초기화된(학습되지 않은) 에이전트로 진행합니다.")
    
    daehan = DaehanPlayer("Daehan Player")
    
    players = [
        {'type': 'agent', 'obj': agent, 'name': 'DQN Agent'},
        {'type': 'daehan', 'obj': daehan, 'name': 'Daehan Player'}
    ]
    
    # 통계 저장소 초기화
    stats = {p['name']: {
        'rounds_played': 0,
        'flip_7_count': 0,
        'bust_count': 0,
        'total_score_sum': 0, # Bust가 아닌 라운드의 점수 합계
        'non_bust_rounds': 0,
        'high_risk_situations': 0, # Bust 확률 > 30% 인 상황
        'high_risk_hits': 0
    } for p in players}
    
    for game_idx in range(num_games):
        # 각 플레이어별 점수 추적
        scores = {p['name']: 0 for p in players}
        
        # 공정한 비교를 위해 매 게임마다 덱 초기화
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        winner = None
        
        while winner is None:
            for p in players:
                if winner: break
                
                # 환경 상태 설정 (점수 동기화)
                env.total_score = scores[p['name']]
                obs, info = env.reset()
                
                terminated = False
                
                while not terminated:
                    # 행동 전 위험도 계산
                    bust_prob = calculate_bust_prob(obs)
                    
                    # 행동 선택
                    if p['type'] == 'agent':
                        action = p['obj'].select_action(obs, eval_mode=True)
                    elif p['type'] == 'daehan':
                        action = p['obj'].select_action(obs, info)
                    
                    # 위험 감수 성향 추적
                    if bust_prob > 0.30: # 고위험 기준 (30%)
                        stats[p['name']]['high_risk_situations'] += 1
                        if action == 1: # Hit
                            stats[p['name']]['high_risk_hits'] += 1
                    
                    next_obs, reward, terminated, _, info = env.step(action)
                    obs = next_obs
                
                # 라운드 종료 후 처리
                stats[p['name']]['rounds_played'] += 1
                
                # 이번 라운드 획득 점수 계산
                current_total_score = info["total_game_score"]
                score_gain = current_total_score - scores[p['name']]
                
                if score_gain == 0:
                    # 점수가 오르지 않았으면 Bust로 간주
                    stats[p['name']]['bust_count'] += 1
                else:
                    stats[p['name']]['non_bust_rounds'] += 1
                    stats[p['name']]['total_score_sum'] += score_gain
                    
                    # 대승 (Big Win) 체크 (40점 이상)
                    if score_gain >= 40:
                         stats[p['name']]['flip_7_count'] += 1
                
                scores[p['name']] = info["total_game_score"]
                
                # 승리 조건 체크 (200점)
                if scores[p['name']] >= 200:
                    winner = p['name']
                    
        if (game_idx + 1) % 1000 == 0:
            print(f"게임 {game_idx + 1}/{num_games} 완료.")

    # ========================================================================
    # 결과 보고
    # ========================================================================
    print("\n" + "=" * 70)
    print("우위 분석 결과")
    print("=" * 70)
    
    metrics = ['Bust Rate', 'Big Win Rate (Score >= 40)', 'Avg Score (Safe Rounds)', 'High Risk Hit Rate']
    results_data = {m: [] for m in metrics}
    
    for p_name in [p['name'] for p in players]:
        s = stats[p_name]
        rounds = s['rounds_played']
        
        bust_rate = (s['bust_count'] / rounds) * 100 if rounds > 0 else 0
        big_win_rate = (s['flip_7_count'] / rounds) * 100 if rounds > 0 else 0
        avg_score = s['total_score_sum'] / s['non_bust_rounds'] if s['non_bust_rounds'] > 0 else 0
        risk_hit_rate = (s['high_risk_hits'] / s['high_risk_situations']) * 100 if s['high_risk_situations'] > 0 else 0
        
        results_data['Bust Rate'].append(bust_rate)
        results_data['Big Win Rate (Score >= 40)'].append(big_win_rate)
        results_data['Avg Score (Safe Rounds)'].append(avg_score)
        results_data['High Risk Hit Rate'].append(risk_hit_rate)
        
        print(f"[{p_name}]")
        print(f"  - Bust Rate: {bust_rate:.2f}%")
        print(f"  - Big Win Rate: {big_win_rate:.2f}%")
        print(f"  - Avg Score (Safe): {avg_score:.2f}")
        print(f"  - High Risk Hit Rate: {risk_hit_rate:.2f}%")
        print("-" * 30)

    # 그래프 그리기
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    agent_vals = [results_data[m][0] for m in metrics]
    daehan_vals = [results_data[m][1] for m in metrics]
    
    rects1 = ax.bar(x - width/2, agent_vals, width, label='DQN Agent', color='#4ECDC4')
    rects2 = ax.bar(x + width/2, daehan_vals, width, label='Daehan Player', color='#FF6B6B')
    
    ax.set_ylabel('Percentage / Score')
    ax.set_title('Agent vs Daehan Superiority Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    # 그래프 저장 (config.PLOTS_DIR 사용)
    plot_path = os.path.join(config.PLOTS_DIR, 'superiority_analysis.png')
    plt.savefig(plot_path)
    print(f"\n분석 그래프 저장 완료: {plot_path}")

if __name__ == "__main__":
    run_analysis(num_games=10000)
