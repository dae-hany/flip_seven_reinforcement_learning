"""
DQN 에이전트 vs Daehan Player 1:1 대결 시뮬레이션

이 스크립트는 학습된 DQN 에이전트와 합리적 카드 카운팅 전략을 사용하는
'Daehan Player' 간의 1:1 대결을 시뮬레이션합니다.
"""

import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple

from flip_seven_env import FlipSevenCoreEnv
from agent import DQNAgent
import config

# OpenMP 라이브러리 충돌 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DAEHAN PLAYER (합리적 카드 카운터)
# ============================================================================
class DaehanPlayer:
    """
    카드 카운팅과 기댓값(EV) 계산을 사용하는 합리적인 플레이어입니다.
    """
    def __init__(self, name="Daehan Player"):
        self.name = name
    
    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        기댓값(Expected Value)에 기반하여 행동을 선택합니다.
        
        Action 0: Stay
        Action 1: Hit
        """
        # 1. 현재 상태 정보 가져오기
        current_round_score = info.get("current_round_score_if_stay", 0)
        
        # 점수가 0이면 (라운드 시작) 항상 Hit
        if current_round_score == 0:
            return 1
            
        # 2. 덱 구성 분석 (카드 카운팅)
        deck_composition = obs["deck_composition"] # shape (19,)
        total_cards_in_deck = np.sum(deck_composition)
        
        if total_cards_in_deck == 0:
            return 0 
            
        # 3. Bust 카드 vs Safe 카드 식별
        # Bust 카드: 이미 손에 있는 숫자 카드
        # Safe 카드: 손에 없는 숫자 카드 + 수정자 카드
        
        current_hand_numbers = obs["current_hand_numbers"] # shape (13,) binary
        
        bust_card_count = 0
        safe_card_count = 0
        expected_gain_sum = 0.0
        
        # 모든 카드 타입 순회
        for card_idx, count in enumerate(deck_composition):
            if count == 0:
                continue
                
            # 카드 타입 및 가치 결정
            if card_idx < 13: # 숫자 카드 (0-12)
                card_val = card_idx
                if current_hand_numbers[card_val] == 1:
                    # 이것은 Bust 카드입니다
                    bust_card_count += count
                else:
                    # 이것은 Safe 숫자 카드입니다
                    safe_card_count += count
                    expected_gain_sum += (card_val * count)
            else: # 수정자 카드
                # 수정자는 항상 Safe 합니다 (x2 주의 필요)
                safe_card_count += count
                
                # 수정자 가치 추정
                # +2, +4, +6, +8, +10, x2
                # 인덱스: 13, 14, 15, 16, 17, 18
                mod_idx = card_idx - 13
                if mod_idx == 0: val = 2
                elif mod_idx == 1: val = 4
                elif mod_idx == 2: val = 6
                elif mod_idx == 3: val = 8
                elif mod_idx == 4: val = 10
                elif mod_idx == 5: val = current_round_score # x2는 현재 점수를 두 배로
                else: val = 5 # 예외 처리
                
                expected_gain_sum += (val * count)
        
        # 4. 확률 계산
        p_bust = bust_card_count / total_cards_in_deck
        p_safe = safe_card_count / total_cards_in_deck
        
        # 5. 기댓값(EV) 계산
        # E[Gain] = Safe 카드의 평균 가치
        avg_gain = expected_gain_sum / safe_card_count if safe_card_count > 0 else 0
        
        # EV(Stay) = 현재 점수
        ev_stay = current_round_score
        
        # EV(Hit) = P(Safe) * (현재 점수 + 평균 이득) + P(Bust) * 0
        # (Bust 시 라운드 점수는 0이라고 가정)
        ev_hit = p_safe * (current_round_score + avg_gain)
        
        # 6. 결정
        if ev_hit > ev_stay:
            return 1 # Hit
        else:
            return 0 # Stay

# ============================================================================
# 대결 시뮬레이션
# ============================================================================
def simulate_duel(num_games=100, goal_score=200):
    print(f"\n1:1 대결 시뮬레이션 시작: DQN Agent vs {DaehanPlayer().name}")
    print(f"게임 수: {num_games} | 목표 점수: {goal_score}")
    print("=" * 70)
    
    # 환경 초기화
    env = FlipSevenCoreEnv() # 공유 환경
    
    # 에이전트 로드
    agent = DQNAgent(device=DEVICE)
    if os.path.exists(config.FINAL_MODEL_PATH):
        agent.load(config.FINAL_MODEL_PATH)
    else:
        print(f"경고: 모델 파일을 찾을 수 없습니다: {config.FINAL_MODEL_PATH}")
        print("랜덤 에이전트로 진행합니다.")
    
    daehan = DaehanPlayer()
    
    # 통계
    results = {
        "Agent_Wins": 0,
        "Daehan_Wins": 0,
        "Agent_Total_Rounds": [],
        "Daehan_Total_Rounds": [],
        "Agent_Final_Scores": [],
        "Daehan_Final_Scores": []
    }
    
    for game_idx in range(num_games):
        # 게임 점수 초기화
        agent_score = 0
        daehan_score = 0
        
        # 게임을 위한 덱 초기화
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        env._shuffle_discard_into_deck()
        
        game_rounds = 0
        winner = None
        
        # 게임 루프
        while agent_score < goal_score and daehan_score < goal_score:
            game_rounds += 1
            
            # --- 턴 1: DQN Agent ---
            # 에이전트를 위해 환경 점수 동기화
            env.total_score = agent_score
            obs, info = env.reset() # 손패 리셋, 덱 유지
            
            terminated = False
            while not terminated:
                action = agent.select_action(obs, eval_mode=True)
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
            
            # 에이전트 점수 업데이트
            agent_score = info["total_game_score"]
            
            if agent_score >= goal_score:
                winner = "Agent"
                break
                
            # --- 턴 2: Daehan Player ---
            # Daehan을 위해 환경 점수 동기화
            env.total_score = daehan_score
            obs, info = env.reset() # 손패 리셋, 덱 유지
            
            terminated = False
            while not terminated:
                action = daehan.select_action(obs, info)
                next_obs, reward, terminated, _, info = env.step(action)
                obs = next_obs
                
            # Daehan 점수 업데이트
            daehan_score = info["total_game_score"]
            
            if daehan_score >= goal_score:
                winner = "Daehan"
                break
        
        # 결과 기록
        if winner == "Agent":
            results["Agent_Wins"] += 1
            results["Agent_Total_Rounds"].append(game_rounds)
        else:
            results["Daehan_Wins"] += 1
            results["Daehan_Total_Rounds"].append(game_rounds)
            
        results["Agent_Final_Scores"].append(agent_score)
        results["Daehan_Final_Scores"].append(daehan_score)
        
        if (game_idx + 1) % 100 == 0:
            print(f"게임 {game_idx + 1}/{num_games} 완료 | 승자: {winner} | 점수: {agent_score} vs {daehan_score}")

    # ========================================================================
    # 분석 및 시각화
    # ========================================================================
    print("\n" + "=" * 70)
    print("대결 결과 요약")
    print("=" * 70)
    
    agent_win_rate = (results["Agent_Wins"] / num_games) * 100
    daehan_win_rate = (results["Daehan_Wins"] / num_games) * 100
    
    print(f"총 게임 수: {num_games}")
    print(f"DQN Agent 승리: {results['Agent_Wins']} ({agent_win_rate:.1f}%)")
    print(f"Daehan Player 승리: {results['Daehan_Wins']} ({daehan_win_rate:.1f}%)")
    
    avg_rounds_agent = np.mean(results["Agent_Total_Rounds"]) if results["Agent_Total_Rounds"] else 0
    avg_rounds_daehan = np.mean(results["Daehan_Total_Rounds"]) if results["Daehan_Total_Rounds"] else 0
    
    print(f"평균 승리 소요 라운드 (Agent): {avg_rounds_agent:.2f}")
    print(f"평균 승리 소요 라운드 (Daehan): {avg_rounds_daehan:.2f}")
    
    # 그래프 그리기
    labels = ['DQN Agent', 'Daehan Player']
    wins = [results['Agent_Wins'], results['Daehan_Wins']]
    colors = ['#4ECDC4', '#FF6B6B']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, wins, color=colors)
    
    # 카운트 라벨 추가
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height} ({height/num_games*100:.1f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title(f'Flip Seven Duel Results ({num_games} Games)', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Wins', fontsize=12)
    plt.ylim(0, num_games * 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(config.PLOTS_DIR, 'duel_simulation_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n결과 그래프 저장 완료: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    simulate_duel(num_games=10000)
