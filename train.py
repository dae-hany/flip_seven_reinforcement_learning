"""
Flip Seven DQN 훈련 스크립트

이 스크립트는 FlipSevenCoreEnv 환경에서 DQN 에이전트를 훈련합니다.
"""

import torch
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flip_seven_env import FlipSevenCoreEnv
from agent import DQNAgent
from config import (
    NUM_TOTAL_GAMES_TO_TRAIN,
    TARGET_UPDATE_FREQUENCY,
    REPLAY_BUFFER_SIZE,
    BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    USE_END_BONUS,
    OUTPUT_DIR,
    LOG_INTERVAL,
    SAVE_INTERVAL,
    EVAL_GAMES
)

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================================
# 메인 훈련 루프
# ============================================================================
def train():
    """
    게임/라운드 구조를 올바르게 처리하는 메인 학습 루프입니다.
    """
    # 환경 및 에이전트 초기화
    env = FlipSevenCoreEnv(use_end_bonus=USE_END_BONUS)
    agent = DQNAgent(device=DEVICE)
    
    # 학습 통계
    all_game_rounds = []
    all_game_avg_loss = []
    total_scores_per_game = []
    
    print("=" * 70)
    print("Starting DQN Training on FlipSevenCoreEnv")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total games to train: {NUM_TOTAL_GAMES_TO_TRAIN}")
    print(f"Use end bonus: {USE_END_BONUS}")
    print(f"Replay buffer size: {REPLAY_BUFFER_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gamma: {GAMMA}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epsilon: {EPSILON_START} -> {EPSILON_END} (decay: {EPSILON_DECAY})")
    print(f"Target update frequency: every {TARGET_UPDATE_FREQUENCY} games")
    print("=" * 70)
    
    # 메인 학습 루프: 게임 단위로 반복
    for game in range(NUM_TOTAL_GAMES_TO_TRAIN):
        
        # ====================================================================
        # 1. 게임 전체를 수동으로 리셋
        # ====================================================================
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()  # 모든 85장 카드를 다시 초기화
        
        # 첫 번째 라운드 준비
        obs, info = env.reset()
        
        game_total_rounds = 0
        game_total_reward = 0.0
        game_total_loss = 0.0
        steps_in_game = 0
        
        # ====================================================================
        # 2. 게임 루프 (total_score >= 200이 될 때까지 계속)
        # ====================================================================
        while info.get("total_game_score", 0) < 200:
            game_total_rounds += 1
            terminated = False
            round_reward = 0.0
            
            # ================================================================
            # 3. 라운드 (에피소드) 루프
            # ================================================================
            while not terminated:
                
                # epsilon-greedy 정책을 사용하여 행동 선택
                action = agent.select_action(obs)
                
                # 환경에서 한 단계 진행
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # 리플레이 버퍼에 전이 저장
                agent.store_transition(obs, action, reward, next_obs, terminated)
                
                # 한 번의 학습 단계 수행
                loss = agent.learn()
                if loss is not None:
                    game_total_loss += loss
                    steps_in_game += 1
                
                # 관찰 업데이트
                obs = next_obs
                round_reward += reward
            
            game_total_reward += round_reward
            
            # ================================================================
            # 4. 라운드 종료 (terminated=True)
            # ================================================================
            # 다음 라운드를 준비하기 위해 reset() 호출
            # 이는 손패를 초기화하지만 total_score는 초기화하지 않음
            if info.get("total_game_score", 0) < 200:
                obs, info = env.reset()
        
        # ====================================================================
        # 5. 게임 종료
        # ====================================================================
        final_score = info.get("total_game_score", 0)
        all_game_rounds.append(game_total_rounds)
        avg_loss = game_total_loss / steps_in_game if steps_in_game > 0 else 0.0
        all_game_avg_loss.append(avg_loss)
        total_scores_per_game.append(final_score)
        
        # 엡실론 감소
        agent.decay_epsilon(EPSILON_DECAY)
        
        # 주기적으로 타겟 네트워크 업데이트
        if (game + 1) % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
        
        # 로깅
        if (game + 1) % LOG_INTERVAL == 0:
            avg_rounds = np.mean(all_game_rounds[-LOG_INTERVAL:])
            avg_score = np.mean(total_scores_per_game[-LOG_INTERVAL:])
            avg_loss_recent = np.mean(all_game_avg_loss[-LOG_INTERVAL:])
            print(f"Game {game + 1}/{NUM_TOTAL_GAMES_TO_TRAIN} | "
                  f"Rounds: {game_total_rounds} | "
                  f"Score: {final_score} | "
                  f"Avg Rounds (last {LOG_INTERVAL}): {avg_rounds:.2f} | "
                  f"Avg Score (last {LOG_INTERVAL}): {avg_score:.2f} | "
                  f"Avg Loss (last {LOG_INTERVAL}): {avg_loss_recent:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
        
        # 주기적으로 모델 저장
        if (game + 1) % SAVE_INTERVAL == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            agent.save(f"{OUTPUT_DIR}/dqn_flip7_game_{game + 1}.pth")
    
    # ========================================================================
    # TRAINING 완료
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)
    print(f"Average rounds per game: {np.mean(all_game_rounds):.2f}")
    print(f"Min rounds in a game: {np.min(all_game_rounds)}")
    print(f"Max rounds in a game: {np.max(all_game_rounds)}")
    print(f"Average final score: {np.mean(total_scores_per_game):.2f}")
    print("=" * 70)
    
    # 최종 모델 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    agent.save(f"{OUTPUT_DIR}/dqn_flip7_final.pth")
    
    # ========================================================================
    # TRAINING 기록 저장 (데이터 & 플롯)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training 기록 저장 중...")
    print("=" * 70)
    
    history_df = pd.DataFrame({
        'Rounds': all_game_rounds,
        'Avg_Loss': all_game_avg_loss
    })
    
    # 원시 데이터를 CSV로 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history_csv_path = f'{OUTPUT_DIR}/training_history_data.csv'
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training data saved to: {history_csv_path}")
    
    # 50게임 이동 평균 계산
    history_df['Rounds_MA50'] = history_df['Rounds'].rolling(window=50, min_periods=1).mean()
    history_df['Avg_Loss_MA50'] = history_df['Avg_Loss'].rolling(window=50, min_periods=1).mean()
    
    # 2-서브플롯 그림 생성
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # 서브플롯 1: 게임당 라운드 수
    ax[0].plot(history_df.index, history_df['Rounds'], color='blue', alpha=0.2, label='Raw')
    ax[0].plot(history_df.index, history_df['Rounds_MA50'], color='blue', linewidth=2, label='50-Game MA')
    ax[0].set_title('Rounds to Reach 200 Points', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Rounds', fontsize=12)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # 서브플롯 2: 게임당 평균 손실
    ax[1].plot(history_df.index, history_df['Avg_Loss'], color='red', alpha=0.2, label='Raw')
    ax[1].plot(history_df.index, history_df['Avg_Loss_MA50'], color='red', linewidth=2, label='50-Game MA')
    ax[1].set_title('Average Training Loss per Game', fontsize=14, fontweight='bold')
    ax[1].set_xlabel('Game Number', fontsize=12)
    ax[1].set_ylabel('Avg. MSE Loss', fontsize=12)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'{OUTPUT_DIR}/training_history_plot.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to: {plot_path}")
    print("=" * 70)
    
    return agent, env


# ============================================================================
# 평가 함수
# ============================================================================
def evaluate(agent: DQNAgent, env: FlipSevenCoreEnv, num_games: int = EVAL_GAMES):
    """
    훈련된 에이전트를 평가합니다.
    
    인자:
        agent: 훈련된 DQN 에이전트
        env: 환경
        num_games: 평가할 게임 수
    """
    print("\n" + "=" * 70)
    print(f"Evaluating agent for {num_games} games...")
    print("=" * 70)
    
    eval_rounds_per_game = []
    eval_scores_per_game = []
    
    for game in range(num_games):
        # 게임 리셋
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        
        obs, info = env.reset()
        game_total_rounds = 0
        
        # 200점에 도달할 때까지 플레이
        while info.get("total_game_score", 0) < 200:
            game_total_rounds += 1
            terminated = False
            
            while not terminated:
                # 항상 탐욕적 행동 선택 (eval_mode=True)
                action = agent.select_action(obs, eval_mode=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs
            
            if info.get("total_game_score", 0) < 200:
                obs, info = env.reset()
        
        final_score = info.get("total_game_score", 0)
        eval_rounds_per_game.append(game_total_rounds)
        eval_scores_per_game.append(final_score)
        
        print(f"Eval Game {game + 1}: {game_total_rounds} rounds, Final Score: {final_score}")
    
    print("=" * 70)
    print(f"Evaluation Results (over {num_games} games):")
    print(f"  Average rounds: {np.mean(eval_rounds_per_game):.2f}")
    print(f"  Min rounds: {np.min(eval_rounds_per_game)}")
    print(f"  Max rounds: {np.max(eval_rounds_per_game)}")
    print(f"  Average final score: {np.mean(eval_scores_per_game):.2f}")
    print("=" * 70)


# ============================================================================
# 진입점
# ============================================================================
if __name__ == "__main__":
    # 에이전트 훈련
    trained_agent, trained_env = train()
    
    # 훈련된 에이전트 평가
    evaluate(trained_agent, trained_env, num_games=EVAL_GAMES)
