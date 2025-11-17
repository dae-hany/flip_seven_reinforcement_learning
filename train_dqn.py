import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random
from typing import Dict, Tuple, Any, Optional
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# from flip_seven_env import FlipSevenCoreEnv (original version)
from flip_seven_env_considering_end_bonus import FlipSevenCoreEnv # (considering end-game bonus version)

# ============================================================================
# 훈련 하이퍼파라미터
# ============================================================================
NUM_TOTAL_GAMES_TO_TRAIN = 1000 # 학습할 전체 게임 수 
TARGET_UPDATE_FREQUENCY = 10  # 타겟 네트워크를 N 게임마다 업데이트
REPLAY_BUFFER_SIZE = 50000 # 리플레이 버퍼 크기
BATCH_SIZE = 64 # 배치 크기
GAMMA = 0.99  # 할인률
LEARNING_RATE = 1e-4 # 학습률 
EPSILON_START = 1.0 # 초기 epsilon
EPSILON_END = 0.01 # 최소 epsilon
EPSILON_DECAY = 0.995  # 게임마다 epsilon 감소
MIN_REPLAY_SIZE = 1000  # 이만큼의 transition이 쌓인 후 학습 시작

OUTPUT_DIR = "./runs_end_bonus"
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================================
# Q-NETWORK ARCHITECTURE (Handles Dict Observation Space)
# ============================================================================
class QNetwork(nn.Module):
    """
    FlipSevenCoreEnv의 관측 공간을 처리하는 큐 네트워크

    구조:
    1. 4가지 관측 구성 요소 각각을 별도로 처리
    2. 모든 처리된 특징들을 연결
    3. 공유 MLP를 통과
    4. 2개의 Q값 출력 ('Stay' 및 'Hit' 행동에 대해)
    """

    def __init__(
        self,
        hand_numbers_dim: int = 13,
        hand_modifiers_dim: int = 6,
        deck_composition_dim: int = 19,
        score_dim: int = 1,
        hidden_dim: int = 128
    ):
        super(QNetwork, self).__init__()
        
        # Dict 관측 공간의 4가지 요소 각각을 처리하기 위한 4개의 독립된 입력 레이어 정의
        self.hand_numbers_net = nn.Sequential(
            nn.Linear(hand_numbers_dim, 32),
            nn.ReLU()
        )
        
        self.hand_modifiers_net = nn.Sequential(
            nn.Linear(hand_modifiers_dim, 16),
            nn.ReLU()
        )
        
        self.deck_composition_net = nn.Sequential(
            nn.Linear(deck_composition_dim, 64),
            nn.ReLU()
        )
        
        self.score_net = nn.Sequential(
            nn.Linear(score_dim, 8),
            nn.ReLU()
        )
        
        # 총 연결된 특징 차원 계산
        concat_dim = 32 + 16 + 64 + 8  # = 120
        
        # 공유 MLP 레이어
        self.shared_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 출력: Q(s, Stay), Q(s, Hit)
        )
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        순전파 프로세스
        
        Args:
            obs_dict: Dictionary containing batched observations
                - "current_hand_numbers": (batch_size, 13)
                - "current_hand_modifiers": (batch_size, 6)
                - "deck_composition": (batch_size, 19)
                - "total_game_score": (batch_size, 1)
        
        Returns:
            Q-values: (batch_size, 2)
        """
        # 각 구성 요소를 별도로 처리
        hand_numbers_feat = self.hand_numbers_net(obs_dict["current_hand_numbers"])
        hand_modifiers_feat = self.hand_modifiers_net(obs_dict["current_hand_modifiers"])
        deck_composition_feat = self.deck_composition_net(obs_dict["deck_composition"])
        score_feat = self.score_net(obs_dict["total_game_score"])
        
        # 모든 특징들을 연결
        combined_feat = torch.cat([
            hand_numbers_feat,
            hand_modifiers_feat,
            deck_composition_feat,
            score_feat
        ], dim=1)
        
        # 공유 MLP를 통과
        q_values = self.shared_net(combined_feat)
        
        return q_values


# ============================================================================
# 리플레이 버퍼
# ============================================================================
class ReplayBuffer:
    """
    경험 재생 버퍼 (Experience Replay Buffer) - 전이(transition)를 저장합니다.
    """
    
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool
    ):
        """전이를 버퍼에 저장합니다"""
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """전이 배치를 샘플링합니다"""
        batch = random.sample(self.buffer, batch_size)
        
        # 배치 분리
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for obs, action, reward, next_obs, done in batch:
            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# DQN 에이전트
# ============================================================================
class DQNAgent:
    """
    Flip 7을 학습하는 DQN 에이전트입니다.
    """
    
    def __init__(
        self,
        action_space_size: int = 2,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        device: torch.device = DEVICE
    ):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.device = device
        
        # Q-네트워크 초기화
        self.q_network = QNetwork().to(device)
        self.target_network = QNetwork().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is in eval mode
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # epsilon-greedy policy를 위한 초기 epsilon 값
        self.epsilon = EPSILON_START
    
    def _dict_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """관찰 딕셔너리를 텐서 딕셔너리로 변환합니다."""
        return {
            key: torch.FloatTensor(value).unsqueeze(0).to(self.device)
            for key, value in obs_dict.items()
        }
    
    def _batch_dict_to_tensor(self, obs_batch: list) -> Dict[str, torch.Tensor]:
        """관찰 딕셔너리 배치를 텐서 딕셔너리로 변환합니다."""
        batched_dict = {}
        
        # 첫 번째 관찰에서 모든 키를 가져옵니다
        keys = obs_batch[0].keys()
        
        for key in keys:
            # 이 키에 대한 모든 관찰을 쌓습니다
            batched_dict[key] = torch.FloatTensor(
                np.array([obs[key] for obs in obs_batch])
            ).to(self.device)
        
        return batched_dict
    
    def select_action(self, obs: Dict[str, np.ndarray], eval_mode: bool = False) -> int:
        """
        epsilon-greedy 정책을 사용하여 행동을 선택합니다.
        
        인자:
            obs: 관찰 딕셔너리
            eval_mode: True인 경우, 항상 탐험 없이 탐욕적 행동 선택

        반환:
            선택된 행동 (0 = Stay, 1 = Hit)
        """
        # exploration: 무작위 행동
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        # exploitation: 탐욕적 행동
        with torch.no_grad():
            obs_tensor = self._dict_to_tensor(obs)
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool
    ):
        """리플레이 버퍼에 전이(transition)를 저장합니다."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)
    
    def learn(self) -> Optional[float]:
        """한 번의 학습 단계를 수행합니다 (배치 샘플링 및 Q-네트워크 업데이트).
        
        반환:
            학습이 수행된 경우 손실 값, 그렇지 않으면 None
        """
        # 충분한 샘플이 쌓일 때까지 학습하지 않습니다
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None
        
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
        
        # 리플레이 버퍼에서 배치 샘플링
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(BATCH_SIZE)
        
        # 텐서로 변환
        obs_tensor = self._batch_dict_to_tensor(obs_batch)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_tensor = self._batch_dict_to_tensor(next_obs_batch)
        done_tensor = torch.FloatTensor(done_batch).to(self.device)
        
        # 현재 Q-값 계산
        current_q_values = self.q_network(obs_tensor)
        current_q_values = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # 목표 Q-값 계산
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_tensor)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q_values
        
        # 손실 계산
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Q-네트워크 최적화
        self.optimizer.zero_grad()
        loss.backward()

        # 그래디언트 클리핑: 그래디언트 폭주 방지
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """현재 Q-네트워크 가중치로 타겟 네트워크를 업데이트합니다."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """epsilon-greedy 정책을 위한 epsilon 값을 감소시킵니다."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save(self, filepath: str):
        """Q-네트워크를 저장합니다."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Q-네트워크를 불러옵니다."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train():
    """
    게임/라운드 구조를 올바르게 처리하는 메인 학습 루프입니다.
    """
    # 환경 및 에이전트 초기화
    env = FlipSevenCoreEnv()
    agent = DQNAgent()
    
    # 학습 통계
    all_game_rounds = []
    all_game_avg_loss = []
    total_scores_per_game = []
    
    print("=" * 70)
    print("Starting DQN Training on FlipSevenCoreEnv")
    print("=" * 70)
    print(f"Total games to train: {NUM_TOTAL_GAMES_TO_TRAIN}")
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
        agent.decay_epsilon()
        
        # 주기적으로 타겟 네트워크 업데이트
        if (game + 1) % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
        
        # 로깅
        if (game + 1) % 10 == 0:
            avg_rounds = np.mean(all_game_rounds[-10:])
            avg_score = np.mean(total_scores_per_game[-10:])
            avg_loss_recent = np.mean(all_game_avg_loss[-10:])
            print(f"Game {game + 1}/{NUM_TOTAL_GAMES_TO_TRAIN} | "
                  f"Rounds: {game_total_rounds} | "
                  f"Score: {final_score} | "
                  f"Avg Rounds (last 10): {avg_rounds:.2f} | "
                  f"Avg Score (last 10): {avg_score:.2f} | "
                  f"Avg Loss (last 10): {avg_loss_recent:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
        
        # 주기적으로 모델 저장
        if (game + 1) % 100 == 0:
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
    history_df.to_csv('./runs/training_history_data.csv', index=False)
    print("Training data saved to: ./runs/training_history_data.csv")
    
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
    plt.savefig(f'{OUTPUT_DIR}/training_history_plot.png', dpi=150)
    print(f"Training plot saved to: {OUTPUT_DIR}/training_history_plot.png")
    print("=" * 70)
    
    return agent, env


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate(agent: DQNAgent, env: FlipSevenCoreEnv, num_games: int = 10):
    """
    Evaluate the trained agent.
    
    Args:
        agent: Trained DQN agent
        env: Environment
        num_games: Number of games to evaluate
    """
    print("\n" + "=" * 70)
    print(f"Evaluating agent for {num_games} games...")
    print("=" * 70)
    
    eval_rounds_per_game = []
    eval_scores_per_game = []
    
    for game in range(num_games):
        # Reset game
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()
        
        obs, info = env.reset()
        game_total_rounds = 0
        
        # Play until 200 points
        while info.get("total_game_score", 0) < 200:
            game_total_rounds += 1
            terminated = False
            
            while not terminated:
                # Always select greedy action (eval_mode=True)
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
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Train the agent
    trained_agent, trained_env = train()
    
    # Evaluate the trained agent
    evaluate(trained_agent, trained_env, num_games=10)
