"""
Flip Seven DQN을 위한 에이전트 및 리플레이 버퍼

이 모듈은 DQN 에이전트와 경험 재생 버퍼를 정의합니다.
"""

import collections
import random
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from network import QNetwork

# ============================================================================
# 리플레이 버퍼
# ============================================================================
class ReplayBuffer:
    """
    경험 재생 버퍼 (Experience Replay Buffer)

    에이전트가 환경과 상호작용하며 얻은 전이(transition) 데이터를 저장하고,
    학습 시 무작위로 샘플링하여 데이터 간의 상관관계를 끊고 학습 안정성을 높입니다.
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
        """전이(transition)를 버퍼에 저장합니다."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        """버퍼에서 배치 크기만큼 무작위로 샘플링합니다."""
        batch = random.sample(self.buffer, batch_size)

        # 배치 데이터 분리
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# DQN 에이전트
# ============================================================================
class DQNAgent:
    """
    DQN (Deep Q-Network) 에이전트

    Flip Seven 게임을 학습하기 위한 강화학습 에이전트입니다.
    Standard DQN 및 Double DQN, Dueling Network 기능을 지원합니다.
    """

    def __init__(
        self,
        action_space_size: int = 2,
        learning_rate: float = config.LEARNING_RATE,
        gamma: float = config.GAMMA,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.device = device

        # Q-네트워크 및 타겟 네트워크 초기화
        self.q_network = QNetwork(
            use_dueling=config.USE_DUELING_NETWORK
        ).to(device)

        self.target_network = QNetwork(
            use_dueling=config.USE_DUELING_NETWORK
        ).to(device)

        # 타겟 네트워크 가중치 동기화 및 평가 모드 설정
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 리플레이 버퍼 초기화
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

        # 탐험을 위한 Epsilon 초기화
        self.epsilon = config.EPSILON_START

    def _dict_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """단일 관측 딕셔너리를 텐서 딕셔너리로 변환합니다 (배치 차원 추가)."""
        return {
            key: torch.FloatTensor(value).unsqueeze(0).to(self.device)
            for key, value in obs_dict.items()
        }

    def _batch_dict_to_tensor(self, obs_batch: Tuple[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """관측 딕셔너리 배치를 텐서 딕셔너리로 변환합니다."""
        batched_dict = {}
        keys = obs_batch[0].keys()

        for key in keys:
            # 해당 키의 모든 관측값을 하나의 텐서로 결합
            batched_dict[key] = torch.FloatTensor(
                np.array([obs[key] for obs in obs_batch])
            ).to(self.device)

        return batched_dict

    def select_action(self, obs: Dict[str, np.ndarray], eval_mode: bool = False) -> int:
        """
        현재 관측에 대해 행동을 선택합니다.

        Args:
            obs: 현재 관측값
            eval_mode: True일 경우 탐험 없이 항상 최적의 행동(Greedy)을 선택

        Returns:
            선택된 행동 (0: Stay, 1: Hit)
        """
        # 탐험 (Exploration): 무작위 행동 선택
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)

        # 활용 (Exploitation): Q-네트워크를 통한 최적 행동 선택
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
        """리플레이 버퍼에 경험을 저장합니다."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def learn(self) -> Optional[float]:
        """
        리플레이 버퍼에서 배치를 샘플링하여 네트워크를 학습시킵니다.

        Returns:
            loss: 학습 손실값 (학습이 수행되지 않았으면 None)
        """
        # 학습 시작 조건 확인
        if len(self.replay_buffer) < config.MIN_REPLAY_SIZE:
            return None
        if len(self.replay_buffer) < config.BATCH_SIZE:
            return None

        # 배치 샘플링
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(config.BATCH_SIZE)

        # 텐서 변환
        obs_tensor = self._batch_dict_to_tensor(obs_batch)
        action_tensor = torch.LongTensor(action_batch).to(self.device)
        reward_tensor = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_tensor = self._batch_dict_to_tensor(next_obs_batch)
        done_tensor = torch.FloatTensor(done_batch).to(self.device)

        # 1. 현재 상태의 Q값 계산 (예측값)
        current_q_values = self.q_network(obs_tensor)
        # 선택한 행동에 대한 Q값만 추출
        current_q_values = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # 2. 다음 상태의 Q값 계산 (목표값)
        with torch.no_grad():
            if config.USE_DOUBLE_DQN:
                # Double DQN: 행동 선택은 Main Net, 가치 평가는 Target Net
                next_actions = self.q_network(next_obs_tensor).argmax(dim=1)
                next_q_values = self.target_network(next_obs_tensor)
                max_next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: 행동 선택과 가치 평가 모두 Target Net (max Q)
                next_q_values = self.target_network(next_obs_tensor)
                max_next_q_values = next_q_values.max(dim=1)[0]

            # Bellman Optimality Equation
            target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q_values

        # 3. 손실 계산 (MSE Loss)
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 4. 역전파 및 가중치 업데이트
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (학습 안정화)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """타겟 네트워크의 가중치를 메인 네트워크와 동기화합니다."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Epsilon 값을 감소시켜 탐험 비율을 줄입니다."""
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)

    def save(self, filepath: str):
        """모델 및 학습 상태를 저장합니다."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"모델 저장 완료: {filepath}")

    def load(self, filepath: str):
        """저장된 모델 및 학습 상태를 불러옵니다."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', config.EPSILON_START)
        print(f"모델 로드 완료: {filepath}")
