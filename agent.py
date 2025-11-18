"""
Flip Seven DQN을 위한 에이전트 및 리플레이 버퍼

이 모듈은 DQN 에이전트와 경험 재생 버퍼를 정의합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random
from typing import Dict, Tuple, Optional

from network import QNetwork
from config import (
    REPLAY_BUFFER_SIZE,
    BATCH_SIZE,
    GAMMA,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_END,
    MIN_REPLAY_SIZE
)


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
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        replay_buffer_size: int = REPLAY_BUFFER_SIZE,
        device: Optional[torch.device] = None
    ):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon_end = epsilon_end
        
        # Q-네트워크 초기화
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is in eval mode
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # epsilon-greedy policy를 위한 초기 epsilon 값
        self.epsilon = epsilon_start
    
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
    
    def decay_epsilon(self, decay_rate: float):
        """epsilon-greedy 정책을 위한 epsilon 값을 감소시킵니다."""
        self.epsilon = max(self.epsilon_end, self.epsilon * decay_rate)
    
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
