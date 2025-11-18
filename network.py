"""
Flip Seven DQN을 위한 Q-네트워크 아키텍처

이 모듈은 FlipSevenCoreEnv의 Dict 관측 공간을 처리하는 Q-네트워크를 정의합니다.
"""

import torch
import torch.nn as nn
from typing import Dict


class QNetwork(nn.Module):
    """
    FlipSevenCoreEnv의 관측 공간을 처리하는 큐 네트워크

    구조:
    1. 4가지 관측 구성 요소 각각을 별도로 처리
    2. 모든 처리된 특징들을 연결
    3. 공유 MLP를 통과
    4. Standard DQN: 2개의 Q값 직접 출력
       Dueling DQN: 상태 가치(V)와 행동 이점(A)을 분리하여 Q값 계산
    """

    def __init__(
        self,
        hand_numbers_dim: int = 13,
        hand_modifiers_dim: int = 6,
        deck_composition_dim: int = 19,
        score_dim: int = 1,
        hidden_dim: int = 128,
        action_space_size: int = 2,
        use_dueling: bool = True
    ):
        super(QNetwork, self).__init__()
        
        self.use_dueling = use_dueling
        self.action_space_size = action_space_size
        
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
        
        # 공유 MLP 레이어 (Dueling과 Standard 모두 사용)
        self.shared_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if self.use_dueling:
            # Dueling DQN 아키텍처: 가치 스트림과 이점 스트림 분리
            # 상태 가치 스트림 (V): 상태 자체의 가치를 학습
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # 행동 이점 스트림 (A): 각 행동의 상대적 이점을 학습
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_space_size)
            )
        else:
            # Standard DQN: Q값을 직접 출력
            self.output_layer = nn.Linear(hidden_dim, action_space_size)
    
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
            Q-values: (batch_size, action_space_size)
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
        shared_features = self.shared_net(combined_feat)
        
        if self.use_dueling:
            # Dueling DQN: V(s)와 A(s,a)를 계산하여 Q(s,a) 도출
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            # 평균을 빼는 이유: V와 A의 식별 가능성을 높이기 위함
            value = self.value_stream(shared_features)  # (batch_size, 1)
            advantage = self.advantage_stream(shared_features)  # (batch_size, action_space_size)
            
            # Q값 계산: 이점의 평균을 빼서 상태 가치와 행동 이점을 명확히 분리
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN: Q값을 직접 출력
            q_values = self.output_layer(shared_features)
        
        return q_values
