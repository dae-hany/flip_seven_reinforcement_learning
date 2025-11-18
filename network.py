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
