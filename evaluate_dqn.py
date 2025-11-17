# evaluate_dqn.py
#
# 훈련된 DQN 에이전트의 성능을 평가하는 스크립트입니다.
# 'train_dqn.py'의 QNetwork, DQNAgent 클래스 정의를 포함하고 있습니다.
# 'flip_seven_env.py'의 환경을 사용합니다.
#
import torch
import torch.nn as nn
import torch.optim as optim  # DQNAgent 클래스 로드에 필요
import numpy as np
import collections
import random
from typing import Dict, Tuple, Any
import gymnasium as gym
import time

# 'flip_seven_env.py' 파일에서 환경 클래스를 임포트합니다.
try:
    from flip_seven_env import FlipSevenCoreEnv
except ImportError:
    print("="*50)
    print("오류: 'flip_seven_env.py' 파일을 찾을 수 없습니다.")
    print("evaluate_dqn.py와 같은 디렉토리에 있는지 확인하세요.")
    print("="*50)
    exit()

# ============================================================================
# 평가 설정
# ============================================================================
NUM_EVAL_GAMES = 100  # 평가할 총 게임 횟수
MODEL_PATH = "./runs/dqn_flip7_final.pth"  # 불러올 모델 경로
GAME_GOAL_SCORE = 200  # 게임 종료 목표 점수

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================================
# Q-NETWORK ARCHITECTURE (train_dqn.py와 동일해야 함)
# ============================================================================
class QNetwork(nn.Module):
    """
    FlipSevenCoreEnv의 Dict 관측 공간을 처리하는 Q-Network.
    train_dqn.py의 정의와 정확히 일치해야 모델 로드가 가능합니다.
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
        
        # 각 관측 요소별 별도 처리 레이어
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
        hand_numbers_feat = self.hand_numbers_net(obs_dict["current_hand_numbers"])
        hand_modifiers_feat = self.hand_modifiers_net(obs_dict["current_hand_modifiers"])
        deck_composition_feat = self.deck_composition_net(obs_dict["deck_composition"])
        score_feat = self.score_net(obs_dict["total_game_score"])
        
        combined_feat = torch.cat([
            hand_numbers_feat,
            hand_modifiers_feat,
            deck_composition_feat,
            score_feat
        ], dim=1)
        
        q_values = self.shared_net(combined_feat)
        return q_values


# ============================================================================
# DQN AGENT (train_dqn.py와 동일, 'learn' 관련 제외)
# ============================================================================
class DQNAgent:
    """
    DQN 에이전트. 모델 로드 및 행동 선택 기능만 사용합니다.
    """
    
    def __init__(
        self,
        action_space_size: int = 2,
        device: torch.device = DEVICE
    ):
        self.action_space_size = action_space_size
        self.device = device
        
        # Q-networks 초기화
        self.q_network = QNetwork().to(device)
        self.target_network = QNetwork().to(device) # 로드에 필요
        self.optimizer = optim.Adam(self.q_network.parameters()) # 로드에 필요
        self.epsilon = 0.0 # 평가 모드이므로 0으로 설정
    
    def _dict_to_tensor(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """관측 딕셔너리를 텐서 딕셔너리로 변환 (배치 크기 1)"""
        return {
            key: torch.FloatTensor(value).unsqueeze(0).to(self.device)
            for key, value in obs_dict.items()
        }
    
    def select_action(self, obs: Dict[str, np.ndarray], eval_mode: bool = False) -> int:
        """
        엡실론-그리디 정책에 따라 행동 선택.
        eval_mode=True일 경우, 항상 그리디(Greedy) 행동만 선택합니다.
        """
        # 평가 모드에서는 항상 최적의 행동(exploitation)을 선택
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                obs_tensor = self._dict_to_tensor(obs)
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax(dim=1).item()
            return action
        else:
            # (eval_mode=False이고 랜덤 확률에 걸린 경우 - 평가 시에는 사용되지 않음)
            return random.randint(0, self.action_space_size - 1)
    
    def load(self, filepath: str):
        """저장된 Q-network 가중치를 불러옵니다."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 0.0) # 훈련 중 엡실론 (평가 시 무관)
            self.q_network.eval() # 평가 모드로 설정! (필수)
            self.target_network.eval() # 평가 모드로 설정! (필수)
            print(f"모델을 {filepath} 에서 성공적으로 불러왔습니다.")
        except FileNotFoundError:
            print(f"오류: {filepath} 에서 모델 파일을 찾을 수 없습니다.")
            exit()
        except Exception as e:
            print(f"오류: 모델 로드 중 문제 발생. {e}")
            print("train_dqn.py의 QNetwork 구조와 현재 스크립트의 구조가 일치하는지 확인하세요.")
            exit()


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def run_evaluation(agent: DQNAgent, env: gym.Env, num_games: int):
    """
    훈련된 에이전트의 성능을 'num_games' 만큼 평가합니다.
    """
    print("\n" + "=" * 70)
    print(f"에이전트 평가 시작 (총 {num_games} 게임)...")
    print("=" * 70)
    
    eval_rounds_per_game = []
    eval_scores_per_game = []
    total_start_time = time.time()
    
    for game in range(num_games):
        game_start_time = time.time()
        
        # --- 1. '게임' 시작 시 전체 상태 수동 초기화 ---
        # (train_dqn.py와 동일한 게임 초기화 로직)
        env.total_score = 0
        env.draw_deck = collections.deque()
        env.discard_pile = []
        env._initialize_deck_to_discard()  # 85장 덱 생성
        
        # 첫 라운드를 위해 env.reset() 호출
        obs, info = env.reset(seed=42 + game)
        
        game_total_rounds = 0
        
        # --- 2. '게임' 루프 (목표 점수 도달까지) ---
        while info.get("total_game_score", 0) < GAME_GOAL_SCORE:
            game_total_rounds += 1
            terminated = False  # '라운드' 종료 플래그
            
            # --- 3. '라운드' 루프 (Bust, Stay, Flip 7 전까지) ---
            while not terminated:
                
                # 행동 선택 (eval_mode=True: Epsilon-greedy가 아닌 Greedy 선택)
                action = agent.select_action(obs, eval_mode=True)
                
                # 환경과 상호작용
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # 다음 상태로 업데이트
                obs = next_obs
            
            # --- 4. 라운드 종료 ---
            # 게임이 끝나지 않았다면, 다음 라운드를 위해 reset() 호출 (손패만 비움)
            if info.get("total_game_score", 0) < GAME_GOAL_SCORE:
                obs, info = env.reset()

        # --- 5. 게임 종료 ---
        game_end_time = time.time()
        final_score = info.get("total_game_score", 0)
        eval_rounds_per_game.append(game_total_rounds)
        eval_scores_per_game.append(final_score)
        
        print(f"  [게임 {game + 1:03d}/{num_games}] "
              f"최종 점수: {final_score:03d} | "
              f"총 라운드: {game_total_rounds:02d} | "
              f"소요 시간: {game_end_time - game_start_time:.2f}초")

    # --- 6. 최종 결과 요약 ---
    total_end_time = time.time()
    print("=" * 70)
    print(f"🏁 평가 완료 (총 {num_games} 게임) 🏁")
    print(f"  - 총 소요 시간: {total_end_time - total_start_time:.2f}초")
    print(f"  - 평균 라운드 수: {np.mean(eval_rounds_per_game):.2f} 라운드")
    print(f"  - 최소 라운드 수: {np.min(eval_rounds_per_game)}")
    print(f"  - 최대 라운드 수: {np.max(eval_rounds_per_game)}")
    print(f"  - 평균 최종 점수: {np.mean(eval_scores_per_game):.2f} 점")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    
    # 1. 환경 생성
    try:
        env = FlipSevenCoreEnv()
        print("[성공] FlipSevenCoreEnv 환경을 생성했습니다.")
    except Exception as e:
        print(f"[실패] 환경 생성 중 오류 발생: {e}")
        exit()

    # 2. 에이전트 생성 및 모델 로드
    agent = DQNAgent(device=DEVICE)
    agent.load(MODEL_PATH)
    
    # 3. 평가 실행
    run_evaluation(agent, env, num_games=NUM_EVAL_GAMES)