"""
Flip Seven 강화학습 프로젝트 설정 파일

이 파일은 프로젝트 전반에서 사용되는 모든 하이퍼파라미터, 파일 경로, 상수를 정의합니다.
설정의 핵심 역할을 합니다.
"""

import os
from datetime import datetime

# ============================================================================
# 1. 파일 경로 및 디렉토리 설정
# ============================================================================
# 모든 결과물이 저장될 최상위 디렉토리
BASE_OUTPUT_DIR = "./runs"

# 고정된 실행 이름 (타임스탬프 폴더 생성 방지 및 경로 고정)
RUN_NAME = "latest_run"

# 현재 실행의 메인 출력 디렉토리
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)

# 분석 결과 그래프 및 이미지를 저장할 디렉토리
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# 정책 분석 그래프 경로
POLICY_ANALYSIS_PLOT_PATH = os.path.join(PLOTS_DIR, "policy_analysis_12_11_10.png")

# 모델 체크포인트를 저장할 디렉토리
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# 최종 학습된 모델의 경로
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "dqn_flip7_final.pth")

# 필요한 디렉토리들이 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ============================================================================
# 2. 학습 하이퍼파라미터
# ============================================================================
NUM_TOTAL_GAMES_TO_TRAIN = 1000      # 학습할 전체 게임 수
TARGET_UPDATE_FREQUENCY = 10         # 타겟 네트워크 업데이트 주기 (게임 단위)
REPLAY_BUFFER_SIZE = 50000           # 리플레이 버퍼 크기
BATCH_SIZE = 64                      # 미니 배치 크기
GAMMA = 0.99                         # 할인율 (Discount Factor)
LEARNING_RATE = 1e-4                 # Adam 옵티마이저 학습률
EPSILON_START = 1.0                  # 초기 탐험 확률 (Epsilon)
EPSILON_END = 0.01                   # 최소 탐험 확률
EPSILON_DECAY = 0.995                # 게임당 Epsilon 감소율
MIN_REPLAY_SIZE = 1000               # 학습 시작을 위한 최소 버퍼 크기

# ============================================================================
# 3. 고급 DQN 설정
# ============================================================================
USE_DOUBLE_DQN = False               # Double DQN 사용 여부 (과대평가 방지)
USE_DUELING_NETWORK = False          # Dueling Network 사용 여부 (가치/이점 분리)

# ============================================================================
# 4. 네트워크 아키텍처 설정
# ============================================================================
HAND_NUMBERS_DIM = 13                # 숫자 카드 입력 차원 (0~12)
HAND_MODIFIERS_DIM = 6               # 수정자 카드 입력 차원
DECK_COMPOSITION_DIM = 19            # 덱 구성 정보 입력 차원 (카운팅용)
SCORE_DIM = 1                        # 점수 정보 입력 차원
HIDDEN_DIM = 128                     # 은닉층 크기

# ============================================================================
# 5. 로깅 및 평가 설정
# ============================================================================
LOG_INTERVAL = 100                   # 로그 출력 주기 (게임 단위)
SAVE_INTERVAL = 100                  # 모델 저장 주기 (게임 단위)
EVAL_GAMES = 100                     # 평가 시 플레이할 게임 수
