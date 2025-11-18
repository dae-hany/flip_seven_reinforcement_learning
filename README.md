# Flip 7 - DQN 강화학습 프로젝트

카드 게임 "Flip 7"의 솔로 플레이 "Core Game" 변형을 마스터하기 위한 Deep Q-Network (DQN) 구현입니다. 이 프로젝트는 심층 강화학습이 카드 카운팅, 위험 관리, 목표 지향적 의사결정을 어떻게 학습하는지 보여줍니다.

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [게임 규칙](#-게임-규칙)
3. [설치 및 실행](#-설치-및-실행)
4. [파일 구조](#-파일-구조)
5. [사용 방법](#-사용-방법)
6. [환경 세부사항](#-환경-세부사항)
7. [DQN 아키텍처](#-dqn-아키텍처)
8. [정책 분석 도구](#-정책-분석-도구)

---

## 🎯 프로젝트 개요

### 목표
**최소 라운드 수**로 **200점**에 도달하는 것을 목표로 하는 프레스-유어-럭(press-your-luck) 카드 게임 "Flip 7"의 Core Game 변형을 위한 DQN 에이전트를 학습시킵니다.

### 주요 특징
- ✅ **커스텀 Gymnasium 환경**: `gymnasium.Env` API와 완전 호환
- ✅ **Dict 관측 공간**: 덱 구성 추적을 통한 카드 카운팅 지원
- ✅ **멀티 라운드 게임 구조**: 라운드(에피소드)와 게임(전체 매치) 구분
- ✅ **종합 분석 도구**: 학습 곡선 시각화 및 정책 평가 스크립트

---

## 🎮 게임 규칙

### 핵심 메커니즘
게임 규칙은 `[core_game]flip_seven_rulebook_for_ai_agent.txt`에 상세히 정의되어 있습니다.

**목표**: 총점 200점 도달 (최소 라운드 수)

**덱 구성** (총 85장):
- 숫자 카드 79장: "1"×1, "2"×2, ..., "12"×12, "0"×1
- 수정자 카드 6장: +2, +4, +6, +8, +10, x2

**라운드 진행**:
1. 각 턴마다 **Hit**(카드 뽑기) 또는 **Stay**(라운드 종료) 선택
2. **Bust 조건**: 손패에 이미 있는 숫자 카드를 뽑으면 0점으로 라운드 종료
3. **Flip 7 보너스**: 7개의 서로 다른 숫자 카드를 모으면 +15점 보너스 후 자동 종료

**점수 계산**:
```
라운드 점수 = (숫자 카드 합 × x2배수) + 수정자 보너스 + Flip 7 보너스
```

**카드 카운팅**: 덱은 라운드 간 재섞이지 않으며, 버림 더미가 비었을 때만 재섞임

---

## 🚀 설치 및 실행

### 필수 요구사항
```bash
Python 3.8+
PyTorch 2.0+
Gymnasium
NumPy
Matplotlib
Pandas
```

### 설치
```bash
# 저장소 클론
git clone https://github.com/dae-hany/flip_seven_reinforcement_learning.git
cd flip_seven_reinforcement_learning

# 의존성 설치
pip install torch gymnasium numpy matplotlib pandas
```

### 빠른 시작

#### 1. 환경 테스트
```bash
python test_env.py
```

#### 2. DQN 에이전트 학습
```bash
python train.py
```
- 학습된 모델: `./runs_end_bonus/dqn_flip7_final.pth`
- 학습 곡선: `./runs_end_bonus/training_history_plot.png`
- **하이퍼파라미터 설정**: `config.py`에서 쉽게 수정 가능

#### 3. 에이전트 평가
```bash
python evaluate_dqn.py
```

#### 4. 정책 분석
```bash
# 기본 시나리오 테스트
python test_policy_scenarios.py

# 카드 카운팅 학습 검증
python test_policy_with_card_counting_test.py

# 목표 인식 분석
python test_policy_with_goal_awareness_test.py

# 위험 대비 보상 평가
python test_policy_with_risk_vs_reward.py

# 수정자 카드 효과 분석
python test_policy_with_modifier_card_effect.py

# 고위험 카드 카운팅 시나리오
python test_policy_with_card_counting_risky.py
```

---

## 📁 파일 구조

```
flip_seven_reinforcement_learning/
├── [core_game]flip_seven_rulebook_for_ai_agent.txt  # 게임 규칙 정의
│
├── config.py                              # 훈련 하이퍼파라미터 설정
├── network.py                             # Q-네트워크 아키텍처
├── agent.py                               # DQN 에이전트 및 리플레이 버퍼
├── flip_seven_env.py                      # Gymnasium 환경 구현 (통합 버전)
│
├── train.py                               # DQN 학습 메인 스크립트
├── train_dqn.py                           # DQN 학습 스크립트 (레거시)
├── evaluate_dqn.py                        # 체크포인트 간 정책 진화 분석
├── test_env.py                            # 환경 테스트 (랜덤 에이전트)
│
├── test_policy_scenarios.py               # 기본 정책 시나리오 테스트
├── test_policy_with_card_counting_test.py # 카드 카운팅 학습 검증 (전체 카드)
├── test_policy_with_card_counting_risky.py # 고위험 카드 카운팅 시나리오
├── test_policy_with_goal_awareness_test.py # 200점 목표 인식 분석
├── test_policy_with_risk_vs_reward.py     # 위험 대비 보상 평가
├── test_policy_with_modifier_card_effect.py # 수정자 카드 효과 분석
│
├── runs/                                  # 학습 결과물
│   ├── dqn_flip7_final.pth               # 최종 학습 모델
│   ├── dqn_flip7_game_*.pth              # 체크포인트 모델들
│   ├── training_history_plot.png         # 학습 곡선
│   ├── training_history_data.csv         # 학습 메트릭 데이터
│   ├── policy_evolution_plot.png         # 정책 진화 그래프
│   ├── policy_evolution_data.csv         # 체크포인트 평가 데이터
│   ├── policy_analysis_card_counting.png # 카드 카운팅 분석 결과
│   ├── policy_analysis_goal_awareness.png # 목표 인식 분석 결과
│   ├── policy_analysis_risk_vs_reward.png # 위험-보상 분석 결과
│   ├── policy_analysis_modifier_effect.png # 수정자 효과 분석 결과
│   └── policy_analysis_high_risk_counting.png # 고위험 시나리오 분석
│
└── README.md                              # 이 파일
```

### 주요 파일 설명

#### 핵심 모듈
- **`config.py`**:
  - 모든 훈련 하이퍼파라미터를 한 곳에서 관리
  - 환경 설정 (게임 종료 보너스 사용 여부 등)
  - 네트워크 아키텍처 파라미터
  - 로깅 및 저장 간격 설정

- **`network.py`**:
  - `QNetwork` 클래스: Dict 관측 공간을 위한 다중 분기 신경망 구조
  - 4개의 독립된 입력 브랜치 (손패 숫자, 손패 수정자, 덱 구성, 총점)
  - 공유 MLP를 통한 Q-value 출력

- **`agent.py`**:
  - `DQNAgent` 클래스: 학습 로직 및 행동 선택
  - `ReplayBuffer` 클래스: 경험 재생 버퍼
  - 타겟 네트워크, ε-greedy 정책, 그래디언트 클리핑 포함

#### 환경 파일
- **`flip_seven_env.py`**: 
  - 전체 게임 로직을 구현한 `FlipSevenCoreEnv` 클래스
  - **`use_end_bonus` 파라미터**: 200점 달성 시 게임 승리 보너스 활성화 여부
  - `gym.spaces.Dict` 관측 공간으로 카드 카운팅 지원
  - 덱 관리, 점수 계산, 멀티 라운드 구조 처리
  - 상태 포함: 손패 숫자 카드, 수정자 카드, 덱 구성, 총점

#### 학습 및 평가 파일
- **`train.py`**: 
  - DQN 에이전트 학습 메인 스크립트 (리팩토링 버전)
  - 모듈화된 구조로 코드 가독성 및 유지보수성 향상
  - `config.py`에서 하이퍼파라미터 자동 로드
  - 학습 곡선 및 메트릭 자동 저장

- **`train_dqn.py`** (레거시):
  - 이전 버전의 통합 학습 스크립트
  - 하위 호환성을 위해 유지

- **`evaluate_dqn.py`**: 
  - 체크포인트 모델들의 성능 평가
  - 정책 진화 과정 시각화
  - 50게임 평균 성능 측정

- **`test_env.py`**: 
  - 환경 동작 검증 스크립트
  - 랜덤 에이전트로 전체 게임 실행

#### 정책 분석 파일
- **`test_policy_scenarios.py`**: 
  - 기본 정책 시나리오 테스트
  - 카드 카운팅, 목표 인식, 위험 관리, 수정자 효과 검증

- **`test_policy_with_card_counting_test.py`**:
  - 모든 숫자 카드(0-12)에 대한 카드 카운팅 학습 검증
  - 덱 상태별 Q-value 비교 시각화

- **`test_policy_with_card_counting_risky.py`**:
  - 고위험 손패({12,11,10,7})로 카드 카운팅 심층 분석
  - Bust 위험 상황에서의 정책 변화 관찰

- **`test_policy_with_goal_awareness_test.py`**:
  - 200점 목표 인식 여부 검증
  - total_game_score 변화에 따른 Q-value 추이 분석

- **`test_policy_with_risk_vs_reward.py`**:
  - 라운드 점수별 위험 감수 성향 분석
  - 정책 전환 지점(crossover point) 탐지

- **`test_policy_with_modifier_card_effect.py`**:
  - 6종 수정자 카드의 영향력 분석
  - 점수 증가에 따른 Q-value 변화 측정

---

## 🎮 사용 방법

### 1. 환경 테스트
먼저 환경이 정상 동작하는지 확인합니다:
```bash
python test_env.py
```

### 2. DQN 에이전트 학습
```bash
python train.py
```

**학습 하이퍼파라미터** (`config.py`에서 수정 가능):
- 총 학습 게임 수: 1000
- 배치 크기: 64
- 학습률: 1e-4
- 할인율 (γ): 0.99
- ε-greedy: 1.0 → 0.01 (decay=0.995)
- 리플레이 버퍼 크기: 50,000
- 타겟 네트워크 업데이트: 매 10게임
- **게임 종료 보너스**: True (200점 달성 시 +100 보상)

**하이퍼파라미터 수정 방법**:
`config.py` 파일을 열어 원하는 값으로 변경:
```python
# config.py 예시
NUM_TOTAL_GAMES_TO_TRAIN = 2000  # 게임 수 증가
LEARNING_RATE = 5e-5              # 학습률 조정
USE_END_BONUS = False             # 게임 종료 보너스 비활성화
```

**출력물**:
- `./runs_end_bonus/dqn_flip7_final.pth`: 최종 모델
- `./runs_end_bonus/dqn_flip7_game_*.pth`: 체크포인트 모델 (매 100게임)
- `./runs_end_bonus/training_history_plot.png`: 학습 곡선
- `./runs_end_bonus/training_history_data.csv`: 학습 메트릭

### 3. 에이전트 평가
```bash
python evaluate_dqn.py
```

체크포인트별 성능 비교:
- 각 모델을 50게임 평가
- 평균 라운드 수 및 최종 점수 측정
- 정책 진화 그래프 생성

### 4. 정책 심층 분석

각 분석 스크립트는 독립적으로 실행 가능하며, 시각화 결과를 `./runs/` 디렉토리에 저장합니다.

#### 카드 카운팅 검증
```bash
# 모든 숫자 카드(0-12) 테스트
python test_policy_with_card_counting_test.py

# 고위험 시나리오 테스트
python test_policy_with_card_counting_risky.py
```

#### 목표 및 위험 관리 분석
```bash
# 200점 목표 인식
python test_policy_with_goal_awareness_test.py

# 위험 대비 보상 평가
python test_policy_with_risk_vs_reward.py
```

#### 수정자 카드 효과
```bash
python test_policy_with_modifier_card_effect.py
```

#### 기본 시나리오 테스트
```bash
python test_policy_scenarios.py
```

---

## 🏗 환경 세부사항

### FlipSevenCoreEnv

#### 초기화 파라미터
```python
env = FlipSevenCoreEnv(use_end_bonus=False)
```

**파라미터**:
- `use_end_bonus` (bool, 기본값: False):
  - `True`: 200점 달성 시 게임 승리 보너스 (+100) 보상에 추가
  - `False`: 기본 동작 (라운드 점수만 보상으로 사용)

#### 관측 공간 (`gym.spaces.Dict`)
```python
{
    "current_hand_numbers": MultiBinary(13),      # 손패의 숫자 카드 (0-12)
    "current_hand_modifiers": MultiBinary(6),     # 손패의 수정자 카드 (6종)
    "deck_composition": Box(0, 12, (19,), int),   # 덱 내 각 카드 개수
    "total_game_score": Box(0, inf, (1,), int)    # 현재 총점
}
```

**카드 카운팅 지원**:
- `deck_composition`은 덱에 남은 19종 카드의 개수를 추적
- 에이전트가 Bust 위험도를 계산하고 최적 전략 수립 가능

#### 행동 공간 (`gym.spaces.Discrete(2)`)
- **0 = Stay**: 라운드 종료, 현재 점수 획득
- **1 = Hit**: 카드 1장 뽑기

#### 보상 구조
- **Stay 선택**: `reward = round_score` (현재 라운드 점수)
- **Hit 후 Bust**: `reward = 0`
- **Hit 후 Flip 7**: `reward = round_score` (자동 종료)
- **Hit 후 계속**: `reward = 0` (라운드 진행 중)

#### 종료 조건
라운드(에피소드)는 다음 중 하나일 때 종료:
- Stay 선택
- Bust 발생 (중복 숫자 뽑기)
- Flip 7 달성 (7개 고유 숫자)

게임은 `total_game_score >= 200`일 때 완료되지만, 환경은 게임 레벨을 관리하지 않습니다. (학습 루프에서 처리)

---

## 🧠 DQN 아키텍처

### QNetwork 구조

Dict 관측 공간을 처리하기 위한 **다중 분기(Multi-branch) 신경망**:

```
관측 입력:
├─ current_hand_numbers (13) ──→ Linear(13→32) + ReLU
├─ current_hand_modifiers (6) ──→ Linear(6→16) + ReLU
├─ deck_composition (19) ────────→ Linear(19→64) + ReLU
└─ total_game_score (1) ─────────→ Linear(1→8) + ReLU

결합 (120차원) ──→ Linear(120→128) + ReLU
                 └→ Linear(128→128) + ReLU
                    └→ Linear(128→2)  [Q(Stay), Q(Hit)]
```

**특징**:
- 각 관측 구성요소를 독립적으로 처리
- 분리된 특징 추출 후 결합하여 최종 Q-value 계산
- 총 파라미터 수: ~20,000개 (경량 네트워크)

### DQNAgent 구성요소

1. **경험 재생 버퍼** (Replay Buffer)
   - 용량: 50,000 transitions
   - 배치 샘플링으로 상관관계 제거

2. **타겟 네트워크** (Target Network)
   - 매 10게임마다 업데이트
   - 학습 안정성 향상

3. **ε-Greedy 정책**
   - 초기: ε = 1.0 (완전 탐험)
   - 최종: ε = 0.01 (거의 활용)
   - 감소율: 0.995/게임

4. **손실 함수**
   - MSE Loss: `(Q(s,a) - [r + γ·max Q(s',a')]²`
   - 옵티마이저: Adam (lr=1e-4)
   - 그래디언트 클리핑: max_norm=10.0

---

## 📊 정책 분석 도구

프로젝트에는 6개의 정책 분석 스크립트가 포함되어 있으며, 각각 학습된 에이전트의 특정 능력을 검증합니다.

### 1. 기본 시나리오 테스트
**파일**: `test_policy_scenarios.py`

4가지 핵심 시나리오 테스트:
- 카드 카운팅 (덱 상태별 Q-value 차이)
- 목표 인식 (200점 근접 시 행동 변화)
- 위험 vs 보상 (점수별 위험 감수 성향)
- 수정자 효과 (수정자 카드의 영향)

### 2. 카드 카운팅 전체 검증
**파일**: `test_policy_with_card_counting_test.py`

- 모든 숫자 카드(0-12) 개별 테스트
- 각 카드에 대해 "덱에 있음" vs "덱에 없음" Q-value 비교
- 시각화: `policy_analysis_card_counting.png`

### 3. 고위험 카드 카운팅
**파일**: `test_policy_with_card_counting_risky.py`

- 고가치 손패({12,11,10,7}, 40점) 사용
- Bust 위험이 높은 상황에서의 정책 변화 관찰
- 시각화: `policy_analysis_high_risk_counting.png`

### 4. 목표 인식 분석
**파일**: `test_policy_with_goal_awareness_test.py`

- 고정 손패(22점)로 total_game_score 0~200 변화
- 200점 목표 근접 시 Stay 선호도 증가 확인
- 시각화: `policy_analysis_goal_awareness.png`

### 5. 위험 대비 보상 평가
**파일**: `test_policy_with_risk_vs_reward.py`

- 12개 손패 시나리오 (3점~47점)
- 점수별 Hit vs Stay 선택 경향 분석
- 정책 전환 지점(crossover point) 탐지
- 시각화: `policy_analysis_risk_vs_reward.png`

### 6. 수정자 카드 효과 분석
**파일**: `test_policy_with_modifier_card_effect.py`

- 기본 손패({10,5})에 7가지 수정자 조합 테스트
- 수정자별 Q-value 변화 측정
- x2 카드의 2배 효과 인식 검증
- 시각화: `policy_analysis_modifier_effect.png`

---

## 🎓 핵심 개념

### 카드 카운팅이 중요한 이유

Flip 7은 **완전 정보 게임**이 아니지만, **관측 가능한 상태**입니다:
- 버린 카드는 라운드 간 재섞이지 않음
- 에이전트가 `deck_composition`을 통해 남은 카드 추적 가능
- 최적 정책은 Bust 확률을 계산하여 Hit/Stay 결정

**예시**:
- 손패에 "12"가 있고 덱에 "12"가 11장 남음 → Hit 위험 높음
- 손패에 "12"가 있지만 덱에 "12"가 0장 → Hit 안전

### 멀티 라운드 게임 구조

**라운드** (에피소드):
- 하나의 `reset()` ~ `terminated=True` 주기
- 단일 점수 획득 기회

**게임** (전체 매치):
- 200점 도달까지의 여러 라운드
- 학습 루프에서 수동 관리
- `total_score`와 덱 상태는 라운드 간 유지

이 구조는 **시간적 신용 할당 문제**를 단순화합니다:
- 각 라운드의 보상이 즉시 제공됨
- 장기적 전략 학습은 `total_game_score`를 통해 암묵적으로 처리

### 보상 구조의 특징

보상은 **희소(sparse)** 하지만 **즉각적(immediate)** 입니다:
- Hit 중에는 보상 0 (라운드 진행 중)
- Stay/Bust/Flip7 시점에만 보상 발생
- 이는 에이전트가 라운드 내 최적 타이밍을 학습하도록 유도

---

## 💡 향후 개선 방향

### 알고리즘 개선
- [ ] **Double DQN**: 과대평가 편향 감소
- [ ] **Dueling DQN**: 상태 가치와 행동 이점 분리
- [ ] **Prioritized Experience Replay**: 중요한 전이에 우선순위 부여
- [ ] **Rainbow DQN**: 여러 개선 기법 통합

### 환경 확장
- [ ] Action 카드 추가 (Freeze, Flip Three, Second Chance)
- [ ] 멀티 플레이어 지원
- [ ] 다양한 점수 목표 및 덱 구성 실험

### 분석 도구 확장
- [ ] 실시간 학습 모니터링 대시보드
- [ ] Q-value 히트맵 시각화
- [ ] 정책 네트워크 해석 가능성 분석
- [ ] 인간 플레이어와의 비교 벤치마크

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

## 🙏 감사의 말

이 프로젝트는 강화학습의 교육적 목적으로 개발되었습니다. Flip 7 게임 메커니즘은 카드 카운팅과 위험 관리를 결합한 흥미로운 학습 환경을 제공합니다.

  [Case B] 덱에 '8'이 전혀 없음 (Bust 불가능)
    Q(Stay):   12.34 | Q(Hit):   18.67
    → 선택: Hit (Q-value 차이: 6.33)

  ✓ 예상 결과: Case B에서 Hit의 Q-value가 더 높아야 함
  ✓ 이는 에이전트가 카드 카운팅을 학습했음을 의미함
```

### d. 환경 테스트

환경이 올바르게 작동하는지 확인하려면:

```bash
python test_env.py
```

이는 랜덤 에이전트로 2번의 전체 게임(200점까지)을 실행하고 환경 로직을 검증합니다.

---

## 🎮 환경 세부사항

### 관측 공간

환경은 4개의 컴포넌트를 가진 **`gym.spaces.Dict`**를 사용합니다:

| 키 | 타입 | 형태 | 설명 |
|-----|------|-------|-------------|
| `current_hand_numbers` | MultiBinary | (13,) | 이진 벡터: 손에 있는 숫자 (0-12) |
| `current_hand_modifiers` | MultiBinary | (6,) | 이진 벡터: 손에 있는 수정자 |
| `deck_composition` | Box | (19,) | 덱에 남아있는 각 카드 타입의 개수 |
| `total_game_score` | Box | (1,) | 모든 라운드에 걸친 누적 점수 |

**왜 Dict 공간인가?**
- **카드 카운팅** 가능: 에이전트가 남은 카드를 추적할 수 있음
- 더 나은 신경망 처리를 위해 다른 정보 타입 분리
- `total_game_score`를 통한 목표 인식 제공

### 행동 공간

Discrete(2):
- **0**: `Stay` - 라운드를 종료하고 점수 저장
- **1**: `Hit` - 덱에서 카드 뽑기

### 보상 구조

- **스텝당 보상**: 0 (라운드 종료 시 제외)
- **라운드 종료 보상**: 
  - 버스트 시 `0`
  - Stay 또는 Flip 7 시 `round_score`
  - 보상은 `total_score`에 추가됨

### 에피소드 구조

**핵심 구분**:
- **에피소드 (라운드)**: `env.step()` 및 `env.reset()`으로 관리
- **게임**: `total_score >= 200`까지의 외부 루프

`env.reset()`은 `total_score`를 리셋하거나 덱을 재섞지 않습니다—다음 라운드를 위해 손패만 비웁니다.

---

## 🧠 DQN 아키텍처

### Q-Network 설계

`QNetwork`는 Dict 관측 공간을 처리하도록 특별히 설계되었습니다:

```python
입력: 4개의 키를 가진 Dict
  ├─ current_hand_numbers (13) → Linear(13, 32) → ReLU
  ├─ current_hand_modifiers (6) → Linear(6, 16) → ReLU
  ├─ deck_composition (19) → Linear(19, 64) → ReLU
  └─ total_game_score (1) → Linear(1, 8) → ReLU
        ↓ (120차원 벡터로 연결)
  공유 MLP:
    → Linear(120, 128) → ReLU
    → Linear(128, 128) → ReLU
    → Linear(128, 2) [Q(Stay), Q(Hit)]
```

**설계 근거**:
- **별도 처리 분기**: 각 관측 컴포넌트를 융합 전에 독립적으로 처리
- **비대칭 은닉 차원**: `deck_composition`에 더 큰 은닉층 (가장 복잡한 특징)
- **공유 MLP**: 모든 특징 간의 상호작용 학습
- **출력**: 2개의 Q-values (행동당 하나)

### 학습 하이퍼파라미터

| 파라미터 | 값 | 목적 |
|-----------|-------|---------|
| Replay Buffer Size | 50,000 | 다양한 경험 저장 |
| Batch Size | 64 | 속도와 안정성 간 균형 |
| Learning Rate | 1e-4 | Adam 옵티마이저로 불안정성 방지 |
| Gamma (γ) | 0.99 | 미래 보상에 대한 강한 고려 |
| Epsilon Decay | 게임당 0.995 | 탐색에서 활용으로 점진적 전환 |
| Target Update | 10게임마다 | 학습 안정화 |
| Min Replay Size | 1,000 | 학습 전 다양성 보장 |

### 주요 DQN 컴포넌트

1. **경험 재생**: 연속적인 경험 간의 상관관계 제거
2. **타겟 네트워크**: 학습 중 Q-value 타겟 안정화
3. **Epsilon-Greedy**: 탐색(랜덤) vs. 활용(그리디) 균형
4. **그래디언트 클리핑**: 기울기 폭발 방지 (max_norm=10.0)

---

## 📊 학습 결과

### 학습 히스토리

![Training History](./runs/training_history_plot.png)

**주요 관찰 사항**:

1. **게임당 라운드** (상단 서브플롯):
   - 초기 성능: 게임당 약 20-25 라운드
   - 최종 성능: 게임당 약 8-10 라운드
   - **개선**: 라운드 수 약 55-60% 감소
   - 수렴: 게임 600-700 근처에서 안정화

2. **게임당 평균 손실** (하단 서브플롯):
   - 높은 초기 분산 (탐색 단계)
   - 약 400게임 후 꾸준한 감소 및 안정화
   - 최종 손실: 약 0.5-1.5 (MSE)
   - 성공적인 Q-value 수렴을 나타냄

### 정책 진화

![Policy Evolution](./runs/policy_evolution_plot.png)

**분석**:
- **게임 100**: 약 15 라운드 (기본 랜덤 정책)
- **게임 400**: 약 11 라운드 (카드 카운팅 학습 중)
- **게임 800**: 약 9 라운드 (거의 최적 전략)
- **최종 모델**: 약 8.5 라운드 (수렴된 정책)

**학습 단계**:
1. **1단계 (게임 0-200)**: 에이전트가 기본 Hit/Stay 결정을 학습하면서 빠른 개선
2. **2단계 (게임 200-600)**: 카드 카운팅 전략이 나타나면서 점진적 개선
3. **3단계 (게임 600-1000)**: 미세 조정 및 안정화

---

## 🔍 정책 분석

### 시나리오 테스트 결과

`test_policy_scenarios.py`의 결과는 에이전트의 학습된 행동을 보여줍니다:

#### 1. 카드 카운팅 테스트 ✅

**설정**: Hand = {8}, '8' 카드가 남아있는/없는 덱 비교

**결과**:
- **덱에 '8' 카드 있음**: `Stay` 선호 (버스트 위험)
- **덱에 '8' 카드 없음**: `Hit` 선호 (버스트 불가능)

**결론**: ✅ 에이전트가 카드 카운팅을 성공적으로 학습함

#### 2. 목표 인식 테스트 ✅

**설정**: Hand = {12, 7, 6} (25점), 총점 변경

**결과**:
- **총점 = 100** (Stay → 125): `Hit` 선호 (더 많은 점수 필요)
- **총점 = 180** (Stay → 205): **강하게** `Stay` 선호 (게임 승리!)

**결론**: ✅ 에이전트가 200점 목표를 인식함

#### 3. 위험 관리 테스트 ✅

**설정**: 낮은 점수 손패 vs. 높은 점수 손패 비교

**결과**:
- **Hand = {5}** (5점): `Hit` 선호 (저장하기엔 너무 낮음)
- **Hand = {12, 11, 10, 7}** (40점): `Stay` 선호 (좋은 점수, 높은 위험)

**결론**: ✅ 에이전트가 현재 점수에 따라 위험 허용도를 조정함

#### 4. 수정자 효과 테스트 ✅

**설정**: Hand = {10, 5}, `x2` 수정자 유무 비교

**결과**:
- **수정자 없음** (15점): 중립 또는 약간 `Hit` 선호
- **x2 있음** (30점): **강하게** `Stay` 선호 (높은 가치)

**결론**: ✅ 에이전트가 수정자 카드를 올바르게 평가함

---

## ⚠️ 비판적 분석

### 강점

1. **✅ 올바른 환경 구현**
   - Gymnasium API와 완전히 호환
   - Game/Round 구분을 적절하게 처리
   - 정확한 점수 계산 및 덱 관리

2. **✅ 정교한 상태 표현**
   - Dict 관측 공간으로 카드 카운팅 가능
   - 모든 필요한 정보를 관측 가능
   - 숨겨진 상태 문제 없음

3. **✅ 효과적인 DQN 아키텍처**
   - Dict 구조를 활용한 다중 분기 설계
   - 작업 복잡도에 적합한 네트워크 용량
   - 타겟 네트워크와 리플레이 버퍼로 안정적인 학습

4. **✅ 성공적인 학습**
   - 성능 약 60% 개선 (25 → 10 라운드)
   - 복잡한 행동 학습 (카드 카운팅, 목표 인식)
   - 안정적인 수렴

5. **✅ 종합적인 분석 도구**
   - 학습 시각화
   - 정책 진화 추적
   - 정성적 시나리오 테스트

### 약점 및 한계

1. **⚠️ 준최적 성능**
   - **현재**: 게임당 약 8-10 라운드
   - **이론적 최적**: 약 6-8 라운드 (덱 확률 기반)
   - **격차**: 에이전트는 좋지만 최적은 아님

2. **⚠️ 보상 구조 문제**
   - **희소 보상**: 라운드 종료 시에만
   - **버스트에 대한 페널티 없음**: 버스트 보상 = 0, 시도하지 않은 것과 동일
   - **잠재적 해결책**: 버스트에 부정적 보상을 주어 신중함 장려

3. **⚠️ 제한적인 탐색 전략**
   - 단순한 epsilon-greedy는 드문 유익한 상태를 탐색하지 못할 수 있음
   - **잠재적 해결책**: Upper Confidence Bound (UCB) 또는 엔트로피 정규화

4. **⚠️ 행동 마스킹 부재**
   - 에이전트가 이론적으로 첫 턴에 "Stay" 가능 (무의미)
   - 환경이 유효하지 않은 행동을 마스킹하지 않음
   - **영향**: 명백히 나쁜 행동에 학습 용량 낭비

5. **⚠️ 샘플 효율성**
   - 수렴에 1000게임 (약 50,000+ 라운드) 필요
   - **잠재적 해결책**: 중요한 전환에 집중하기 위한 우선순위 경험 재생

6. **⚠️ 절제 연구 부족**
   - 각 설계 선택의 기여도 미확인
   - 베이스라인(랜덤, 휴리스틱, 더 단순한 네트워크)과의 비교 없음

### 잠재적 버그/문제

1. **덱 구성 관측**
   - `draw_deck`에 남아있는 카드만 카운트
   - `discard_pile` 또는 현재 손패의 카드를 포함하지 않음
   - **질문**: 의도된 것인가? 전체 덱 지식은 다음과 같아야 함:
     ```
     remaining = initial_deck - (draw_deck + discard_pile + hand)
     ```

2. **Flip 7 처리**
   - Flip 7 발생 시, 라운드가 종료되고 보상이 `total_score`에 추가됨
   - 그러나 `step()` 함수에서 Flip 7에 대해 `self.total_score += reward` 라인도 있음
   - **잠재적 중복 카운팅?** 검증 필요.

3. **Epsilon 감소 타이밍**
   - Epsilon은 게임당 한 번 감소 (스텝당이 아님)
   - 게임당 약 10 라운드로, epsilon이 약 50-100 스텝마다 감소함을 의미
   - **질문**: 너무 느린가? 표준 DQN은 스텝당 감소.

---

## 🚀 향후 개선사항

### 높은 우선순위

1. **보상 형성**
   ```python
   # 현재: 버스트 시 reward = 0
   # 제안: 버스트 시 reward = -5 (나쁜 결정에 대한 페널티)
   ```

2. **우선순위 경험 재생**
   - TD-error로 전환에 가중치 부여
   - "놀라운" 결과에 학습 집중
   - 예상 개선: 약 30% 더 빠른 수렴

3. **행동 마스킹**
   - 턴 1에서 `Stay` 마스킹 (아직 카드를 뽑지 않음)
   - 행동 공간 낭비 감소

4. **하이퍼파라미터 튜닝**
   - 학습률, 배치 크기, 네트워크 깊이에 대한 그리드 검색
   - 현재 파라미터는 합리적이지만 최적화되지 않음

### 중간 우선순위

5. **고급 RL 알고리즘**
   - **Double DQN**: 과대평가 편향 감소
   - **Dueling DQN**: 가치와 이점 스트림 분리
   - **Rainbow DQN**: 여러 개선사항 결합

6. **커리큘럼 학습**
   - 더 쉬운 목표로 시작 (예: 100점)
   - 점진적으로 200점으로 증가
   - 학습 가속화 가능

7. **다중 작업 학습**
   - 여러 목표 점수에서 동시에 학습
   - 일반화 개선

### 낮은 우선순위

8. **절제 연구**
   - 더 단순한 상태 표현 테스트
   - 휴리스틱 베이스라인과 비교
   - 각 네트워크 분기의 기여도 측정

9. **해석 가능성**
   - 어떤 상태 컴포넌트가 중요한지 보기 위한 어텐션 메커니즘
   - 결정 시각화를 위한 현저성 맵

10. **배포 최적화**
    - 더 빠른 추론을 위한 모델 양자화
    - 배포를 위한 ONNX 내보내기

---

## 📚 참고문헌

### 강화학습
- [DQN 논문 (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

### 관련 연구
- Blackjack RL: 유사한 카드 카운팅 문제
- Press-your-luck 게임: 불확실한 환경에서의 위험 관리

---

## 📄 라이선스

이 프로젝트는 교육 목적입니다. 학습 및 연구를 위해 코드를 자유롭게 사용하고 수정하세요.

---

## 👤 작성자

**dae-hany**
- GitHub: [@dae-hany](https://github.com/dae-hany)
- Repository: [flip_seven_reinforcement_learning](https://github.com/dae-hany/flip_seven_reinforcement_learning)

---

## 🙏 감사의 말

- 훌륭한 RL 환경 프레임워크를 제공한 **Gymnasium**
- 딥러닝 인프라를 제공한 **PyTorch**
- 기초 알고리즘을 제공한 DQN 연구 커뮤니티

---

**마지막 업데이트**: 2025년 11월 17일
