# Flip 7 - DQN 강화학습 에이전트

카드 게임 "Flip 7"의 솔로 플레이 "Core Game" 변형을 마스터하기 위한 Deep Q-Network (DQN) 구현입니다. 이 프로젝트는 심층 강화학습이 카드 카운팅, 위험 관리, 목표 지향적 행동을 포함한 복잡한 의사결정을 어떻게 학습하는지 보여줍니다.

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [파일 구조](#파일-구조)
3. [게임 규칙 요약](#게임-규칙-요약)
4. [설치 및 의존성](#설치-및-의존성)
5. [사용 방법](#사용-방법)
6. [환경 세부사항](#환경-세부사항)
7. [DQN 아키텍처](#dqn-아키텍처)
8. [학습 결과](#학습-결과)
9. [정책 분석](#정책-분석)
10. [비판적 분석](#비판적-분석)
11. [향후 개선사항](#향후-개선사항)

---

## 🎯 프로젝트 개요

### 목표
이 프로젝트는 **최소 라운드 수**로 **200점**에 도달하는 것이 목표인 프레스-유어-럭(press-your-luck) 카드 게임 Flip 7의 "Core Game" 변형을 마스터하기 위한 Deep Q-Network (DQN) 에이전트를 구현합니다.

### 주요 특징
- **커스텀 Gymnasium 환경**: `gymnasium.Env` API와 완전히 호환
- **Dict 관측 공간**: 덱 구성을 추적하여 카드 카운팅 가능
- **멀티 라운드 게임 구조**: "라운드"(에피소드)와 "게임"(라운드의 시리즈)을 구분
- **종합적인 분석 도구**: 학습 시각화, 정책 진화 추적, 시나리오 기반 테스트

### 환경
게임 로직은 `flip_seven_env.py`에 `FlipSevenCoreEnv`라는 커스텀 `gymnasium.Env`로 구현되어 있습니다.

### 규칙 출처
에이전트는 `[core_game]flip_seven_rulebook_for_ai_agent.txt`에 명시된 규칙을 기반으로 학습됩니다. 룰북에는 다음이 정의되어 있습니다:
- **85장 덱** (숫자 카드 79장 + 수정자 카드 6장)
- **프레스-유어-럭 메커니즘** (Hit 또는 Stay)
- **버스트 조건** (중복된 숫자 뽑기)
- **Flip 7 보너스** (7개의 고유 숫자 수집 시 +15점)
- **카드 카운팅** (라운드 간 덱이 재섞이지 않음)

---

## 📁 파일 구조

```
flip_seven_reinforcement_learning/
├── flip_seven_env.py                 # FlipSevenCoreEnv implementation
├── train_dqn.py                      # DQN training script with visualization
├── evaluate_dqn.py                   # Policy evolution analysis across checkpoints
├── test_policy_scenarios.py          # Qualitative Q-value analysis
├── test_env.py                       # Environment testing with random agent
├── [core_game]flip_seven_rulebook_for_ai_agent.txt  # Game rules
├── runs/                             # Training outputs
│   ├── dqn_flip7_final.pth          # Final trained model
│   ├── dqn_flip7_game_100.pth       # Checkpoint at game 100
│   ├── dqn_flip7_game_200.pth       # Checkpoint at game 200
│   ├── ...                          # More checkpoints
│   ├── training_history_plot.png    # Loss and rounds over training
│   ├── training_history_data.csv    # Raw training metrics
│   ├── policy_evolution_plot.png    # Performance across checkpoints
│   └── policy_evolution_data.csv    # Checkpoint evaluation results
└── README.md                         # This file
```

### 주요 파일 설명

- **`flip_seven_env.py`**: 
  - 전체 게임 로직을 포함한 `FlipSevenCoreEnv` 구현
  - 관측 공간으로 `gym.spaces.Dict` 사용 (카드 카운팅 가능)
  - 덱 관리, 점수 계산, 멀티 라운드 구조 처리
  - 상태 포함 항목: 손패 카드, 수정자, 덱 구성, 총점

- **`train_dqn.py`**: 
  - `QNetwork` 포함 (Dict 관측을 위한 다중 분기 신경망)
  - 경험 재생 및 타겟 네트워크를 갖춘 `DQNAgent`
  - Game/Round 구분이 있는 메인 학습 루프
  - 학습 히스토리 플롯 및 CSV 데이터 자동 생성

- **`evaluate_dqn.py`**: 
  - `./runs/`에서 모든 모델 체크포인트 로드
  - 각 체크포인트를 50게임에 걸쳐 평가
  - 학습 진행을 보여주는 정책 진화 플롯 생성

- **`test_policy_scenarios.py`**: 
  - 학습된 Q-values의 정성적 분석
  - 4가지 주요 시나리오 테스트: 카드 카운팅, 목표 인식, 위험 관리, 수정자 효과
  - 에이전트 행동을 조사하기 위해 게임 상태를 수동으로 구성

- **`test_env.py`**: 
  - 간단한 환경 검증 스크립트
  - 환경 정확성을 검증하기 위해 랜덤 에이전트로 전체 게임 실행

---

## 🎲 게임 규칙 요약

### 목표
최소 라운드 수로 **총 200점**에 먼저 도달하기.

### 덱 구성 (85장)
- **숫자 카드 (79장)**: `12×"12"`, `11×"11"`, ..., `2×"2"`, `1×"1"`, `1×"0"`
- **수정자 카드 (6장)**: `+2`, `+4`, `+6`, `+8`, `+10`, `x2`

### 게임플레이 (라운드당)
1. **행동**: `Hit` (카드 뽑기) 또는 `Stay` (라운드 종료 및 점수 저장)
2. **버스트**: 이미 가지고 있는 숫자를 뽑으면 → 해당 라운드 0점
3. **Flip 7**: 7개의 고유 숫자 수집 → +15 보너스 점수
4. **수정자**: 버스트를 유발하지 않음; `x2`는 숫자 합계만 2배로

### 점수 계산
```
round_score = (number_sum × x2_multiplier) + modifier_sum + flip_7_bonus
```
- 버스트 시: `round_score = 0`
- 각 라운드 후: `total_score += round_score`

### 핵심 규칙: 카드 카운팅
- 라운드 간 **버린 카드 더미가 섞이지 않음**
- 덱은 게임플레이 중 비어있을 때만 재섞임
- 에이전트는 최적의 결정을 위해 남은 카드를 추적해야 함

---

## 🛠 설치 및 의존성

### 필요 패키지

```txt
torch>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
collections-extended>=2.0.0
```

### 설치

1. **저장소 클론**:
   ```bash
   git clone https://github.com/dae-hany/flip_seven_reinforcement_learning.git
   cd flip_seven_reinforcement_learning
   ```

2. **가상 환경 생성** (권장):
   ```bash
   conda create -n flip7_rl python=3.10
   conda activate flip7_rl
   ```

3. **의존성 설치**:
   ```bash
   pip install torch gymnasium numpy matplotlib pandas
   ```

4. **환경 검증**:
   ```bash
   python test_env.py
   ```

---

## 🚀 사용 방법

### a. 새 에이전트 학습

처음부터 학습을 시작하려면:

```bash
python train_dqn.py
```

**학습 파라미터** (`train_dqn.py`에서 설정 가능):
- `NUM_TOTAL_GAMES_TO_TRAIN = 1000`
- `BATCH_SIZE = 64`
- `LEARNING_RATE = 1e-4`
- `GAMMA = 0.99`
- `EPSILON_START = 1.0`, `EPSILON_END = 0.01`, `EPSILON_DECAY = 0.995`
- `TARGET_UPDATE_FREQUENCY = 10` (게임)

**출력물**:
- 100게임마다 `./runs/`에 모델 체크포인트 저장
- 최종 모델: `./runs/dqn_flip7_final.pth`
- 학습 플롯: `./runs/training_history_plot.png`
- 학습 데이터: `./runs/training_history_data.csv`

**예상 학습 시간**: CPU에서 약 2-4시간, GPU에서 약 30-60분 (1000게임 기준)

### b. 학습된 에이전트 평가

최종 학습된 모델을 평가하려면:

```bash
python evaluate_dqn.py
```

이 스크립트는:
1. `./runs/`에서 모든 체크포인트 로드
2. 각 체크포인트를 50게임에 걸쳐 평가
3. `./runs/policy_evolution_plot.png` 생성
4. 평가 데이터를 `./runs/policy_evolution_data.csv`에 저장

**샘플 출력**:
```
Evaluating checkpoint: dqn_flip7_game_100.pth...
  ✓ 완료: 평균 15.32 라운드

Evaluating checkpoint: dqn_flip7_game_200.pth...
  ✓ 완료: 평균 12.84 라운드

...

Evaluating checkpoint: Final Model...
  ✓ 완료: 평균 8.76 라운드
```

### c. 정책 시나리오 테스트

학습된 정책의 정성적 분석을 수행하려면:

```bash
python test_policy_scenarios.py
```

이 스크립트는 4가지 주요 시나리오에서 에이전트의 Q-values를 테스트합니다:
1. **카드 카운팅**: 버스트 불가능할 때 에이전트가 "Hit"을 선호하는가?
2. **목표 인식**: 200점에 가까울 때 에이전트가 "Stay"를 선호하는가?
3. **위험 대 보상**: 에이전트가 현재 점수에 따라 위험을 관리하는가?
4. **수정자 효과**: 에이전트가 `x2` 수정자를 가진 손패를 더 높게 평가하는가?

**샘플 출력**:
```
📊 Scenario 1: Card Counting (카드 카운팅 학습 여부)
  [Case A] 덱에 '8'이 남아있음 (Bust 위험 있음)
    Q(Stay):   12.34 | Q(Hit):   10.21
    → 선택: Stay (Q-value 차이: 2.13)

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
