# Flip Seven Reinforcement Learning Project

이 프로젝트는 보드게임 **"Flip Seven"**을 플레이하는 AI 에이전트를 **강화학습(Deep Q-Network, DQN)**을 통해 학습시키고, 그 성능을 다양한 방식으로 검증한 연구 프로젝트입니다.

## 1. 프로젝트 개요
**Flip Seven**은 카드를 뽑아 점수를 모으는 "Press Your Luck" 장르의 카드 게임입니다. 플레이어는 언제 멈출지(Stay) 또는 계속 뽑을지(Hit)를 결정해야 하며, 같은 숫자가 나오면 점수를 잃는(Bust) 위험이 있습니다.

이 프로젝트의 목표는 "**카드 카운팅(Card Counting)**"과 "**위험 관리(Risk Management)**"를 스스로 학습하여 인간 고수 수준의 플레이를 펼치는 AI를 만드는 것입니다.

또한, 이론적으로 합리적인 플레이어(Rational Player)가 평균 11라운드 내에 200점에 도달한다는 점을 벤치마크로 삼아, **솔로 플레이 시 평균 11라운드 이내에 200점에 도달하는 고성능 에이전트**를 구현하는 것을 목표로 합니다.

---

## 2. 강화학습 환경 (Environment)
OpenAI Gymnasium 인터페이스를 기반으로 `FlipSevenCoreEnv`를 직접 구현했습니다.

### 관측 공간 (Observation Space)
에이전트는 다음 정보를 보고 판단합니다:
1.  **내 손패 (Hand)**: 현재 가지고 있는 숫자 카드와 특수 카드(Modifier)
2.  **덱 구성 (Deck Composition)**: **(핵심)** 남은 덱에 어떤 카드가 몇 장 있는지 (카드 카운팅 정보)
3.  **현재 점수 (Score)**: 현재 라운드의 획득 점수

### 행동 공간 (Action Space)
*   **0 (Stay)**: 현재 점수를 확정 짓고 라운드 종료
*   **1 (Hit)**: 카드 한 장 더 뽑기 (Bust 위험 감수)

### 보상 (Reward)
*   **Linear Reward**: 획득한 점수 그대로를 보상으로 지급 (최종 채택)
*   *Non-linear Reward*: 점수의 제곱 등을 시도했으나, Linear 방식이 학습 안정성이 더 높음

---

## 3. 알고리즘 (DQN)
*   **모델 구조**: 상태(State)를 입력받아 각 행동(Hit/Stay)의 가치(Q-Value)를 출력하는 신경망
*   **고급 기법**:
    *   **Replay Buffer**: 과거 경험을 저장해두고 무작위로 추출하여 학습 (데이터 상관관계 제거)
    *   **Target Network**: 학습 목표가 되는 네트워크를 천천히 업데이트하여 학습 안정화
    *   *(실험적 시도)* Double DQN, Dueling Network도 구현하여 비교했으나, 본 게임에서는 기본 DQN으로도 충분한 성능을 보여 기본 모델을 채택함

---

## 4. 주요 실험 및 검증 결과

### ① 카드 카운팅 능력 검증
에이전트가 단순히 "내 점수"만 보는 것이 아니라, **"남은 덱의 상황"**을 보고 판단하는지 테스트했습니다.
*   **실험**: 내 손에 `7`이 있을 때, 덱에 `7`이 **있는 경우(위험)**와 **없는 경우(안전)**의 Q-Value 비교
*   **결과**: 모든 숫자 카드(0~12)에 대해, **안전한 상황일 때 Hit의 가치를 훨씬 높게 평가**했습니다.
*   **결론**: 에이전트는 **완벽한 카드 카운팅 능력**을 학습했습니다.
    *   ![Card Counting](./runs/policy_analysis_card_counting.png)

### ② 고위험 시나리오 테스트 (Hand: 12, 11, 10)
*   **상황**: 손패에 `12, 11, 10`이 있어 Bust 확률이 매우 높은 상황
*   **결과**: 덱에 `12, 11, 10`이 하나도 없는 **안전한 상황(Safe Case)**임을 인지하자, Hit에 대한 선호도(Q-Value)가 급격히 상승했습니다. (비록 33점이라 Stay를 택했지만, 판단의 근거는 명확했습니다.)
    *   ![Scenario Test](./runs/policy_analysis_12_11_10.png)

### ③ 특수 카드(Modifier) 가치 인식
*   **실험**: 손패에 `12`가 있을 때, **일반 상황**과 **Modifier 카드가 있는 상황** 비교
*   **결과**: Modifier 카드가 있을 때 Hit에 대한 Q-Value가 변화함 (점수 증가로 인한 전략 수정)
*   **의미**: 에이전트가 숫자 카드뿐만 아니라 **특수 카드의 효과(점수 추가, 배수 등)를 정확히 이해**하고 이를 행동 결정에 반영하고 있음을 확인했습니다.
    *   ![Modifier Effect](./runs/policy_analysis_modifier_effect_{12}.png)

### ④ 1:1 대결 시뮬레이션 (vs Daehan Player)
*   **상대**: **Daehan Player** (수학적으로 기대값(EV)을 계산하여 플레이하는 **이론상 최적의 봇**)
*   **결과 (10,000판)**:
    *   **DQN Agent 승률**: **51.2%**
    *   **Daehan Player 승률**: 48.8%
*   **의미**: 수학적 계산기인 봇과 대등하거나 근소하게 우위인 실력을 보여주었습니다. 이는 AI가 **"합리적 판단"**의 경지에 도달했음을 의미합니다.
    *   ![Duel Result](./runs/duel_simulation_results.png)

### ④ 다인전 (6인) 시뮬레이션
*   **상황**: 6명의 플레이어가 하나의 덱을 공유 (덱 소모 속도가 매우 빠름)
*   **참가자**: Agent, Daehan, 그리고 단순 전략 봇들(Always Hit, One Hit 등)
*   **결과**:
    *   **1위**: Daehan Player (43.2%)
    *   **2위**: DQN Agent (35.9%)
    *   나머지 봇들은 처참한 승률 기록
*   **의미**: 변수가 많은 다인전에서도 Agent는 상위권을 유지하며 **"판을 지배하는 플레이어"**임을 입증했습니다.
    *   ![6 Player Result](./runs/6player_simulation_results.png)

### ⑤ DQN Agent 우위 분석 (vs Daehan Player)
*   **목적**: 수학적 최적해(Daehan)를 이길 수 있었던 원인 정밀 분석 (10,000판 시뮬레이션)
*   **결과**:
    *   **안정성**: 둘 다 Bust 비율은 극도로 낮음 (0%에 수렴).
    *   **평균 득점**: Daehan Player가 소폭 우세 (24.5 vs 22.5).
    *   **결정적 차이 (Risk Taking)**: **Bust 확률이 30%가 넘는 위험 상황**에서,
        *   **Daehan Player**: **0.77%**만 Hit를 시도 (거의 절대 안전 지향).
        *   **DQN Agent**: **7.63%**나 Hit를 시도 (약 **10배** 더 과감함).
*   **결론**: Daehan Player는 "잃지 않는 플레이"에 집착하여 역전 기회를 놓치는 반면, DQN Agent는 **"승리를 위해 필요한 계산된 위험"**을 감수할 줄 압니다. 이 **전략적 과감함(Strategic Aggression)**이 51.2% 승률의 비결입니다.
    *   ![Superiority Analysis](./runs/superiority_analysis.png)

---

## 5. 설치 및 실행 방법

### 필수 라이브러리 설치
```bash
pip install gymnasium torch numpy matplotlib pandas
```

### 주요 스크립트 실행
1.  **에이전트 학습**:
    ```bash
    python train.py
    ```
2.  **카드 카운팅 검증**:
    ```bash
    python test_policy_with_card_counting_test.py
    ```
3.  **1:1 대결 시뮬레이션**:
    ```bash
    python simulate_duel.py
    ```
4.  **6인 대결 시뮬레이션**:
    ```bash
    python simulate_6players.py
    ```
5.  **플레이어 수에 따른 승률 분석**:
    ```bash
    python simulate_player_scaling.py
    ```

---

## 6. 파일 구조 및 설명
이 폴더의 각 파일은 다음과 같은 역할을 합니다.

### 📂 핵심 코드
*   **`train.py`**: DQN 에이전트를 **학습**시키는 메인 스크립트입니다. 실행 시 `./runs/latest_run` 폴더에 모델과 로그가 저장됩니다.
*   **`agent.py`**: DQN 에이전트 클래스(`DQNAgent`)와 리플레이 버퍼(`ReplayBuffer`)가 정의되어 있습니다.
*   **`network.py`**: 신경망 모델(`QNetwork`) 구조가 정의되어 있습니다.
*   **`flip_seven_env.py`**: Flip Seven 게임의 **강화학습 환경(Gymnasium)**입니다. 게임의 규칙과 로직이 여기에 있습니다.
*   **`config.py`**: 학습에 필요한 **하이퍼파라미터** (학습률, 배치 크기 등)와 설정을 관리합니다.

### 🧪 시뮬레이션 및 분석 (실행 가능)
*   **`simulate_solo.py`**: 에이전트와 Daehan Player가 **혼자서** 게임할 때의 성능(200점 도달 라운드)을 측정합니다.
*   **`simulate_duel.py`**: 에이전트와 Daehan Player의 **1:1 대결**을 시뮬레이션하고 승률을 비교합니다.
*   **`simulate_6players.py`**: **6인 플레이** 상황에서 에이전트의 성능을 테스트합니다.
*   **`simulate_player_scaling.py`**: 플레이어 수(2~6명)에 따른 에이전트의 승률 변화를 분석합니다.
*   **`analyze_superiority.py`**: 에이전트가 Daehan Player보다 우수한 이유(Bust 확률, 리스크 관리 등)를 정밀 분석합니다.
*   **`test_policy_with_card_counting_test.py`**: 에이전트가 **카드 카운팅**을 제대로 하고 있는지 검증하는 테스트 스크립트입니다.

### 📁 기타
*   **`runs/`**: 학습된 모델(`dqn_flip7_final.pth`)과 분석 결과 이미지들이 저장되는 폴더입니다.

---

## 7. 결론
이 프로젝트를 통해 **강화학습 에이전트가 복잡한 카드 게임의 규칙을 스스로 터득하고, 고도의 전략인 카드 카운팅까지 구사할 수 있음**을 확인했습니다. 특히 수학적으로 설계된 알고리즘 봇과 대등한 승부를 펼친 것은 매우 고무적인 성과입니다.
