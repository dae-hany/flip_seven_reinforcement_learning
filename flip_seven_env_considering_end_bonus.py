import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import collections

# --- 관측 공간 정의 ---
# 숫자 카드는 0 ~ 12까지 총 13 종류
NUMBER_CARD_TYPES = [str(i) for i in range(13)]
# 수정자 카드는 +2, +4, +6, +8, +10, x2 총 6 종류
MODIFIER_CARD_TYPES = ["+2", "+4", "+6", "+8", "+10", "x2"]
# 총 19 종류의 카드
ALL_CARD_TYPES = NUMBER_CARD_TYPES + MODIFIER_CARD_TYPES

# 관측 벡터를 위한 매핑 딕셔너리 생성
# e.g., {"0": 0, "1": 1, ..., "12": 12, "+2": 13, ..., "x2": 18}
CARD_TO_IDX = {card: i for i, card in enumerate(ALL_CARD_TYPES)}
MODIFIER_TO_IDX = {card: i for i, card in enumerate(MODIFIER_CARD_TYPES)}
# ----------------------------------------

# [수정 1] 게임 승리 보너스 및 목표 점수 상수 정의
GAME_GOAL_SCORE = 200
GAME_WIN_BONUS = 100.0  # (하이퍼파라미터: 50.0, 100.0 등 자유롭게 설정)

class FlipSevenCoreEnv(gym.Env):
    """
    [core_game]flip_seven_rulebook_for_ai_agent.txt를
    기반으로 하는 gynamsium 환경 구현
    
    이 환경은 다음을 구분함:
    - 에피소드 (라운드): step() 및 reset()으로 관리됨.
    - 게임 (전체 매치): 외부에서 관리됨. reset()은 total_score
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        # 1. 행동 공간 정의
        # 0: Stay(라운드를 종료하고 점수를 획득)
        # 1: Hit(카드를 한 장 뽑음)
        self.action_space = spaces.Discrete(2)

        # 2. 관측 공간 정의 
        # 명확성을 위해 gym.spaces.Dict 사용
        self.observation_space = spaces.Dict({
            # 'current_hand_numbers': 손패에 있는 숫자 카드들
            # 13 slots, 0 = 손에 없음, 1 = 손에 있음.
            "current_hand_numbers": spaces.MultiBinary(13),

            # 'current_hand_modifiers': 손패에 있는 수정자 카드들 (+2 ~ +10, x2)
            # 6 slots (use MODIFIER_TO_IDX), 0 = 손에 없음, 1 = 손에 있음.
            "current_hand_modifiers": spaces.MultiBinary(6),

            # 'deck_composition': 카드 카운팅을 위한 덱 구성 정보
            # draw deck에 남아 있는 19종 카드 각각의 개수
            # Box를 사용하여 low=0, high=12 (최대 '12' 카드 개수)
            "deck_composition": spaces.Box(low=0, high=12, shape=(19,), dtype=np.int32),

            # 'total_game_score': 현재까지 획득한 총 점수
            "total_game_score": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })

        # --- 내부 환경 상태 ---
        self.draw_deck = collections.deque() # 효율적인 pop을 위해 deque 사용
        self.discard_pile = []
        self.total_score = 0
        
        # 이 헬퍼 함수는 self.discard_pile에 85장의 모든 카드를 채움.
        self._initialize_deck_to_discard() 

        # --- 라운드(에피소드)별 상태  ---
        self.current_numbers_in_hand = set()
        self.current_modifiers_in_hand = []

    def _initialize_deck_to_discard(self):
        """
        85장의 모든 카드를 생성 (규칙 2.1, 2.2)
        그리고 self.discard_pile에 저장함.
        """
        self.discard_pile = []
        # 규칙 2.1: 숫자 카드 (총 79장)
        for i in range(1, 13):
            self.discard_pile.extend([str(i)] * i)
        self.discard_pile.append("0") # 1x "0" 카드
        
        # 규칙 2.2: 수정자 카드 (총 6장)
        self.discard_pile.extend(MODIFIER_CARD_TYPES)
        
        # 85장의 카드가 생성되었는지 확인
        assert len(self.discard_pile) == 85

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """관측 딕셔너리를 구성하는 헬퍼 함수."""
        # 1. current_hand_numbers
        hand_num_obs = np.zeros(13, dtype=np.int32)
        for num_val in self.current_numbers_in_hand:
            hand_num_obs[num_val] = 1

        # 2. current_hand_modifiers
        hand_mod_obs = np.zeros(6, dtype=np.int32)
        for mod_str in self.current_modifiers_in_hand:
            hand_mod_obs[MODIFIER_TO_IDX[mod_str]] = 1

        # 3. deck_composition (Card Counting)
        deck_comp_obs = np.zeros(19, dtype=np.int32)
        for card_str in self.draw_deck:
            deck_comp_obs[CARD_TO_IDX[card_str]] += 1

        # 4. total_game_score
        total_score_obs = np.array([self.total_score], dtype=np.int32)

        return {
            "current_hand_numbers": hand_num_obs,
            "current_hand_modifiers": hand_mod_obs,
            "deck_composition": deck_comp_obs,
            "total_game_score": total_score_obs
        }

    def _get_info(self) -> Dict[str, Any]:
        """정보 딕셔너리를 구성하는 헬퍼 함수."""
        return {
            "total_game_score": self.total_score,
            "current_round_score_if_stay": self._calculate_score(bust=False),
            "cards_in_deck": len(self.draw_deck),
            "cards_in_discard": len(self.discard_pile)
        }

    def _calculate_score(self, bust: bool) -> int:
        """
        현재 라운드의 점수를 계산하는 헬퍼 함수 (규칙 4에 기반).
        """
        # 규칙 4: "Bust"인 경우, round_score = 0.
        if bust:
            return 0

        # 규칙 4.1: 숫자 카드의 값 합산
        number_sum = sum(self.current_numbers_in_hand)
        
        # 규칙 4.2: x2 배수 적용 (존재하는 경우)
        # "x2는 다른 수정자 카드에서 얻은 점수를 두 배로 만들지 않습니다."
        if "x2" in self.current_modifiers_in_hand:
            number_sum = number_sum * 2
            
        # 규칙 4.3: 추가 보너스 점수 합산
        modifier_sum = 0
        for mod_str in self.current_modifiers_in_hand:
            if mod_str.startswith("+"):
                modifier_sum += int(mod_str[1:]) # e.g., "+10" -> 10
        
        # 규칙 4.4: Flip 7 보너스 점수 합산
        flip_7_bonus = 0
        if len(self.current_numbers_in_hand) == 7:
            flip_7_bonus = 15

        # 규칙 4.5: 최종 점수 계산
        round_score = number_sum + modifier_sum + flip_7_bonus
        return int(round_score)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        새로운 라운드(에피소드)를 위해 환경을 리셋합니다.
        핵심 설계 원칙에 따라, 이 메서드는 total_score나 덱/버림 더미를 리셋하지 않습니다.
        단지 손패만 초기화합니다.
        -> 따라서 카드 카운팅이 가능합니다.
        """
        super().reset(seed=seed)

        # 1. 라운드별 상태 초기화
        self.current_numbers_in_hand = set()
        self.current_modifiers_in_hand = []

        # 2. 라운드 시작 시 덱이 비어있는지 확인
        #    (첫 라운드이거나, 이전 라운드의 마지막 카드로 덱이 완전히 비워진 경우 발생)
        if not self.draw_deck:
            self._shuffle_discard_into_deck()

        # 3. 관측치와 정보 얻기
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _shuffle_discard_into_deck(self):
        """
        규칙 5.2를 구현합니다.
        버림 더미를 섞어 새로운 뽑기 덱을 만듭니다.
        """
        self.draw_deck = collections.deque(self.discard_pile)
        self.discard_pile = []
        self.np_random.shuffle(self.draw_deck)

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        현재 라운드 내에서 한 번의 타임 스텝(턴)을 실행합니다.
        """
        terminated = False # 에피소드(라운드) 종료 여부
        truncated = False  # 이 게임에서는 사용하지 않음
        reward = 0.0

        # [수정] : 승리 보너스 추가
        # ACTION 0: STAY (규칙 3.4)
        if action == 0:
            terminated = True
            # 1. 라운드 점수 계산
            round_score = self._calculate_score(bust=False)
            reward = round_score  # 기본 보상은 라운드 점수

            # [수정 2] 승리 보너스 체크
            current_total_score = self.total_score
            new_total_score = current_total_score + round_score

            # 200점을 넘기지 못한 상태에서, 이번 라운드 점수로 200점을 넘긴 경우
            if current_total_score < GAME_GOAL_SCORE and new_total_score >= GAME_GOAL_SCORE:
                reward += GAME_WIN_BONUS # 보상에 승리 보너스 추가

            # 3. 실제 총점 업데이트 (보너스가 아닌 순수 라운드 점수만)
            self.total_score += round_score

        # ACTION 1: HIT (규칙 3.3)
        elif action == 1:
            # 1. 덱이 비어있는지 확인 (규칙 5.2)
            if not self.draw_deck:
                self._shuffle_discard_into_deck()

            # 2. 카드 한 장 뽑기
            drawn_card_str = self.draw_deck.popleft() # 덱 맨 위의 카드 뽑기 
            
            # 3. 카드 유형에 따라 처리
            try:
                # 숫자 카드인지 확인 (예: "8", "0", "12")
                card_value = int(drawn_card_str)
                is_number_card = True
            except ValueError:
                # 수정자 카드인지 확인 (예: "+4", "x2")
                card_value = None
                is_number_card = False

            # --- 3a. 숫자 카드일 경우 ---
            if is_number_card:
                # BUST인지 확인 (규칙 3.3)
                if card_value in self.current_numbers_in_hand:
                    terminated = True
                    reward = self._calculate_score(bust=True) # 0 points
                
                # BUST가 아닌 경우
                else:
                    self.current_numbers_in_hand.add(card_value)
                    # [수정] 승리 보너스 추가
                    # FLIP 7인지 확인 (규칙 3.3)
                    if len(self.current_numbers_in_hand) == 7:
                        terminated = True
                        # 1. 라운드 점수 계산
                        round_score = self._calculate_score(bust=False) # +15 포함
                        reward = round_score # 기본 보상

                        # [수정 3] 승리 보너스 체크 (Stay와 동일한 로직)
                        current_total_score = self.total_score
                        new_total_score = current_total_score + round_score

                        if current_total_score < GAME_GOAL_SCORE and new_total_score >= GAME_GOAL_SCORE:
                            reward += GAME_WIN_BONUS
                        
                        # 3. 실제 총점 업데이트
                        self.total_score += round_score
                    else:
                        # 라운드 계속, 이번 스텝의 보상은 0
                        reward = 0.0
            
            # --- 3b. 수정자 카드일 경우 ---
            else:
                self.current_modifiers_in_hand.append(drawn_card_str)
                # 라운드 계속, 이번 스텝의 보상은 0
                reward = 0.0
        
        # --- End of Step Logic ---
        
        # 규칙 5.1 : 라운드 종료 시 손패를 버림 더미로 이동
        if terminated:
            # 숫자 카드들을 버림 더미에 추가
            for num_val in self.current_numbers_in_hand:
                self.discard_pile.append(str(num_val))
            # 수정자 카드들을 버림 더미에 추가
            self.discard_pile.extend(self.current_modifiers_in_hand)

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print("---")
            print(f"  Total Game Score: {self.total_score}")
            print(f"  Current Hand Numbers: {sorted(list(self.current_numbers_in_hand))}")
            print(f"  Current Hand Modifiers: {self.current_modifiers_in_hand}")
            print(f"  Score (if 'Stay'): {self._calculate_score(bust=False)}")
            print(f"  Cards in Deck: {len(self.draw_deck)} | Cards in Discard: {len(self.discard_pile)}")

    def close(self):
        pass
