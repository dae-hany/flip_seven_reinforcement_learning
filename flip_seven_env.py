import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import collections

# --- Constants for Observation Space ---
# 13 Number Cards (0-12)
NUMBER_CARD_TYPES = [str(i) for i in range(13)]
# 6 Modifier Cards
MODIFIER_CARD_TYPES = ["+2", "+4", "+6", "+8", "+10", "x2"]
# Total 19 unique card types
ALL_CARD_TYPES = NUMBER_CARD_TYPES + MODIFIER_CARD_TYPES

# Create mapping dictionaries for the observation vector
# e.g., {"0": 0, "1": 1, ..., "12": 12, "+2": 13, ..., "x2": 18}
CARD_TO_IDX = {card: i for i, card in enumerate(ALL_CARD_TYPES)}
MODIFIER_TO_IDX = {card: i for i, card in enumerate(MODIFIER_CARD_TYPES)}
# ----------------------------------------

class FlipSevenCoreEnv(gym.Env):
    """
    A Gymnasium environment for the "Core Game" of Flip 7, based *only* on
    [core_game]flip_seven_rulebook_for_ai_agent.txt.
    
    This environment distinguishes between:
    - Episode (Round): Managed by step() and reset().
    - Game (Full Match): Managed externally. reset() does NOT reset total_score.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        # 1. DEFINE ACTION SPACE (Rule 3.2)
        # 0: Stay
        # 1: Hit
        self.action_space = spaces.Discrete(2)

        # 2. DEFINE OBSERVATION SPACE (STATE)
        # Use a gym.spaces.Dict for clarity.
        self.observation_space = spaces.Dict({
            # 'current_hand_numbers': Which numbers (0-12) are in hand.
            # 13 slots, 0 = not in hand, 1 = in hand.
            "current_hand_numbers": spaces.MultiBinary(13),

            # 'current_hand_modifiers': Which modifiers (+2 to +10, x2) are in hand.
            # 6 slots (use MODIFIER_TO_IDX), 0 = not in hand, 1 = in hand.
            "current_hand_modifiers": spaces.MultiBinary(6),

            # 'deck_composition': "Card Counting" state.
            # Count of each of the 19 card types *remaining in the draw deck*.
            # Use Box with low=0 and high=12 (max count for '12' card).
            "deck_composition": spaces.Box(low=0, high=12, shape=(19,), dtype=np.int32),

            # 'total_game_score': Cumulative score across all rounds.
            "total_game_score": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })

        # --- Internal Environment State ---
        self.draw_deck = collections.deque() # Use deque for efficient pop
        self.discard_pile = []
        self.total_score = 0
        
        # This helper populates self.discard_pile with all 85 cards.
        self._initialize_deck_to_discard() 

        # --- Per-Round (Episode) State ---
        self.current_numbers_in_hand = set()
        self.current_modifiers_in_hand = []

    def _initialize_deck_to_discard(self):
        """
        Creates a list of all 85 cards (Rule 2.1, 2.2)
        and stores it in self.discard_pile.
        """
        self.discard_pile = []
        # Rule 2.1: Number Cards (79 total)
        for i in range(1, 13):
            self.discard_pile.extend([str(i)] * i)
        self.discard_pile.append("0") # 1x "0" card
        
        # Rule 2.2: Modifier Cards (6 total)
        self.discard_pile.extend(MODIFIER_CARD_TYPES)
        
        # Ensure 85 cards are created
        assert len(self.discard_pile) == 85

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Helper function to construct the observation dictionary."""
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
        """Helper function to construct the info dictionary."""
        return {
            "total_game_score": self.total_score,
            "current_round_score_if_stay": self._calculate_score(bust=False),
            "cards_in_deck": len(self.draw_deck),
            "cards_in_discard": len(self.discard_pile)
        }

    def _calculate_score(self, bust: bool) -> int:
        """
        Calculates the score for the current round based on rules section 4.
        """
        # Rule 4: If "Bust", round_score = 0.
        if bust:
            return 0

        # Rule 4.1: Add value of number cards
        number_sum = sum(self.current_numbers_in_hand)
        
        # Rule 4.2: Apply x2 multiplier (if present)
        # "x2 does not double the points gained from other Modifier cards."
        if "x2" in self.current_modifiers_in_hand:
            number_sum = number_sum * 2
            
        # Rule 4.3: Add additional bonus points
        modifier_sum = 0
        for mod_str in self.current_modifiers_in_hand:
            if mod_str.startswith("+"):
                modifier_sum += int(mod_str[1:]) # e.g., "+10" -> 10
        
        # Rule 4.4: Add Flip 7 bonus
        flip_7_bonus = 0
        if len(self.current_numbers_in_hand) == 7:
            flip_7_bonus = 15

        # Rule 4.5: Final score
        round_score = number_sum + modifier_sum + flip_7_bonus
        return int(round_score)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets the environment for a NEW ROUND (Episode).
        As per the Core Design Principle, this method does NOT reset
        total_score or the deck/discard piles. It only clears the hand.
        """
        super().reset(seed=seed)

        # 1. Clear per-round state
        self.current_numbers_in_hand = set()
        self.current_modifiers_in_hand = []

        # 2. Check if deck is empty at the *start* of a round
        #    (This happens on the very first round, or if the deck was
        #    perfectly emptied by the last card of the previous round).
        if not self.draw_deck:
            self._shuffle_discard_into_deck()

        # 3. Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _shuffle_discard_into_deck(self):
        """
        Implements Rule 5.2.
        Shuffles the discard pile to create a new draw deck.
        """
        self.draw_deck = collections.deque(self.discard_pile)
        self.discard_pile = []
        # Use the Gymnasium-provided RNG
        self.np_random.shuffle(self.draw_deck)

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step (turn) within the current round.
        """
        terminated = False # An episode (round) ends
        truncated = False  # Not used in this game
        reward = 0.0

        # ACTION 0: STAY (Rule 3.4)
        if action == 0:
            terminated = True
            # Calculate final score for the round (not a bust)
            reward = self._calculate_score(bust=False)
            self.total_score += reward

        # ACTION 1: HIT (Rule 3.3)
        elif action == 1:
            # 1. Check if deck is empty (Rule 5.2)
            if not self.draw_deck:
                self._shuffle_discard_into_deck()

            # 2. Draw a card
            drawn_card_str = self.draw_deck.popleft() # Draw from top
            
            # 3. Process card based on type
            try:
                # Check if it's a Number Card (e.g., "8", "0", "12")
                card_value = int(drawn_card_str)
                is_number_card = True
            except ValueError:
                # It's a Modifier Card (e.g., "+4", "x2")
                card_value = None
                is_number_card = False

            # --- 3a. If NUMBER CARD ---
            if is_number_card:
                # Check for BUST (Rule 3.3)
                if card_value in self.current_numbers_in_hand:
                    terminated = True
                    reward = self._calculate_score(bust=True) # 0 points
                
                # If SAFE
                else:
                    self.current_numbers_in_hand.add(card_value)
                    # Check for FLIP 7 (Rule 3.3)
                    if len(self.current_numbers_in_hand) == 7:
                        terminated = True
                        reward = self._calculate_score(bust=False) # Will include +15
                        self.total_score += reward
                    else:
                        # Round continues, reward is 0 for this step
                        reward = 0.0
            
            # --- 3b. If MODIFIER CARD ---
            else:
                self.current_modifiers_in_hand.append(drawn_card_str)
                # Round continues, reward is 0 for this step
                reward = 0.0
        
        # --- End of Step Logic ---
        
        # Rule 5.1: If the round ended (Stay, Bust, or Flip 7),
        # move all hand cards to the discard pile.
        if terminated:
            # Add all number cards from set to discard
            for num_val in self.current_numbers_in_hand:
                self.discard_pile.append(str(num_val))
            # Add all modifier cards from list to discard
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
