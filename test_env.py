import gymnasium as gym
from gymnasium.utils.env_checker import check_env
# Import the custom environment
from flip_seven_env import FlipSevenCoreEnv  # 가정: flip_seven_env.py에 클래스가 있음
import time

def run_random_agent(env, num_games=1):
    """
    Plays a number of full games (to 200 points) with a random agent.
    A "game" consists of multiple "rounds" (episodes) until the
    total score reaches 200.
    """
    
    # 룰북에 정의된 게임 종료 점수
    GAME_END_SCORE = 200 

    for game in range(num_games):
        print(f"\n--- STARTING GAME {game + 1} ---")
        
        # 1. 게임 시작 시 환경을 "완전 리셋"
        #    (total_score, deck, discard_pile 모두 초기화)
        #    env.reset()은 라운드 리셋이므로, 수동으로 리셋합니다.
        
        # 'full_reset' 같은 별도 메서드가 없다면 수동 초기화:
        env.total_score = 0
        env.draw_deck = []
        env.discard_pile = []
        env._initialize_deck() # _initialize_deck이 discard_pile을 채운다고 가정
        # (만약 _initialize_deck이 self.draw_deck을 채운다면 env.draw_deck = env._initialize_deck() )
        
        obs, info = env.reset(seed=42 + game) # 매 게임 다른 시드
        
        game_total_rounds = 0
        game_start_time = time.time()

        # 2. total_score가 200점이 될 때까지 라운드(에피소드) 반복
        while info.get('total_game_score', 0) < GAME_END_SCORE:
            game_total_rounds += 1
            print(f"\n--- Game {game+1}, Round {game_total_rounds} ---")
            
            terminated = False
            round_steps = 0
            
            # 3. 한 라운드(에피소드) 실행
            while not terminated:
                # Take a random action
                action = env.action_space.sample()
                
                print(f"Step: {round_steps}, Action: {'Stay' if action == 0 else 'Hit'}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                env.render()
                print(f"Reward: {reward}")
                print(f"Terminated: {terminated}")
                
                round_steps += 1
                if round_steps > 30: # 안전 브레이크 (수정카드를 모두 뽑는 등)
                    print("Warning: Round exceeded 30 steps, breaking.")
                    break
            
            print(f"End of Round. Final Reward: {reward}")
            print(f"Total Game Score: {info.get('total_game_score')}")
            
            # 다음 라운드를 위해 reset() 호출
            if info.get('total_game_score', 0) < GAME_END_SCORE:
                obs, info = env.reset()

        game_end_time = time.time()
        print(f"\n--- GAME {game + 1} FINISHED ---")
        print(f"Total rounds to reach {GAME_END_SCORE} points: {game_total_rounds}")
        print(f"Final Score: {info.get('total_game_score')}")
        print(f"Time taken: {game_end_time - game_start_time:.2f} seconds")


if __name__ == "__main__":
    # 1. Create the environment
    env = FlipSevenCoreEnv()

    # 2. Check the environment with Gymnasium's checker
    print("Running Gymnasium Environment Checker...")
    passed_check = False
    try:
        check_env(env) # env.unwrapped는 래퍼가 있을 때 사용, 기본은 env
        print("Success! Environment passed the check.")
        passed_check = True
    except Exception as e:
        print("ERROR: Environment failed the check.")
        print(e)

    # 3. Run the random agent test *only if* the check passed
    if passed_check:
        print("\nRunning random agent test (Full Game Simulation)...")
        run_random_agent(env, num_games=2) # 200점 도달 게임 2회 실행