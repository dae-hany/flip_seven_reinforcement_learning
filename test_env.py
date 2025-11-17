import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import time
import collections  # <--- 이 줄이 추가되었습니다.

# flip_seven_env.py 파일에서 FlipSevenCoreEnv 클래스를 임포트합니다.
try:
    from flip_seven_env import FlipSevenCoreEnv
except ImportError:
    print("="*50)
    print("오류: 'flip_seven_env.py' 파일에서 'FlipSevenCoreEnv' 클래스를 찾을 수 없습니다.")
    print("두 파일이 같은 디렉토리에 있는지 확인하세요.")
    print("="*50)
    exit()

def run_full_game_test(env: gym.Env, num_games: int = 2):
    """
    환경을 가지고 '무작위 에이전트'로 200점에 도달하는
    '풀 게임(Full Game)'을 여러 번 실행합니다.
    
    이 테스트는 env.reset()이 '라운드'를 리셋하고,
    '게임' 루프가 'total_score'를 관리하는지 검증합니다.
    """
    
    GAME_END_SCORE = 200 # 룰북에 명시된 게임 종료 점수

    print(f"\n--- {num_games}회의 풀 게임(200점 도달) 테스트 시작 ---")

    for game in range(num_games):
        print(f"\n=========================================")
        print(f" 🎲 [ 게임 {game + 1} 시작 ] 🎲")
        print(f"=========================================")
        game_start_time = time.time()
        
        # --- 1. '게임' 시작 시 수동으로 '전체' 상태 초기화 ---
        # env.reset()은 '라운드'만 초기화하므로,
        # '게임'을 새로 시작하기 위해 '전체' 상태를 강제로 리셋합니다.
        env.total_score = 0
        env.draw_deck = collections.deque() # 이제 'collections'가 정의되었습니다.
        env.discard_pile = []
        env._initialize_deck_to_discard() # discard_pile을 85장으로 채움
        
        # 첫 라운드를 위해 env.reset() 호출
        # (이때 _shuffle_discard_into_deck()이 호출될 것입니다)
        obs, info = env.reset(seed=42 + game)
        
        game_total_rounds = 0

        # --- 2. '게임' 루프 (200점에 도달할 때까지) ---
        while info.get("total_game_score", 0) < GAME_END_SCORE:
            game_total_rounds += 1
            print(f"\n--- [ 라운드 {game_total_rounds} | 현재 총 점수: {info.get('total_game_score', 0)} ] ---")
            
            terminated = False # '라운드' 종료 플래그
            round_step_count = 0

            # --- 3. '라운드' 루프 (Bust, Stay, Flip 7 전까지) ---
            while not terminated:
                round_step_count += 1
                
                # 무작위 행동 선택 (0: Stay, 1: Hit)
                action = env.action_space.sample() 
                
                print(f"  (스텝 {round_step_count:02d}) 행동: {'STAY' if action == 0 else 'HIT'}", end=" | ")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # '인간' 모드로 렌더링 (현재 손패, 덱 상태 등 출력)
                # env.render() # 너무 길면 주석 처리
                
                print(f"손패: {sorted(list(env.current_numbers_in_hand))}", end=" | ")
                print(f"수정: {env.current_modifiers_in_hand}")

                if terminated:
                    print(f"  >>> 라운드 종료! <<<")
                    if reward == 0:
                        print(f"  결과: BUST! 💥")
                    else:
                        print(f"  결과: 점수 획득! 💰 (이번 라운드 보상: {reward})")
            
            # --- 라운드 종료 후 다음 라운드 준비 ---
            if info.get("total_game_score", 0) < GAME_END_SCORE:
                # 다음 라운드를 위해 reset() 호출 (손패만 비워짐)
                obs, info = env.reset()

        # --- 게임 종료 ---
        game_end_time = time.time()
        print(f"\n=========================================")
        print(f" 🏆 [ 게임 {game + 1} 종료! ] 🏆")
        print(f"  - 최종 점수: {info.get('total_game_score', 0)} 점")
        print(f"  - 200점 도달까지 걸린 라운드: {game_total_rounds} 라운드")
        print(f"  - 소요 시간: {game_end_time - game_start_time:.2f} 초")
        print(f"=========================================")


if __name__ == "__main__":
    
    print("1. FlipSevenCoreEnv 환경 인스턴스 생성 중...")
    try:
        env = FlipSevenCoreEnv()
        print("   [성공] 환경이 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"  [실패] 환경 생성 중 오류 발생: {e}")
        exit()

    # 2. Gymnasium 표준 환경 검사 (API 준수 여부)
    print("\n2. Gymnasium 환경 검사기(check_env) 실행 중...")
    passed_check = False
    try:
        check_env(env)
        print("   [성공] ⭐️ Gymnasium API 규격을 완벽하게 준수합니다! ⭐️")
        passed_check = True
    except Exception as e:
        print(f"  [실패] 환경 검사 실패. 오류: {e}")
        print("       환경 코드에 수정이 필요할 수 있습니다.")

    # 3. 환경 검사를 통과한 경우에만 무작위 에이전트 테스트 실행
    if passed_check:
        run_full_game_test(env, num_games=2)