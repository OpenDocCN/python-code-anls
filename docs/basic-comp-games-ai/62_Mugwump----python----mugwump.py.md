# `basic-computer-games\62_Mugwump\python\mugwump.py`

```

# 从 math 模块中导入 sqrt 函数
# 从 random 模块中导入 randint 函数
# 从 typing 模块中导入 List 和 Tuple 类型
def introduction() -> None:
    # 打印游戏介绍
    print(
        """The object of this game is to find 4 mugwumps
hidden on a 10*10 grid.  Homebase is position 0,0.
Any guess you make must be two numbers with each
number between 0 and 9 inclusive.  First number
is distance to right of homebase, and second number
is the distance above homebase."""
    )
    # 打印游戏规则
    print(
        """You get 10 tries.  After each try, I will tell
you how far you are from each mugwump."""
    )


# 生成随机位置的 mugwumps
def generate_mugwumps(n: int = 4) -> List[List[int]]:
    mugwumps = []
    for _ in range(n):
        current = [randint(0, 9), randint(0, 9)]
        mugwumps.append(current)
    return mugwumps


# 打印 mugwumps 的位置
def reveal_mugwumps(mugwumps: List[List[int]]) -> None:
    print("Sorry, that's 10 tries.  Here's where they're hiding.")
    for idx, mugwump in enumerate(mugwumps, 1):
        if mugwump[0] != -1:
            print(f"Mugwump {idx} is at {mugwump[0]},{mugwump[1]}")


# 计算猜测位置和 mugwump 位置之间的距离
def calculate_distance(guess: Tuple[int, int], mugwump: List[int]) -> float:
    d = sqrt(((mugwump[0] - guess[0]) ** 2) + ((mugwump[1] - guess[1]) ** 2))
    return d


# 询问玩家是否再玩一次
def play_again() -> None:
    print("THAT WAS FUN! LET'S PLAY AGAIN.......")
    choice = input("Press Enter to play again, any other key then Enter to quit.")
    if choice == "":
        print("Four more mugwumps are now in hiding.")
    else:
        exit()


# 进行一轮游戏
def play_round() -> None:
    mugwumps = generate_mugwumps()
    turns = 1
    score = 0
    while turns <= 10 and score != 4:
        m = -1
        while m == -1:
            try:
                m, n = map(int, input(f"Turn {turns} - what is your guess? ").split())
            except ValueError:
                m = -1
        for idx, mugwump in enumerate(mugwumps):
            if m == mugwump[0] and n == mugwump[1]:
                print(f"You found mugwump {idx + 1}")
                mugwumps[idx][0] = -1
                score += 1
            if mugwump[0] == -1:
                continue
            print(
                f"You are {calculate_distance((m, n), mugwump):.1f} units from mugwump {idx + 1}"
            )
        turns += 1
    if score == 4:
        print(f"Well done! You got them all in {turns} turns.")
    else:
        reveal_mugwumps(mugwumps)


# 主程序入口
if __name__ == "__main__":
    introduction()
    while True:
        play_round()
        play_again()

```