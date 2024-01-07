# `basic-computer-games\79_Slalom\python\slalom.py`

```

# 从 random 模块中导入 random 函数
from random import random

# 初始化奖牌字典
medals = {
    "gold": 0,
    "silver": 0,
    "bronze": 0,
}

# 定义一个函数，用于向用户提问并返回用户输入的字符串
def ask(question: str) -> str:
    print(question, end="? ")
    return input().upper()

# 定义一个函数，用于向用户提问并返回用户输入的整数
def ask_int(question: str) -> int:
    reply = ask(question)
    return int(reply) if reply.isnumeric() else -1

# 预备阶段，包括显示指令和最大速度等信息
def pre_run(gates, max_speeds) -> None:
    # 显示指令
    print('\nType "INS" for instructions')
    print('Type "MAX" for approximate maximum speeds')
    print('Type "RUN" for the beginning of the race')
    cmd = ask("Command--")
    while cmd != "RUN":
        # 显示赛道介绍
        if cmd == "INS":
            # ...
        # 显示最大速度
        elif cmd == "MAX":
            # ...
        else:
            cmd = ask(f'"{cmd}" is an illegal command--Retry')

# 开始比赛
def run(gates, lvl, max_speeds) -> None:
    global medals
    # ...
    # 初始化时间和速度
    time: float = 0
    speed = int(random() * (18 - 9) + 9)
    # ...
    # 循环处理每个门
    for i in range(0, gates):
        while True:
            # ...
            # 用户选择操作
            opt = ask_int("Option")
            # ...
            # 处理用户选择的操作
            if opt == 8:
                # ...
            else:
                # ...
            # ...
    # ...
    # 计算比赛结果
    print(f"\nYou took {int(time + random())} seconds.")
    avg = time / gates
    if avg < 1.5 - (lvl * 0.1):
        print("Yout won a gold medal!")
        medals["gold"] += 1
    elif avg < 2.9 - (lvl * 0.1):
        print("You won a silver medal!")
        medals["silver"] += 1
    elif avg < 4.4 - (lvl * 0.01):
        print("You won a bronze medal!")
        medals["bronze"] += 1

# 主函数
def main() -> None:
    # 显示比赛信息
    print("Slalom".rjust(39))
    print("Creative Computing Morristown, New Jersey\n\n\n".rjust(57))

    # 初始化最大速度列表
    max_speeds = [
        # ...
    ]

    # 获取赛道门数
    while True:
        gates = ask_int("How many gates does this course have (1 to 25)")
        if gates < 1:
            print("Try again,")
        else:
            if gates > 25:
                print("25 is the limit.")
            break

    # 进入预备阶段
    pre_run(gates, max_speeds)

    # 获取滑雪水平
    while True:
        lvl = ask_int("Rate yourself as a skier, (1=Worst, 3=Best)")
        if lvl < 1 or lvl > 3:
            print("The bounds are 1-3.")
        else:
            break

    # 开始比赛
    while True:
        run(gates, lvl, max_speeds)
        # 询问是否再次比赛
        while True:
            answer = ask("Do you want to play again?")
            if answer == "YES" or answer == "NO":
                break
            else:
                print('Please type "YES" or "NO"')
        if answer == "NO":
            break

    # 显示奖牌情况
    print("Thanks for the race")
    if medals["gold"] > 0:
        print(f"Gold medals: {medals['gold']}")
    if medals["silver"] > 0:
        print(f"Silver medals: {medals['silver']}")
    if medals["bronze"] > 0:
        print(f"Bronze medals: {medals['bronze']}")

# 程序入口
if __name__ == "__main__":
    main()

```