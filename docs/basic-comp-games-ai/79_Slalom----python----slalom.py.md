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

# 定义一个函数，用于向用户提问并返回大写形式的输入
def ask(question: str) -> str:
    print(question, end="? ")
    return input().upper()

# 定义一个函数，用于向用户提问并返回整数形式的输入
def ask_int(question: str) -> int:
    reply = ask(question)
    return int(reply) if reply.isnumeric() else -1

# 定义一个函数，用于准备比赛前的操作
def pre_run(gates, max_speeds) -> None:
    print('\nType "INS" for instructions')
    print('Type "MAX" for approximate maximum speeds')
    print('Type "RUN" for the beginning of the race')
    cmd = ask("Command--")
    # 循环直到用户输入"RUN"
    while cmd != "RUN":
        if cmd == "INS":
            # 显示比赛说明和操作指南
            print("\n*** Slalom: This is the 1976 Winter Olypic Giant Slalom.  You are")
            print("            the American team's only hope for a gold medal.\n")
            print("     0 -- Type this if you want to see how long you've taken.")
            # 其他操作指南...
            print("                Good Luck!\n")
            cmd = ask("Command--")
        elif cmd == "MAX":
            # 显示各个门的最大速度
            print("Gate Max")
            print(" # M.P.H.")
            print("----------")
            for i in range(0, gates):
                print(f" {i + 1}  {max_speeds[i]}")
            cmd = ask("Command--")
        else:
            # 提示用户输入的命令不合法
            cmd = ask(f'"{cmd}" is an illegal command--Retry')

# 定义一个函数，用于开始比赛
def run(gates, lvl, max_speeds) -> None:
# 这里是 run 函数的代码，未提供，需要继续补充
    # 声明变量 medals 为全局变量
    global medals
    # 打印倒计时开始信息
    print("The starter counts down...5...4...3...2...1...Go!")
    # 初始化时间变量为浮点数 0
    time: float = 0
    # 生成随机速度，范围在 9 到 18 之间
    speed = int(random() * (18 - 9) + 9)
    # 打印出发信息
    print("You're off")
    # 打印随机时间加上当前时间的整数部分
    print(f"\nYou took {int(time + random())} seconds.")
    # 计算平均时间
    avg = time / gates
    # 根据平均时间和等级判断获得的奖牌
    if avg < 1.5 - (lvl * 0.1):
        print("Yout won a gold medal!")
        medals["gold"] += 1
    elif avg < 2.9 - (lvl * 0.1):
        print("You won a silver medal!")
        medals["silver"] += 1
    elif avg < 4.4 - (lvl * 0.01):
        print("You won a bronze medal!")
        medals["bronze"] += 1
# 定义主函数
def main() -> None:
    # 打印标题
    print("Slalom".rjust(39))
    print("Creative Computing Morristown, New Jersey\n\n\n".rjust(57))

    # 定义最大速度列表
    max_speeds = [
        14,
        18,
        26,
        29,
        18,
        25,
        28,
        32,
        29,
        20,
        29,
        29,
        25,
        21,
        26,
        29,
        20,
        21,
        20,
        18,
        26,
        25,
        33,
        31,
        22,
    ]

    # 循环直到输入符合要求的门数
    while True:
        gates = ask_int("How many gates does this course have (1 to 25)")
        if gates < 1:
            print("Try again,")
        else:
            if gates > 25:
                print("25 is the limit.")
            break

    # 运行前的准备工作
    pre_run(gates, max_speeds)

    # 循环直到输入符合要求的滑雪水平
    while True:
        lvl = ask_int("Rate yourself as a skier, (1=Worst, 3=Best)")
        if lvl < 1 or lvl > 3:
            print("The bounds are 1-3.")
        else:
            break

    # 循环直到游戏结束
    while True:
        run(gates, lvl, max_speeds)
        # 循环直到输入符合要求的答案
        while True:
            answer = ask("Do you want to play again?")
            if answer == "YES" or answer == "NO":
                break
            else:
                print('Please type "YES" or "NO"')
        if answer == "NO":
            break

    # 打印感谢信息和奖牌情况
    print("Thanks for the race")
    if medals["gold"] > 0:
        print(f"Gold medals: {medals['gold']}")
    if medals["silver"] > 0:
        print(f"Silver medals: {medals['silver']}")
    if medals["bronze"] > 0:
        print(f"Bronze medals: {medals['bronze']}")


# 如果作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```