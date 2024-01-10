# `basic-computer-games\37_Football\python\football.py`

```
"""
FOOTBALL

A game.

Ported to Python by Martin Thoma in 2022.
The JavaScript version by Oscar Toledo G. (nanochess) was used
"""
# NOTE: The newlines might be wrong

# 导入所需模块
import json
from math import floor
from pathlib import Path
from random import randint, random
from typing import List, Tuple

# 从 data.json 文件中读取数据
with open(Path(__file__).parent / "data.json") as f:
    data = json.load(f)

# 从数据中获取球员数据和动作数据
player_data = [num - 1 for num in data["players"]]
actions = data["actions"]

# 初始化一些变量
aa: List[int] = [-100 for _ in range(20)]
ba: List[int] = [-100 for _ in range(20)]
ca: List[int] = [-100 for _ in range(40)]
score: List[int] = [0, 0]
ta: Tuple[int, int] = (1, 0)
wa: Tuple[int, int] = (-1, 1)
xa: Tuple[int, int] = (100, 0)
ya: Tuple[int, int] = (1, -1)
za: Tuple[int, int] = (0, 100)
marker: Tuple[str, str] = ("--->", "<---")
t: int = 0
p: int = 0
winning_score: int

# 定义函数，用于询问用户是否为布尔值
def ask_bool(prompt: str) -> bool:
    while True:
        answer = input(prompt).lower()
        if answer in ["yes", "y"]:
            return True
        elif answer in ["no", "n"]:
            return False

# 定义函数，用于询问用户输入整数
def ask_int(prompt: str) -> int:
    while True:
        answer = input(prompt)
        try:
            int_answer = int(answer)
            return int_answer
        except Exception:
            pass

# 定义函数，用于获取进攻和防守的选择
def get_offense_defense() -> Tuple[int, int]:
    while True:
        input_str = input("INPUT OFFENSIVE PLAY, DEFENSIVE PLAY: ")
        try:
            p1, p2 = (int(n) for n in input_str.split(","))
            return p1, p2
        except Exception:
            pass

# 定义函数，用于打印场地头部信息
def field_headers() -> None:
    print("TEAM 1 [0   10   20   30   40   50   60   70   80   90   100] TEAM 2")
    print("\n\n")

# 定义函数，用于打印分隔线
def separator() -> None:
    print("+" * 72 + "\n")

# 定义函数，用于展示球的位置
def show_ball() -> None:
    da: Tuple[int, int] = (0, 3)
    print(" " * (da[t] + 5 + int(p / 2)) + marker[t] + "\n")
    field_headers()

# 定义函数，用于展示比分
def show_scores() -> bool:
    print()
    print(f"TEAM 1 SCORE IS {score[0]}")
    print(f"TEAM 2 SCORE IS {score[1]}")
    print()
    # 如果当前队伍的得分大于或等于获胜分数
    if score[t] >= winning_score:
        # 打印出当前队伍获胜的消息
        print(f"TEAM {t+1} WINS*******************")
        # 返回True，表示当前队伍获胜
        return True
    # 如果当前队伍的得分小于获胜分数
    # 返回False，表示当前队伍未获胜
    return False
# 定义失去球权的函数，不返回任何内容
def loss_posession() -> None:
    # 声明全局变量 t
    global t
    # 打印失去球权的消息，包括当前队伍和下一个队伍的编号
    print()
    print(f"** LOSS OF POSSESSION FROM TEAM {t+1} TO TEAM {ta[t]+1}")
    print()
    # 调用分隔线函数
    separator()
    print()
    # 将 t 的值更新为 ta[t]，表示失去球权的队伍变为下一个队伍
    t = ta[t]


# 定义 Touchdown 函数，不返回任何内容
def touchdown() -> None:
    # 打印 Touchdown 的消息，包括得分队伍的编号
    print()
    print(f"TOUCHDOWN BY TEAM {t+1} *********************YEA TEAM")
    # 初始化得分 q 为 7，生成随机数 g
    q = 7
    g = random()
    # 如果随机数小于等于 0.1，则将得分 q 更新为 6，并打印额外点球不中的消息
    if g <= 0.1:
        q = 6
        print("EXTRA POINT NO GOOD")
    else:
        # 否则打印额外点球成功的消息
        print("EXTRA POINT GOOD")
    # 将得分 q 加到得分队伍的分数上
    score[t] = score[t] + q


# 定义打印标题的函数，不返回任何内容
def print_header() -> None:
    # 打印游戏标题和信息
    print(" " * 32 + "FOOTBALL")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("PRESENTING N.F.U. FOOTBALL (NO FORTRAN USED)\n\n")


# 定义打印游戏说明的函数，不返回任何内容
def print_instructions() -> None:
    # 打印游戏说明
    print(
        """THIS IS A FOOTBALL GAME FOR TWO TEAMS IN WHICH PLAYERS MUST
PREPARE A TAPE WITH A DATA STATEMENT (1770 FOR TEAM 1,
1780 FOR TEAM 2) IN WHICH EACH TEAM SCRAMBLES NOS. 1-20
THESE NUMBERS ARE THEN ASSIGNED TO TWENTY GIVEN PLAYS.
A LIST OF NOS. AND THEIR PLAYS IS PROVIDED WITH
BOTH TEAMS HAVING THE SAME PLAYS. THE MORE SIMILAR THE
PLAYS THE LESS YARDAGE GAINED.  SCORES ARE GIVEN
WHENEVER SCORES ARE MADE. SCORES MAY ALSO BE OBTAINED
BY INPUTTING 99,99 FOR PLAY NOS. TO PUNT OR ATTEMPT A
FIELD GOAL, INPUT 77,77 FOR PLAY NUMBERS. QUESTIONS WILL BE
ASKED THEN. ON 4TH DOWN, YOU WILL ALSO BE ASKED WHETHER
YOU WANT TO PUNT OR ATTEMPT A FIELD GOAL. IF THE ANSWER TO
BOTH QUESTIONS IS NO IT WILL BE ASSUMED YOU WANT TO
TRY AND GAIN YARDAGE. ANSWER ALL QUESTIONS YES OR NO.
THE GAME IS PLAYED UNTIL PLAYERS TERMINATE (CONTROL-C).
PLEASE PREPARE A TAPE AND RUN.
"""
    )


# 定义主函数，不返回任何内容
def main() -> None:
    # 声明全局变量 winning_score
    global winning_score
    # 调用打印标题的函数
    print_header()
    # 询问是否需要游戏说明
    want_instructions = ask_bool("DO YOU WANT INSTRUCTIONS? ")
    if want_instructions:
        # 如果需要，调用打印游戏说明的函数
        print_instructions()
    print()
    # 询问设定游戏的得分上限
    winning_score = ask_int("PLEASE INPUT SCORE LIMIT ON GAME: ")
    # 循环40次，i从0到39
    for i in range(40):
        # 获取player_data中第i-1个元素的值
        index = player_data[i - 1]
        # 如果i小于20
        if i < 20:
            # 将i赋值给aa字典中的index位置
            aa[index] = i
        # 如果i大于等于20
        else:
            # 将i-20赋值给ba字典中的index位置
            ba[index] = i - 20
        # 将index赋值给ca列表中的第i个位置
        ca[i] = index
    # 初始化offset为0
    offset = 0
    # 遍历[0, 1]列表
    for t in [0, 1]:
        # 打印TEAM t+1 PLAY CHART
        print(f"TEAM {t+1} PLAY CHART")
        # 打印表头
        print("NO.      PLAY")
        # 遍历20次
        for i in range(20):
            # 将ca[i+offset]转换为字符串赋值给input_str
            input_str = f"{ca[i + offset]}"
            # 如果input_str长度小于6，添加空格使其长度为6
            while len(input_str) < 6:
                input_str += " "
            # 将actions[i]添加到input_str末尾
            input_str += actions[i]
            # 打印input_str
            print(input_str)
        # offset增加20
        offset += 20
        # 将t赋值为1
        t = 1
        # 打印空行
        print()
        # 打印分隔线
        print("TEAR OFF HERE----------------------------------------------")
        # 打印10个空行
        print("\n" * 10)

    # 调用field_headers函数
    field_headers()
    # 打印TEAM 1 DEFEND 0 YD GOAL -- TEAM 2 DEFENDS 100 YD GOAL.
    print("TEAM 1 DEFEND 0 YD GOAL -- TEAM 2 DEFENDS 100 YD GOAL.")
    # 随机生成0或1赋值给t
    t = randint(0, 1)
    # 打印空行
    print()
    # 打印THE COIN IS FLIPPED
    print("THE COIN IS FLIPPED")
    # 将routine赋值为1
    routine = 1
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```