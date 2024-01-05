# `d:/src/tocomm/basic-computer-games\37_Football\python\football.py`

```
"""
FOOTBALL

A game.

Ported to Python by Martin Thoma in 2022.
The JavaScript version by Oscar Toledo G. (nanochess) was used
"""
# NOTE: The newlines might be wrong

import json  # 导入 json 模块
from math import floor  # 从 math 模块导入 floor 函数
from pathlib import Path  # 从 pathlib 模块导入 Path 类
from random import randint, random  # 从 random 模块导入 randint 和 random 函数
from typing import List, Tuple  # 从 typing 模块导入 List 和 Tuple 类型

with open(Path(__file__).parent / "data.json") as f:  # 打开当前文件所在目录下的 data.json 文件
    data = json.load(f)  # 读取并解析 JSON 文件内容，存储到 data 变量中

player_data = [num - 1 for num in data["players"]]  # 从 data 字典中的 "players" 键获取数据，将每个元素减去 1 后存储到 player_data 列表中
# 从数据字典中获取键为"actions"的值
actions = data["actions"]

# 初始化列表aa，ba，ca，score，分别存储20个-100，20个-100，40个-100，和两个0
aa: List[int] = [-100 for _ in range(20)]
ba: List[int] = [-100 for _ in range(20)]
ca: List[int] = [-100 for _ in range(40)]
score: List[int] = [0, 0]

# 初始化元组ta，wa，xa，ya，za，分别为(1, 0)，(-1, 1)，(100, 0)，(1, -1)，(0, 100)
ta: Tuple[int, int] = (1, 0)
wa: Tuple[int, int] = (-1, 1)
xa: Tuple[int, int] = (100, 0)
ya: Tuple[int, int] = (1, -1)
za: Tuple[int, int] = (0, 100)

# 初始化元组marker，存储字符串"--->"和"<---"
marker: Tuple[str, str] = ("--->", "<---")

# 初始化整数t和p，分别为0
t: int = 0
p: int = 0

# 初始化整数winning_score
winning_score: int

# 定义函数ask_bool，参数为字符串prompt，返回值为布尔类型
def ask_bool(prompt: str) -> bool:
    # 进入无限循环
    while True:
        answer = input(prompt).lower()  # 从用户输入中获取答案，并转换为小写
        if answer in ["yes", "y"]:  # 如果答案是yes或者y
            return True  # 返回True
        elif answer in ["no", "n"]:  # 如果答案是no或者n
            return False  # 返回False


def ask_int(prompt: str) -> int:  # 定义一个函数，接受一个字符串参数，返回一个整数
    while True:  # 无限循环
        answer = input(prompt)  # 从用户输入中获取答案
        try:  # 尝试执行以下代码
            int_answer = int(answer)  # 将答案转换为整数
            return int_answer  # 返回整数答案
        except Exception:  # 如果出现异常
            pass  # 继续循环


def get_offense_defense() -> Tuple[int, int]:  # 定义一个函数，返回一个元组，包含两个整数
    while True:  # 无限循环
        input_str = input("INPUT OFFENSIVE PLAY, DEFENSIVE PLAY: ")  # 从用户输入中获取字符串
        try:
            # 尝试将输入字符串按逗号分隔，并转换为整数，分别赋值给 p1 和 p2
            p1, p2 = (int(n) for n in input_str.split(","))
            # 如果成功转换并赋值，则返回 p1 和 p2
            return p1, p2
        except Exception:
            # 如果出现异常，则不做任何操作
            pass


def field_headers() -> None:
    # 打印比赛场地头部信息
    print("TEAM 1 [0   10   20   30   40   50   60   70   80   90   100] TEAM 2")
    # 打印空行
    print("\n\n")


def separator() -> None:
    # 打印分隔线
    print("+" * 72 + "\n")


def show_ball() -> None:
    # 定义一个元组 da，包含两个整数
    da: Tuple[int, int] = (0, 3)
    # 打印球的位置信息
    print(" " * (da[t] + 5 + int(p / 2)) + marker[t] + "\n")
    # 调用 field_headers 函数打印比赛场地头部信息
    field_headers()
def show_scores() -> bool:
    # 打印空行
    print()
    # 打印第一队的得分
    print(f"TEAM 1 SCORE IS {score[0]}")
    # 打印第二队的得分
    print(f"TEAM 2 SCORE IS {score[1]}")
    # 打印空行
    print()
    # 如果当前队伍的得分大于等于获胜分数
    if score[t] >= winning_score:
        # 打印获胜队伍的信息
        print(f"TEAM {t+1} WINS*******************")
        # 返回True
        return True
    # 返回False
    return False


def loss_posession() -> None:
    # 声明全局变量t
    global t
    # 打印空行
    print()
    # 打印失去球权的信息
    print(f"** LOSS OF POSSESSION FROM TEAM {t+1} TO TEAM {ta[t]+1}")
    # 打印空行
    print()
    # 调用separator函数
    separator()
    # 打印空行
    print()
    t = ta[t]  # 将数组 ta 中索引为 t 的元素赋值给变量 t


def touchdown() -> None:
    print()  # 打印空行
    print(f"TOUCHDOWN BY TEAM {t+1} *********************YEA TEAM")  # 打印带有变量 t 的字符串
    q = 7  # 初始化变量 q 为 7
    g = random()  # 生成一个随机数并赋值给变量 g
    if g <= 0.1:  # 如果 g 小于等于 0.1
        q = 6  # 将变量 q 的值改为 6
        print("EXTRA POINT NO GOOD")  # 打印字符串
    else:
        print("EXTRA POINT GOOD")  # 打印字符串
    score[t] = score[t] + q  # 将 score[t] 的值加上 q 赋值给 score[t]


def print_header() -> None:
    print(" " * 32 + "FOOTBALL")  # 打印带有空格的字符串和 "FOOTBALL"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印带有空格的字符串和 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n"
    print("PRESENTING N.F.U. FOOTBALL (NO FORTRAN USED)\n\n")  # 打印字符串
# 定义一个打印说明的函数，不返回任何值
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
```
# 定义一个主函数，没有返回值
def main() -> None:
    # 声明全局变量 winning_score
    global winning_score
    # 打印游戏标题
    print_header()
    # 询问用户是否需要游戏说明
    want_instructions = ask_bool("DO YOU WANT INSTRUCTIONS? ")
    # 如果需要游戏说明，则打印游戏说明
    if want_instructions:
        print_instructions()
    # 打印空行
    print()
    # 询问用户输入游戏的得分上限
    winning_score = ask_int("PLEASE INPUT SCORE LIMIT ON GAME: ")
    # 循环40次
    for i in range(40):
        # 获取玩家数据中的索引
        index = player_data[i - 1]
        # 如果索引小于20，则将其赋值给aa列表
        if i < 20:
            aa[index] = i
        # 如果索引大于等于20，则将其减去20后赋值给ba列表
        else:
            ba[index] = i - 20
        # 将索引赋值给ca列表
        ca[i] = index
    # 初始化偏移量为0
    offset = 0
    for t in [0, 1]:  # 循环遍历列表[0, 1]，t分别取值0和1
        print(f"TEAM {t+1} PLAY CHART")  # 打印输出团队t+1的比赛图表
        print("NO.      PLAY")  # 打印输出编号和比赛动作
        for i in range(20):  # 循环遍历范围在0到19的整数
            input_str = f"{ca[i + offset]}"  # 将ca[i + offset]的值转换为字符串并赋给input_str
            while len(input_str) < 6:  # 当input_str的长度小于6时执行循环
                input_str += " "  # 在input_str末尾添加空格
            input_str += actions[i]  # 将actions[i]的值添加到input_str末尾
            print(input_str)  # 打印输出input_str
        offset += 20  # offset增加20
        t = 1  # 将t的值设为1
        print()  # 打印输出空行
        print("TEAR OFF HERE----------------------------------------------")  # 打印输出分隔线
        print("\n" * 10)  # 打印输出10个换行符

    field_headers()  # 调用field_headers函数
    print("TEAM 1 DEFEND 0 YD GOAL -- TEAM 2 DEFENDS 100 YD GOAL.")  # 打印输出比赛信息
    t = randint(0, 1)  # 生成一个0到1之间的随机整数并赋给t
    print()  # 打印输出空行
    print("THE COIN IS FLIPPED")  # 打印输出硬币翻转的信息
    routine = 1  # 初始化变量routine为1
    while True:  # 进入无限循环
        if routine <= 1:  # 如果routine小于等于1
            p = xa[t] - ya[t] * 40  # 计算p的值
            separator()  # 调用separator函数
            print(f"TEAM {t+1} RECEIVES KICK-OFF")  # 打印接球队伍信息
            k = floor(26 * random() + 40)  # 计算k的值
        if routine <= 2:  # 如果routine小于等于2
            p = p - ya[t] * k  # 更新p的值
        if routine <= 3:  # 如果routine小于等于3
            if wa[t] * p >= za[t] + 10:  # 判断条件
                print("BALL WENT OUT OF ENDZONE --AUTOMATIC TOUCHBACK--")  # 打印信息
                p = za[t] - wa[t] * 20  # 更新p的值
                if routine <= 4:  # 如果routine小于等于4
                    routine = 5  # 更新routine的值为5
            else:
                print(f"BALL WENT {k} YARDS.  NOW ON {p}")  # 打印信息
                show_ball()  # 调用show_ball函数

        if routine <= 4:  # 如果routine小于等于4
            # 询问球队是否要进行再次进攻
            want_runback = ask_bool(f"TEAM {t+1} DO YOU WANT TO RUNBACK? ")

            # 如果要再次进攻
            if want_runback:
                # 随机生成一个1到9之间的整数
                k = floor(9 * random() + 1)
                # 计算r的值
                r = floor(((xa[t] - ya[t] * p + 25) * random() - 15) / k)
                # 更新p的值
                p = p - wa[t] * r
                # 打印输出再次进攻的信息
                print(f"RUNBACK TEAM {t+1} {r} YARDS")
                # 生成一个0到1之间的随机数
                g = random()
                # 如果随机数小于0.25
                if g < 0.25:
                    # 失去控球
                    loss_posession()
                    # 更新routine的值
                    routine = 4
                    # 继续下一轮循环
                    continue
                # 如果ya[t] * p大于等于xa[t]
                elif ya[t] * p >= xa[t]:
                    # 进球
                    touchdown()
                    # 如果显示比分
                    if show_scores():
                        # 返回
                        return
                    # 更新t的值
                    t = ta[t]
                    # 更新routine的值
                    routine = 1
                    # 继续下一轮循环
                    continue
                # 如果wa[t] * p大于等于za[t]
                    # 打印安全得分信息，增加得分
                    print(f"SAFETY AGAINST TEAM {t+1} **********************OH-OH")
                    score[ta[t]] = score[ta[t]] + 2
                    # 如果需要展示得分情况，则返回
                    if show_scores():
                        return

                    # 计算需要的得分
                    p = za[t] - wa[t] * 20
                    # 询问是否要进行开球而不是踢球
                    want_punt = ask_bool(
                        f"TEAM {t+1} DO YOU WANT TO PUNT INSTEAD OF A KICKOFF? "
                    )
                    # 如果选择开球
                    if want_punt:
                        print(f"TEAM {t+1} WILL PUNT")
                        # 随机生成一个概率值
                        g = random()
                        # 如果概率小于0.25，则失去控球
                        if g < 0.25:
                            loss_posession()
                            routine = 4
                            continue

                        # 打印空行和分隔符
                        print()
                        separator()
                        # 随机生成一个距离值
                        k = floor(25 * random() + 35)
                        t = ta[t]  # 将变量t赋值为数组ta中索引为t的元素的值
                        routine = 2  # 将变量routine赋值为2
                        continue  # 继续执行下一次循环

                    touchdown()  # 调用touchdown函数
                    if show_scores():  # 如果show_scores函数返回True
                        return  # 返回
                    t = ta[t]  # 将变量t赋值为数组ta中索引为t的元素的值
                    routine = 1  # 将变量routine赋值为1
                    continue  # 继续执行下一次循环
                else:
                    routine = 5  # 将变量routine赋值为5
                    continue  # 继续执行下一次循环

            else:
                if wa[t] * p >= za[t]:  # 如果数组wa中索引为t的元素乘以p大于等于数组za中索引为t的元素
                    p = za[t] - wa[t] * 20  # 将变量p赋值为za[t]减去wa[t]乘以20的值

        if routine <= 5:  # 如果变量routine小于等于5
            d = 1  # 将变量d赋值为1
            s = p  # 将变量p的值赋给变量s

        if routine <= 6:  # 如果routine小于等于6
            print("=" * 72 + "\n")  # 打印72个等号
            print(f"TEAM {t+1} DOWN {d} ON {p}")  # 打印TEAM t+1 DOWN d ON p
            if d == 1:  # 如果d等于1
                if ya[t] * (p + ya[t] * 10) >= xa[t]:  # 如果ya[t]乘以(p + ya[t]乘以10)大于等于xa[t]
                    c = 8  # 将变量c的值设为8
                else:
                    c = 4  # 否则将变量c的值设为4

            if c != 8:  # 如果c不等于8
                yards = 10 - (ya[t] * p - ya[t] * s)  # 计算yards的值
                print(" " * 27 + f"{yards} YARDS TO 1ST DOWN")  # 打印yards的值
            else:  # 否则
                yards = xa[t] - ya[t] * p  # 计算yards的值
                print(" " * 27 + f"{yards} YARDS")  # 打印yards的值

            show_ball()  # 调用show_ball函数
            if d == 4:  # 如果d等于4
                routine = 8  # 设置变量routine的值为8

        if routine <= 7:  # 如果routine的值小于等于7
            u = floor(3 * random() - 1)  # 计算3倍随机数减1的值并向下取整，赋给变量u
            while True:  # 进入无限循环
                p1, p2 = get_offense_defense()  # 调用函数get_offense_defense()，并将返回的两个值分别赋给p1和p2
                if t != 1:  # 如果t不等于1
                    p2, p1 = p1, p2  # 交换p1和p2的值

                if p1 == 99:  # 如果p1等于99
                    if show_scores():  # 调用函数show_scores()，如果返回True
                        return  # 返回
                    if p1 == 99:  # 如果p1等于99
                        continue  # 继续下一次循环

                if p1 < 1 or p1 > 20 or p2 < 1 or p2 > 20:  # 如果p1小于1或大于20，或者p2小于1或大于20
                    print("ILLEGAL PLAY NUMBER, CHECK AND ", end="")  # 打印错误信息
                    continue  # 继续下一次循环

                break  # 跳出循环
            p1 -= 1  # 将变量p1减1
            p2 -= 1  # 将变量p2减1

        if d == 4 or p1 == 77:  # 如果d等于4或者p1等于77
            want_punt = ask_bool(f"DOES TEAM {t+1} WANT TO PUNT? ")  # 调用ask_bool函数询问用户是否要进行punt，并将结果存储在want_punt变量中

            if want_punt:  # 如果want_punt为True
                print()  # 打印空行
                print(f"TEAM {t+1} WILL PUNT")  # 打印团队t+1将进行punt
                g = random()  # 生成一个随机数并存储在变量g中
                if g < 0.25:  # 如果g小于0.25
                    loss_posession()  # 调用loss_posession函数
                    routine = 4  # 将变量routine设置为4
                    continue  # 继续循环

                print()  # 打印空行
                separator()  # 调用separator函数
                k = floor(25 * random() + 35)  # 计算一个随机数并存储在变量k中
                t = ta[t]  # 将ta[t]的值赋给变量t
                routine = 2  # 将变量routine设置为2
                continue  # 继续循环，跳过当前迭代的剩余部分

            attempt_field_goal = ask_bool(
                f"DOES TEAM {t+1} WANT TO ATTEMPT A FIELD GOAL? "
            )  # 询问当前队伍是否想要尝试射门

            if attempt_field_goal:  # 如果队伍选择尝试射门
                print()
                print(f"TEAM {t+1} WILL ATTEMPT A FIELD GOAL")  # 打印队伍将尝试射门的消息
                g = random()  # 生成一个随机数 g

                if g < 0.025:  # 如果随机数 g 小于 0.025
                    loss_posession()  # 失去球权
                    routine = 4  # 设置 routine 为 4
                    continue  # 继续循环，跳过当前迭代的剩余部分
                else:  # 如果随机数 g 不小于 0.025
                    f = floor(35 * random() + 20)  # 生成一个随机数 f，表示射门的距离
                    print()
                    print(f"KICK IS {f} YARDS LONG")  # 打印射门距离
                    p = p - wa[t] * f  # 更新得分
                    g = random()  # 重新生成一个随机数 g
                    if g < 0.35:  # 如果 g 小于 0.35
                        print("BALL WENT WIDE")  # 打印“球偏了”
                    elif ya[t] * p >= xa[t]:  # 否则如果 ya[t] 乘以 p 大于等于 xa[t]
                        print(
                            f"FIELD GOLD GOOD FOR TEAM {t+1} *********************YEA"
                        )  # 打印“对于球队 t+1，场地黄金好*********************耶”
                        q = 3  # q 等于 3
                        score[t] = score[t] + q  # 球队 t 的得分加上 q
                        if show_scores():  # 如果显示得分
                            return  # 返回
                        t = ta[t]  # t 等于 ta[t]
                        routine = 1  # routine 等于 1
                        continue  # 继续
                    print(f"FIELD GOAL UNSUCCESFUL TEAM {t+1}-----------------TOO BAD")  # 打印“场地目标不成功，球队 t+1-----------------太糟糕”
                    print()  # 打印空行
                    separator()  # 分隔符
                    if ya[t] * p < xa[t] + 10:  # 如果 ya[t] 乘以 p 小于 xa[t] 加上 10
                        print()  # 打印空行
                        print(f"BALL NOW ON {p}")  # 打印“球现在在 p”
                        t = ta[t]  # 将变量t赋值为ta[t]
                        show_ball()  # 调用show_ball函数
                        routine = 4  # 将变量routine赋值为4
                        continue  # 继续下一次循环

                    else:  # 如果条件不成立
                        t = ta[t]  # 将变量t赋值为ta[t]
                        routine = 3  # 将变量routine赋值为3
                        continue  # 继续下一次循环

            else:  # 如果条件不成立
                routine = 7  # 将变量routine赋值为7
                continue  # 继续下一次循环

        y = floor(  # 将y赋值为floor函数的返回值
            abs(aa[p1] - ba[p2]) / 19 * ((xa[t] - ya[t] * p + 25) * random() - 15)
        )
        print()  # 打印空行
        if t == 1 and aa[p1] < 11 or t == 2 and ba[p2] < 11:  # 如果条件成立
            print("THE BALL WAS RUN")  # 打印"THE BALL WAS RUN"
        elif u == 0:  # 如果条件不成立
            # 打印输出信息，表示未完成的球队
            print(f"PASS INCOMPLETE TEAM {t+1}")
            # 重置y为0
            y = 0
        else:
            # 生成一个随机数
            g = random()
            # 如果随机数小于等于0.025并且y大于2，则打印“PASS COMPLETED”
            if g <= 0.025 and y > 2:
                print("PASS COMPLETED")
            else:
                # 否则打印“QUARTERBACK SCRAMBLED”
                print("QUARTERBACK SCRAMBLED")

        # 计算p的值
        p = p - wa[t] * y
        # 打印空行
        print()
        # 打印输出信息，表示在第d次进攻中净码数为y
        print(f"NET YARDS GAINED ON DOWN {d} ARE {y}")

        # 生成一个随机数
        g = random()
        # 如果随机数小于等于0.025，则失去球权，设置routine为4，并继续循环
        if g <= 0.025:
            loss_posession()
            routine = 4
            continue
        # 否则如果ya[t] * p大于等于xa[t]，则进球得分
        elif ya[t] * p >= xa[t]:
            touchdown()
            # 如果显示比分，则返回
            if show_scores():
                return
            # 将 t 对应的值赋给 t
            t = ta[t]
            # 将 routine 设为 1
            routine = 1
            # 继续执行下一轮循环
            continue
        # 如果 wa[t] 乘以 p 大于等于 za[t]
        elif wa[t] * p >= za[t]:
            # 打印信息
            print()
            print(f"SAFETY AGAINST TEAM {t+1} **********************OH-OH")
            # 将 score[ta[t]] 的值加上 2
            score[ta[t]] = score[ta[t]] + 2
            # 如果显示比分，则返回
            if show_scores():
                return
            # 将 p 设为 za[t] 减去 wa[t] 乘以 20
            p = za[t] - wa[t] * 20
            # 询问是否要进行 punt 而不是 kickoff
            want_punt = ask_bool(
                f"TEAM {t+1} DO YOU WANT TO PUNT INSTEAD OF A KICKOFF? "
            )
            # 如果选择 punt
            if want_punt:
                # 打印信息
                print()
                print(f"TEAM {t+1} WILL PUNT")
                # 生成一个随机数 g
                g = random()
                # 如果 g 小于 0.25
                if g < 0.25:
                    loss_posession()  # 调用 loss_posession 函数
                    routine = 4  # 将变量 routine 的值设为 4
                    continue  # 跳出当前循环，继续执行下一次循环

                print()  # 打印空行
                separator()  # 调用 separator 函数
                k = floor(25 * random() + 35)  # 计算一个随机数并赋值给 k
                t = ta[t]  # 将 ta[t] 的值赋给 t
                routine = 2  # 将变量 routine 的值设为 2
                continue  # 跳出当前循环，继续执行下一次循环

            touchdown()  # 调用 touchdown 函数
            if show_scores():  # 如果 show_scores 函数返回 True
                return  # 返回
            t = ta[t]  # 将 ta[t] 的值赋给 t
            routine = 1  # 将变量 routine 的值设为 1
        elif ya[t] * p - ya[t] * s >= 10:  # 如果 ya[t] * p - ya[t] * s 大于等于 10
            routine = 5  # 将变量 routine 的值设为 5
        else:  # 否则
            d += 1  # 变量 d 的值加 1
            if d != 5:  # 如果变量d不等于5
                routine = 6  # 则将变量routine赋值为6
            else:  # 否则
                print()  # 打印空行
                print(f"CONVERSION UNSUCCESSFUL TEAM {t+1}")  # 打印带有团队编号的提示信息
                t = ta[t]  # 将变量t赋值为ta列表中索引为t的值
                print()  # 打印空行
                separator()  # 调用名为separator的函数
                routine = 5  # 将变量routine赋值为5

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用名为main的函数
```