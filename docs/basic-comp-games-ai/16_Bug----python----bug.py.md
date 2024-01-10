# `basic-computer-games\16_Bug\python\bug.py`

```
# 导入随机数模块
import random
# 导入时间模块
import time
# 导入数据类模块
from dataclasses import dataclass
# 导入类型提示模块
from typing import Literal

# 定义状态类
@dataclass
class State:
    # 是否是玩家
    is_player: bool
    # 身体
    body: int = 0
    # 颈部
    neck: int = 0
    # 头部
    head: int = 0
    # 触角
    feelers: int = 0
    # 尾巴
    tail: int = 0
    # 腿
    legs: int = 0

    # 判断是否完成
    def is_finished(self) -> bool:
        return (
            self.feelers == 2
            and self.tail == 1
            and self.legs == 6
            and self.head == 1
            and self.neck == 1
        )

    # 显示状态
    def display(self) -> None:
        # 如果有触角，则打印触角
        if self.feelers != 0:
            print_feelers(self.feelers, is_player=self.is_player)
        # 如果有头部，则打印头部
        if self.head != 0:
            print_head()
        # 如果有颈部，则打印颈部
        if self.neck != 0:
            print_neck()
        # 如果有身体，则打印身体
        if self.body != 0:
            print_body(True) if self.tail == 1 else print_body(False)
        # 如果有腿，则打印腿
        if self.legs != 0:
            print_legs(self.legs)

# 打印指定行数的空行
def print_n_newlines(n: int) -> None:
    for _ in range(n):
        print()

# 打印触角
def print_feelers(n_feelers: int, is_player: bool = True) -> None:
    for _ in range(4):
        print(" " * 10, end="")
        for _ in range(n_feelers):
            print("A " if is_player else "F ", end="")
        print()

# 打印头部
def print_head() -> None:
    print("        HHHHHHH")
    print("        H     H")
    print("        H O O H")
    print("        H     H")
    print("        H  V  H")
    print("        HHHHHHH")

# 打印颈部
def print_neck() -> None:
    print("          N N")
    print("          N N")

# 打印身体
def print_body(has_tail: bool = False) -> None:
    print("     BBBBBBBBBBBB")
    print("     B          B")
    print("     B          B")
    print("TTTTTB          B") if has_tail else ""
    print("     BBBBBBBBBBBB")

# 打印腿
def print_legs(n_legs: int) -> None:
    for _ in range(2):
        print(" " * 5, end="")
        for _ in range(n_legs):
            print(" L", end="")
        print()

# 处理掷骰子结果
def handle_roll(diceroll: Literal[1, 2, 3, 4, 5, 6], state: State) -> bool:
    who = "YOU" if state.is_player else "I"
    changed = False
    # 打印掷骰子的结果
    print(f"{who} ROLLED A", diceroll)
    # 如果骰子结果为1
    if diceroll == 1:
        # 打印身体部位
        print("1=BODY")
        # 如果状态中已经有身体
        if state.body:
            # 打印不需要身体的消息
            print(f"{who} DO NOT NEED A BODY.")
        else:
            # 打印现在有身体的消息
            print(f"{who} NOW HAVE A BODY.")
            # 将状态中的身体部位标记为1
            state.body = 1
            # 标记状态已经改变
            changed = True
    # 如果骰子结果为2
    elif diceroll == 2:
        # 打印颈部部位
        print("2=NECK")
        # 如果状态中已经有颈部
        if state.neck:
            # 打印不需要颈部的消息
            print(f"{who} DO NOT NEED A NECK.")
        # 如果状态中没有身体
        elif state.body == 0:
            # 打印没有身体的消息
            print(f"{who} DO NOT HAVE A BODY.")
        else:
            # 打印现在有颈部的消息
            print(f"{who} NOW HAVE A NECK.")
            # 将状态中的颈部部位标记为1
            state.neck = 1
            # 标记状态已经改变
            changed = True
    # 如果骰子结果为3
    elif diceroll == 3:
        # 打印头部部位
        print("3=HEAD")
        # 如果状态中没有颈部
        if state.neck == 0:
            # 打印没有颈部的消息
            print(f"{who} DO NOT HAVE A NECK.")
        # 如果状态中已经有头部
        elif state.head:
            # 打印已经有头部的消息
            print(f"{who} HAVE A HEAD.")
        else:
            # 打印需要头部的消息
            print(f"{who} NEEDED A HEAD.")
            # 将状态中的头部部位标记为1
            state.head = 1
            # 标记状态已经改变
            changed = True
    # 如果骰子结果为4
    elif diceroll == 4:
        # 打印触角部位
        print("4=FEELERS")
        # 如果状态中没有头部
        if state.head == 0:
            # 打印没有头部的消息
            print(f"{who} DO NOT HAVE A HEAD.")
        # 如果状态中已经有两个触角
        elif state.feelers == 2:
            # 打印已经有两个触角的消息
            print(f"{who} HAVE TWO FEELERS ALREADY.")
        else:
            # 如果是玩家角色
            if state.is_player:
                # 打印现在给你一个触角的消息
                print("I NOW GIVE YOU A FEELER.")
            else:
                # 打印现在有一个触角的消息
                print(f"{who} GET A FEELER.")
            # 触角数量加1
            state.feelers += 1
            # 标记状态已经改变
            changed = True
    # 如果骰子结果为5
    elif diceroll == 5:
        # 打印尾部部位
        print("5=TAIL")
        # 如果状态中没有身体
        if state.body == 0:
            # 打印没有身体的消息
            print(f"{who} DO NOT HAVE A BODY.")
        # 如果状态中已经有尾部
        elif state.tail:
            # 打印已经有尾部的消息
            print(f"{who} ALREADY HAVE A TAIL.")
        else:
            # 如果是玩家角色
            if state.is_player:
                # 打印现在给你一个尾部的消息
                print("I NOW GIVE YOU A TAIL.")
            else:
                # 打印现在有一个尾部的消息
                print(f"{who} NOW HAVE A TAIL.")
            # 将状态中的尾部部位标记为1
            state.tail = 1
            # 标记状态已经改变
            changed = True
    # 如果骰子点数为6
    elif diceroll == 6:
        # 打印输出“6=LEG”
        print("6=LEG")
        # 如果状态中的legs已经为6
        if state.legs == 6:
            # 打印输出“{who}已经有6条腿了。”
            print(f"{who} HAVE 6 FEET ALREADY.")
        # 如果状态中的body为0
        elif state.body == 0:
            # 打印输出“{who}没有身体。”
            print(f"{who} DO NOT HAVE A BODY.")
        else:
            # 增加状态中的legs数量
            state.legs += 1
            # 将changed标记为True
            changed = True
            # 打印输出“{who}现在有{state.legs}条腿”
            print(f"{who} NOW HAVE {state.legs} LEGS")
    # 返回changed标记
    return changed
# 定义主函数，不返回任何结果
def main() -> None:
    # 打印 BUG 字样
    print(" " * 34 + "BUG")
    # 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY 字样
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印三个空行
    print_n_newlines(3)

    # 打印游戏标题
    print("THE GAME BUG")
    # 打印提示语句
    print("I HOPE YOU ENJOY THIS GAME.")
    print()
    # 询问玩家是否需要游戏说明
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ")
    # 如果玩家不需要游戏说明
    if want_instructions != "NO":
        # 打印游戏规则说明
        print("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH")
        print("MINE. EACH NUMBER STANDS FOR A PART OF THE BUG BODY.")
        print("I WILL ROLL THE DIE FOR YOU, TELL YOU WHAT I ROLLED FOR YOU")
        print("WHAT THE NUMBER STANDS FOR, AND IF YOU CAN GET THE PART.")
        print("IF YOU CAN GET THE PART I WILL GIVE IT TO YOU.")
        print("THE SAME WILL HAPPEN ON MY TURN.")
        print("IF THERE IS A CHANGE IN EITHER BUG I WILL GIVE YOU THE")
        print("OPTION OF SEEING THE PICTURES OF THE BUGS.")
        print("THE NUMBERS STAND FOR PARTS AS FOLLOWS:")
        # 创建包含游戏规则表格的列表
        table = [
            ["NUMBER", "PART", "NUMBER OF PART NEEDED"],
            ["1", "BODY", "1"],
            ["2", "NECK", "1"],
            ["3", "HEAD", "1"],
            ["4", "FEELERS", "2"],
            ["5", "TAIL", "1"],
            ["6", "LEGS", "6"],
        ]
        # 遍历游戏规则表格，打印每一行内容
        for row in table:
            print(f"{row[0]:<16}{row[1]:<16}{row[2]:<20}")
        # 打印两个空行
        print_n_newlines(2)

    # 创建玩家状态对象
    player = State(is_player=True)
    # 创建对手状态对象
    opponent = State(is_player=False)
    # 初始化已完成的 BUG 数量为 0
    bugs_finished = 0
    # 当完成的错误数量小于等于0时，执行循环
    while bugs_finished <= 0:
        # 产生1到6之间的随机整数作为骰子点数
        diceroll = random.randint(1, 6)
        # 打印空行
        print()
        # 调用handle_roll函数处理骰子点数，传入玩家对象，忽略类型检查
        changed = handle_roll(diceroll, player)  # type: ignore

        # 重新产生1到6之间的随机整数作为骰子点数
        diceroll = random.randint(1, 6)
        # 打印空行
        print()
        # 等待2秒
        time.sleep(2)

        # 调用handle_roll函数处理骰子点数，传入对手对象，忽略类型检查
        changed_op = handle_roll(diceroll, opponent)  # type: ignore

        # 更新changed变量，如果有任何一个handle_roll函数返回True，则为True
        changed = changed or changed_op

        # 如果玩家完成了所有错误
        if player.is_finished():
            # 打印消息
            print("YOUR BUG IS FINISHED.")
            # 完成的错误数量加1
            bugs_finished += 1
        # 如果对手完成了所有错误
        if opponent.is_finished():
            # 打印消息
            print("MY BUG IS FINISHED.")
            # 完成的错误数量加1
            bugs_finished += 1
        # 如果没有错误被修复，则继续下一次循环
        if not changed:
            continue
        # 询问玩家是否需要图片
        want_pictures = input("DO YOU WANT THE PICTURES? ")
        # 如果玩家需要图片
        if want_pictures != "NO":
            # 打印消息和玩家的错误信息
            print("*****YOUR BUG*****")
            print_n_newlines(2)
            player.display()
            print_n_newlines(4)
            # 打印消息和对手的错误信息
            print("*****MY BUG*****")
            print_n_newlines(3)
            opponent.display()

            # 如果完成的错误数量不为0，则跳出循环
            if bugs_finished != 0:
                break

    # 打印结束游戏的消息
    print("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!")
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```