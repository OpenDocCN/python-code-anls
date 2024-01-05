# `16_Bug\python\bug.py`

```
import random  # 导入 random 模块，用于生成随机数
import time  # 导入 time 模块，用于处理时间相关操作
from dataclasses import dataclass  # 从 dataclasses 模块中导入 dataclass 装饰器，用于创建数据类
from typing import Literal  # 从 typing 模块中导入 Literal 类型提示，用于指定特定的值类型


@dataclass  # 使用 dataclass 装饰器，声明 State 类为数据类
class State:  # 定义 State 类
    is_player: bool  # 声明 is_player 属性为布尔类型
    body: int = 0  # 声明 body 属性为整数类型，默认值为 0
    neck: int = 0  # 声明 neck 属性为整数类型，默认值为 0
    head: int = 0  # 声明 head 属性为整数类型，默认值为 0
    feelers: int = 0  # 声明 feelers 属性为整数类型，默认值为 0
    tail: int = 0  # 声明 tail 属性为整数类型，默认值为 0
    legs: int = 0  # 声明 legs 属性为整数类型，默认值为 0

    def is_finished(self) -> bool:  # 定义 is_finished 方法，返回布尔类型
        return (  # 返回以下条件的逻辑与结果
            self.feelers == 2  # 判断 feelers 属性是否等于 2
            and self.tail == 1  # 判断 tail 属性是否等于 1
            and self.legs == 6  # 检查昆虫是否有6条腿
            and self.head == 1  # 检查昆虫是否有1个头部
            and self.neck == 1  # 检查昆虫是否有1个颈部
        )

    def display(self) -> None:
        if self.feelers != 0:  # 如果昆虫有触角
            print_feelers(self.feelers, is_player=self.is_player)  # 打印触角
        if self.head != 0:  # 如果昆虫有头部
            print_head()  # 打印头部
        if self.neck != 0:  # 如果昆虫有颈部
            print_neck()  # 打印颈部
        if self.body != 0:  # 如果昆虫有身体
            print_body(True) if self.tail == 1 else print_body(False)  # 如果有尾巴则打印身体，否则不打印尾巴
        if self.legs != 0:  # 如果昆虫有腿
            print_legs(self.legs)  # 打印腿部


def print_n_newlines(n: int) -> None:
    for _ in range(n):  # 打印指定数量的换行符
# 打印空行
        print()


# 打印触角
def print_feelers(n_feelers: int, is_player: bool = True) -> None:
    # 循环4次，打印10个空格
    for _ in range(4):
        print(" " * 10, end="")
        # 根据参数is_player决定打印"A "或"F "，打印n_feelers次
        for _ in range(n_feelers):
            print("A " if is_player else "F ", end="")
        # 换行
        print()


# 打印头部
def print_head() -> None:
    # 打印头部的图案
    print("        HHHHHHH")
    print("        H     H")
    print("        H O O H")
    print("        H     H")
    print("        H  V  H")
    print("        HHHHHHH")
# 定义一个打印“脖子”的函数，不返回任何值
def print_neck() -> None:
    # 打印脖子的形状
    print("          N N")
    print("          N N")


# 定义一个打印“身体”的函数，可以选择是否有尾巴，默认为没有尾巴，不返回任何值
def print_body(has_tail: bool = False) -> None:
    # 打印身体的形状
    print("     BBBBBBBBBBBB")
    print("     B          B")
    print("     B          B")
    # 如果有尾巴，则打印尾巴的形状
    print("TTTTTB          B") if has_tail else ""
    print("     BBBBBBBBBBBB")


# 定义一个打印“腿”的函数，根据传入的腿的数量打印相应数量的腿，不返回任何值
def print_legs(n_legs: int) -> None:
    # 循环打印两次
    for _ in range(2):
        # 打印空格
        print(" " * 5, end="")
        # 根据传入的腿的数量打印相应数量的腿
        for _ in range(n_legs):
            print(" L", end="")
        print()
def handle_roll(diceroll: Literal[1, 2, 3, 4, 5, 6], state: State) -> bool:
    # 定义一个处理掷骰子结果的函数，参数为骰子点数和当前状态对象，返回布尔值

    who = "YOU" if state.is_player else "I"
    # 根据当前状态对象的is_player属性确定是玩家还是电脑

    changed = False
    # 初始化一个变量用于标记状态是否发生改变

    print(f"{who} ROLLED A", diceroll)
    # 打印出是玩家还是电脑掷出了什么点数的骰子

    if diceroll == 1:
        # 如果骰子点数为1
        print("1=BODY")
        # 打印出骰子点数对应的部位
        if state.body:
            # 如果当前状态对象的body属性不为0
            print(f"{who} DO NOT NEED A BODY.")
            # 打印出玩家或电脑不需要身体
        else:
            print(f"{who} NOW HAVE A BODY.")
            # 打印出玩家或电脑现在有了身体
            state.body = 1
            # 将状态对象的body属性设置为1
            changed = True
            # 将状态改变标记设置为True
    elif diceroll == 2:
        # 如果骰子点数为2
        print("2=NECK")
        # 打印出骰子点数对应的部位
        if state.neck:
            # 如果当前状态对象的neck属性不为0
            print(f"{who} DO NOT NEED A NECK.")
            # 打印出玩家或电脑不需要脖子
        elif state.body == 0:
            # 如果当前状态对象的body属性为0
            print(f"{who} DO NOT HAVE A BODY.")
            # 打印出玩家或电脑没有身体
```

    else:
        # 如果骰子点数为4，打印提示信息
        print("4=FEELERS")
        # 如果状态中头部为0，打印提示信息
        if state.head == 0:
            print(f"{who} DO NOT HAVE A HEAD.")
        # 如果状态中触角为2，打印提示信息
        elif state.feelers == 2:
            print(f"{who} HAVE TWO FEELERS ALREADY.")
    elif diceroll == 5:  # 如果骰子点数为5
        print("5=TAIL")  # 打印出"5=TAIL"
        if state.body == 0:  # 如果状态中的身体部分为0
            print(f"{who} DO NOT HAVE A BODY.")  # 打印出"{who} DO NOT HAVE A BODY."
        elif state.tail:  # 否则，如果状态中已经有尾巴
            print(f"{who} ALREADY HAVE A TAIL.")  # 打印出"{who} ALREADY HAVE A TAIL."
        else:  # 否则
            if state.is_player:  # 如果状态是玩家
                print("I NOW GIVE YOU A TAIL.")  # 打印出"I NOW GIVE YOU A TAIL."
            else:  # 否则
                print(f"{who} NOW HAVE A TAIL.")  # 打印出"{who} NOW HAVE A TAIL."
            state.tail = 1  # 将状态中的尾巴部分设置为1
            changed = True  # 将changed标记为True
    elif diceroll == 6:  # 如果骰子掷出的点数为6
        print("6=LEG")  # 打印输出“6=LEG”
        if state.legs == 6:  # 如果状态中的legs属性值为6
            print(f"{who} HAVE 6 FEET ALREADY.")  # 打印输出“{who} HAVE 6 FEET ALREADY.”
        elif state.body == 0:  # 如果状态中的body属性值为0
            print(f"{who} DO NOT HAVE A BODY.")  # 打印输出“{who} DO NOT HAVE A BODY.”
        else:  # 否则
            state.legs += 1  # 将状态中的legs属性值加1
            changed = True  # 将changed变量设为True
            print(f"{who} NOW HAVE {state.legs} LEGS")  # 打印输出“{who} NOW HAVE {state.legs} LEGS”
    return changed  # 返回changed变量的值
    # 打印空行
    print()
    # 获取用户输入，询问是否需要说明
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ")
    # 如果用户输入不是"NO"
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
        # 创建包含游戏规则说明的表格
        table = [
            ["NUMBER", "PART", "NUMBER OF PART NEEDED"],
            ["1", "BODY", "1"],
            ["2", "NECK", "1"],
            ["3", "HEAD", "1"],
            ["4", "FEELERS", "2"],
            ["5", "TAIL", "1"],
            ["6", "LEGS", "6"],
    ]
    # 遍历表格中的每一行，并打印每行的前三列内容
    for row in table:
        print(f"{row[0]:<16}{row[1]:<16}{row[2]:<20}")
    # 打印两个空行
    print_n_newlines(2)

player = State(is_player=True)  # 创建一个玩家状态对象
opponent = State(is_player=False)  # 创建一个对手状态对象
bugs_finished = 0  # 初始化已完成的错误数量为0

while bugs_finished <= 0:  # 当已完成的错误数量小于等于0时循环
    diceroll = random.randint(1, 6)  # 生成一个1到6的随机数作为骰子点数
    print()  # 打印一个空行
    changed = handle_roll(diceroll, player)  # 处理玩家根据骰子点数做出的动作，并记录是否有变化

    diceroll = random.randint(1, 6)  # 生成另一个1到6的随机数作为骰子点数
    print()  # 打印一个空行
    time.sleep(2)  # 程序暂停2秒

    changed_op = handle_roll(diceroll, opponent)  # 处理对手根据骰子点数做出的动作，并记录是否有变化
        changed = changed or changed_op  # 如果 changed 或 changed_op 有任何一个为 True，则将 changed 设为 True

        if player.is_finished():  # 如果玩家的 bug 完成了
            print("YOUR BUG IS FINISHED.")  # 打印消息
            bugs_finished += 1  # 完成的 bug 数加一
        if opponent.is_finished():  # 如果对手的 bug 完成了
            print("MY BUG IS FINISHED.")  # 打印消息
            bugs_finished += 1  # 完成的 bug 数加一
        if not changed:  # 如果没有发生改变
            continue  # 继续下一次循环
        want_pictures = input("DO YOU WANT THE PICTURES? ")  # 获取用户输入是否想要图片
        if want_pictures != "NO":  # 如果用户输入不是 "NO"
            print("*****YOUR BUG*****")  # 打印消息
            print_n_newlines(2)  # 打印两行空行
            player.display()  # 显示玩家的 bug
            print_n_newlines(4)  # 打印四行空行
            print("*****MY BUG*****")  # 打印消息
            print_n_newlines(3)  # 打印三行空行
            opponent.display()  # 显示对手的 bug
            if bugs_finished != 0:  # 如果已完成的错误数量不等于0
                break  # 退出循环
    print("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!")  # 打印消息

if __name__ == "__main__":  # 如果当前文件被作为主程序运行
    main()  # 调用主函数
```