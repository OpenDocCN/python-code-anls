# `11_Bombardment\python\bombardment.py`

```
#!/usr/bin/env python3  # 指定使用 Python3 解释器来执行脚本

import random  # 导入 random 模块，用于生成随机数
from functools import partial  # 导入 functools 模块中的 partial 函数，用于创建偏函数
from typing import Callable, List, Set  # 导入 typing 模块中的 Callable、List、Set 类型提示

# 定义一个函数，没有返回值
def print_intro() -> None:
    print(" " * 33 + "BOMBARDMENT")  # 打印游戏标题
    print(" " * 15 + " CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印游戏信息
    print("\n\n")  # 打印两行空行
    print("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU")  # 打印游戏背景信息
    print("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.")  # 打印游戏背景信息
    print("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.")  # 打印游戏背景信息
    print("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.")  # 打印游戏背景信息
    print()
    print("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE")  # 打印游戏目标
    print("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.")  # 打印游戏目标
    print("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS")  # 打印游戏目标
    print("FIRST IS THE WINNER.")  # 打印游戏目标
    print()
    # 打印祝福语
    print("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!")
    # 打印空行
    print()
    # 打印提示信息
    print("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.")
    # 打印4个空行
    print("\n" * 4)


def display_field() -> None:
    # 遍历5行
    for row in range(5):
        # 计算每行的初始值
        initial = row * 5 + 1
        # 打印每行的数字，用制表符分隔
        print("\t".join([str(initial + column) for column in range(5)]))
    # 打印9个空行
    print("\n" * 9)


def positions_list() -> List[int]:
    # 返回1到25的数字列表
    return list(range(1, 26, 1))


def generate_enemy_positions() -> Set[int]:
    """随机选择4个1到25的数字作为敌人的位置"""
    # 获取1到25的数字列表
    positions = positions_list()
    random.shuffle(positions)  # 随机打乱位置列表，以便随机选择4个位置
    return set(positions[:4])  # 返回打乱后的位置列表的前4个位置作为玩家的位置选择


def is_valid_position(pos: int) -> bool:
    return pos in positions_list()  # 检查给定的位置是否在有效的位置列表中


def prompt_for_player_positions() -> Set[int]:
    while True:
        raw_positions = input("WHAT ARE YOUR FOUR POSITIONS? ")  # 提示玩家输入4个位置
        positions = {int(pos) for pos in raw_positions.split()}  # 将玩家输入的位置转换为整数集合
        # Verify user inputs (for example, if the player gives a
        # a position for 26, the enemy can never hit it)
        if len(positions) != 4:  # 如果玩家输入的位置数量不等于4
            print("PLEASE ENTER 4 UNIQUE POSITIONS\n")  # 提示玩家输入4个唯一的位置
            continue  # 继续循环，等待玩家重新输入
        elif any(not is_valid_position(pos) for pos in positions):  # 如果玩家输入的位置中有任何一个不在有效位置列表中
            print("ALL POSITIONS MUST RANGE (1-25)\n")  # 提示玩家所有位置必须在1到25的范围内
            continue  # 继续循环，等待玩家重新输入
        else:
            return positions  # 如果不满足条件，返回已有的位置集合

def prompt_player_for_target() -> int:
    # 循环直到玩家输入有效的目标位置
    while True:
        target = int(input("WHERE DO YOU WISH TO FIRE YOUR MISSLE? "))  # 提示玩家输入目标位置
        if not is_valid_position(target):  # 如果目标位置不在有效范围内
            print("POSITIONS MUST RANGE (1-25)\n")  # 打印错误信息
            continue  # 继续循环
        return target  # 返回有效的目标位置

def attack(
    target: int,
    positions: Set[int],
    hit_message: str,
    miss_message: str,
```

需要注释的代码已经添加了注释。
    progress_messages: str,  # 用于存储进度消息的字符串
) -> bool:  # 指定函数返回的数据类型为布尔值

    """Performs attack procedure returning True if we are to continue."""
    # 执行攻击程序，如果需要继续则返回 True

    if target in positions:  # 如果目标在位置列表中
        print(hit_message.format(target))  # 打印命中消息并格式化目标
        positions.remove(target)  # 从位置列表中移除目标
        print(progress_messages[len(positions)].format(target))  # 打印进度消息并格式化目标
    else:  # 否则
        print(miss_message.format(target))  # 打印未命中消息并格式化目标

    return len(positions) > 0  # 返回位置列表的长度是否大于 0 的布尔值


def init_enemy() -> Callable[[], int]:  # 指定函数返回的数据类型为无参数函数，返回整数
    """
    Return a closure analogous to prompt_player_for_target.

    Will choose from a unique sequence of positions to avoid picking the
    same position twice.
    """
    # 返回类似于 prompt_player_for_target 的闭包
    # 将从唯一的位置序列中选择，以避免两次选择相同的位置
    # 生成位置序列并随机打乱顺序
    position_sequence = positions_list()
    random.shuffle(position_sequence)
    # 创建位置迭代器
    position = iter(position_sequence)

    # 选择下一个位置的函数
    def choose() -> int:
        return next(position)

    # 返回选择函数
    return choose


# 玩家进度对应的消息（3, 2, 1, 0）
PLAYER_PROGRESS_MESSAGES = (
    "YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\nMY TRANSISTO&S RECUP%RA*E!",
    "THREE DOWN, ONE TO GO.\n\n",
    "TWO DOWN, TWO TO GO.\n\n",
    "ONE DOWN, THREE TO GO.\n\n",
)
# 定义敌人进攻时的不同消息，根据不同情况选择不同的消息进行显示
ENEMY_PROGRESS_MESSAGES = (
    "YOU'RE DEAD. YOUR LAST OUTPOST WAS AT {}. HA, HA, HA.\nBETTER LUCK NEXT TIME.",
    "YOU HAVE ONLY ONE OUTPOST LEFT.\n\n",
    "YOU HAVE ONLY TWO OUTPOSTS LEFT.\n\n",
    "YOU HAVE ONLY THREE OUTPOSTS LEFT.\n\n",
)

# 定义主函数
def main() -> None:
    # 打印游戏介绍
    print_intro()
    # 显示游戏场地
    display_field()

    # 生成敌人位置
    enemy_positions = generate_enemy_positions()
    # 提示玩家输入自己的位置
    player_positions = prompt_for_player_positions()

    # 构建部分函数，只需要目标作为输入
    player_attacks = partial(
        attack,
        positions=enemy_positions,
        hit_message="YOU GOT ONE OF MY OUTPOSTS!",  # 设置击中消息
        miss_message="HA, HA YOU MISSED. MY TURN NOW:\n\n",  # 设置未击中消息
        progress_messages=PLAYER_PROGRESS_MESSAGES,  # 设置玩家进度消息
    )

    enemy_attacks = partial(
        attack,
        positions=player_positions,  # 设置玩家位置
        hit_message="I GOT YOU. IT WON'T BE LONG NOW. POST {} WAS HIT.",  # 设置敌人击中消息
        miss_message="I MISSED YOU, YOU DIRTY RAT. I PICKED {}. YOUR TURN:\n\n",  # 设置敌人未击中消息
        progress_messages=ENEMY_PROGRESS_MESSAGES,  # 设置敌人进度消息
    )

    enemy_position_choice = init_enemy()  # 初始化敌人位置选择

    # Play as long as both player_attacks and enemy_attacks allow to continue
    while player_attacks(prompt_player_for_target()) and enemy_attacks(
        enemy_position_choice()  # 当玩家和敌人都可以继续攻击时进行游戏
    ):
        pass
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
```

这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以使代码更具可重用性和模块化。
```