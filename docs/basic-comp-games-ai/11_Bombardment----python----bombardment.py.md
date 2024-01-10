# `basic-computer-games\11_Bombardment\python\bombardment.py`

```
#!/usr/bin/env python3
# 导入 random 模块
import random
# 导入 functools 模块中的 partial 函数
from functools import partial
# 导入 typing 模块中的 Callable、List、Set 类型
from typing import Callable, List, Set

# 打印游戏介绍
def print_intro() -> None:
    print(" " * 33 + "BOMBARDMENT")
    print(" " * 15 + " CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU")
    print("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.")
    print("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.")
    print("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.")
    print()
    print("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE")
    print("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.")
    print("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS")
    print("FIRST IS THE WINNER.")
    print()
    print("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!")
    print()
    print("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.")
    print("\n" * 4)

# 显示战场
def display_field() -> None:
    for row in range(5):
        initial = row * 5 + 1
        print("\t".join([str(initial + column) for column in range(5)]))
    print("\n" * 9)

# 生成位置列表
def positions_list() -> List[int]:
    return list(range(1, 26, 1))

# 生成敌方位置
def generate_enemy_positions() -> Set[int]:
    """Randomly choose 4 'positions' out of a range of 1 to 25"""
    positions = positions_list()
    random.shuffle(positions)
    return set(positions[:4])

# 检查位置是否有效
def is_valid_position(pos: int) -> bool:
    return pos in positions_list()

# 提示玩家选择位置
def prompt_for_player_positions() -> Set[int]:
    # 无限循环，直到用户输入正确的四个位置
    while True:
        # 从用户输入中获取四个位置，并转换成整数集合
        raw_positions = input("WHAT ARE YOUR FOUR POSITIONS? ")
        positions = {int(pos) for pos in raw_positions.split()}
        # 验证用户输入（例如，如果玩家给出一个26的位置，敌人永远无法击中它）
        if len(positions) != 4:
            # 如果位置数量不等于4，提示用户重新输入
            print("PLEASE ENTER 4 UNIQUE POSITIONS\n")
            continue
        elif any(not is_valid_position(pos) for pos in positions):
            # 如果任何一个位置不在有效范围内，提示用户重新输入
            print("ALL POSITIONS MUST RANGE (1-25)\n")
            continue
        else:
            # 如果用户输入正确，返回位置集合
            return positions
# 从玩家输入中获取目标位置
def prompt_player_for_target() -> int:

    while True:
        # 获取玩家输入的目标位置
        target = int(input("WHERE DO YOU WISH TO FIRE YOUR MISSLE? "))
        # 检查目标位置是否有效
        if not is_valid_position(target):
            print("POSITIONS MUST RANGE (1-25)\n")
            continue

        return target


# 执行攻击程序，如果需要继续则返回 True
def attack(
    target: int,
    positions: Set[int],
    hit_message: str,
    miss_message: str,
    progress_messages: str,
) -> bool:
    """Performs attack procedure returning True if we are to continue."""

    if target in positions:
        # 如果目标位置在敌方位置集合中，打印命中信息并移除目标位置
        print(hit_message.format(target))
        positions.remove(target)
        # 打印进度信息
        print(progress_messages[len(positions)].format(target))
    else:
        # 如果目标位置不在敌方位置集合中，打印未命中信息
        print(miss_message.format(target))

    return len(positions) > 0


# 初始化敌方位置选择器
def init_enemy() -> Callable[[], int]:
    """
    Return a closure analogous to prompt_player_for_target.

    Will choose from a unique sequence of positions to avoid picking the
    same position twice.
    """

    # 生成敌方位置序列并随机打乱
    position_sequence = positions_list()
    random.shuffle(position_sequence)
    position = iter(position_sequence)

    # 返回一个函数，每次调用返回敌方位置序列中的下一个位置
    def choose() -> int:
        return next(position)

    return choose


# 玩家剩余据点数量对应的消息
PLAYER_PROGRESS_MESSAGES = (
    "YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\nMY TRANSISTO&S RECUP%RA*E!",
    "THREE DOWN, ONE TO GO.\n\n",
    "TWO DOWN, TWO TO GO.\n\n",
    "ONE DOWN, THREE TO GO.\n\n",
)


# 敌方剩余据点数量对应的消息
ENEMY_PROGRESS_MESSAGES = (
    "YOU'RE DEAD. YOUR LAST OUTPOST WAS AT {}. HA, HA, HA.\nBETTER LUCK NEXT TIME.",
    "YOU HAVE ONLY ONE OUTPOST LEFT.\n\n",
    "YOU HAVE ONLY TWO OUTPOSTS LEFT.\n\n",
    "YOU HAVE ONLY THREE OUTPOSTS LEFT.\n\n",
)


# 主函数
def main() -> None:
    # 打印游戏介绍
    print_intro()
    # 显示游戏场地
    display_field()

    # 生成敌方位置
    enemy_positions = generate_enemy_positions()
    # 获取玩家位置
    player_positions = prompt_for_player_positions()

    # 构建只需要目标位置作为输入的部分函数
    # 创建一个名为 player_attacks 的函数，使用 attack 函数的部分参数，并设置特定的消息和进度消息
    player_attacks = partial(
        attack,
        positions=enemy_positions,
        hit_message="YOU GOT ONE OF MY OUTPOSTS!",
        miss_message="HA, HA YOU MISSED. MY TURN NOW:\n\n",
        progress_messages=PLAYER_PROGRESS_MESSAGES,
    )

    # 创建一个名为 enemy_attacks 的函数，使用 attack 函数的部分参数，并设置特定的消息和进度消息
    enemy_attacks = partial(
        attack,
        positions=player_positions,
        hit_message="I GOT YOU. IT WON'T BE LONG NOW. POST {} WAS HIT.",
        miss_message="I MISSED YOU, YOU DIRTY RAT. I PICKED {}. YOUR TURN:\n\n",
        progress_messages=ENEMY_PROGRESS_MESSAGES,
    )

    # 初始化敌方的位置选择
    enemy_position_choice = init_enemy()

    # 只要 player_attacks 和 enemy_attacks 允许继续，就一直进行游戏
    while player_attacks(prompt_player_for_target()) and enemy_attacks(
        enemy_position_choice()
    ):
        pass
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```