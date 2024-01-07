# `basic-computer-games\11_Bombardment\python\bombardment.py`

```

#!/usr/bin/env python3
# 指定脚本解释器为python3

import random
# 导入random模块，用于生成随机数
from functools import partial
# 导入functools模块中的partial函数，用于创建偏函数
from typing import Callable, List, Set
# 导入typing模块中的Callable、List、Set类型，用于类型提示

def print_intro() -> None:
    # 打印游戏介绍
    ...

def display_field() -> None:
    # 打印游戏场地
    ...

def positions_list() -> List[int]:
    # 返回一个包含1到25的整数列表
    ...

def generate_enemy_positions() -> Set[int]:
    # 随机选择4个1到25的整数，返回一个集合
    ...

def is_valid_position(pos: int) -> bool:
    # 判断位置是否合法
    ...

def prompt_for_player_positions() -> Set[int]:
    # 提示玩家输入4个位置
    ...

def prompt_player_for_target() -> int:
    # 提示玩家选择目标位置
    ...

def attack(
    target: int,
    positions: Set[int],
    hit_message: str,
    miss_message: str,
    progress_messages: str,
) -> bool:
    # 执行攻击过程，返回是否继续游戏
    ...

def init_enemy() -> Callable[[], int]:
    # 返回一个类似于prompt_player_for_target的闭包函数
    ...

# Messages correspond to outposts remaining (3, 2, 1, 0)
# 消息对应剩余的哨所数量（3, 2, 1, 0）

def main() -> None:
    # 主函数
    ...

if __name__ == "__main__":
    main()
    # 如果当前脚本为主程序，则执行main函数

```