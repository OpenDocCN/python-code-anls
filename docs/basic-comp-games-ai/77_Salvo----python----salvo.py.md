# `basic-computer-games\77_Salvo\python\salvo.py`

```

import random  # 导入random模块，用于生成随机数
import re  # 导入re模块，用于正则表达式匹配
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

BoardType = List[List[Optional[int]]]  # 定义BoardType类型，表示一个二维列表
CoordinateType = Tuple[int, int]  # 定义CoordinateType类型，表示一个包含两个整数的元组

BOARD_WIDTH = 10  # 定义游戏板的宽度
BOARD_HEIGHT = 10  # 定义游戏板的高度

# 定义游戏中船只的信息，包括名称、长度和射击次数
SHIPS = [
    ("BATTLESHIP", 5, 3),
    ("CRUISER", 3, 2),
    ("DESTROYER<A>", 2, 1),
    ("DESTROYER<B>", 2, 1),
]

# 定义允许的移动方向
VALID_MOVES = [
    [-1, 0],  # North
    [-1, 1],  # North East
    [0, 1],  # East
    [1, 1],  # South East
    [1, 0],  # South
    [1, -1],  # South West
    [0, -1],  # West
    [-1, -1],  # North West
]

COORD_REGEX = "[ \t]{0,}(-?[0-9]{1,3})[ \t]{0,},[ \t]{0,}(-?[0-9]{1,2})"  # 定义坐标的正则表达式

# 定义玩家和计算机的游戏板
player_board: BoardType = []  # 玩家游戏板
computer_board: BoardType = []  # 计算机游戏板
computer_ship_coords: List[List[CoordinateType]] = []  # 计算机船只的坐标

# 以下是游戏中射击相关的变量和函数
print_computer_shots = False  # 是否打印计算机的射击情况
num_computer_shots = 7  # 计算机的射击次数
num_player_shots = 7  # 玩家的射击次数

# 标识当前轮到谁的回合
COMPUTER = False
PLAYER = True
active_turn = COMPUTER  # 当前轮到计算机先行

# 初始化随机数生成器
random.seed()

# 以下是一些游戏函数的定义，包括生成坐标、输入坐标、生成船只坐标、创建空游戏板、打印游戏板、放置船只等

# 初始化游戏
def initialize_game() -> None:
    # 初始化玩家和计算机的游戏板
    global player_board
    player_board = create_blank_board()
    global computer_board
    global computer_ship_coords
    computer_board, computer_ship_coords = generate_board()

    # 询问玩家是否想要打印计算机的船只位置和是否想要先行
    input_loop = True
    player_start = "YES"
    while input_loop:
        player_start = input("DO YOU WANT TO START? ")
        if player_start == "WHERE ARE YOUR SHIPS?":
            for ship_index in range(len(SHIPS)):
                print(SHIPS[ship_index][0])
                coords = computer_ship_coords[ship_index]
                for coord in coords:
                    x = coord[0]
                    y = coord[1]
                    print(f"{x:2}", f"{y:2}")
        else:
            input_loop = False

    see_computer_shots = input("DO YOU WANT TO SEE MY SHOTS? ")
    if see_computer_shots.lower() == "yes":
        print_computer_shots = True

    global first_turn
    if player_start.lower() != "yes":
        first_turn = COMPUTER

    global num_computer_shots, num_player_shots
    num_player_shots = calculate_shots(player_board)
    num_computer_shots = calculate_shots(computer_board)

# 执行游戏的主要逻辑
def main() -> None:
    current_turn = 0
    initialize_game()

    game_over = False
    while not game_over:
        current_turn += 1

        print("\n")
        print("TURN", current_turn)

        if (
            execute_turn(first_turn, current_turn) == 0
            or execute_turn(not first_turn, current_turn) == 0
        ):
            game_over = True
            continue

if __name__ == "__main__":
    main()

```