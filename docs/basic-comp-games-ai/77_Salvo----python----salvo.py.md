# `basic-computer-games\77_Salvo\python\salvo.py`

```py
# 导入 random 模块
import random
# 导入 re 模块
import re
# 导入 List、Optional、Tuple 类型
from typing import List, Optional, Tuple

# 定义 BoardType 类型
BoardType = List[List[Optional[int]]]
# 定义 CoordinateType 类型
CoordinateType = Tuple[int, int]

# 定义棋盘宽度和高度
BOARD_WIDTH = 10
BOARD_HEIGHT = 10

# 定义船只信息的数据结构
SHIPS = [
    ("BATTLESHIP", 5, 3),  # 战舰
    ("CRUISER", 3, 2),  # 巡洋舰
    ("DESTROYER<A>", 2, 1),  # 驱逐舰A
    ("DESTROYER<B>", 2, 1),  # 驱逐舰B
]

# 定义有效移动的方向
VALID_MOVES = [
    [-1, 0],  # 北
    [-1, 1],  # 东北
    [0, 1],  # 东
    [1, 1],  # 东南
    [1, 0],  # 南
    [1, -1],  # 西南
    [0, -1],  # 西
    [-1, -1],  # 西北
]

# 定义坐标的正则表达式
COORD_REGEX = "[ \t]{0,}(-?[0-9]{1,3})[ \t]{0,},[ \t]{0,}(-?[0-9]{1,2})"

# 代表玩家和计算机的棋盘数组
player_board: BoardType = []  # 玩家棋盘
computer_board: BoardType = []  # 计算机棋盘

# 代表玩家和计算机的船只坐标数组，与 SHIPS 的顺序一致
computer_ship_coords: List[List[CoordinateType]] = []  # 计算机船只坐标

# 标志是否打印计算机的射击信息
print_computer_shots = False

# 记录计算机可用的射击次数，初始为 7
num_computer_shots = 7

# 记录玩家可用的射击次数，初始为 7
num_player_shots = 7

# 标志当前轮到谁的回合
COMPUTER = False
PLAYER = True
active_turn = COMPUTER
# 游戏功能
#
####################

# 随机数函数
#
# 设置随机数生成器的种子
random.seed()


# random_x_y
#

def random_x_y() -> CoordinateType:
    """生成棋盘上有效的 x，y 坐标"""

    x = random.randrange(1, BOARD_WIDTH + 1)  # 生成 1 到 BOARD_WIDTH 之间的随机 x 坐标
    y = random.randrange(1, BOARD_HEIGHT + 1)  # 生成 1 到 BOARD_HEIGHT 之间的随机 y 坐标
    return (x, y)


def input_coord() -> CoordinateType:
    """
    请求用户输入单个 (x, y) 坐标

    验证坐标是否在棋盘宽度和高度范围内。模仿原始程序的行为，如果坐标超出数组边界，则退出并显示错误消息。
    如果输入不是数字，则向用户打印错误消息，并让他们重试。
    """
    match = None
    while not match:
        coords = input("? ")  # 请求用户输入坐标
        match = re.match(COORD_REGEX, coords)  # 使用正则表达式验证输入的坐标格式
        if not match:
            print("!NUMBER EXPECTED - RETRY INPUT LINE")  # 如果输入不是数字，则提示用户重新输入
    x = int(match.group(1))  # 将 x 坐标转换为整数
    y = int(match.group(2))  # 将 y 坐标转换为整数

    if x > BOARD_HEIGHT or y > BOARD_WIDTH:  # 如果 x 或 y 坐标超出了棋盘边界
        print("!OUT OF ARRAY BOUNDS IN LINE 1540")  # 显示错误消息
        exit()  # 退出程序

    if x <= 0 or y <= 0:  # 如果 x 或 y 坐标为负数
        print("!NEGATIVE ARRAY DIM IN LINE 1540")  # 显示错误消息
        exit()  # 退出程序

    return x, y  # 返回坐标值


def generate_ship_coordinates(ship: int) -> List[CoordinateType]:
    """
    给定 SHIPS 数组中的一艘船，生成船的坐标。
    随机生成船的起始坐标。
    一旦确定了起始坐标，就确定了船的可能方向，考虑到棋盘的边缘。
    一旦找到可能的方向，就随机确定一个方向，并通过加法或减法从起始坐标生成剩余的坐标。

    参数：
      ship - SHIPS 数组中的索引

    返回：
      坐标集合数组 (x, y)
    """
    # 随机生成起始 x，y 坐标
    # 生成随机的起始坐标
    start_x, start_y = random_x_y()

    # 根据起始坐标和船只类型，生成可能放置船只的方向向量。
    # 方向按照罗盘点（N, NE, E, SE, S, SW, W, NW）编号为0-7。
    # 顺时针方向。确定船只不会离开棋盘的有效方向向量
    ship_len = SHIPS[ship][1] - 1
    dirs = [False for x in range(8)]
    dirs[0] = (start_x - ship_len) >= 1
    dirs[2] = (start_y + ship_len) <= BOARD_WIDTH
    dirs[1] = dirs[0] and dirs[2]
    dirs[4] = (start_x + ship_len) <= BOARD_HEIGHT
    dirs[3] = dirs[2] and dirs[4]
    dirs[6] = (start_y - ship_len) >= 1
    dirs[5] = dirs[4] and dirs[6]
    dirs[7] = dirs[6] and dirs[0]
    directions = [p for p in range(len(dirs)) if dirs[p]]

    # 从有效方向向量中随机选择一个方向来放置船只
    dir_idx = random.randrange(len(directions))
    direction = directions[dir_idx]

    # 根据起始坐标、方向和船只类型，返回船只每个点的坐标。
    # VALID_MOVES是一个静态数组，包含了从起始坐标到选择方向的终点坐标的坐标偏移量
    ship_len = SHIPS[ship][1] - 1
    d_x = VALID_MOVES[direction][0]
    d_y = VALID_MOVES[direction][1]

    coords = [(start_x, start_y)]
    x_coord = start_x
    y_coord = start_y
    for _ in range(ship_len):
        x_coord = x_coord + d_x
        y_coord = y_coord + d_y
        coords.append((x_coord, y_coord))
    return coords
# 创建一个空的游戏棋盘
def create_blank_board() -> BoardType:
    # 返回一个二维数组，每个元素为 None，表示空白
    return [[None for _y in range(BOARD_WIDTH)] for _x in range(BOARD_HEIGHT)]


# 打印游戏棋盘，用于测试目的
def print_board(board: BoardType) -> None:
    # 打印棋盘头部（列号）
    print("  ", end="")
    for z in range(BOARD_WIDTH):
        print(f"{z+1:3}", end="")
    print()

    # 遍历棋盘，打印每个格子的内容
    for x in range(len(board)):
        print(f"{x+1:2}", end="")
        for y in range(len(board[x])):
            if board[x][y] is None:
                print(f"{' ':3}", end="")
            else:
                print(f"{board[x][y]:3}", end="")
        print()


# 在给定的棋盘上放置一艘船
def place_ship(board: BoardType, coords: List[CoordinateType], ship: int) -> None:
    """
    在给定的棋盘上放置一艘船。

    更新棋盘上给定坐标的行、列值，表示船在棋盘上的位置。

    输入：board - BOARD_HEIGHT x BOARD_WIDTH 的数组
          coords - 每艘船的 (x, y) 坐标集合的数组
          ship - 表示船的类型的整数（在 SHIPS 中给出）
    """
    for coord in coords:
        board[coord[0] - 1][coord[1] - 1] = ship


# 生成游戏棋盘
def generate_board() -> Tuple[BoardType, List[List[CoordinateType]]]:
    """
    注意：这里和原始游戏中存在一个小问题：船可以相互交叉！
          例如：2艘长度为2的驱逐舰，一个在[(1,1),(2,2)]，另一个在[(2,1),(1,2)]
    """
    # 创建一个空的游戏棋盘
    board = create_blank_board()

    # 存储船的坐标
    ship_coords = []
    # 遍历船只列表的长度范围
    for ship in range(len(SHIPS)):
        # 标记船只是否已经放置
        placed = False
        # 初始化坐标列表
        coords = []
        # 当船只未被放置时循环
        while not placed:
            # 生成船只的坐标
            coords = generate_ship_coordinates(ship)
            # 标记是否所有坐标都为空
            clear = True
            # 遍历坐标列表
            for coord in coords:
                # 如果坐标对应的位置不为空
                if board[coord[0] - 1][coord[1] - 1] is not None:
                    # 标记为不清空
                    clear = False
                    # 跳出循环
                    break
            # 如果所有坐标都为空
            if clear:
                # 标记船只已经放置
                placed = True
        # 放置船只到棋盘上
        place_ship(board, coords, ship)
        # 将船只的坐标添加到船只坐标列表中
        ship_coords.append(coords)
    # 返回放置好船只的棋盘和船只坐标列表
    return board, ship_coords
# 执行射击，根据给定的坐标和当前回合数，在棋盘上执行射击，如果射击有效则返回True，否则返回False
def execute_shot(
    turn: bool, board: BoardType, x: int, y: int, current_turn: int
) -> int:
    """
    given a board and x, y coordinates,
    execute a shot. returns True if the shot
    is valid, False if not
    """
    # 获取指定坐标上的方块
    square = board[x - 1][y - 1]
    # 初始化击中的船的索引
    ship_hit = -1
    # 如果方块不为空且值大于等于0且小于船只数量，则表示击中了船只
    if square is not None and square >= 0 and square < len(SHIPS):
        ship_hit = square
    # 在棋盘上标记当前回合的射击
    board[x - 1][y - 1] = 10 + current_turn
    # 返回击中的船的索引
    return ship_hit


# 计算剩余的射击次数
def calculate_shots(board: BoardType) -> int:
    """Examine each board and determine how many shots remaining"""
    # 初始化已发现的船只列表
    ships_found = [0 for x in range(len(SHIPS))]
    # 遍历整个棋盘
    for x in range(BOARD_HEIGHT):
        for y in range(BOARD_WIDTH):
            # 获取指定坐标上的方块
            square = board[x - 1][y - 1]
            # 如果方块不为空且值大于等于0且小于船只数量，则表示发现了船只
            if square is not None and square >= 0 and square < len(SHIPS):
                ships_found[square] = 1
    # 初始化射击次数
    shots = 0
    # 统计已发现的船只需要的射击次数
    for ship in range(len(ships_found)):
        if ships_found[ship] == 1:
            shots += SHIPS[ship][2]

    return shots


# 初始化游戏
def initialize_game() -> None:
    # 初始化全局玩家和计算机的棋盘
    global player_board
    player_board = create_blank_board()

    # 生成计算机棋盘上的船只
    global computer_board
    global computer_ship_coords
    computer_board, computer_ship_coords = generate_board()

    # 打印标题
    print("{:>38}".format("SALVO"))
    print("{:>57s}".format("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"))
    print()
    print("{:>52s}".format("ORIGINAL BY LAWRENCE SIEGEL, 1973"))
    print("{:>56s}".format("PYTHON 3 PORT BY TODD KAISER, MARCH 2021"))
    print("\n")

    # 询问玩家船只的坐标
    print("ENTER COORDINATES FOR...")
    ship_coords = []
    for ship in SHIPS:
        print(ship[0])
        list = []
        for _ in range(ship[1]):
            x, y = input_coord()
            list.append((x, y))
        ship_coords.append(list)

    # 将船只添加到玩家的棋盘
    # 遍历玩家的船只列表，依次放置船只到玩家的游戏板上
    for ship_index in range(len(SHIPS)):
        place_ship(player_board, ship_coords[ship_index], ship_index)

    # 询问玩家是否想要打印出计算机的船只位置，并且询问玩家是否想要开始游戏
    input_loop = True
    player_start = "YES"
    while input_loop:
        player_start = input("DO YOU WANT TO START? ")
        if player_start == "WHERE ARE YOUR SHIPS?":
            # 如果玩家想要知道计算机的船只位置，打印出计算机的船只位置
            for ship_index in range(len(SHIPS)):
                print(SHIPS[ship_index][0])
                coords = computer_ship_coords[ship_index]
                for coord in coords:
                    x = coord[0]
                    y = coord[1]
                    print(f"{x:2}", f"{y:2}")
        else:
            input_loop = False

    # 询问玩家是否想要打印出计算机每一轮的射击位置
    global print_computer_shots
    see_computer_shots = input("DO YOU WANT TO SEE MY SHOTS? ")
    if see_computer_shots.lower() == "yes":
        print_computer_shots = True

    # 如果玩家不想要开始游戏，将首次轮到计算机进行射击
    global first_turn
    if player_start.lower() != "yes":
        first_turn = COMPUTER

    # 计算初始时玩家和计算机的射击次数
    global num_computer_shots, num_player_shots
    num_player_shots = calculate_shots(player_board)
    num_computer_shots = calculate_shots(computer_board)
# 初始化第一个轮次为玩家的轮次
first_turn = PLAYER

# 定义执行轮次的函数，根据当前轮次和玩家的轮次数返回整数
def execute_turn(turn: bool, current_turn: int) -> int:
    global num_computer_shots, num_player_shots

    # 打印当前玩家的射击次数
    board = None
    num_shots = 0
    if turn == COMPUTER:
        print(f"I HAVE {num_computer_shots} SHOTS.")
        board = player_board
        num_shots = num_computer_shots
    else:
        print(f"YOU HAVE {num_player_shots} SHOTS.")
        board = computer_board
        num_shots = num_player_shots

    shots = []
    for _shot in range(num_shots):
        valid_shot = False
        x = -1
        y = -1

        # 循环直到获得有效的射击。对于计算机，随机选择一个射击位置。对于玩家，请求射击位置
        while not valid_shot:
            if turn == COMPUTER:
                x, y = random_x_y()
            else:
                x, y = input_coord()
            square = board[x - 1][y - 1]
            if square is not None and square > 10:
                if turn == PLAYER:
                    print("YOU SHOT THERE BEFORE ON TURN", square - 10)
                continue
            shots.append((x, y))
            valid_shot = True

    hits = []
    for shot in shots:
        # 执行射击，并将结果添加到hits列表中
        hit = execute_shot(turn, board, shot[0], shot[1], current_turn)
        if hit >= 0:
            hits.append(hit)
        if turn == COMPUTER and print_computer_shots:
            print(shot[0], shot[1])

    for hit in hits:
        if turn == COMPUTER:
            print("I HIT YOUR", SHIPS[hit][0])
        else:
            print("YOU HIT MY", SHIPS[hit][0])
    # 如果轮到计算机行动
    if turn == COMPUTER:
        # 计算玩家的射击次数
        num_player_shots = calculate_shots(board)
        # 返回玩家的射击次数
        return num_player_shots
    # 如果轮到玩家行动
    else:
        # 计算计算机的射击次数
        num_computer_shots = calculate_shots(board)
        # 返回计算机的射击次数
        return num_computer_shots
#
# Turn Control
#
######################################

# 定义主函数
def main() -> None:
    # 当前回合数初始化为0
    current_turn = 0
    # 初始化游戏
    initialize_game()

    # 执行回合直到有人获胜或者没有可射击的方块
    game_over = False
    while not game_over:
        # 回合数加1
        current_turn += 1

        # 打印当前回合数
        print("\n")
        print("TURN", current_turn)

        # 如果执行第一个回合或者执行第二个回合的结果为0，则游戏结束
        if (
            execute_turn(first_turn, current_turn) == 0
            or execute_turn(not first_turn, current_turn) == 0
        ):
            game_over = True
            continue

# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```