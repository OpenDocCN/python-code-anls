# `d:/src/tocomm/basic-computer-games\77_Salvo\python\salvo.py`

```
import random  # 导入 random 模块，用于生成随机数
import re  # 导入 re 模块，用于处理正则表达式
from typing import List, Optional, Tuple  # 从 typing 模块中导入 List、Optional、Tuple 类型

BoardType = List[List[Optional[int]]]  # 定义 BoardType 类型，表示一个二维列表，列表元素为可选的整数
CoordinateType = Tuple[int, int]  # 定义 CoordinateType 类型，表示一个包含两个整数的元组

BOARD_WIDTH = 10  # 定义 BOARD_WIDTH 常量，表示游戏板的宽度
BOARD_HEIGHT = 10  # 定义 BOARD_HEIGHT 常量，表示游戏板的高度


# data structure keeping track of information
# about the ships in the game. for each ship,
# the following information is provided:
#
#   name - string representation of the ship
#   length - number of "parts" on the ship that
#            can be shot
#   shots - number of shots the ship counts for
SHIPS = [  # 定义 SHIPS 列表，用于存储游戏中船只的信息
    ("BATTLESHIP", 5, 3),  # 定义了一个元组，包含船只名称、长度和数量
    ("CRUISER", 3, 2),  # 定义了一个元组，包含船只名称、长度和数量
    ("DESTROYER<A>", 2, 1),  # 定义了一个元组，包含船只名称、长度和数量
    ("DESTROYER<B>", 2, 1),  # 定义了一个元组，包含船只名称、长度和数量
]

VALID_MOVES = [
    [-1, 0],  # 北方向
    [-1, 1],  # 东北方向
    [0, 1],  # 东方向
    [1, 1],  # 东南方向
    [1, 0],  # 南方向
    [1, -1],  # 西南方向
    [0, -1],  # 西方向
    [-1, -1],  # 西北方向
]

COORD_REGEX = "[ \t]{0,}(-?[0-9]{1,3})[ \t]{0,},[ \t]{0,}(-?[0-9]{1,2})"  # 定义了一个正则表达式，用于匹配坐标格式
# array of BOARD_HEIGHT arrays, BOARD_WIDTH in length,
# representing the human player and computer
# 代表人类玩家和计算机的 BOARD_HEIGHT 个数组，每个数组的长度为 BOARD_WIDTH
player_board: BoardType = []  # 用于存储玩家的游戏板
computer_board: BoardType = []  # 用于存储计算机的游戏板

# array representing the coordinates
# for each ship for player and computer
# array is in the same order as SHIPS
# 代表玩家和计算机每艘船的坐标
# 数组的顺序与 SHIPS 相同
computer_ship_coords: List[List[CoordinateType]] = []  # 用于存储计算机每艘船的坐标

####################################
#
# SHOTS
#
# The number of shots computer/player
# has is determined by the shot "worth"
# of each ship the computer/player
# possesses. As long as the ship has one
# part not hit (i.e., ship was not
# 计算机/玩家的射击次数由计算机/玩家拥有的每艘船的射击价值确定。
# 只要船只有一个部分没有被击中（即船没有被击中
# sunk), the player gets all the shots
# from that ship.

# flag indicating if computer's shots are
# printed out during computer's turn
print_computer_shots = False  # 初始化一个变量，用于表示是否在计算机回合时打印计算机的射击情况

# keep track of the number
# of available computer shots
# inital shots are 7
num_computer_shots = 7  # 初始化一个变量，用于跟踪计算机的可用射击次数，初始值为7

# keep track of the number
# of available player shots
# initial shots are 7
num_player_shots = 7  # 初始化一个变量，用于跟踪玩家的可用射击次数，初始值为7

#
# SHOTS
#
####################################

# 表示当前轮到谁的标志
COMPUTER = False  # 计算机的回合
PLAYER = True  # 玩家的回合
active_turn = COMPUTER  # 当前活动的回合，默认为计算机的回合

####################
#
# 游戏函数
#
####################

# 随机数函数
#
# 设置随机数生成器的种子
random.seed()


# random_x_y
# 生成一个有效的 x，y 坐标
def random_x_y() -> CoordinateType:
    """Generate a valid x,y coordinate on the board"""

    # 生成一个随机的 x 坐标，范围在 1 到 BOARD_WIDTH + 1 之间
    x = random.randrange(1, BOARD_WIDTH + 1)
    # 生成一个随机的 y 坐标，范围在 1 到 BOARD_HEIGHT + 1 之间
    y = random.randrange(1, BOARD_HEIGHT + 1)
    # 返回生成的坐标
    return (x, y)


# 询问用户输入单个 (x,y) 坐标
def input_coord() -> CoordinateType:
    """
    Ask user for single (x,y) coordinate

    validate the coordinates are within the bounds
    of the board width and height. mimic the behavior
    of the original program which exited with error
    messages if coordinates where outside of array bounds.
    if input is not numeric, print error out to user and
    """
    let them try again.
    """
    # 初始化 match 变量
    match = None
    # 循环直到输入的坐标符合要求
    while not match:
        # 获取用户输入的坐标
        coords = input("? ")
        # 使用正则表达式匹配输入的坐标格式
        match = re.match(COORD_REGEX, coords)
        # 如果输入的坐标格式不符合要求，提示用户重新输入
        if not match:
            print("!NUMBER EXPECTED - RETRY INPUT LINE")
    # 获取匹配到的 x 和 y 坐标值
    x = int(match.group(1))
    y = int(match.group(2))

    # 检查坐标是否超出数组边界
    if x > BOARD_HEIGHT or y > BOARD_WIDTH:
        print("!OUT OF ARRAY BOUNDS IN LINE 1540")
        exit()

    # 检查坐标是否为负数
    if x <= 0 or y <= 0:
        print("!NEGATIVE ARRAY DIM IN LINE 1540")
        exit()

    # 返回有效的 x 和 y 坐标值
    return x, y
def generate_ship_coordinates(ship: int) -> List[CoordinateType]:
    """
    # 给定 SHIPS 数组中的一个船只，生成船只的坐标。船只的第一个坐标的起始点是随机生成的。
    # 一旦确定了起始坐标，就确定了船只可能的方向，考虑到棋盘的边缘。一旦找到可能的方向，就随机确定一个方向，然后通过加法或减法从起始坐标生成剩余的坐标，由方向确定。

    参数：
      ship - SHIPS 数组中的索引

    返回：
      坐标集合的数组 (x,y)
    """
    # 随机生成起始 x, y 坐标
    start_x, start_y = random_x_y()

    # 使用起始坐标和船只类型，生成可能放置船只的方向向量。
    # 方向按照罗盘点（N, NE, E, SE, S, SW, W, NW）编号为 0-7。
    # 顺时针方向。确定一个有效方向向量，使船只不会离开棋盘。
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
    # 根据有效方向的向量，选择一个随机方向来放置船只
    # 使用有效方向的向量，选择一个随机方向来放置船只
    dir_idx = random.randrange(len(directions))
    direction = directions[dir_idx]

    # 使用起始 x、y 坐标、方向和船只类型，返回船只每个点的坐标
    # VALID_MOVES 是一个静态数组，包含了从起始坐标到选择方向的结束坐标的坐标偏移量
    ship_len = SHIPS[ship][1] - 1
    d_x = VALID_MOVES[direction][0]
    d_y = VALID_MOVES[direction][1]

    coords = [(start_x, start_y)]
    x_coord = start_x
    y_coord = start_y
    for _ in range(ship_len):
        # 循环遍历船的长度，根据船的方向更新 x 和 y 坐标
        x_coord = x_coord + d_x
        y_coord = y_coord + d_y
        # 将更新后的坐标添加到坐标列表中
        coords.append((x_coord, y_coord))
    # 返回更新后的坐标列表
    return coords


def create_blank_board() -> BoardType:
    """创建一个空的游戏棋盘"""
    # 使用列表推导式创建一个空的游戏棋盘
    return [[None for _y in range(BOARD_WIDTH)] for _x in range(BOARD_HEIGHT)]


def print_board(board: BoardType) -> None:
    """打印游戏棋盘，用于测试目的"""
    # 打印棋盘的列号
    print("  ", end="")
    for z in range(BOARD_WIDTH):
        print(f"{z+1:3}", end="")
    print()
    for x in range(len(board)):  # 遍历二维数组的行
        print(f"{x+1:2}", end="")  # 打印行号
        for y in range(len(board[x])):  # 遍历二维数组的列
            if board[x][y] is None:  # 如果当前位置为空
                print(f"{' ':3}", end="")  # 打印空格
            else:
                print(f"{board[x][y]:3}", end="")  # 打印当前位置的值
        print()  # 换行

def place_ship(board: BoardType, coords: List[CoordinateType], ship: int) -> None:
    """
    Place a ship on a given board.

    updates
    the board's row,column value at the given
    coordinates to indicate where a ship is
    on the board.

    inputs: board - array of BOARD_HEIGHT by BOARD_WIDTH
    """
            coords - array of sets of (x,y) coordinates of each
                     part of the given ship
            ship - integer representing the type of ship (given in SHIPS)
    """
    # 遍历给定船只的坐标集合，将每个船只的坐标在游戏板上标记为对应的船只类型
    for coord in coords:
        board[coord[0] - 1][coord[1] - 1] = ship


def generate_board() -> Tuple[BoardType, List[List[CoordinateType]]]:
    """
    NOTE: A little quirk that exists here and in the orginal
          game: Ships are allowed to cross each other!
          For example: 2 destroyers, length 2, one at
          [(1,1),(2,2)] and other at [(2,1),(1,2)]
    """
    # 创建一个空白的游戏板
    board = create_blank_board()

    # 存储船只的坐标
    ship_coords = []
    # 遍历所有船只类型
    for ship in range(len(SHIPS)):
        placed = False
        coords = []  # 创建一个空列表用于存储坐标
        while not placed:  # 当未放置船时执行循环
            coords = generate_ship_coordinates(ship)  # 生成船的坐标
            clear = True  # 初始化一个布尔变量clear为True
            for coord in coords:  # 遍历船的坐标
                if board[coord[0] - 1][coord[1] - 1] is not None:  # 如果船的坐标在棋盘上不为空
                    clear = False  # 将clear设置为False
                    break  # 跳出循环
            if clear:  # 如果clear为True
                placed = True  # 将placed设置为True
        place_ship(board, coords, ship)  # 在棋盘上放置船
        ship_coords.append(coords)  # 将船的坐标添加到ship_coords列表中
    return board, ship_coords  # 返回棋盘和船的坐标列表


def execute_shot(
    turn: bool, board: BoardType, x: int, y: int, current_turn: int
) -> int:
    """
    given a board and x, y coordinates,
```  # 定义一个函数execute_shot，接受一个布尔类型的turn，一个BoardType类型的board，两个整数类型的x和y，一个整数类型的current_turn作为参数，并返回一个整数类型的值
    # 执行一次射击。如果射击有效，则返回True，否则返回False
    square = board[x - 1][y - 1]  # 获取指定坐标上的方块
    ship_hit = -1  # 初始化被击中的船的索引
    if square is not None and square >= 0 and square < len(SHIPS):  # 如果方块不为空且在船的索引范围内
        ship_hit = square  # 记录被击中的船的索引
    board[x - 1][y - 1] = 10 + current_turn  # 在指定坐标上标记为被射击
    return ship_hit  # 返回被击中的船的索引

def calculate_shots(board: BoardType) -> int:
    """检查每个棋盘并确定剩余的射击次数"""
    ships_found = [0 for x in range(len(SHIPS))]  # 初始化每艘船是否被发现的列表
    for x in range(BOARD_HEIGHT):  # 遍历棋盘的高度
        for y in range(BOARD_WIDTH):  # 遍历棋盘的宽度
            square = board[x - 1][y - 1]  # 获取指定坐标上的方块
            if square is not None and square >= 0 and square < len(SHIPS):  # 如果方块不为空且在船的索引范围内
                ships_found[square] = 1  # 将对应船的发现状态标记为1
    shots = 0  # 初始化射击次数为0
    for ship in range(len(ships_found)):  # 遍历 ships_found 列表的索引
        if ships_found[ship] == 1:  # 如果 ships_found 中对应索引的值为 1
            shots += SHIPS[ship][2]  # 将 SHIPS 列表中对应索引的第三个元素的值加到 shots 上

    return shots  # 返回 shots 的值


def initialize_game() -> None:  # 定义一个名为 initialize_game 的函数，不返回任何值
    # initialize the global player and computer boards  # 初始化全局变量 player_board 和 computer_board
    global player_board  # 声明 player_board 为全局变量
    player_board = create_blank_board()  # 调用 create_blank_board 函数，将返回的值赋给 player_board

    # generate the ships for the computer's board  # 为计算机的游戏板生成船只
    global computer_board  # 声明 computer_board 为全局变量
    global computer_ship_coords  # 声明 computer_ship_coords 为全局变量
    computer_board, computer_ship_coords = generate_board()  # 调用 generate_board 函数，将返回的值分别赋给 computer_board 和 computer_ship_coords

    # print out the title 'screen'  # 打印标题屏幕
    print("{:>38}".format("SALVO"))  # 打印标题 "SALVO"，右对齐，总宽度为 38
    print("{:>57s}".format("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"))  # 打印 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，右对齐，总宽度为 57
    # 打印原始作者和 Python 3 移植作者的信息
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

    # 将船只添加到用户的游戏板上
    for ship_index in range(len(SHIPS)):
        place_ship(player_board, ship_coords[ship_index], ship_index)
    # 看玩家是否想要打印出计算机的船只位置，并且玩家是否想要开始游戏
    input_loop = True
    player_start = "YES"
    while input_loop:
        player_start = input("DO YOU WANT TO START? ")
        if player_start == "WHERE ARE YOUR SHIPS?":  # 如果玩家输入"WHERE ARE YOUR SHIPS?"，则打印出计算机的船只位置
            for ship_index in range(len(SHIPS)):
                print(SHIPS[ship_index][0])  # 打印出船只的名称
                coords = computer_ship_coords[ship_index]  # 获取计算机船只的坐标
                for coord in coords:
                    x = coord[0]  # 获取x坐标
                    y = coord[1]  # 获取y坐标
                    print(f"{x:2}", f"{y:2}")  # 打印出坐标
        else:
            input_loop = False  # 如果玩家输入其他内容，则结束循环

    # 询问玩家是否希望在每个回合打印出计算机的射击位置
    global print_computer_shots  # 声明全局变量 print_computer_shots
    see_computer_shots = input("DO YOU WANT TO SEE MY SHOTS? ")  # 获取用户输入是否想要查看计算机的射击记录
    if see_computer_shots.lower() == "yes":  # 如果用户输入是 "yes"，则将 print_computer_shots 设置为 True
        print_computer_shots = True

    global first_turn  # 声明全局变量 first_turn
    if player_start.lower() != "yes":  # 如果玩家不想先手，则将 first_turn 设置为 COMPUTER
        first_turn = COMPUTER

    # 计算每方初始的射击次数
    global num_computer_shots, num_player_shots  # 声明全局变量 num_computer_shots 和 num_player_shots
    num_player_shots = calculate_shots(player_board)  # 计算玩家的射击次数
    num_computer_shots = calculate_shots(computer_board)  # 计算计算机的射击次数
# initialize the first_turn function to the player's turn
# 将第一轮的玩家设为初始值
first_turn = PLAYER


def execute_turn(turn: bool, current_turn: int) -> int:
    # 声明全局变量
    global num_computer_shots, num_player_shots

    # print out the number of shots the current player has
    # 打印当前玩家的射击次数
    board = None
    num_shots = 0
    if turn == COMPUTER:
        # 如果轮到电脑，打印电脑的射击次数
        print(f"I HAVE {num_computer_shots} SHOTS.")
        # 设置board为玩家的游戏板
        board = player_board
        # 设置num_shots为电脑的射击次数
        num_shots = num_computer_shots
    else:
        # 如果不是第一次射击，则打印玩家还剩余的射击次数
        print(f"YOU HAVE {num_player_shots} SHOTS.")
        # 将玩家的射击板赋值给电脑的射击板
        board = computer_board
        # 将玩家的射击次数赋值给总的射击次数
        num_shots = num_player_shots

    # 初始化射击列表
    shots = []
    # 循环进行射击
    for _shot in range(num_shots):
        # 初始化射击是否有效的标志
        valid_shot = False
        x = -1
        y = -1

        # 循环直到得到有效的射击坐标。对于电脑来说，随机选择一个射击坐标。对于玩家来说，请求输入射击坐标
        while not valid_shot:
            if turn == COMPUTER:
                # 对于电脑，随机选择一个射击坐标
                x, y = random_x_y()
            else:
                # 对于玩家，请求输入射击坐标
                x, y = input_coord()
            # 获取射击坐标对应的方块
            square = board[x - 1][y - 1]
            if square is not None and square > 10:  # 如果方块不为空且大于10
                if turn == PLAYER:  # 如果轮到玩家
                    print("YOU SHOT THERE BEFORE ON TURN", square - 10)  # 打印“你在第(square-10)轮之前就在那里开过火”
                continue  # 继续下一次循环
            shots.append((x, y))  # 将(x, y)添加到射击列表中
            valid_shot = True  # 有效射击为真

    hits = []  # 创建一个空列表用于存储命中的射击
    for shot in shots:  # 遍历射击列表
        hit = execute_shot(turn, board, shot[0], shot[1], current_turn)  # 执行射击并将结果存储在hit中
        if hit >= 0:  # 如果命中大于等于0
            hits.append(hit)  # 将命中结果添加到hits列表中
        if turn == COMPUTER and print_computer_shots:  # 如果轮到电脑并且需要打印电脑的射击
            print(shot[0], shot[1])  # 打印射击的坐标

    for hit in hits:  # 遍历命中列表
        if turn == COMPUTER:  # 如果轮到电脑
            print("I HIT YOUR", SHIPS[hit][0])  # 打印“我击中了你的”和被击中船只的名称
        else:  # 否则
            print("YOU HIT MY", SHIPS[hit][0])  # 打印“你击中了我的”和被击中船只的名称
    if turn == COMPUTER:  # 如果轮到电脑
        num_player_shots = calculate_shots(board)  # 计算玩家的射击次数
        return num_player_shots  # 返回玩家的射击次数
    else:
        num_computer_shots = calculate_shots(board)  # 计算电脑的射击次数
        return num_computer_shots  # 返回电脑的射击次数


#
# Turn Control
#
######################################


def main() -> None:
    current_turn = 0  # 初始化当前轮数为0
    initialize_game()  # 初始化游戏

    # execute turns until someone wins or we run  # 执行轮次直到有人获胜或游戏结束
    # out of squares to shoot
    # 没有可以射击的方块
    game_over = False
    # 当游戏未结束时循环执行
    while not game_over:
        current_turn += 1

        print("\n")
        print("TURN", current_turn)

        # 如果执行当前回合时返回值为0，或者执行下一回合时返回值为0，则游戏结束
        if (
            execute_turn(first_turn, current_turn) == 0
            or execute_turn(not first_turn, current_turn) == 0
        ):
            game_over = True
            continue

# 如果当前文件为主程序，则执行main函数
if __name__ == "__main__":
    main()
```