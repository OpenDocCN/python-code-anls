# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\5_tic_tac_toe\artifacts_out\tic_tac_toe.py`

```py
# 导入 pprint 模块，用于漂亮打印数据结构
import pprint

# 获取矩阵中指定列的元素列表
def column(matrix, i):
    return [row[i] for row in matrix]

# 检查列表中的元素是否相同，如果相同且不为 0，则返回该元素，否则返回 None
def check(list):
    if len(set(list)) <= 1:
        if list[0] != 0:
            return list[0]
    return None

# 检查左对角线上的元素是否相同，如果相同且不为 0，则返回该元素，否则返回 None
def checkDiagLeft(board):
    if board[0][0] == board[1][1] and board[1][1] == board[2][2]:
        if board[0][0] != 0:
            return board[0][0]
    return None

# 检查右对角线上的元素是否相同，如果相同且不为 0，则返回该元素，否则返回 None
def checkDiagRight(board):
    if board[2][0] == board[1][1] and board[1][1] == board[0][2]:
        if board[2][0] != 0:
            return board[2][0]
    return None

# 在棋盘的指定行列放置当前玩家的棋子
def placeItem(row, column, board, current_player):
    if board[row][column] != 0:
        return None
    else:
        board[row][column] = current_player

# 切换玩家
def swapPlayers(player):
    if player == 2:
        return 1
    else:
        return 2

# 检查是否有玩家获胜，返回获胜玩家的编号或 0（平局）
def winner(board):
    for rowIndex in board:
        if check(rowIndex) is not None:
            return check(rowIndex)
    for columnIndex in range(len(board[0])):
        if check(column(board, columnIndex)) is not None:
            return check(column(board, columnIndex))
    if checkDiagLeft(board) is not None:
        return checkDiagLeft(board)
    if checkDiagRight(board) is not None:
        return checkDiagRight(board)
    return 0

# 获取玩家输入的位置坐标
def getLocation():
    location = input(
        "Choose where to play. Enter two numbers separated by a comma, for example: 1,1 "
    )
    print(f"\nYou picked {location}")
    coordinates = [int(x) for x in location.split(",")]
    while (
        len(coordinates) != 2
        or coordinates[0] < 0
        or coordinates[0] > 2
        or coordinates[1] < 0
        or coordinates[1] > 2
    ):
        print("You inputted a location in an invalid format")
        location = input(
            "Choose where to play. Enter two numbers separated by a comma, for example: 1,1 "
        )
        coordinates = [int(x) for x in location.split(",")]
    return coordinates

# 游戏主循环
def gamePlay():
    num_moves = 0
    # 创建一个 PrettyPrinter 对象，用于漂亮打印数据结构
    pp = pprint.PrettyPrinter(width=20)
    # 当前玩家，默认为1
    current_player = 1
    # 初始化一个3x3的棋盘，用0表示空位
    board = [[0 for x in range(3)] for x in range(3)]
    
    # 当前移动次数小于9且没有获胜者时，继续游戏
    while num_moves < 9 and winner(board) == 0:
        # 打印当前棋盘状态
        print("This is the current board: ")
        pp.pprint(board)
        # 获取玩家输入的坐标
        coordinates = getLocation()
        # 在指定坐标放置当前玩家的棋子
        placeItem(coordinates[0], coordinates[1], board, current_player)
        # 切换当前玩家
        current_player = swapPlayers(current_player)
        # 如果有获胜者，则打印获胜信息
        if winner(board) != 0:
            print(f"Player {winner(board)} won!")
        # 移动次数加1
        num_moves += 1
    
    # 如果没有获胜者，则打印平局信息
    if winner(board) == 0:
        print("Draw")
# 如果当前脚本被直接执行，则调用gamePlay函数
if __name__ == "__main__":
    gamePlay()
```