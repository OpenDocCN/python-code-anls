# `basic-computer-games\04_Awari\python\awari.py`

```py
"""
AWARI

An ancient African game (see also Kalah, Mancala).

Ported by Dave LeCompte
"""

# PORTING NOTES
#
# This game started out as 70 lines of BASIC, and I have ported it
# before. I find it somewhat amazing how efficient (densely packed) the
# original code is. Of course, the original code has fairly cryptic
# variable names (as was forced by BASIC's limitation on long (2+
# character) variable names). I have done my best here to interpret what
# each variable is doing in context, and rename them appropriately.
#
# I have endeavored to leave the logic of the code in place, as it's
# interesting to see a 2-ply game tree evaluation written in BASIC,
# along with what a reader in 2021 would call "machine learning".
#
# As each game is played, the move history is stored as base-6
# digits stored losing_book[game_number]. If the human player wins or
# draws, the computer increments game_number, effectively "recording"
# that loss to be referred to later. As the computer evaluates moves, it
# checks the potential game state against these losing game records, and
# if the potential move matches with the losing game (up to the current
# number of moves), that move is evaluated at a two point penalty.
#
# Compare this, for example with MENACE, a mechanical device for
# "learning" tic-tac-toe:
# https://en.wikipedia.org/wiki/Matchbox_Educable_Noughts_and_Crosses_Engine
#
# The base-6 representation allows game history to be VERY efficiently
# represented. I considered whether to rewrite this representation to be
# easier to read, but I elected to TRY to document it, instead.
#
# Another place where I have made a difficult decision between accuracy
# and correctness is inside the "wrapping" code where it considers
# "while human_move_end > 13". The original BASIC code reads:
#
# 830 IF L>13 THEN L=L-14:R=1:GOTO 830
#
# I suspect that the intention is not to assign 1 to R, but to increment
# R. I discuss this more in a porting note comment next to the
# 游戏编号
game_number: int = 0
# 移动计数
move_count: int = 0
# 输棋记录
losing_book: List[int] = []
# 临时变量
n = 0

# 最大历史记录
MAX_HISTORY = 9
# 输棋记录大小
LOSING_BOOK_SIZE = 50

# 绘制一个坑
def draw_pit(line: str, board, pit_index) -> str:
    # 获取坑的值
    val = board[pit_index]
    line = line + " "
    # 如果值小于10，补充一个空格
    if val < 10:
        line = line + " "
    # 添加值到字符串
    line = line + str(val) + " "
    return line

# 绘制游戏棋盘
def draw_board(board) -> None:
    print()

    # 绘制顶部（电脑）坑
    line = "   "
    for i in range(12, 6, -1):
        line = draw_pit(line, board, i)
    print(line)

    # 绘制侧边（家）坑
    line = draw_pit("", board, 13)
    line += " " * 24
    line = draw_pit(line, board, 6)
    print(line)

    # 绘制底部（玩家）坑
    line = "   "
    # 循环6次，每次调用draw_pit函数，并将返回的结果赋给line
    for i in range(0, 6):
        line = draw_pit(line, board, i)
    # 打印line的值
    print(line)
    # 打印空行
    print()
    # 打印空行
    print()
def play_game(board: List[int]) -> None:
    # Place the beginning stones
    # 在棋盘上放置起始的棋子
    for i in range(0, 13):
        board[i] = 3

    # Empty the home pits
    # 清空家的坑
    board[6] = 0
    board[13] = 0

    global move_count
    move_count = 0

    # clear the history record for this game
    # 清除这个游戏的历史记录
    losing_book[game_number] = 0

    while True:
        draw_board(board)

        print("YOUR MOVE")
        # 玩家进行移动
        landing_spot, is_still_going, home = player_move(board)
        if not is_still_going:
            break
        if landing_spot == home:
            # 如果落点是家，玩家再次移动
            landing_spot, is_still_going, home = player_move_again(board)
        if not is_still_going:
            break

        print("MY MOVE")
        # 电脑进行移动
        landing_spot, is_still_going, home, msg = computer_move("", board)

        if not is_still_going:
            print(msg)
            break
        if landing_spot == home:
            # 如果落点是家，电脑再次移动
            landing_spot, is_still_going, home, msg = computer_move(msg + " , ", board)
        if not is_still_going:
            print(msg)
            break
        print(msg)

    game_over(board)


def computer_move(msg: str, board) -> Tuple[int, bool, int, str]:
    # This function does a two-ply lookahead evaluation; one computer
    # move plus one human move.
    # 这个函数进行两层深度的前瞻评估；一次电脑移动加一次玩家移动。
    #
    # To do this, it makes a copy (temp_board) of the board, plays
    # each possible computer move and then uses math to work out what
    # the scoring heuristic is for each possible human move.
    # 为了实现这一点，它复制了棋盘（temp_board），进行每个可能的电脑移动，然后使用数学方法来计算每个可能的玩家移动的得分启发式。
    #
    # Additionally, if it detects that a potential move puts it on a
    # series of moves that it has recorded in its "losing book", it
    # penalizes that move by two stones.
    # 另外，如果它发现一个潜在的移动将它放在它在“失败记录”中记录的一系列移动中，它会用两颗棋子来惩罚这个移动。

    best_quality = -99

    # Make a copy of the board, so that we can experiment. We'll put
    # everything back, later.
    # 复制棋盘，以便我们可以进行实验。稍后我们会把一切都放回去。
    temp_board = board[:]

    # For each legal computer move 7-12
    # 对于每个合法的电脑移动 7-12
    selected_move = best_move

    move_str = chr(42 + selected_move)
    if msg:
        msg += ", " + move_str
    else:
        msg = move_str
    # 调用 execute_move 函数执行选定的移动，返回移动次数、是否仍在进行、和家的位置
    move_number, is_still_going, home = execute_move(selected_move, 13, board)
    # 返回移动次数、是否仍在进行、家的位置和消息
    return move_number, is_still_going, home, msg
# 定义游戏结束函数，打印游戏结束信息
def game_over(board) -> None:
    print()
    print("GAME OVER")

    # 计算双方分数差
    pit_difference = board[6] - board[13]
    # 如果分数差小于0，打印电脑获胜信息
    if pit_difference < 0:
        print(f"I WIN BY {-pit_difference} POINTS")
    # 如果分数差大于等于0
    else:
        global n
        n = n + 1
        # 如果分数差为0，打印平局信息
        if pit_difference == 0:
            print("DRAWN GAME")
        # 如果分数差大于0，打印玩家获胜信息
        else:
            print(f"YOU WIN BY {pit_difference} POINTS")


# 执行吃子操作
def do_capture(m, home, board) -> None:
    board[home] += board[12 - m] + 1
    board[m] = 0
    board[12 - m] = 0


# 执行移动操作
def do_move(m, home, board) -> int:
    move_stones = board[m]
    board[m] = 0

    # 将石子逐个移动
    for _stones in range(move_stones, 0, -1):
        m = m + 1
        if m > 13:
            m = m - 14
        board[m] += 1
    # 如果落子位置只有一个石子，并且不是玩家和电脑的大坑，并且对面位置有石子，则执行吃子操作
    if board[m] == 1 and (m != 6) and (m != 13) and (board[12 - m] != 0):
        do_capture(m, home, board)
    return m


# 判断玩家是否还有石子
def player_has_stones(board) -> bool:
    return any(board[i] > 0 for i in range(6))


# 判断电脑是否还有石子
def computer_has_stones(board: Dict[int, int]) -> bool:
    return any(board[i] > 0 for i in range(7, 13))


# 执行移动操作，并返回最后落子位置、游戏是否继续、落子位置所属大坑
def execute_move(move, home: int, board) -> Tuple[int, bool, int]:
    move_digit = move
    last_location = do_move(move, home, board)

    # 如果落子位置大于6，将其转换为0-5的数字
    if move_digit > 6:
        move_digit = move_digit - 7

    global move_count
    move_count += 1
    if move_count < MAX_HISTORY:
        # 记录游戏历史，用于电脑评估落子
        losing_book[game_number] = losing_book[game_number] * 6 + move_digit

    # 判断玩家和电脑是否还有石子
    if player_has_stones(board) and computer_has_stones(board):
        is_still_going = True
    else:
        is_still_going = False
    return last_location, is_still_going, home
# 玩家再次移动，调用 player_move 函数
def player_move_again(board) -> Tuple[int, bool, int]:
    # 打印提示信息
    print("AGAIN")
    # 返回 player_move 函数的结果
    return player_move(board)


# 玩家移动，返回移动结果
def player_move(board) -> Tuple[int, bool, int]:
    # 循环直到合法移动
    while True:
        # 打印提示信息
        print("SELECT MOVE 1-6")
        # 获取玩家输入的移动位置
        m = int(input()) - 1

        # 判断移动是否合法
        if m > 5 or m < 0 or board[m] == 0:
            # 打印非法移动信息
            print("ILLEGAL MOVE")
            # 继续循环
            continue

        # 合法移动，跳出循环
        break

    # 执行移动操作，返回结果
    ending_spot, is_still_going, home = execute_move(m, 6, board)

    # 绘制游戏棋盘
    draw_board(board)

    # 返回移动结果
    return ending_spot, is_still_going, home


# 主函数
def main() -> None:
    # 打印游戏标题
    print(" " * 34 + "AWARI")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n")

    # 初始化游戏棋盘
    board = [0] * 14  # clear the board representation
    # 初始化 losing_book
    global losing_book
    losing_book = [0] * LOSING_BOOK_SIZE  # clear the "machine learning" state

    # 循环进行游戏
    while True:
        play_game(board)


# 如果作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```