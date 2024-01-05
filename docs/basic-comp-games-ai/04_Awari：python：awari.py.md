# `04_Awari\python\awari.py`

```
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
"""
# 每次游戏结束后，移动历史记录以六进制数字的形式存储在losing_book[game_number]中。如果人类玩家赢了或打成平局，计算机会增加game_number，实际上是“记录”了那次失败以便以后参考。当计算机评估移动时，它会将潜在的游戏状态与这些失败的游戏记录进行比较，如果潜在移动与失败的游戏匹配（直到当前移动的数量为止），那么该移动将被评估为扣除两分。
# 例如，将此与MENACE进行比较，MENACE是用于“学习”井字棋的机械设备：https://en.wikipedia.org/wiki/Matchbox_Educable_Noughts_and_Crosses_Engine
# 六进制表示法允许非常高效地表示游戏历史。我考虑过是否重写这个表示法以使其更易读，但我选择尝试对其进行文档化。
# 我在另一个地方做出了一个艰难的决定，即在“wrapping”代码内部考虑“while human_move_end > 13”。原始的BASIC代码如下：
# 830 IF L>13 THEN L=L-14:R=1:GOTO 830
# 如果L大于13，则将L减去14，将R赋值为1，然后跳转到830行
# 我怀疑意图不是将1赋值给R，而是要对R进行递增。我在翻译代码旁边的移植注释中更详细地讨论了这一点。如果您希望玩一种更准确的游戏版本，可以将递增转换回赋值。
# 我对这款游戏的精华仍然印象深刻；一旦我让AI和我对战，它就能打败我。我已经能够在与计算机对战中取得一些胜利，但即使在其2层搜索的情况下，它几乎总是能打败我。我想要变得更擅长这个游戏，以探索“失败书”机器学习的有效性。
# 读者的练习
# 这个游戏可以有很多方向：
# - 改变每个坑中的初始石头数量
# - 改变坑的数量
# 游戏编号，用于记录游戏的次数
game_number: int = 0
# 移动计数，用于记录游戏中的移动次数
move_count: int = 0
# 失败记录，用于记录失败的情况
losing_book: List[int] = []
# 变量 n，未知具体用途
n = 0

# 最大历史记录数
MAX_HISTORY = 9
LOSING_BOOK_SIZE = 50  # 设置一个常量 LOSING_BOOK_SIZE 的值为 50


def draw_pit(line: str, board, pit_index) -> str:  # 定义一个函数 draw_pit，接受一个字符串 line，一个列表 board，一个整数 pit_index 作为参数，并返回一个字符串
    val = board[pit_index]  # 从列表 board 中获取索引为 pit_index 的值，赋给变量 val
    line = line + " "  # 在字符串 line 后面添加一个空格
    if val < 10:  # 如果 val 小于 10
        line = line + " "  # 在字符串 line 后面再添加一个空格
    line = line + str(val) + " "  # 在字符串 line 后面添加 val 的字符串形式，并再添加一个空格
    return line  # 返回拼接后的字符串 line


def draw_board(board) -> None:  # 定义一个函数 draw_board，接受一个列表 board 作为参数，并返回空值
    print()  # 打印一个空行

    # Draw the top (computer) pits
    line = "   "  # 初始化一个字符串 line 为 "   "
    for i in range(12, 6, -1):  # 遍历从 12 到 7 的整数
        line = draw_pit(line, board, i)  # 调用函数 draw_pit，传入参数 line, board, i，并将返回值赋给 line
    print(line)  # 打印字符串 line
    # Draw the side (home) pits
    # 画出侧面（家）坑
    line = draw_pit("", board, 13)  # 调用draw_pit函数，传入空字符串、board列表和13作为参数，返回一个字符串
    line += " " * 24  # 将空格乘以24后添加到line字符串末尾
    line = draw_pit(line, board, 6)  # 调用draw_pit函数，传入line字符串、board列表和6作为参数，返回一个字符串
    print(line)  # 打印line字符串

    # Draw the bottom (player) pits
    # 画出底部（玩家）坑
    line = "   "  # 将三个空格赋值给line字符串
    for i in range(0, 6):  # 循环6次
        line = draw_pit(line, board, i)  # 调用draw_pit函数，传入line字符串、board列表和i作为参数，返回一个字符串
    print(line)  # 打印line字符串
    print()  # 打印空行
    print()  # 再次打印空行


def play_game(board: List[int]) -> None:
    # Place the beginning stones
    # 放置初始的石头
    for i in range(0, 13):  # 循环13次
        board[i] = 3  # 将3赋值给board列表中的每个元素
    # 清空主要的棋子位置
    board[6] = 0
    board[13] = 0

    # 设置全局变量 move_count 为 0
    global move_count
    move_count = 0

    # 清空这场比赛的历史记录
    losing_book[game_number] = 0

    # 进入游戏循环
    while True:
        # 绘制游戏棋盘
        draw_board(board)

        # 打印提示信息
        print("YOUR MOVE")

        # 玩家进行移动，获取落点、是否继续移动、是否回家
        landing_spot, is_still_going, home = player_move(board)

        # 如果不再继续移动，则跳出循环
        if not is_still_going:
            break

        # 如果落点是回家的位置，则玩家再次移动
        if landing_spot == home:
            landing_spot, is_still_going, home = player_move_again(board)
        if not is_still_going:  # 如果游戏已经结束，退出循环
            break

        print("MY MOVE")  # 打印信息，表示轮到计算机移动
        landing_spot, is_still_going, home, msg = computer_move("", board)  # 调用计算机移动函数，获取移动位置、游戏是否继续、家的位置和消息

        if not is_still_going:  # 如果游戏已经结束，打印消息并退出循环
            print(msg)
            break
        if landing_spot == home:  # 如果移动位置等于家的位置
            landing_spot, is_still_going, home, msg = computer_move(msg + " , ", board)  # 继续调用计算机移动函数，获取移动位置、游戏是否继续、家的位置和消息
        if not is_still_going:  # 如果游戏已经结束，打印消息并退出循环
            print(msg)
            break
        print(msg)  # 打印消息

    game_over(board)  # 调用游戏结束函数，传入游戏板参数


def computer_move(msg: str, board) -> Tuple[int, bool, int, str]:  # 定义计算机移动函数，接受消息和游戏板参数，返回移动位置、游戏是否继续、家的位置和消息的元组类型
    # 初始化最佳质量为-99
    best_quality = -99

    # 复制棋盘，以便进行实验。稍后我们会把一切都放回去。
    temp_board = board[:]

    # 对于每个合法的计算机移动7-12
    for computer_move in range(7, 13):
        if board[computer_move] == 0:
            # 如果当前位置为空
            continue  # 跳过当前循环，继续执行下一次循环

        do_move(computer_move, 13, board)  # 尝试进行一次移动（向前看1步）

        best_player_move_quality = 0  # 初始化最佳玩家移动质量为0
        # 遍历所有合法的玩家移动 0-5（作为对电脑移动computer_move的响应）
        for human_move_start in range(0, 6):
            if board[human_move_start] == 0:  # 如果该位置没有棋子，则跳过
                continue

            human_move_end = board[human_move_start] + human_move_start  # 计算玩家移动的终点位置
            this_player_move_quality = 0  # 初始化当前玩家移动质量为0

            # 如果这个移动绕过了整个棋盘，向后绕回来
            #
            # 移植注意：细心的读者会注意到，我为每次绕回来都增加了this_player_move_quality，
            # 而原始代码只将其设置为1。
            #
            # 我认为这可能是一个笔误或疏忽，但我也意识到，你可能需要绕过棋盘更多次
            # 如果玩家的结束位置大于13，则将其减去14，并增加玩家移动的质量
            while human_move_end > 13:
                human_move_end = human_move_end - 14
                this_player_move_quality += 1

            # 如果玩家结束位置上没有棋子，并且结束位置不是6和13，则评分捕获
            if (
                (board[human_move_end] == 0)
                and (human_move_end != 6)
                and (human_move_end != 13)
            ):
                # 评分捕获
                this_player_move_quality += board[12 - human_move_end]

            # 如果当前玩家移动的质量大于最佳玩家移动的质量，则更新最佳玩家移动的质量
            if this_player_move_quality > best_player_move_quality:
                best_player_move_quality = this_player_move_quality
        # 这是一个零和游戏，所以人类玩家的移动越好，对计算机玩家越不利。
        computer_move_quality = board[13] - board[6] - best_player_move_quality

        if move_count < MAX_HISTORY:
            move_digit = computer_move
            if move_digit > 6:
                move_digit = move_digit - 7

            # 计算游戏的基数-6历史表示，如果该历史在我们的“失败书”中，惩罚该移动。
            for prev_game_number in range(game_number):
                if losing_book[game_number] * 6 + move_digit == int(
                    losing_book[prev_game_number] / 6 ^ (7 - move_count) + 0.1  # type: ignore
                ):
                    computer_move_quality -= 2

        # 从临时棋盘复制回来
        for i in range(14):
    # 将temp_board中的值复制到board中
    board[i] = temp_board[i]

    # 如果计算机移动的质量大于等于最佳质量，则更新最佳移动和最佳质量
    if computer_move_quality >= best_quality:
        best_move = computer_move
        best_quality = computer_move_quality

    # 选择最佳移动
    selected_move = best_move

    # 将选择的移动转换为字符形式
    move_str = chr(42 + selected_move)

    # 如果消息存在，则在消息后面添加选择的移动字符，否则将选择的移动字符作为消息
    if msg:
        msg += ", " + move_str
    else:
        msg = move_str

    # 执行选择的移动，获取移动编号、游戏是否仍在进行、家的位置
    move_number, is_still_going, home = execute_move(selected_move, 13, board)

    # 返回移动编号、游戏是否仍在进行、家的位置和消息
    return move_number, is_still_going, home, msg


# 定义游戏结束函数，参数为board
def game_over(board) -> None:
    print()  # 打印空行
    print("GAME OVER")  # 打印游戏结束提示

    pit_difference = board[6] - board[13]  # 计算玩家和对手的分数差
    if pit_difference < 0:  # 如果分数差小于0
        print(f"I WIN BY {-pit_difference} POINTS")  # 打印玩家获胜的分数差

    else:  # 如果分数差大于等于0
        global n  # 声明全局变量n
        n = n + 1  # n加1

        if pit_difference == 0:  # 如果分数差为0
            print("DRAWN GAME")  # 打印平局提示
        else:  # 如果分数差大于0
            print(f"YOU WIN BY {pit_difference} POINTS")  # 打印对手获胜的分数差


def do_capture(m, home, board) -> None:
    board[home] += board[12 - m] + 1  # 将对手的棋子和当前位置的棋子全部移动到玩家的家中
    board[m] = 0  # 当前位置的棋子数量置为0
    board[12 - m] = 0  # 将索引为12-m的位置上的石子数量设置为0，表示清空该位置的石子


def do_move(m, home, board) -> int:
    move_stones = board[m]  # 获取索引为m的位置上的石子数量
    board[m] = 0  # 将索引为m的位置上的石子数量设置为0，表示清空该位置的石子

    for _stones in range(move_stones, 0, -1):  # 遍历移动的石子数量
        m = m + 1  # 移动到下一个位置
        if m > 13:  # 如果超过了索引范围
            m = m - 14  # 回到起始位置
        board[m] += 1  # 在当前位置上增加一颗石子
    if board[m] == 1 and (m != 6) and (m != 13) and (board[12 - m] != 0):  # 如果当前位置上只有一颗石子，并且不是家的位置，并且对面位置上有石子
        do_capture(m, home, board)  # 进行对面石子的吃掉操作
    return m  # 返回最终停留的位置


def player_has_stones(board) -> bool:
    return any(board[i] > 0 for i in range(6))  # 判断玩家是否还有石子，只要有一个位置上有石子就返回True
# 检查棋盘上是否有棋子
def computer_has_stones(board: Dict[int, int]) -> bool:
    return any(board[i] > 0 for i in range(7, 13))

# 执行移动
def execute_move(move, home: int, board) -> Tuple[int, bool, int]:
    move_digit = move
    last_location = do_move(move, home, board)  # 执行移动并记录最后位置

    if move_digit > 6:  # 如果移动大于6，则减去7
        move_digit = move_digit - 7

    global move_count  # 声明全局变量
    move_count += 1  # 移动次数加1
    if move_count < MAX_HISTORY:  # 如果移动次数小于最大历史记录
        # 计算机通过将一系列移动存储为基数为6的数字的数字序列来在losing_book中保留移动链。
        #
        # game_number表示当前游戏，
        # losing_book[game_number]记录进行中游戏的历史记录
        # game.  When the computer evaluates moves, it tries to avoid
        # moves that will lead it into paths that have led to previous
        # losses.
        # 将当前游戏的编号和玩家的移动数字加入到失利记录中，用于计算避免之前导致失败的路径
        losing_book[game_number] = losing_book[game_number] * 6 + move_digit

    # 如果玩家和计算机都还有棋子
    if player_has_stones(board) and computer_has_stones(board):
        # 游戏仍在进行中
        is_still_going = True
    else:
        # 游戏已结束
        is_still_going = False
    # 返回上一次移动的位置、游戏是否仍在进行中、玩家的家的位置
    return last_location, is_still_going, home


def player_move_again(board) -> Tuple[int, bool, int]:
    # 打印提示信息
    print("AGAIN")
    # 返回玩家的移动
    return player_move(board)


def player_move(board) -> Tuple[int, bool, int]:
    # 循环直到玩家选择有效的移动
    while True:
        # 打印提示信息
        print("SELECT MOVE 1-6")
        m = int(input()) - 1  # 从用户输入中获取移动的位置，并将其减去1，以匹配程序中的索引

        if m > 5 or m < 0 or board[m] == 0:  # 如果移动位置超出范围或者该位置没有种子，则打印"ILLEGAL MOVE"并继续循环
            print("ILLEGAL MOVE")
            continue

        break  # 如果移动合法，则跳出循环

    ending_spot, is_still_going, home = execute_move(m, 6, board)  # 执行移动操作，返回移动结束的位置、是否继续移动、以及家的情况

    draw_board(board)  # 绘制游戏棋盘

    return ending_spot, is_still_going, home  # 返回移动结束的位置、是否继续移动、以及家的情况


def main() -> None:  # 主函数定义，不返回任何结果
    print(" " * 34 + "AWARI")  # 打印游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n")  # 打印创意计算的信息

    board = [0] * 14  # 清空棋盘的表示
    global losing_book  # 声明 losing_book 变量为全局变量
    losing_book = [0] * LOSING_BOOK_SIZE  # 使用 0 初始化 losing_book 列表，长度为 LOSING_BOOK_SIZE，用于清空“机器学习”状态

    while True:  # 无限循环
        play_game(board)  # 调用 play_game 函数进行游戏

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用 main 函数
```