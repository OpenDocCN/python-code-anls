# `basic-computer-games\46_Hexapawn\python\hexapawn.py`

```
"""
HEXAPAWN

A machine learning game, an interpretation of HEXAPAWN game as
presented in Martin Gardner's "The Unexpected Hanging and Other
Mathematical Diversions", Chapter Eight: A Matchbox Game-Learning
Machine.

Original version for H-P timeshare system by R.A. Kaapke 5/5/76
Instructions by Jeff Dalton
Conversion to MITS BASIC by Steve North


Port to Python by Dave LeCompte
"""

# PORTING NOTES:
#
# I printed out the BASIC code and hand-annotated what each little block
# of code did, which feels amazingly retro.
#
# I encourage other porters that have a complex knot of GOTOs and
# semi-nested subroutines to do hard-copy hacking, it might be a
# different perspective that helps.
#
# A spoiler - the objective of the game is not documented, ostensibly to
# give the human player a challenge. If a player (human or computer)
# advances a pawn across the board to the far row, that player wins. If
# a player has no legal moves (either by being blocked, or all their
# pieces having been captured), that player loses.
#
# The original BASIC had 2 2-dimensional tables stored in DATA at the
# end of the program. This encoded all 19 different board configurations
# (Hexapawn is a small game), with reflections in one table, and then in
# a parallel table, for each of the 19 rows, a list of legal moves was
# encoded by turning them into 2-digit decimal numbers. As gameplay
# continued, the AI would overwrite losing moves with 0 in the second
# array.
#
# My port takes this "parallel array" structure and turns that
# information into a small Python class, BoardLayout. BoardLayout stores
# the board description and legal moves, but stores the moves as (row,
# column) 2-tuples, which is easier to read. The logic for checking if a
# BoardLayout matches the current board, as well as removing losing move
# have been moved into methods of this class.

import random
from typing import Iterator, List, NamedTuple, Optional, Tuple

PAGE_WIDTH = 64

HUMAN_PIECE = 1
EMPTY_SPACE = 0
# 定义计算机棋子的值
COMPUTER_PIECE = -1

# 定义表示计算机移动的命名元组
class ComputerMove(NamedTuple):
    board_index: int
    move_index: int
    m1: int
    m2: int

# 初始化胜利和失败次数
wins = 0
losses = 0

# 打印居中的消息
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 打印标题
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

# 打印游戏说明
def print_instructions() -> None:
    print(
        """
        ...
        (游戏说明内容)
        ...
        """
    )

# 提示用户输入yes或no，并返回布尔值
def prompt_yes_no(msg: str) -> bool:
    while True:
        print(msg)
        response = input().upper()
        if response[0] == "Y":
            return True
        elif response[0] == "N":
            return False

# 反转空格名称
def reverse_space_name(space_name: int) -> int:
    # 反转范围在1-9之间的空格名称，从左到右
    assert 1 <= space_name <= 9
    # 创建一个字典，表示每个数字对应的镜像数字
    reflections = {1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7}
    # 返回给定数字对应的镜像数字
    return reflections[space_name]
# 检查给定的空间名称是否在中心列
def is_space_in_center_column(space_name: int) -> bool:
    return reverse_space_name(space_name) == space_name


class BoardLayout:
    def __init__(self, cells: List[int], move_list: List[Tuple[int, int]]) -> None:
        self.cells = cells
        self.moves = move_list

    # 检查是否匹配，不考虑镜像
    def _check_match_no_mirror(self, cell_list: List[int]) -> bool:
        return all(
            board_contents == cell_list[space_index]
            for space_index, board_contents in enumerate(self.cells)
        )

    # 检查是否匹配，考虑镜像
    def _check_match_with_mirror(self, cell_list: List[int]) -> bool:
        for space_index, board_contents in enumerate(self.cells):
            reversed_space_index = reverse_space_name(space_index + 1) - 1
            if board_contents != cell_list[reversed_space_index]:
                return False
        return True

    # 检查匹配情况，返回是否匹配和是否考虑镜像
    def check_match(self, cell_list: List[int]) -> Tuple[bool, Optional[bool]]:
        if self._check_match_with_mirror(cell_list):
            return True, True
        elif self._check_match_no_mirror(cell_list):
            return True, False
        return False, None

    # 获取随机移动，可以选择是否镜像
    def get_random_move(
        self, reverse_board: Optional[bool]
    ) -> Optional[Tuple[int, int, int]]:
        if not self.moves:
            return None
        move_index = random.randrange(len(self.moves))

        m1, m2 = self.moves[move_index]
        if reverse_board:
            m1 = reverse_space_name(m1)
            m2 = reverse_space_name(m2)

        return move_index, m1, m2


# 创建多个 BoardLayout 实例
boards = [
    BoardLayout([-1, -1, -1, 1, 0, 0, 0, 1, 1], [(2, 4), (2, 5), (3, 6)]),
    BoardLayout([-1, -1, -1, 0, 1, 0, 1, 0, 1], [(1, 4), (1, 5), (3, 6)]),
    BoardLayout([-1, 0, -1, -1, 1, 0, 0, 0, 1], [(1, 5), (3, 5), (3, 6), (4, 7)]),
    BoardLayout([0, -1, -1, 1, -1, 0, 0, 0, 1], [(3, 6), (5, 8), (5, 9)]),
    BoardLayout([-1, 0, -1, 1, 1, 0, 0, 1, 0], [(1, 5), (3, 5), (3, 6)]),
    BoardLayout([-1, -1, 0, 1, 0, 1, 0, 0, 1], [(2, 4), (2, 5), (2, 6)]),
    # 创建 BoardLayout 对象，传入初始棋盘状态和需要翻转的棋子坐标
    BoardLayout([0, -1, -1, 0, -1, 1, 1, 0, 0], [(2, 6), (5, 7), (5, 8)]),
    BoardLayout([0, -1, -1, -1, 1, 1, 1, 0, 0], [(2, 6), (3, 5)]),
    BoardLayout([-1, 0, -1, -1, 0, 1, 0, 1, 0], [(4, 7), (4, 8)]),
    BoardLayout([0, -1, -1, 0, 1, 0, 0, 0, 1], [(3, 5), (3, 6)]),
    BoardLayout([0, -1, -1, 0, 1, 0, 1, 0, 0], [(3, 5), (3, 6)]),
    BoardLayout([-1, 0, -1, 1, 0, 0, 0, 0, 1], [(3, 6)]),
    BoardLayout([0, 0, -1, -1, -1, 1, 0, 0, 0], [(4, 7), (5, 8)]),
    BoardLayout([-1, 0, 0, 1, 1, 1, 0, 0, 0], [(1, 5)]),
    BoardLayout([0, -1, 0, -1, 1, 1, 0, 0, 0], [(2, 6), (4, 7)]),
    BoardLayout([-1, 0, 0, -1, -1, 1, 0, 0, 0], [(4, 7), (5, 8)]),
    BoardLayout([0, 0, -1, -1, 1, 0, 0, 0, 0], [(3, 5), (3, 6), (4, 7)]),
    BoardLayout([0, -1, 0, 1, -1, 0, 0, 0, 0], [(2, 8), (5, 8)]),
    BoardLayout([-1, 0, 0, -1, 1, 0, 0, 0, 0], [(1, 5), (4, 7)]),
# 获取指定棋盘和移动索引对应的移动坐标
def get_move(board_index: int, move_index: int) -> Tuple[int, int]:
    # 确保棋盘索引在有效范围内
    assert board_index >= 0 and board_index < len(boards)
    # 获取指定索引对应的棋盘
    board = boards[board_index]

    # 确保移动索引在有效范围内
    assert move_index >= 0 and move_index < len(board.moves)

    # 返回移动坐标
    return board.moves[move_index]


# 移除指定棋盘和移动索引对应的移动
def remove_move(board_index: int, move_index: int) -> None:
    # 确保棋盘索引在有效范围内
    assert board_index >= 0 and board_index < len(boards)
    # 获取指定索引对应的棋盘
    board = boards[board_index]

    # 确保移动索引在有效范围内
    assert move_index >= 0 and move_index < len(board.moves)

    # 移除指定索引对应的移动
    del board.moves[move_index]


# 初始化棋盘
def init_board() -> List[int]:
    # 返回初始棋盘状态
    return [COMPUTER_PIECE] * 3 + [EMPTY_SPACE] * 3 + [HUMAN_PIECE] * 3


# 打印棋盘状态
def print_board(board: List[int]) -> None:
    # 定义棋子对应的符号
    piece_dict = {COMPUTER_PIECE: "X", EMPTY_SPACE: ".", HUMAN_PIECE: "O"}

    # 打印棋盘状态
    space = " " * 10
    print()
    for row in range(3):
        line = ""
        for column in range(3):
            line += space
            space_number = row * 3 + column
            space_contents = board[space_number]
            line += piece_dict[space_contents]
        print(line)
    print()


# 获取玩家输入的移动坐标
def get_coordinates() -> Tuple[int, int]:
    while True:
        try:
            print("YOUR MOVE?")
            response = input()
            m1, m2 = (int(c) for c in response.split(","))
            return m1, m2
        except ValueError:
            # 打印非法移动提示
            print_illegal()


# 打印非法移动提示
def print_illegal() -> None:
    print("ILLEGAL MOVE.")


# 获取指定棋盘和空间索引对应的内容
def board_contents(board: List[int], space_number: int) -> int:
    return board[space_number - 1]


# 设置指定棋盘和空间索引对应的内容
def set_board(board: List[int], space_number: int, new_value: int) -> None:
    board[space_number - 1] = new_value


# 判断玩家移动是否合法
def is_legal_human_move(board: List[int], m1: int, m2: int) -> bool:
    if board_contents(board, m1) != HUMAN_PIECE:
        # 起始空间不包含玩家的棋子
        return False
    if board_contents(board, m2) == HUMAN_PIECE:
        # 目标空间包含玩家的棋子（不能吃掉自己的棋子）
        return False

    is_capture = m2 - m1 != -3
    # 如果是捕获移动，并且目标位置不包含计算机棋子，则返回 False
    if is_capture and board_contents(board, m2) != COMPUTER_PIECE:
        # Destination does not contain computer piece
        return False

    # 如果目标位置大于起始位置，则返回 False，因为不能向后移动
    if m2 > m1:
        # can't move backwards
        return False

    # 如果不是捕获移动，并且目标位置不是空格，则返回 False
    if (not is_capture) and board_contents(board, m2) != EMPTY_SPACE:
        # Destination is not open
        return False

    # 如果移动距离小于-4，则返回 False，因为移动距离太远
    if m2 - m1 < -4:
        # too far
        return False

    # 如果起始位置是7，目标位置是3，则返回 False，因为不能从角落跳到角落（绕过棋盘）
    if m1 == 7 and m2 == 3:
        # can't jump corner to corner (wrapping around the board)
        return False
    # 其他情况返回 True
    return True
# 检查玩家的棋子是否在棋盘的后排
def player_piece_on_back_row(board: List[int]) -> bool:
    return any(board_contents(board, space) == HUMAN_PIECE for space in range(1, 4))


# 检查计算机的棋子是否在棋盘的前排
def computer_piece_on_front_row(board: List[int]) -> bool:
    return any(board_contents(board, space) == COMPUTER_PIECE for space in range(7, 10))


# 检查所有玩家的棋子是否被全部吃掉
def all_human_pieces_captured(board: List[int]) -> bool:
    return len(list(get_human_spaces(board))) == 0


# 检查所有计算机的棋子是否被全部吃掉
def all_computer_pieces_captured(board: List[int]) -> bool:
    return len(list(get_computer_spaces(board))) == 0


# 玩家获胜时的处理
def human_win(last_computer_move: ComputerMove) -> None:
    print("YOU WIN")
    remove_move(last_computer_move.board_index, last_computer_move.move_index)
    global losses
    losses += 1


# 计算机获胜时的处理
def computer_win(has_moves: bool) -> None:
    if not has_moves:
        msg = "YOU CAN'T MOVE, SO "
    else:
        msg = ""
    msg += "I WIN"
    print(msg)
    global wins
    wins += 1


# 显示比分
def show_scores() -> None:
    print(f"I HAVE WON {wins} AND YOU {losses} OUT OF {wins + losses} GAMES.\n")


# 检查玩家是否有可行的移动
def human_has_move(board: List[int]) -> bool:
    for i in get_human_spaces(board):
        if board_contents(board, i - 3) == EMPTY_SPACE:
            # 可以向前移动棋子
            return True
        elif is_space_in_center_column(i):
            if (board_contents(board, i - 2) == COMPUTER_PIECE) or (
                board_contents(board, i - 4) == COMPUTER_PIECE
            ):
                # 可以从中间吃掉对方的棋子
                return True
            else:
                continue
        elif i < 7:
            assert (i == 4) or (i == 6)
            if board_contents(board, 2) == COMPUTER_PIECE:
                # 可以吃掉位置2的计算机棋子
                return True
            else:
                continue
        elif board_contents(board, 5) == COMPUTER_PIECE:
            assert (i == 7) or (i == 9)
            # 可以吃掉位置5的计算机棋子
            return True
        else:
            continue
    return False
# 生成空格名称（1-9）
def get_board_spaces() -> Iterator[int]:
    """generates the space names (1-9)"""
    yield from range(1, 10)


# 生成包含特定类型值的空格
def get_board_spaces_with(board: List[int], val: int) -> Iterator[int]:
    """generates spaces containing pieces of type val"""
    for i in get_board_spaces():
        if board_contents(board, i) == val:
            yield i


# 生成包含玩家棋子的空格
def get_human_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, HUMAN_PIECE)


# 生成空白空格
def get_empty_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, EMPTY_SPACE)


# 生成包含计算机棋子的空格
def get_computer_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, COMPUTER_PIECE)


# 检查计算机是否有可行动的空格
def has_computer_move(board: List[int]) -> bool:
    for i in get_computer_spaces(board):
        if board_contents(board, i + 3) == EMPTY_SPACE:
            # 可以向前移动（向下）
            return True

        if is_space_in_center_column(i):
            # i在中间列
            if (board_contents(board, i + 2) == HUMAN_PIECE) or (
                board_contents(board, i + 4) == HUMAN_PIECE
            ):
                return True
        else:
            if i > 3:
                # 超出第一行
                if board_contents(board, 8) == HUMAN_PIECE:
                    # 可以在8上捕获
                    return True
                else:
                    continue
            else:
                if board_contents(board, 5) == HUMAN_PIECE:
                    # 可以在5上捕获
                    return True
                else:
                    continue
    return False


# 查找与给定棋盘匹配的棋盘索引
def find_board_index_that_matches_board(board: List[int]) -> Tuple[int, Optional[bool]]:
    for board_index, board_layout in enumerate(boards):
        matches, is_reversed = board_layout.check_match(board)
        if matches:
            return board_index, is_reversed

    # 不应该到达这一点
    # 在将来，mypy可能会通过assert_never来检查穷尽性
    # 抛出运行时错误，指示非法的棋盘模式
    raise RuntimeError("ILLEGAL BOARD PATTERN.")
# 选择计算机的移动，如果没有可行的移动则返回 None
def pick_computer_move(board: List[int]) -> Optional[ComputerMove]:
    # 如果没有计算机可以移动的位置，则返回 None
    if not has_computer_move(board):
        return None

    # 找到匹配当前棋盘的索引和反转后的棋盘
    board_index, reverse_board = find_board_index_that_matches_board(board)

    # 从匹配的棋盘中获取随机移动
    m = boards[board_index].get_random_move(reverse_board)

    # 如果没有可行的移动，则打印"我认输"并返回 None
    if m is None:
        print("I RESIGN")
        return None

    # 获取移动的索引和具体的移动
    move_index, m1, m2 = m

    # 返回计算机的移动
    return ComputerMove(board_index, move_index, m1, m2)


# 获取玩家的移动
def get_human_move(board: List[int]) -> Tuple[int, int]:
    # 循环直到获取合法的玩家移动
    while True:
        m1, m2 = get_coordinates()

        # 如果移动不合法，则打印提示信息
        if not is_legal_human_move(board, m1, m2):
            print_illegal()
        else:
            return m1, m2


# 应用移动到棋盘上
def apply_move(board: List[int], m1: int, m2: int, piece_value: int) -> None:
    # 将棋子从 m1 移动到 m2
    set_board(board, m1, EMPTY_SPACE)
    set_board(board, m2, piece_value)


# 开始游戏
def play_game() -> None:
    # 初始化上一次计算机移动
    last_computer_move = None

    # 初始化棋盘
    board = init_board()

    # 游戏循环
    while True:
        # 打印当前棋盘状态
        print_board(board)

        # 获取玩家的移动
        m1, m2 = get_human_move(board)

        # 应用玩家的移动到棋盘上
        apply_move(board, m1, m2, HUMAN_PIECE)

        # 打印当前棋盘状态
        print_board(board)

        # 如果玩家的棋子到达对方底线或者计算机的所有棋子被吃掉，则玩家获胜
        if player_piece_on_back_row(board) or all_computer_pieces_captured(board):
            assert last_computer_move is not None
            human_win(last_computer_move)
            return

        # 获取计算机的移动
        computer_move = pick_computer_move(board)
        if computer_move is None:
            assert last_computer_move is not None
            human_win(last_computer_move)
            return

        # 更新上一次计算机的移动
        last_computer_move = computer_move

        # 应用计算机的移动到棋盘上
        m1, m2 = last_computer_move.m1, last_computer_move.m2
        print(f"I MOVE FROM {m1} TO {m2}")
        apply_move(board, m1, m2, COMPUTER_PIECE)

        # 打印当前棋盘状态
        print_board(board)

        # 如果计算机的棋子到达对方底线，则计算机获胜
        if computer_piece_on_front_row(board):
            computer_win(True)
            return
        # 如果玩家没有可行的移动或者玩家的所有棋子被吃掉，则计算机获胜
        elif (not human_has_move(board)) or (all_human_pieces_captured(board)):
            computer_win(False)
            return


# 主函数
def main() -> None:
    # 打印游戏标题
    print_header("HEXAPAWN")
    # 如果用户输入 Y 或者 N，则执行下面的代码
    if prompt_yes_no("INSTRUCTIONS (Y-N)?"):
        # 打印游戏说明
        print_instructions()

    # 设置全局变量 wins 和 losses 的初始值
    global wins, losses
    wins = 0
    losses = 0

    # 无限循环，直到游戏结束
    while True:
        # 执行游戏
        play_game()
        # 显示得分
        show_scores()
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```