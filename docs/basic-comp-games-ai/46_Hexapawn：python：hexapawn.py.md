# `d:/src/tocomm/basic-computer-games\46_Hexapawn\python\hexapawn.py`

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
# I encourage other porters that have a complex knot of GOTOs and
# semi-nested subroutines to do hard-copy hacking, it might be a
# different perspective that helps.
# 这段注释是作者对其他程序员的鼓励和建议，鼓励他们尝试不同的方法来解决复杂的问题。

# A spoiler - the objective of the game is not documented, ostensibly to
# give the human player a challenge. If a player (human or computer)
# advances a pawn across the board to the far row, that player wins. If
# a player has no legal moves (either by being blocked, or all their
# pieces having been captured), that player loses.
# 这段注释解释了游戏的目标和规则，玩家需要将一个棋子移动到棋盘的对面一行才能获胜，如果玩家没有合法的移动，则会输掉游戏。

# The original BASIC had 2 2-dimensional tables stored in DATA at the
# end of the program. This encoded all 19 different board configurations
# (Hexapawn is a small game), with reflections in one table, and then in
# a parallel table, for each of the 19 rows, a list of legal moves was
# encoded by turning them into 2-digit decimal numbers. As gameplay
# continued, the AI would overwrite losing moves with 0 in the second
# array.
# 这段注释解释了原始BASIC程序中存储在DATA结尾的两个二维表格，这些表格编码了19种不同的棋盘配置，每一行都有一个合法移动的列表，AI会在第二个数组中用0覆盖输掉的移动。

# My port takes this "parallel array" structure and turns that
# 这段注释似乎不完整，缺少了解释的内容。
# information into a small Python class, BoardLayout. BoardLayout stores
# the board description and legal moves, but stores the moves as (row,
# column) 2-tuples, which is easier to read. The logic for checking if a
# BoardLayout matches the current board, as well as removing losing move
# have been moved into methods of this class.

# 导入需要的模块
import random
from typing import Iterator, List, NamedTuple, Optional, Tuple

# 设置页面宽度常量
PAGE_WIDTH = 64

# 定义玩家和计算机的棋子值
HUMAN_PIECE = 1
EMPTY_SPACE = 0
COMPUTER_PIECE = -1

# 定义计算机移动的命名元组
class ComputerMove(NamedTuple):
    board_index: int
    move_index: int
    m1: int
# 定义整数变量m2
m2: int

# 初始化赢和输的次数
wins = 0
losses = 0

# 定义一个打印居中文本的函数，参数为字符串类型，返回值为空
def print_centered(msg: str) -> None:
    # 计算需要添加的空格数，使得文本居中
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    # 打印居中文本
    print(spaces + msg)

# 定义一个打印标题的函数，参数为字符串类型，返回值为空
def print_header(title: str) -> None:
    # 调用print_centered函数打印标题
    print_centered(title)
    # 打印固定格式的副标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

# 定义一个打印游戏说明的函数，返回值为空
def print_instructions() -> None:
    # 打印游戏说明文本
    print(
        """
# 这个程序是用来玩六角棋的。
# 六角棋是在一个3x3的棋盘上用象棋的兵棋来玩的。
# 兵棋的移动方式和象棋一样 - 向前一格到空格，或者向前一格斜对角吃子。
# 在棋盘上，你的兵棋用'O'表示，计算机的兵棋用'X'表示，空格用'.'表示。
# 输入一个移动，需要输入你要移动的起始格子的编号，后面跟着你要移动到的目标格子的编号，两个数字之间用逗号隔开。

# 计算机开始一系列游戏时只知道如何赢（平局是不可能的）和如何移动。
# 它一开始没有任何策略，只是随机移动。然而，它会从每局游戏中学习。
# 因此，赢得游戏会变得越来越困难。另外，为了抵消你的初始优势，你不会被告知如何赢得游戏，而必须通过游戏来学习。

# 棋盘的编号如下：
# 定义一个函数，参数为字符串类型的消息，返回布尔值
def prompt_yes_no(msg: str) -> bool:
    # 创建一个无限循环，直到得到有效的输入
    while True:
        # 打印消息
        print(msg)
        # 获取用户输入并转换为大写
        response = input().upper()
        # 如果用户输入以Y开头，则返回True
        if response[0] == "Y":
            return True
        # 如果用户输入以N开头，则返回False
        elif response[0] == "N":
            return False
            return False  # 返回 False

def reverse_space_name(space_name: int) -> int:
    # reverse a space name in the range 1-9 left to right
    assert 1 <= space_name <= 9  # 断言 space_name 的取值范围在 1 到 9 之间

    reflections = {1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7}  # 定义一个字典，用于存储 space_name 的反转映射关系
    return reflections[space_name]  # 返回 space_name 的反转值


def is_space_in_center_column(space_name: int) -> bool:
    return reverse_space_name(space_name) == space_name  # 返回 space_name 的反转值是否等于原值的布尔值


class BoardLayout:
    def __init__(self, cells: List[int], move_list: List[Tuple[int, int]]) -> None:
        self.cells = cells  # 初始化实例变量 cells
        self.moves = move_list  # 初始化实例变量 moves
    # 检查给定的 cell_list 是否与当前对象的 cells 匹配，不考虑镜像情况
    def _check_match_no_mirror(self, cell_list: List[int]) -> bool:
        # 使用 all() 函数检查是否所有的 board_contents 与 cell_list 中对应位置的值相等
        return all(
            board_contents == cell_list[space_index]
            for space_index, board_contents in enumerate(self.cells)
        )

    # 检查给定的 cell_list 是否与当前对象的 cells 匹配，考虑镜像情况
    def _check_match_with_mirror(self, cell_list: List[int]) -> bool:
        # 遍历当前对象的 cells
        for space_index, board_contents in enumerate(self.cells):
            # 计算镜像位置的索引
            reversed_space_index = reverse_space_name(space_index + 1) - 1
            # 检查当前位置的值是否与镜像位置的值相等
            if board_contents != cell_list[reversed_space_index]:
                return False
        return True

    # 检查给定的 cell_list 是否与当前对象的 cells 匹配，返回是否匹配以及是否考虑了镜像情况
    def check_match(self, cell_list: List[int]) -> Tuple[bool, Optional[bool]]:
        # 如果考虑了镜像情况匹配，则返回 True 和 True
        if self._check_match_with_mirror(cell_list):
            return True, True
        # 如果不考虑镜像情况匹配，则返回 True 和 False
        elif self._check_match_no_mirror(cell_list):
            return True, False
        # 如果都不匹配，则返回 False 和 None
        return False, None
    def get_random_move(
        self, reverse_board: Optional[bool]
    ) -> Optional[Tuple[int, int, int]]:
        # 定义一个方法，用于获取随机移动
        if not self.moves:
            # 如果没有可用的移动，返回空
            return None
        # 从可用移动中随机选择一个移动
        move_index = random.randrange(len(self.moves))

        m1, m2 = self.moves[move_index]
        # 如果需要翻转棋盘，则翻转移动的位置
        if reverse_board:
            m1 = reverse_space_name(m1)
            m2 = reverse_space_name(m2)

        # 返回选择的移动
        return move_index, m1, m2


boards = [
    # 创建一个包含多个棋盘布局的列表
    BoardLayout([-1, -1, -1, 1, 0, 0, 0, 1, 1], [(2, 4), (2, 5), (3, 6)]),
    BoardLayout([-1, -1, -1, 0, 1, 0, 1, 0, 1], [(1, 4), (1, 5), (3, 6)]),
    BoardLayout([-1, 0, -1, -1, 1, 0, 0, 0, 1], [(1, 5), (3, 5), (3, 6), (4, 7)]),
    BoardLayout([0, -1, -1, 1, -1, 0, 0, 0, 1], [(3, 6), (5, 8), (5, 9)]),
    # 创建多个不同的棋盘布局对象
    # 创建 BoardLayout 对象，传入参数为列表和元组
    BoardLayout([-1, 0, -1, 1, 1, 0, 0, 1, 0], [(1, 5), (3, 5), (3, 6)]),
    BoardLayout([-1, -1, 0, 1, 0, 1, 0, 0, 1], [(2, 4), (2, 5), (2, 6)]),
    ...
    # 创建 BoardLayout 对象的一系列实例，每个实例包含一个列表和一个元组作为参数

def get_move(board_index: int, move_index: int) -> Tuple[int, int]:
    # 断言，确保 board_index 大于等于 0 且小于 boards 列表的长度
    assert board_index >= 0 and board_index < len(boards)
    board = boards[board_index]  # 从boards列表中获取指定索引的棋盘对象

    assert move_index >= 0 and move_index < len(board.moves)  # 确保移动索引在合法范围内

    return board.moves[move_index]  # 返回指定棋盘对象的移动列表中指定索引的移动


def remove_move(board_index: int, move_index: int) -> None:
    assert board_index >= 0 and board_index < len(boards)  # 确保棋盘索引在合法范围内
    board = boards[board_index]  # 从boards列表中获取指定索引的棋盘对象

    assert move_index >= 0 and move_index < len(board.moves)  # 确保移动索引在合法范围内

    del board.moves[move_index]  # 从指定棋盘对象的移动列表中删除指定索引的移动


def init_board() -> List[int]:
    return [COMPUTER_PIECE] * 3 + [EMPTY_SPACE] * 3 + [HUMAN_PIECE] * 3  # 返回一个初始的棋盘状态列表，包括计算机棋子、空格和玩家棋子
    # 询问用户输入坐标
            row = int(input("Enter the row (0, 1, or 2): "))
            column = int(input("Enter the column (0, 1, or 2): "))
            # 返回用户输入的坐标
            return (row, column)
        except ValueError:
            # 如果用户输入的不是整数，提示用户重新输入
            print("Invalid input. Please enter a number.")
            response = input()  # 从用户输入获取响应
            m1, m2 = (int(c) for c in response.split(","))  # 将用户输入的字符串按逗号分隔，并转换为整数赋值给m1和m2
            return m1, m2  # 返回m1和m2
        except ValueError:  # 如果出现值错误
            print_illegal()  # 调用print_illegal函数打印"ILLEGAL MOVE."


def print_illegal() -> None:  # 定义一个返回空类型的print_illegal函数
    print("ILLEGAL MOVE.")  # 打印"ILLEGAL MOVE."


def board_contents(board: List[int], space_number: int) -> int:  # 定义一个返回整数类型的board_contents函数，接受一个整数列表和一个整数作为参数
    return board[space_number - 1]  # 返回列表中索引为space_number-1的元素


def set_board(board: List[int], space_number: int, new_value: int) -> None:  # 定义一个返回空类型的set_board函数，接受一个整数列表和两个整数作为参数
    board[space_number - 1] = new_value  # 将列表中索引为space_number-1的元素赋值为new_value


def is_legal_human_move(board: List[int], m1: int, m2: int) -> bool:  # 定义一个返回布尔类型的is_legal_human_move函数，接受一个整数列表和两个整数作为参数
    if board_contents(board, m1) != HUMAN_PIECE:
        # 如果棋盘上起始位置不包含玩家的棋子
        return False
    if board_contents(board, m2) == HUMAN_PIECE:
        # 如果目标位置包含玩家的棋子（不能吃掉自己的棋子）
        return False

    is_capture = m2 - m1 != -3
    if is_capture and board_contents(board, m2) != COMPUTER_PIECE:
        # 如果是吃子动作，并且目标位置不包含电脑的棋子
        return False

    if m2 > m1:
        # 不能后退移动
        return False

    if (not is_capture) and board_contents(board, m2) != EMPTY_SPACE:
        # 目标位置不是空位
        return False
    if m2 - m1 < -4:
        # 如果 m2 - m1 的值小于 -4，表示跳跃的距离太远，返回 False
        return False

    if m1 == 7 and m2 == 3:
        # 如果 m1 等于 7 并且 m2 等于 3，表示不能从角落跳到角落（绕过棋盘），返回 False
        return False
    # 其他情况返回 True
    return True


def player_piece_on_back_row(board: List[int]) -> bool:
    # 检查玩家的棋子是否在后排
    return any(board_contents(board, space) == HUMAN_PIECE for space in range(1, 4))


def computer_piece_on_front_row(board: List[int]) -> bool:
    # 检查计算机的棋子是否在前排
    return any(board_contents(board, space) == COMPUTER_PIECE for space in range(7, 10))


def all_human_pieces_captured(board: List[int]) -> bool:
    # 检查所有玩家的棋子是否被对方吃掉
    return len(list(get_human_spaces(board))) == 0
def all_computer_pieces_captured(board: List[int]) -> bool:
    # 检查计算机棋子是否全部被对手吃掉
    return len(list(get_computer_spaces(board))) == 0


def human_win(last_computer_move: ComputerMove) -> None:
    # 打印提示信息，表示玩家获胜
    print("YOU WIN")
    # 移除计算机最后一步的移动
    remove_move(last_computer_move.board_index, last_computer_move.move_index)
    # 增加全局变量 losses 的值
    global losses
    losses += 1


def computer_win(has_moves: bool) -> None:
    # 如果计算机没有可行的移动
    if not has_moves:
        msg = "YOU CAN'T MOVE, SO "
    else:
        msg = ""
    # 打印提示信息，表示计算机获胜
    msg += "I WIN"
    print(msg)
    global wins  # 声明 wins 变量为全局变量
    wins += 1  # 增加 wins 变量的值


def show_scores() -> None:
    print(f"I HAVE WON {wins} AND YOU {losses} OUT OF {wins + losses} GAMES.\n")  # 打印游戏得分信息


def human_has_move(board: List[int]) -> bool:
    for i in get_human_spaces(board):  # 遍历玩家的棋盘空间
        if board_contents(board, i - 3) == EMPTY_SPACE:  # 如果可以向前移动棋子
            return True  # 返回 True
        elif is_space_in_center_column(i):  # 如果棋子在中间列
            if (board_contents(board, i - 2) == COMPUTER_PIECE) or (board_contents(board, i - 4) == COMPUTER_PIECE):  # 如果可以从中间列捕获对手的棋子
                return True  # 返回 True
            else:
                # 如果无法移动棋子或捕获对手的棋子
        continue  # 继续下一次循环
        elif i < 7:  # 如果 i 小于 7
            assert (i == 4) or (i == 6)  # 断言 i 等于 4 或者 i 等于 6
            if board_contents(board, 2) == COMPUTER_PIECE:  # 如果在位置 2 可以捕获计算机棋子
                return True  # 返回 True
            else:
                continue  # 继续下一次循环
        elif board_contents(board, 5) == COMPUTER_PIECE:  # 如果在位置 5 可以捕获计算机棋子
            assert (i == 7) or (i == 9)  # 断言 i 等于 7 或者 i 等于 9
            return True  # 返回 True
            # can capture computer piece at 5
        else:
            continue  # 继续下一次循环
    return False  # 返回 False


def get_board_spaces() -> Iterator[int]:
    """generates the space names (1-9)"""
    yield from range(1, 10)  # 生成 1 到 9 的空间名称
def get_board_spaces_with(board: List[int], val: int) -> Iterator[int]:
    """生成包含类型为val的棋子的空格"""
    for i in get_board_spaces():  # 遍历棋盘上的所有空格
        if board_contents(board, i) == val:  # 如果空格上的棋子类型为val
            yield i  # 生成该空格的索引


def get_human_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, HUMAN_PIECE)  # 生成包含玩家棋子的空格


def get_empty_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, EMPTY_SPACE)  # 生成空的空格


def get_computer_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, COMPUTER_PIECE)  # 生成包含电脑棋子的空格
def has_computer_move(board: List[int]) -> bool:
    # 遍历计算机的棋子位置
    for i in get_computer_spaces(board):
        # 如果下方有空位
        if board_contents(board, i + 3) == EMPTY_SPACE:
            # 可以向前移动（向下）
            return True

        # 如果棋子在中间列
        if is_space_in_center_column(i):
            if (board_contents(board, i + 2) == HUMAN_PIECE) or (
                board_contents(board, i + 4) == HUMAN_PIECE
            ):
                return True
        else:
            # 如果在第一行之外
            if i > 3:
                if board_contents(board, 8) == HUMAN_PIECE:
                    # 可以在8号位置进行夺取
                    return True
                else:
                    # 其他情况
# 循环遍历棋盘列表，找到与给定棋盘匹配的索引和是否翻转的信息
def find_board_index_that_matches_board(board: List[int]) -> Tuple[int, Optional[bool]]:
    # 使用enumerate函数遍历boards列表，获取索引和对应的board_layout
    for board_index, board_layout in enumerate(boards):
        # 调用board_layout对象的check_match方法，检查是否匹配给定的棋盘，返回匹配结果和是否翻转的信息
        matches, is_reversed = board_layout.check_match(board)
        # 如果匹配成功，返回匹配的索引和是否翻转的信息
        if matches:
            return board_index, is_reversed

    # 如果没有匹配的棋盘，抛出运行时错误
    # 在将来，mypy可能能够通过assert_never检查穷尽性
    raise RuntimeError("ILLEGAL BOARD PATTERN.")
# 定义函数pick_computer_move，参数为board列表，返回类型为Optional[ComputerMove]
def pick_computer_move(board: List[int]) -> Optional[ComputerMove]:
    # 如果没有计算机可以移动的位置，返回None
    if not has_computer_move(board):
        return None

    # 找到与给定board匹配的board索引和反转后的board
    board_index, reverse_board = find_board_index_that_matches_board(board)

    # 从boards列表中获取随机移动
    m = boards[board_index].get_random_move(reverse_board)

    # 如果没有可用的移动，打印"I RESIGN"并返回None
    if m is None:
        print("I RESIGN")
        return None

    # 解包m元组，获取move_index, m1, m2
    move_index, m1, m2 = m

    # 返回ComputerMove对象，包含board_index, move_index, m1, m2
    return ComputerMove(board_index, move_index, m1, m2)


# 定义函数get_human_move，参数为board列表，返回类型为Tuple[int, int]
def get_human_move(board: List[int]) -> Tuple[int, int]:
    # 无限循环，直到得到有效的人类移动
        m1, m2 = get_coordinates()  # 从用户输入获取两个坐标值

        if not is_legal_human_move(board, m1, m2):  # 检查用户输入的移动是否合法
            print_illegal()  # 如果移动不合法，则打印提示信息
        else:
            return m1, m2  # 如果移动合法，则返回坐标值


def apply_move(board: List[int], m1: int, m2: int, piece_value: int) -> None:
    set_board(board, m1, EMPTY_SPACE)  # 将第一个坐标位置设置为空
    set_board(board, m2, piece_value)  # 将第二个坐标位置设置为指定的棋子值


def play_game() -> None:
    last_computer_move = None  # 初始化上一次电脑移动的变量为None

    board = init_board()  # 初始化游戏棋盘

    while True:
        print_board(board)  # 打印当前游戏棋盘
        # 获取人类玩家的移动位置
        m1, m2 = get_human_move(board)

        # 在棋盘上应用人类玩家的移动
        apply_move(board, m1, m2, HUMAN_PIECE)

        # 打印更新后的棋盘
        print_board(board)

        # 如果人类玩家的棋子到达对方底线或者计算机的所有棋子被吃掉，则人类玩家获胜
        if player_piece_on_back_row(board) or all_computer_pieces_captured(board):
            assert last_computer_move is not None
            human_win(last_computer_move)
            return

        # 计算机选择移动位置
        computer_move = pick_computer_move(board)
        # 如果计算机无法选择移动位置，则人类玩家获胜
        if computer_move is None:
            assert last_computer_move is not None
            human_win(last_computer_move)
            return

        # 更新计算机的最后移动位置
        last_computer_move = computer_move
        m1, m2 = last_computer_move.m1, last_computer_move.m2  # 从上一步计算机移动中获取移动的起始位置和目标位置

        print(f"I MOVE FROM {m1} TO {m2}")  # 打印计算机的移动信息
        apply_move(board, m1, m2, COMPUTER_PIECE)  # 应用计算机的移动到棋盘上

        print_board(board)  # 打印更新后的棋盘

        if computer_piece_on_front_row(board):  # 如果计算机的棋子到达对方的底线
            computer_win(True)  # 计算机获胜
            return
        elif (not human_has_move(board)) or (all_human_pieces_captured(board)):  # 如果玩家无法移动或者所有玩家的棋子都被吃掉
            computer_win(False)  # 计算机获胜
            return


def main() -> None:
    print_header("HEXAPAWN")  # 打印游戏标题
    if prompt_yes_no("INSTRUCTIONS (Y-N)?"):  # 提示用户是否需要游戏说明
        print_instructions()  # 打印游戏说明
    global wins, losses  # 声明 wins 和 losses 为全局变量
    wins = 0  # 初始化 wins 为 0
    losses = 0  # 初始化 losses 为 0

    while True:  # 进入无限循环
        play_game()  # 调用 play_game() 函数进行游戏
        show_scores()  # 调用 show_scores() 函数显示得分


if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用 main() 函数
```