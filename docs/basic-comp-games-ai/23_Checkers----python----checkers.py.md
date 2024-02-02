# `basic-computer-games\23_Checkers\python\checkers.py`

```py
"""
CHECKERS

How about a nice game of checkers?

Ported by Dave LeCompte
"""

# 导入必要的类型
from typing import Iterator, NamedTuple, Optional, Tuple

# 定义页面宽度
PAGE_WIDTH = 64

# 定义玩家和棋子类型的常量
HUMAN_PLAYER = 1
COMPUTER_PLAYER = -1
HUMAN_PIECE = 1
HUMAN_KING = 2
COMPUTER_PIECE = -1
COMPUTER_KING = -2
EMPTY_SPACE = 0

# 定义棋盘的顶部和底部行
TOP_ROW = 7
BOTTOM_ROW = 0

# 定义移动记录的命名元组
class MoveRecord(NamedTuple):
    quality: int
    start_x: int
    start_y: int
    dest_x: int
    dest_y: int

# 打印居中文本
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 打印标题
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

# 获取坐标输入
def get_coordinates(prompt: str) -> Tuple[int, int]:
    err_msg = "ENTER COORDINATES in X,Y FORMAT"
    while True:
        print(prompt)
        response = input()
        if "," not in response:
            print(err_msg)
            continue

        try:
            x, y = (int(c) for c in response.split(","))
        except ValueError:
            print(err_msg)
            continue

        return x, y

# 检查坐标是否合法
def is_legal_board_coordinate(x: int, y: int) -> bool:
    return (0 <= x <= 7) and (0 <= y <= 7)

# 定义棋盘类
class Board:
    def __init__(self) -> None:
        # 初始化棋盘空间
        self.spaces = [[0 for y in range(8)] for x in range(8)]
        # 初始化棋盘上的棋子位置
        for x in range(8):
            if (x % 2) == 0:
                self.spaces[x][6] = COMPUTER_PIECE
                self.spaces[x][2] = HUMAN_PIECE
                self.spaces[x][0] = HUMAN_PIECE
            else:
                self.spaces[x][7] = COMPUTER_PIECE
                self.spaces[x][5] = COMPUTER_PIECE
                self.spaces[x][1] = HUMAN_PIECE
    # 返回表示当前棋盘状态的字符串
    def __str__(self) -> str:
        # 定义棋子的表示方式
        pieces = {
            EMPTY_SPACE: ".",
            HUMAN_PIECE: "O",
            HUMAN_KING: "O*",
            COMPUTER_PIECE: "X",
            COMPUTER_KING: "X*",
        }

        # 初始化字符串
        s = "\n\n\n"
        # 遍历棋盘的每一个位置
        for y in range(7, -1, -1):
            for x in range(0, 8):
                # 获取当前位置的棋子表示，并补齐空格
                piece_str = pieces[self.spaces[x][y]]
                piece_str += " " * (5 - len(piece_str))
                s += piece_str
            s += "\n"
        s += "\n\n"

        # 返回表示棋盘状态的字符串
        return s

    # 获取所有棋盘位置的迭代器
    def get_spaces(self) -> Iterator[Tuple[int, int]]:
        for x in range(0, 8):
            for y in range(0, 8):
                yield x, y

    # 获取所有包含计算机棋子的位置的迭代器
    def get_spaces_with_computer_pieces(self) -> Iterator[Tuple[int, int]]:
        for x, y in self.get_spaces():
            contents = self.spaces[x][y]
            if contents < 0:
                yield x, y

    # 获取所有包含玩家棋子的位置的迭代器
    def get_spaces_with_human_pieces(self) -> Iterator[Tuple[int, int]]:
        for x, y in self.get_spaces():
            contents = self.spaces[x][y]
            if contents > 0:
                yield x, y

    # 获取指定位置的合法移动方向的迭代器
    def get_legal_deltas_for_space(self, x: int, y: int) -> Iterator[Tuple[int, int]]:
        contents = self.spaces[x][y]
        if contents == COMPUTER_PIECE:
            for delta_x in (-1, 1):
                yield (delta_x, -1)
        else:
            for delta_x in (-1, 1):
                for delta_y in (-1, 1):
                    yield (delta_x, delta_y)

    # 获取指定位置的所有合法移动的迭代器
    def get_legal_moves(self, x: int, y: int) -> Iterator[MoveRecord]:
        for delta_x, delta_y in self.get_legal_deltas_for_space(x, y):
            new_move_record = self.check_move(x, y, delta_x, delta_y)

            if new_move_record is not None:
                yield new_move_record
    # 选择计算机移动的最佳着法，并返回移动记录
    def pick_computer_move(self) -> Optional[MoveRecord]:
        # 初始化移动记录为空
        move_record = None

        # 遍历所有计算机棋子的位置
        for start_x, start_y in self.get_spaces_with_computer_pieces():
            # 遍历每个位置的合法移动方向
            for delta_x, delta_y in self.get_legal_deltas_for_space(start_x, start_y):
                # 检查移动是否合法，并返回移动记录
                new_move_record = self.check_move(start_x, start_y, delta_x, delta_y)

                # 如果移动不合法，则继续下一个移动
                if new_move_record is None:
                    continue

                # 如果当前移动记录为空，或者新移动记录的质量比当前记录高
                if (move_record is None) or (
                    new_move_record.quality > move_record.quality
                ):
                    # 更新最佳移动记录
                    move_record = new_move_record

        # 返回最佳移动记录
        return move_record

    # 检查移动是否合法，并返回移动记录
    def check_move(
        self, start_x: int, start_y: int, delta_x: int, delta_y: int
    ) -> Optional[MoveRecord]:
        # 计算新位置的坐标
        new_x = start_x + delta_x
        new_y = start_y + delta_y
        # 如果新位置不在棋盘范围内，则返回空
        if not is_legal_board_coordinate(new_x, new_y):
            return None

        # 获取新位置的内容
        contents = self.spaces[new_x][new_y]
        # 如果新位置为空，则评估移动并返回移动记录
        if contents == EMPTY_SPACE:
            return self.evaluate_move(start_x, start_y, new_x, new_y)
        # 如果新位置有对方棋子，则返回空
        if contents < 0:
            return None

        # 检查跳跃着陆位置，即从新位置再次跳跃的位置
        landing_x = new_x + delta_x
        landing_y = new_y + delta_y

        # 如果着陆位置不在棋盘范围内，则返回空
        if not is_legal_board_coordinate(landing_x, landing_y):
            return None
        # 如果着陆位置为空，则评估移动并返回移动记录
        if self.spaces[landing_x][landing_y] == EMPTY_SPACE:
            return self.evaluate_move(start_x, start_y, landing_x, landing_y)
        # 否则返回空
        return None

    # 评估移动并返回移动记录
    def evaluate_move(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> MoveRecord:
        # 初始化移动质量为0
        quality = 0
        # 如果目标位置的y坐标为0且起始位置的棋子为计算机棋子，则提升棋子是好的
        if dest_y == 0 and self.spaces[start_x][start_y] == COMPUTER_PIECE:
            quality += 2
        # 如果目标位置的y坐标与起始位置的y坐标之差为2，则跳跃是好的
        if abs(dest_y - start_y) == 2:
            quality += 5
        # 如果起始位置的y坐标为7，则更倾向于保卫后排
        if start_y == 7:
            quality -= 2
        # 如果目标位置的x坐标为0或7，则更倾向于移动到边缘列
        if dest_x in (0, 7):
            quality += 1
        # 遍历delta_x为-1和1的情况
        for delta_x in (-1, 1):
            # 如果目标位置的x坐标加上delta_x和dest_y-1不是合法的棋盘坐标，则继续下一次循环
            if not is_legal_board_coordinate(dest_x + delta_x, dest_y - 1):
                continue
            # 如果目标位置的x坐标加上delta_x和dest_y-1处的棋子为负数，则移动到另一个计算机棋子的“阴影”中
            if self.spaces[dest_x + delta_x][dest_y - 1] < 0:
                quality += 1
            # 如果目标位置的x坐标减去delta_x和dest_y+1不是合法的棋盘坐标，则继续下一次循环
            if not is_legal_board_coordinate(dest_x - delta_x, dest_y + 1):
                continue
            # 如果目标位置的x坐标加上delta_x和dest_y-1处的棋子大于0，并且目标位置的x坐标减去delta_x和dest_y+1处为空格，或者目标位置的x坐标减去delta_x等于起始位置的x坐标并且dest_y+1等于起始位置的y坐标
            if (
                (self.spaces[dest_x + delta_x][dest_y - 1] > 0)
                and (self.spaces[dest_x - delta_x][dest_y + 1] == EMPTY_SPACE)
                or ((dest_x - delta_x == start_x) and (dest_y + 1 == start_y))
            ):
                # 我们正在移动到一个可能跳过我们的人类棋子上方
                quality -= 2
        # 返回移动记录对象
        return MoveRecord(quality, start_x, start_y, dest_x, dest_y)

    def remove_r_pieces(self, move_record: MoveRecord) -> None:
        # 调用remove_pieces方法移除棋子
        self.remove_pieces(
            move_record.start_x,
            move_record.start_y,
            move_record.dest_x,
            move_record.dest_y,
        )

    def remove_pieces(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> None:
        # 将目标位置的棋子设置为起始位置的棋子
        self.spaces[dest_x][dest_y] = self.spaces[start_x][start_y]
        # 将起始位置的棋子设置为空格
        self.spaces[start_x][start_y] = EMPTY_SPACE
        # 如果目标位置的x坐标与起始位置的x坐标之差为2
        if abs(dest_x - start_x) == 2:
            # 计算中间位置的坐标
            mid_x = (start_x + dest_x) // 2
            mid_y = (start_y + dest_y) // 2
            # 将中间位置的棋子设置为空格
            self.spaces[mid_x][mid_y] = EMPTY_SPACE

    def try_extend(
        self, start_x: int, start_y: int, delta_x: int, delta_y: int
    # 定义一个方法，用于检查给定的移动是否合法，并返回移动记录
    # 参数：delta_x和delta_y表示移动的增量
    # 返回：MoveRecord对象或None
    def get_legal_moves(self, start_x: int, start_y: int, delta_x: int, delta_y: int) -> Optional[MoveRecord]:
        # 计算新的x坐标和y坐标
        new_x = start_x + delta_x
        new_y = start_y + delta_y

        # 检查新的坐标是否在棋盘范围内
        if not is_legal_board_coordinate(new_x, new_y):
            return None

        # 计算跳跃后的x坐标和y坐标
        jumped_x = start_x + delta_x // 2
        jumped_y = start_y + delta_y // 2

        # 检查目标位置是否为空，并且跳跃位置是否有对方棋子
        if (self.spaces[new_x][new_y] == EMPTY_SPACE) and (
            self.spaces[jumped_x][jumped_y] > 0
        ):
            return self.evaluate_move(start_x, start_y, new_x, new_y)
        return None

    # 获取玩家输入的移动
    # 返回：起始坐标和目标坐标的元组
    def get_human_move(self) -> Tuple[int, int, int, int]:
        is_king = False

        # 循环直到玩家输入合法的起始坐标
        while True:
            start_x, start_y = get_coordinates("FROM?")

            # 获取起始坐标的合法移动列表
            legal_moves = list(self.get_legal_moves(start_x, start_y))
            if not legal_moves:
                print(f"({start_x}, {start_y}) has no legal moves. Choose again.")
                continue
            if self.spaces[start_x][start_y] > 0:
                break

        # 检查起始位置是否为玩家的国王
        is_king = self.spaces[start_x][start_y] == HUMAN_KING

        # 循环直到玩家输入合法的目标坐标
        while True:
            dest_x, dest_y = get_coordinates("TO?")

            # 如果不是国王，且向后移动，则继续循环
            if (not is_king) and (dest_y < start_y):
                # CHEATER! Trying to move non-king backwards
                continue
            is_free = self.spaces[dest_x][dest_y] == 0
            within_reach = abs(dest_x - start_x) <= 2
            is_diagonal_move = abs(dest_x - start_x) == abs(dest_y - start_y)
            # 检查目标位置是否为空、是否在可达范围内、是否为对角线移动
            if is_free and within_reach and is_diagonal_move:
                break
        return start_x, start_y, dest_x, dest_y

    # 获取玩家的扩展移动
    # 参数：start_x和start_y表示起始坐标
    # 定义函数，接受起始坐标和目标坐标，返回布尔值和可选的坐标元组
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        # 判断起始坐标是否为玩家的国王
        is_king = self.spaces[start_x][start_y] == HUMAN_KING

        # 循环直到条件不满足
        while True:
            # 获取目标坐标
            dest_x, dest_y = get_coordinates("+TO?")

            # 如果目标坐标小于0，返回False和空
            if dest_x < 0:
                return False, None
            # 如果不是国王，并且目标y坐标小于起始y坐标，继续循环
            if (not is_king) and (dest_y < start_y):
                # CHEATER! Trying to move non-king backwards
                continue
            # 如果目标坐标为空格，并且横纵坐标的绝对值都为2，返回True和起始坐标到目标坐标的坐标元组
            if (
                (self.spaces[dest_x][dest_y] == EMPTY_SPACE)
                and (abs(dest_x - start_x) == 2)
                and (abs(dest_y - start_y) == 2)
            ):
                return True, (start_x, start_y, dest_x, dest_y)

    # 定义函数，接受起始坐标和目标坐标，不返回任何值
    def play_human_move(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> None:
        # 调用remove_pieces函数移除棋子
        self.remove_pieces(start_x, start_y, dest_x, dest_y)

        # 如果目标y坐标为顶部行，将目标坐标设为玩家国王
        if dest_y == TOP_ROW:
            # KING ME
            self.spaces[dest_x][dest_y] = HUMAN_KING

    # 定义函数，不接受参数，返回布尔值
    def check_pieces(self) -> bool:
        # 如果计算机棋子数量为0，打印玩家获胜信息，返回False
        if len(list(self.get_spaces_with_computer_pieces())) == 0:
            print_human_won()
            return False
        # 如果玩家棋子数量为0，打印计算机获胜信息，返回False
        if len(list(self.get_spaces_with_computer_pieces())) == 0:
            print_computer_won()
            return False
        # 否则返回True
        return True
# 打印游戏说明
def print_instructions() -> None:
    print("THIS IS THE GAME OF CHECKERS.  THE COMPUTER IS X,")
    print("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.")
    print("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.")
    print("(0,0) IS THE LOWER LEFT CORNER")
    print("(0,7) IS THE UPPER LEFT CORNER")
    print("(7,0) IS THE LOWER RIGHT CORNER")
    print("(7,7) IS THE UPPER RIGHT CORNER")
    print("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER")
    print("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.\n\n\n")


# 打印人类赢得游戏的消息
def print_human_won() -> None:
    print("\nYOU WIN.")


# 打印计算机赢得游戏的消息
def print_computer_won() -> None:
    print("\nI WIN.")


# 进行游戏的主要逻辑
def play_game() -> None:
    # 创建棋盘对象
    board = Board()

    while True:
        # 让计算机选择移动
        move_record = board.pick_computer_move()
        # 如果没有可行的移动，则打印人类赢得游戏的消息并返回
        if move_record is None:
            print_human_won()
            return
        # 计算机执行移动
        board.play_computer_move(move_record)

        # 打印当前棋盘状态
        print(board)

        # 检查棋子是否还存在，如果不存在则返回
        if not board.check_pieces():
            return

        # 获取人类玩家的移动
        start_x, start_y, dest_x, dest_y = board.get_human_move()
        # 执行人类玩家的移动
        board.play_human_move(start_x, start_y, dest_x, dest_y)
        # 如果移动距离为2，则继续检查是否有额外的跳吃
        if abs(dest_x - start_x) == 2:
            while True:
                # 获取人类玩家的额外跳吃移动
                extend, move = board.get_human_extension(dest_x, dest_y)
                assert move is not None
                # 如果没有额外的跳吃，则退出循环
                if not extend:
                    break
                start_x, start_y, dest_x, dest_y = move
                # 执行额外的跳吃移动
                board.play_human_move(start_x, start_y, dest_x, dest_y)


# 游戏的入口函数
def main() -> None:
    # 打印游戏标题
    print_header("CHECKERS")
    # 打印游戏说明
    print_instructions()

    # 开始游戏
    play_game()


# 如果作为主程序运行，则调用main函数开始游戏
if __name__ == "__main__":
    main()
```