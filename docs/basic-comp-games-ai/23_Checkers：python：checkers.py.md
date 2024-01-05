# `23_Checkers\python\checkers.py`

```
"""
CHECKERS

How about a nice game of checkers?

Ported by Dave LeCompte
"""

from typing import Iterator, NamedTuple, Optional, Tuple  # 导入需要的类型提示模块

PAGE_WIDTH = 64  # 设置页面宽度为64

HUMAN_PLAYER = 1  # 人类玩家的标识
COMPUTER_PLAYER = -1  # 计算机玩家的标识
HUMAN_PIECE = 1  # 人类玩家的棋子
HUMAN_KING = 2  # 人类玩家的国王棋子
COMPUTER_PIECE = -1  # 计算机玩家的棋子
COMPUTER_KING = -2  # 计算机玩家的国王棋子
EMPTY_SPACE = 0  # 空格
TOP_ROW = 7  # 设置顶部行的位置为第7行
BOTTOM_ROW = 0  # 设置底部行的位置为第0行


class MoveRecord(NamedTuple):  # 定义一个名为MoveRecord的命名元组，包含quality、start_x、start_y、dest_x、dest_y字段
    quality: int  # 定义quality字段为整数类型
    start_x: int  # 定义start_x字段为整数类型
    start_y: int  # 定义start_y字段为整数类型
    dest_x: int  # 定义dest_x字段为整数类型
    dest_y: int  # 定义dest_y字段为整数类型


def print_centered(msg: str) -> None:  # 定义一个名为print_centered的函数，接受一个字符串参数并返回空值
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得字符串在页面宽度中居中显示
    print(spaces + msg)  # 打印居中显示的字符串


def print_header(title: str) -> None:  # 定义一个名为print_header的函数，接受一个字符串参数并返回空值
    print_centered(title)  # 调用print_centered函数打印居中显示的标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印居中显示的副标题
# 定义一个函数，用于获取用户输入的坐标，返回一个元组，包含两个整数类型的坐标值
def get_coordinates(prompt: str) -> Tuple[int, int]:
    # 定义错误消息
    err_msg = "ENTER COORDINATES in X,Y FORMAT"
    # 无限循环，直到用户输入正确的坐标格式
    while True:
        # 打印提示信息
        print(prompt)
        # 获取用户输入
        response = input()
        # 如果输入中不包含逗号，则打印错误消息并继续循环
        if "," not in response:
            print(err_msg)
            continue

        # 尝试将输入的坐标字符串按逗号分割，并转换为整数类型的坐标值
        try:
            x, y = (int(c) for c in response.split(","))
        # 如果转换失败，则打印错误消息并继续循环
        except ValueError:
            print(err_msg)
            continue

        # 如果成功获取到坐标值，则返回坐标的元组
        return x, y
def is_legal_board_coordinate(x: int, y: int) -> bool:
    # 检查坐标是否合法，即在 0 到 7 之间
    return (0 <= x <= 7) and (0 <= y <= 7)


class Board:
    def __init__(self) -> None:
        # 创建一个 8x8 的棋盘，初始化所有位置为 0
        self.spaces = [[0 for y in range(8)] for x in range(8)]
        # 在棋盘上放置计算机和玩家的棋子
        for x in range(8):
            if (x % 2) == 0:
                self.spaces[x][6] = COMPUTER_PIECE
                self.spaces[x][2] = HUMAN_PIECE
                self.spaces[x][0] = HUMAN_PIECE
            else:
                self.spaces[x][7] = COMPUTER_PIECE
                self.spaces[x][5] = COMPUTER_PIECE
                self.spaces[x][1] = HUMAN_PIECE

    def __str__(self) -> str:
        # 定义棋子的显示方式
        pieces = {
            EMPTY_SPACE: ".",
            HUMAN_PIECE: "O",  # 定义人类棋子的表示字符
            HUMAN_KING: "O*",   # 定义人类王子的表示字符
            COMPUTER_PIECE: "X",  # 定义计算机棋子的表示字符
            COMPUTER_KING: "X*",   # 定义计算机王子的表示字符
        }

        s = "\n\n\n"  # 初始化字符串s，用于存储棋盘的字符串表示
        for y in range(7, -1, -1):  # 遍历棋盘的y坐标，从7到0
            for x in range(0, 8):  # 遍历棋盘的x坐标，从0到7
                piece_str = pieces[self.spaces[x][y]]  # 获取当前坐标上的棋子类型，并根据类型获取对应的表示字符
                piece_str += " " * (5 - len(piece_str))  # 将表示字符补齐到5个字符长度
                s += piece_str  # 将表示字符添加到字符串s中
            s += "\n"  # 在每行结束时添加换行符
        s += "\n\n"  # 在棋盘表示结束时添加两个换行符

        return s  # 返回棋盘的字符串表示

    def get_spaces(self) -> Iterator[Tuple[int, int]]:  # 定义一个生成器函数，用于遍历棋盘上的所有空格坐标
        for x in range(0, 8):  # 遍历棋盘的x坐标，从0到7
            for y in range(0, 8):  # 遍历棋盘的y坐标，从0到7
                yield x, y
```
这行代码是一个生成器函数的一部分，用于返回坐标(x, y)。

```
    def get_spaces_with_computer_pieces(self) -> Iterator[Tuple[int, int]]:
```
这是一个方法定义，用于获取棋盘上计算机棋子所在的位置。

```
        for x, y in self.get_spaces():
```
这是一个for循环，用于遍历棋盘上的所有位置。

```
            contents = self.spaces[x][y]
```
这行代码用于获取棋盘上位置(x, y)的内容。

```
            if contents < 0:
                yield x, y
```
这是一个条件语句，如果位置(x, y)的内容小于0，则返回坐标(x, y)。

```
    def get_spaces_with_human_pieces(self) -> Iterator[Tuple[int, int]]:
```
这是一个方法定义，用于获取棋盘上玩家棋子所在的位置。

```
    def get_legal_deltas_for_space(self, x: int, y: int) -> Iterator[Tuple[int, int]]:
```
这是一个方法定义，用于获取位置(x, y)的合法移动方向。

```
        if contents == COMPUTER_PIECE:
```
这是一个条件语句，用于检查位置(x, y)的内容是否为计算机棋子。

```
            for delta_x in (-1, 1):
                yield (delta_x, -1)
```
这是一个for循环，用于返回计算机棋子合法的移动方向。
            for delta_x in (-1, 1):
                for delta_y in (-1, 1):
                    yield (delta_x, delta_y)
```
这段代码是一个生成器函数，用于生成合法的移动方向。它会遍历(-1, 1)和(-1, 1)两个元组，生成所有可能的移动方向。

```
    def get_legal_moves(self, x: int, y: int) -> Iterator[MoveRecord]:
        for delta_x, delta_y in self.get_legal_deltas_for_space(x, y):
            new_move_record = self.check_move(x, y, delta_x, delta_y)

            if new_move_record is not None:
                yield new_move_record
```
这段代码是一个方法，用于获取在特定位置(x, y)的合法移动。它会遍历通过get_legal_deltas_for_space方法获取的合法移动方向，然后通过check_move方法检查移动是否合法，如果合法则返回移动记录。

```
    def pick_computer_move(self) -> Optional[MoveRecord]:
        move_record = None

        for start_x, start_y in self.get_spaces_with_computer_pieces():
            for delta_x, delta_y in self.get_legal_deltas_for_space(start_x, start_y):
                new_move_record = self.check_move(start_x, start_y, delta_x, delta_y)

                if new_move_record is None:
                    continue
```
这段代码是一个方法，用于让计算机选择移动。它会遍历所有计算机棋子的位置，然后通过get_legal_deltas_for_space方法获取合法移动方向，再通过check_move方法检查移动是否合法。如果移动不合法，则继续遍历下一个移动。
        if (move_record is None) or (  # 如果move_record为空或新的移动记录的质量高于当前move_record的质量
            new_move_record.quality > move_record.quality
        ):
            move_record = new_move_record  # 更新move_record为新的移动记录

    return move_record  # 返回最终的move_record

def check_move(  # 定义一个名为check_move的方法，接受start_x, start_y, delta_x, delta_y四个参数，并返回一个可选的MoveRecord对象
    self, start_x: int, start_y: int, delta_x: int, delta_y: int
) -> Optional[MoveRecord]:
    new_x = start_x + delta_x  # 计算新的x坐标
    new_y = start_y + delta_y  # 计算新的y坐标
    if not is_legal_board_coordinate(new_x, new_y):  # 如果新的坐标不合法
        return None  # 返回空值

    contents = self.spaces[new_x][new_y]  # 获取新坐标处的内容
    if contents == EMPTY_SPACE:  # 如果新坐标处为空
        return self.evaluate_move(start_x, start_y, new_x, new_y)  # 调用evaluate_move方法评估移动
    if contents < 0:  # 如果新坐标处有其他棋子
            return None  # 如果条件不满足，返回空值

        # 检查跳跃着陆点，这是从新的 x、y 坐标偏移的额外 dx、dy
        landing_x = new_x + delta_x  # 计算着陆点的 x 坐标
        landing_y = new_y + delta_y  # 计算着陆点的 y 坐标

        if not is_legal_board_coordinate(landing_x, landing_y):  # 如果着陆点坐标不合法
            return None  # 返回空值
        if self.spaces[landing_x][landing_y] == EMPTY_SPACE:  # 如果着陆点是空格
            return self.evaluate_move(start_x, start_y, landing_x, landing_y)  # 调用 evaluate_move 方法
        return None  # 其他情况返回空值

    def evaluate_move(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> MoveRecord:
        quality = 0  # 初始化 quality 变量为 0
        if dest_y == 0 and self.spaces[start_x][start_y] == COMPUTER_PIECE:  # 如果目标 y 坐标为 0 且起始坐标处是计算机棋子
            quality += 2  # quality 值加 2
        if abs(dest_y - start_y) == 2:  # 如果目标 y 坐标与起始 y 坐标的绝对值为 2
            # jumps are good
            # 跳跃是好的
            quality += 5
            # 品质加5
        if start_y == 7:
            # prefer to defend back row
            # 更倾向于防守后排
            quality -= 2
            # 品质减2
        if dest_x in (0, 7):
            # moving to edge column
            # 移动到边缘列
            quality += 1
            # 品质加1
        for delta_x in (-1, 1):
            if not is_legal_board_coordinate(dest_x + delta_x, dest_y - 1):
                continue
            # 如果目标位置不合法，则继续下一次循环

            if self.spaces[dest_x + delta_x][dest_y - 1] < 0:
                # moving into "shadow" of another computer piece
                # 移动到另一台计算机棋子的“阴影”中
                quality += 1
                # 品质加1

            if not is_legal_board_coordinate(dest_x - delta_x, dest_y + 1):
                continue
            # 如果目标位置不合法，则继续下一次循环

            if (
                (self.spaces[dest_x + delta_x][dest_y - 1] > 0)  # 检查目标位置左上方是否有对方棋子
                and (self.spaces[dest_x - delta_x][dest_y + 1] == EMPTY_SPACE)  # 检查目标位置右下方是否为空
                or ((dest_x - delta_x == start_x) and (dest_y + 1 == start_y))  # 检查目标位置是否为起始位置的右下方
            ):
                # we are moving up to a human checker that could jump us
                quality -= 2  # 如果目标位置上有对方棋子，quality减2
        return MoveRecord(quality, start_x, start_y, dest_x, dest_y)  # 返回移动记录

    def remove_r_pieces(self, move_record: MoveRecord) -> None:  # 移除红色棋子
        self.remove_pieces(  # 调用移除棋子的方法
            move_record.start_x,  # 起始位置x坐标
            move_record.start_y,  # 起始位置y坐标
            move_record.dest_x,  # 目标位置x坐标
            move_record.dest_y,  # 目标位置y坐标
        )

    def remove_pieces(  # 移除棋子
        self, start_x: int, start_y: int, dest_x: int, dest_y: int  # 起始位置和目标位置的坐标
    ) -> None:
        self.spaces[dest_x][dest_y] = self.spaces[start_x][start_y]  # 将目标位置的棋子替换为起始位置的棋子
        # 将起始位置标记为空格
        self.spaces[start_x][start_y] = EMPTY_SPACE

        # 如果目标位置和起始位置的横坐标差的绝对值为2，说明是跳吃对方棋子
        if abs(dest_x - start_x) == 2:
            # 计算跳吃对方棋子的中间位置
            mid_x = (start_x + dest_x) // 2
            mid_y = (start_y + dest_y) // 2
            # 将中间位置标记为空格
            self.spaces[mid_x][mid_y] = EMPTY_SPACE

    # 电脑执行移动的方法
    def play_computer_move(self, move_record: MoveRecord) -> None:
        # 打印电脑移动的起始位置和目标位置
        print(
            f"FROM {move_record.start_x} {move_record.start_y} TO {move_record.dest_x} {move_record.dest_y}"
        )

        # 循环直到满足条件退出
        while True:
            if move_record.dest_y == BOTTOM_ROW:
                # 如果电脑移动到底行，将其棋子升级为王
                self.remove_r_pieces(move_record)
                self.spaces[move_record.dest_x][move_record.dest_y] = COMPUTER_KING
                return
            else:
                # 否则将目标位置标记为电脑的棋子
                self.spaces[move_record.dest_x][move_record.dest_y] = self.spaces[
                move_record.start_x  # 获取移动记录的起始 x 坐标
                ][move_record.start_y]  # 获取移动记录的起始 y 坐标
                self.remove_r_pieces(move_record)  # 调用 remove_r_pieces 方法移除指定位置的棋子

                if abs(move_record.dest_x - move_record.start_x) != 2:  # 如果目标 x 坐标与起始 x 坐标的差的绝对值不等于 2，则返回

                landing_x = move_record.dest_x  # 将目标 x 坐标赋值给 landing_x
                landing_y = move_record.dest_y  # 将目标 y 坐标赋值给 landing_y

                best_move = None  # 初始化 best_move 变量为 None
                if self.spaces[landing_x][landing_y] == COMPUTER_PIECE:  # 如果目标位置上是计算机棋子
                    for delta_x in (-2, 2):  # 遍历 (-2, 2) 中的值
                        test_record = self.try_extend(landing_x, landing_y, delta_x, -2)  # 调用 try_extend 方法
                        if (move_record is not None) and (  # 如果移动记录不为空且
                            (best_move is None)  # best_move 为空
                            or (move_record.quality > best_move.quality)  # 或者移动记录的质量大于 best_move 的质量
                        ):
                            best_move = test_record  # 将 test_record 赋值给 best_move
                else:  # 如果目标位置上不是计算机棋子
                    # 确保落点上是计算机的国王
                    assert self.spaces[landing_x][landing_y] == COMPUTER_KING
                    # 遍历所有可能的跳跃方向
                    for delta_x in (-2, 2):
                        for delta_y in (-2, 2):
                            # 尝试扩展跳跃路径
                            test_record = self.try_extend(
                                landing_x, landing_y, delta_x, delta_y
                            )
                            # 如果移动记录不为空，并且当前移动是最佳移动或者比当前最佳移动质量更高
                            if (move_record is not None) and (
                                (best_move is None)
                                or (move_record.quality > best_move.quality)
                            ):
                                best_move = test_record

                # 如果没有最佳移动，则返回
                if best_move is None:
                    return
                else:
                    # 打印最佳移动的目的地坐标
                    print(f"TO {best_move.dest_x} {best_move.dest_y}")
                    move_record = best_move

    def try_extend(
        self, start_x: int, start_y: int, delta_x: int, delta_y: int
        ) -> Optional[MoveRecord]:  # 定义函数返回类型为可选的移动记录
        new_x = start_x + delta_x  # 计算新的 x 坐标
        new_y = start_y + delta_y  # 计算新的 y 坐标

        if not is_legal_board_coordinate(new_x, new_y):  # 如果新坐标不在合法的棋盘范围内
            return None  # 返回空值

        jumped_x = start_x + delta_x // 2  # 计算跳跃后的 x 坐标
        jumped_y = start_y + delta_y // 2  # 计算跳跃后的 y 坐标

        if (self.spaces[new_x][new_y] == EMPTY_SPACE) and (  # 如果新位置为空，并且跳跃位置上有对方棋子
            self.spaces[jumped_x][jumped_y] > 0
        ):
            return self.evaluate_move(start_x, start_y, new_x, new_y)  # 返回评估后的移动记录
        return None  # 否则返回空值

    def get_human_move(self) -> Tuple[int, int, int, int]:  # 定义函数返回类型为元组，包含四个整数
        is_king = False  # 初始化是否为国王的标志为假

        while True:  # 无限循环
            # 获取起始位置的坐标
            start_x, start_y = get_coordinates("FROM?")

            # 获取起始位置的合法移动列表
            legal_moves = list(self.get_legal_moves(start_x, start_y))
            # 如果没有合法移动，则打印提示信息并重新选择起始位置
            if not legal_moves:
                print(f"({start_x}, {start_y}) has no legal moves. Choose again.")
                continue
            # 如果起始位置上没有棋子，则跳出循环
            if self.spaces[start_x][start_y] > 0:
                break

        # 判断起始位置上的棋子是否为王
        is_king = self.spaces[start_x][start_y] == HUMAN_KING

        # 循环直到输入合法的目标位置
        while True:
            # 获取目标位置的坐标
            dest_x, dest_y = get_coordinates("TO?")

            # 如果不是王且目标位置在起始位置的上方，则继续循环
            if (not is_king) and (dest_y < start_y):
                # 作弊！试图让非王棋子向后移动
                continue
            # 判断目标位置是否为空
            is_free = self.spaces[dest_x][dest_y] == 0
            # 判断目标位置是否在起始位置的可达范围内
            within_reach = abs(dest_x - start_x) <= 2
            # 判断移动是否为对角线移动
            is_diagonal_move = abs(dest_x - start_x) == abs(dest_y - start_y)
            if is_free and within_reach and is_diagonal_move:
                break  # 如果条件满足，则跳出循环
        return start_x, start_y, dest_x, dest_y  # 返回起始坐标和目标坐标

    def get_human_extension(
        self, start_x: int, start_y: int
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        is_king = self.spaces[start_x][start_y] == HUMAN_KING  # 检查起始位置是否为王子

        while True:  # 进入无限循环
            dest_x, dest_y = get_coordinates("+TO?")  # 获取目标坐标

            if dest_x < 0:  # 如果目标坐标小于0
                return False, None  # 返回False和空值
            if (not is_king) and (dest_y < start_y):  # 如果不是王子且目标y坐标小于起始y坐标
                # CHEATER! Trying to move non-king backwards
                continue  # 继续下一次循环
            if (
                (self.spaces[dest_x][dest_y] == EMPTY_SPACE)  # 如果目标位置为空
                and (abs(dest_x - start_x) == 2)  # 且目标位置与起始位置的横坐标差为2
    def play_human_move(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> None:
        # 调用 remove_pieces 方法移动棋子
        self.remove_pieces(start_x, start_y, dest_x, dest_y)

        # 如果目的地的 y 坐标为顶部行，则将棋子升级为王
        if dest_y == TOP_ROW:
            # KING ME
            self.spaces[dest_x][dest_y] = HUMAN_KING

    def check_pieces(self) -> bool:
        # 如果计算机棋子数量为 0，则打印人类获胜信息并返回 False
        if len(list(self.get_spaces_with_computer_pieces())) == 0:
            print_human_won()
            return False
        # 如果人类棋子数量为 0，则打印计算机获胜信息并返回 False
        if len(list(self.get_spaces_with_computer_pieces())) == 0:
            print_computer_won()
            return False
        return True  # 返回 True 值

def print_instructions() -> None:
    print("THIS IS THE GAME OF CHECKERS.  THE COMPUTER IS X,")  # 打印游戏说明
    print("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.")  # 打印游戏说明
    print("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.")  # 打印游戏说明
    print("(0,0) IS THE LOWER LEFT CORNER")  # 打印游戏说明
    print("(0,7) IS THE UPPER LEFT CORNER")  # 打印游戏说明
    print("(7,0) IS THE LOWER RIGHT CORNER")  # 打印游戏说明
    print("(7,7) IS THE UPPER RIGHT CORNER")  # 打印游戏说明
    print("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER")  # 打印游戏说明
    print("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.\n\n\n")  # 打印游戏说明

def print_human_won() -> None:
    print("\nYOU WIN.")  # 打印玩家获胜信息

def print_computer_won() -> None:
    print("\nI WIN.")  # 打印"I WIN."消息

def play_game() -> None:
    board = Board()  # 创建一个棋盘对象

    while True:  # 进入游戏循环
        move_record = board.pick_computer_move()  # 让计算机选择移动
        if move_record is None:  # 如果没有可行的移动
            print_human_won()  # 打印人类获胜的消息
            return  # 结束游戏
        board.play_computer_move(move_record)  # 让计算机执行移动

        print(board)  # 打印当前棋盘状态

        if not board.check_pieces():  # 如果没有棋子了
            return  # 结束游戏

        start_x, start_y, dest_x, dest_y = board.get_human_move()  # 获取人类玩家的移动
        board.play_human_move(start_x, start_y, dest_x, dest_y)  # 让人类玩家执行移动
        if abs(dest_x - start_x) == 2:  # 如果目标位置和起始位置的横坐标差的绝对值为2
            while True:  # 进入循环
                extend, move = board.get_human_extension(dest_x, dest_y)  # 调用board对象的get_human_extension方法，获取扩展和移动信息
                assert move is not None  # 断言移动不为空
                if not extend:  # 如果没有扩展
                    break  # 退出循环
                start_x, start_y, dest_x, dest_y = move  # 更新起始位置和目标位置
                board.play_human_move(start_x, start_y, dest_x, dest_y)  # 调用board对象的play_human_move方法，执行人类移动

def main() -> None:  # 主函数声明，返回空值
    print_header("CHECKERS")  # 打印游戏标题
    print_instructions()  # 打印游戏说明

    play_game()  # 调用play_game函数，开始游戏

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用主函数
```