# `basic-computer-games\02_Amazing\python\amazing.py`

```py
# 导入必要的模块和类
import enum
import random
from dataclasses import dataclass
from typing import List, Tuple

# 定义迷宫类
class Maze:
    def __init__(self, width: int, length: int) -> None:
        # 确保迷宫的宽度和长度都大于等于2
        assert width >= 2 and length >= 2
        # 初始化用于标记已使用的二维列表和墙壁的二维列表
        used: List[List[int]] = []
        walls: List[List[int]] = []
        for _ in range(length):
            used.append([0] * width)
            walls.append([0] * width)

        # 随机选择一个入口，并标记为已使用
        enter_col = random.randint(0, width - 1)
        used[0][enter_col] = 1

        # 设置实例变量
        self.used = used
        self.walls = walls
        self.enter_col = enter_col
        self.width = width
        self.length = length

    # 添加出口到迷宫中
    def add_exit(self) -> None:
        """Modifies 'walls' to add an exit to the maze."""
        col = random.randint(0, self.width - 1)
        row = self.length - 1
        self.walls[row][col] = self.walls[row][col] + 1

    # 显示迷宫
    def display(self) -> None:
        for col in range(self.width):
            if col == self.enter_col:
                print(".  ", end="")
            else:
                print(".--", end="")
        print(".")
        for row in range(self.length):
            print("I", end="")
            for col in range(self.width):
                if self.walls[row][col] < 2:
                    print("  I", end="")
                else:
                    print("   ", end="")
            print()
            for col in range(self.width):
                if self.walls[row][col] == 0 or self.walls[row][col] == 2:
                    print(":--", end="")
                else:
                    print(":  ", end="")
            print(".")


# 定义方向枚举类
class Direction(enum.Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


# 定义位置数据类
@dataclass
class Position:
    row: int
    col: int


# 定义出口方向的常量
EXIT_DOWN = 1
EXIT_RIGHT = 2


# 主函数
def main() -> None:
    # 打印介绍信息
    print_intro()
    # 获取迷宫的宽度和长度
    width, length = get_maze_dimensions()
    # 构建迷宫
    maze = build_maze(width, length)
    # 显示迷宫
    maze.display()
def print_intro() -> None:
    # 打印程序介绍
    print(" " * 28 + "AMAZING PROGRAM")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def build_maze(width: int, length: int) -> Maze:
    """Build two 2D arrays."""
    # 初始化迷宫的宽度和长度
    assert width >= 2 and length >= 2

    # 创建迷宫对象
    maze = Maze(width, length)
    # 设置起始位置
    position = Position(row=0, col=maze.enter_col)
    # 计数器
    count = 2

    # 循环直到所有单元格都被处理
    while count != width * length + 1:
        # 获取可能的移动方向
        possible_dirs = get_possible_directions(maze, position)

        # 如果可以移动，移动并打开通路
        if len(possible_dirs) != 0:
            position, count = make_opening(maze, possible_dirs, position, count)
        # 否则，移动到下一个已使用的单元格，并重试
        else:
            while True:
                if position.col != width - 1:
                    position.col += 1
                elif position.row != length - 1:
                    position.row, position.col = position.row + 1, 0
                else:
                    position.row, position.col = 0, 0
                if maze.used[position.row][position.col] != 0:
                    break

    # 添加迷宫出口
    maze.add_exit()
    return maze


def make_opening(
    maze: Maze,
    possible_dirs: List[Direction],
    pos: Position,
    count: int,
) -> Tuple[Position, int]:
    """
    Attention! This modifies 'used' and 'walls'
    """
    # 随机选择一个方向
    direction = random.choice(possible_dirs)
    # 如果方向是向左，则打开右侧通路
    if direction == Direction.LEFT:
        pos.col = pos.col - 1
        maze.walls[pos.row][pos.col] = EXIT_RIGHT
    # 如果方向是向上
    elif direction == Direction.UP:
        # 更新位置的行数
        pos.row = pos.row - 1
        # 在迷宫的对应位置设置上方的出口
        maze.walls[pos.row][pos.col] = EXIT_DOWN
    # 如果方向是向右
    elif direction == Direction.RIGHT:
        # 在迷宫的对应位置设置右方的出口
        maze.walls[pos.row][pos.col] = maze.walls[pos.row][pos.col] + EXIT_RIGHT
        # 更新位置的列数
        pos.col = pos.col + 1
    # 如果方向是向下
    elif direction == Direction.DOWN:
        # 在迷宫的对应位置设置下方的出口
        maze.walls[pos.row][pos.col] = maze.walls[pos.row][pos.col] + EXIT_DOWN
        # 更新位置的行数
        pos.row = pos.row + 1
    # 在迷宫的使用情况记录中标记当前位置已经使用过
    maze.used[pos.row][pos.col] = count
    # 更新计数器
    count = count + 1
    # 返回更新后的位置和计数器
    return pos, count
# 获取可能的方向列表，不包括已经被阻挡的方向
def get_possible_directions(maze: Maze, pos: Position) -> List[Direction]:
    """
    Get a list of all directions that are not blocked.

    Also ignore hit cells that we have already processed
    """
    # 初始化可能的方向列表
    possible_dirs = list(Direction)
    # 如果当前位置在最左边或者左边的位置已经被占用，则移除左方向
    if pos.col == 0 or maze.used[pos.row][pos.col - 1] != 0:
        possible_dirs.remove(Direction.LEFT)
    # 如果当前位置在最上边或者上边的位置已经被占用，则移除上方向
    if pos.row == 0 or maze.used[pos.row - 1][pos.col] != 0:
        possible_dirs.remove(Direction.UP)
    # 如果当前位置在最右边或者右边的位置已经被占用，则移除右方向
    if pos.col == maze.width - 1 or maze.used[pos.row][pos.col + 1] != 0:
        possible_dirs.remove(Direction.RIGHT)
    # 如果当前位置在最下边或者下边的位置已经被占用，则移除下方向
    if pos.row == maze.length - 1 or maze.used[pos.row + 1][pos.col] != 0:
        possible_dirs.remove(Direction.DOWN)
    # 返回可能的方向列表
    return possible_dirs


# 获取迷宫的宽度和长度
def get_maze_dimensions() -> Tuple[int, int]:
    while True:
        # 获取用户输入的宽度和长度
        input_str = input("What are your width and length?")
        # 如果输入包含一个逗号
        if input_str.count(",") == 1:
            # 通过逗号分割宽度和长度
            width_str, length_str = input_str.split(",")
            # 将字符串转换为整数
            width = int(width_str)
            length = int(length_str)
            # 如果宽度和长度大于1，则跳出循环
            if width > 1 and length > 1:
                break
        # 如果输入的宽度和长度无意义，则提示用户重新输入
        print("Meaningless dimensions. Try again.")
    # 返回宽度和长度
    return width, length


# 如果作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```