# `02_Amazing\python\amazing.py`

```
import enum  # 导入枚举类型模块
import random  # 导入随机数模块
from dataclasses import dataclass  # 导入数据类模块
from typing import List, Tuple  # 导入类型提示模块

# Python translation by Frank Palazzolo - 2/2021

class Maze:
    def __init__(self, width: int, length: int) -> None:  # 初始化迷宫类，接受宽度和长度参数
        assert width >= 2 and length >= 2  # 断言宽度和长度必须大于等于2
        used: List[List[int]] = []  # 创建一个二维列表用于标记迷宫中已使用的位置
        walls: List[List[int]] = []  # 创建一个二维列表用于标记迷宫中的墙
        for _ in range(length):  # 遍历迷宫的长度
            used.append([0] * width)  # 将长度为宽度且元素为0的列表添加到used中
            walls.append([0] * width)  # 将长度为宽度且元素为0的列表添加到walls中

        # Pick a random entrance, mark as used
        enter_col = random.randint(0, width - 1)  # 随机选择一个入口列
        used[0][enter_col] = 1  # 将入口位置标记为已使用
        self.used = used  # 将参数 used 赋值给对象的属性 used
        self.walls = walls  # 将参数 walls 赋值给对象的属性 walls
        self.enter_col = enter_col  # 将参数 enter_col 赋值给对象的属性 enter_col
        self.width = width  # 将参数 width 赋值给对象的属性 width
        self.length = length  # 将参数 length 赋值给对象的属性 length

    def add_exit(self) -> None:
        """Modifies 'walls' to add an exit to the maze."""
        col = random.randint(0, self.width - 1)  # 生成一个随机的列数
        row = self.length - 1  # 设置行数为 length - 1
        self.walls[row][col] = self.walls[row][col] + 1  # 在迷宫的出口位置加上一个出口

    def display(self) -> None:
        for col in range(self.width):  # 遍历迷宫的宽度
            if col == self.enter_col:  # 如果当前列是入口列
                print(".  ", end="")  # 打印一个点和空格
            else:
                print(".--", end="")  # 否则打印一个点和横线
        print(".")  # 打印最后一个点
        for row in range(self.length):  # 遍历迷宫的行
            print("I", end="")  # 打印字符"I"，并且不换行
            for col in range(self.width):  # 遍历迷宫的列
                if self.walls[row][col] < 2:  # 如果当前位置的墙壁值小于2
                    print("  I", end="")  # 打印两个空格和字符"I"，并且不换行
                else:  # 如果当前位置的墙壁值大于等于2
                    print("   ", end="")  # 打印三个空格，不换行
            print()  # 换行
            for col in range(self.width):  # 再次遍历迷宫的列
                if self.walls[row][col] == 0 or self.walls[row][col] == 2:  # 如果当前位置的墙壁值为0或2
                    print(":--", end="")  # 打印":--"，不换行
                else:  # 如果当前位置的墙壁值不为0或2
                    print(":  ", end="")  # 打印":  "，不换行
            print(".")  # 打印"."，换行


class Direction(enum.Enum):  # 定义一个枚举类Direction
    LEFT = 0  # 枚举值LEFT的值为0
    UP = 1  # 枚举值UP的值为1
    RIGHT = 2  # 枚举值RIGHT的值为2
    DOWN = 3  # 定义一个常量 DOWN，值为3，表示向下移动

@dataclass
class Position:
    row: int  # 定义一个位置的行数
    col: int  # 定义一个位置的列数

# 给出出口方向取一个友好的名称
EXIT_DOWN = 1  # 定义一个常量 EXIT_DOWN，值为1，表示向下的出口
EXIT_RIGHT = 2  # 定义一个常量 EXIT_RIGHT，值为2，表示向右的出口

def main() -> None:
    print_intro()  # 调用打印介绍的函数
    width, length = get_maze_dimensions()  # 调用获取迷宫尺寸的函数，获取迷宫的宽度和长度
    maze = build_maze(width, length)  # 调用构建迷宫的函数，构建一个迷宫
    maze.display()  # 显示迷宫
def print_intro() -> None:
    # 打印程序介绍
    print(" " * 28 + "AMAZING PROGRAM")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def build_maze(width: int, length: int) -> Maze:
    """Build two 2D arrays."""
    #
    # used:
    #   Initially set to zero, unprocessed cells
    #   Filled in with consecutive non-zero numbers as cells are processed
    #
    # walls:
    #   Initially set to zero, (all paths blocked)
    #   Remains 0 if there is no exit down or right
    #   Set to 1 if there is an exit down
    #   Set to 2 if there is an exit right
    #   Set to 3 if there are exits down and right
    # 检查迷宫的宽度和长度是否大于等于2
    assert width >= 2 and length >= 2
    # 创建迷宫对象，指定宽度和长度
    maze = Maze(width, length)
    # 设置起始位置为迷宫入口
    position = Position(row=0, col=maze.enter_col)
    # 初始化计数器为2，表示已经走过的格子数量

    # 当计数器不等于迷宫格子总数加1时，继续循环
    while count != width * length + 1:
        # 获取当前位置可以移动的方向
        possible_dirs = get_possible_directions(maze, position)

        # 如果存在可以移动的方向，移动并打开通路
        if len(possible_dirs) != 0:
            position, count = make_opening(maze, possible_dirs, position, count)
        # 否则，移动到下一个已经走过的格子，然后重新尝试
        else:
            while True:
                # 如果当前列不是最后一列，向右移动一格
                if position.col != width - 1:
                    position.col += 1
                # 如果当前行不是最后一行，移动到下一行的第一列
                elif position.row != length - 1:
                    position.row, position.col = position.row + 1, 0
                # 否则，回到第一行第一列
                else:
                    position.row, position.col = 0, 0
                if maze.used[position.row][position.col] != 0:
                    break  # 如果迷宫中的当前位置已经被访问过，则跳出循环

    maze.add_exit()  # 在迷宫中添加出口
    return maze  # 返回迷宫对象


def make_opening(
    maze: Maze,
    possible_dirs: List[Direction],
    pos: Position,
    count: int,
) -> Tuple[Position, int]:
    """
    Attention! This modifies 'used' and 'walls'
    """
    direction = random.choice(possible_dirs)  # 从可能的方向中随机选择一个方向
    if direction == Direction.LEFT:  # 如果选择的方向是向左
        pos.col = pos.col - 1  # 更新位置的列数，向左移动
        maze.walls[pos.row][pos.col] = EXIT_RIGHT  # 在迷宫的墙壁中标记出右侧的出口
    elif direction == Direction.UP:  # 如果方向是向上
        pos.row = pos.row - 1  # 更新位置的行数
        maze.walls[pos.row][pos.col] = EXIT_DOWN  # 更新迷宫中对应位置的墙壁状态
    elif direction == Direction.RIGHT:  # 如果方向是向右
        maze.walls[pos.row][pos.col] = maze.walls[pos.row][pos.col] + EXIT_RIGHT  # 更新迷宫中对应位置的墙壁状态
        pos.col = pos.col + 1  # 更新位置的列数
    elif direction == Direction.DOWN:  # 如果方向是向下
        maze.walls[pos.row][pos.col] = maze.walls[pos.row][pos.col] + EXIT_DOWN  # 更新迷宫中对应位置的墙壁状态
        pos.row = pos.row + 1  # 更新位置的行数
    maze.used[pos.row][pos.col] = count  # 更新迷宫中对应位置的使用状态
    count = count + 1  # 更新计数器
    return pos, count  # 返回更新后的位置和计数器值


def get_possible_directions(maze: Maze, pos: Position) -> List[Direction]:
    """
    Get a list of all directions that are not blocked.

    Also ignore hit cells that we have already processed
    """
    # 创建一个包含所有可能方向的列表
    possible_dirs = list(Direction)
    # 如果当前位置在迷宫的最左边或者左边的位置已经被占用，则移除左方向
    if pos.col == 0 or maze.used[pos.row][pos.col - 1] != 0:
        possible_dirs.remove(Direction.LEFT)
    # 如果当前位置在迷宫的最上边或者上边的位置已经被占用，则移除上方向
    if pos.row == 0 or maze.used[pos.row - 1][pos.col] != 0:
        possible_dirs.remove(Direction.UP)
    # 如果当前位置在迷宫的最右边或者右边的位置已经被占用，则移除右方向
    if pos.col == maze.width - 1 or maze.used[pos.row][pos.col + 1] != 0:
        possible_dirs.remove(Direction.RIGHT)
    # 如果当前位置在迷宫的最下边或者下边的位置已经被占用，则移除下方向
    if pos.row == maze.length - 1 or maze.used[pos.row + 1][pos.col] != 0:
        possible_dirs.remove(Direction.DOWN)
    # 返回剩余的可能方向列表
    return possible_dirs


def get_maze_dimensions() -> Tuple[int, int]:
    # 无限循环，直到输入符合要求
    while True:
        # 获取用户输入的迷宫宽度和长度
        input_str = input("What are your width and length?")
        # 如果输入包含一个逗号
        if input_str.count(",") == 1:
            # 通过逗号分割宽度和长度
            width_str, length_str = input_str.split(",")
            # 将宽度和长度转换为整数
            width = int(width_str)
            length = int(length_str)
            # 如果宽度和长度都大于1
            if width > 1 and length > 1:
                break  # 结束循环，跳出当前循环
        print("Meaningless dimensions. Try again.")  # 打印提示信息，表示输入的尺寸无意义，提示重新输入
    return width, length  # 返回输入的宽度和长度


if __name__ == "__main__":  # 如果当前文件被作为主程序运行
    main()  # 调用主函数
```