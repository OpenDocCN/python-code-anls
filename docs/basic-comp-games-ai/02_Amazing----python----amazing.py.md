# `basic-computer-games\02_Amazing\python\amazing.py`

```

# 导入必要的库
import enum
import random
from dataclasses import dataclass
from typing import List, Tuple

# 迷宫类
class Maze:
    def __init__(self, width: int, length: int) -> None:
        # 确保迷宫的宽度和长度大于等于2
        assert width >= 2 and length >= 2
        # 初始化used和walls列表
        used: List[List[int]] = []
        walls: List[List[int]] = []
        for _ in range(length):
            used.append([0] * width)
            walls.append([0] * width)

        # 随机选择一个入口，并标记为已使用
        enter_col = random.randint(0, width - 1)
        used[0][enter_col] = 1

        self.used = used
        self.walls = walls
        self.enter_col = enter_col
        self.width = width
        self.length = length

    # 添加出口
    def add_exit(self) -> None:
        """Modifies 'walls' to add an exit to the maze."""
        col = random.randint(0, self.width - 1)
        row = self.length - 1
        self.walls[row][col] = self.walls[row][col] + 1

    # 显示迷宫
    def display(self) -> None:
        # 显示迷宫的墙和通道
        ...

# 方向枚举类
class Direction(enum.Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

# 位置数据类
@dataclass
class Position:
    row: int
    col: int

# 主函数
def main() -> None:
    # 打印介绍
    print_intro()
    # 获取迷宫的宽度和长度
    width, length = get_maze_dimensions()
    # 构建迷宫并显示
    maze = build_maze(width, length)
    maze.display()

# 打印介绍
def print_intro() -> None:
    print(" " * 28 + "AMAZING PROGRAM")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

# 构建迷宫
def build_maze(width: int, length: int) -> Maze:
    """Build two 2D arrays."""
    # 构建迷宫
    ...

# 创建开口
def make_opening(
    maze: Maze,
    possible_dirs: List[Direction],
    pos: Position,
    count: int,
) -> Tuple[Position, int]:
    """
    Attention! This modifies 'used' and 'walls'
    """
    # 根据可能的方向创建开口
    ...

# 获取可能的方向
def get_possible_directions(maze: Maze, pos: Position) -> List[Direction]:
    """
    Get a list of all directions that are not blocked.

    Also ignore hit cells that we have already processed
    """
    # 获取可能的方向
    ...

# 获取迷宫的宽度和长度
def get_maze_dimensions() -> Tuple[int, int]:
    # 获取用户输入的迷宫宽度和长度
    ...

# 主函数入口
if __name__ == "__main__":
    main()

```