# `MetaGPT\tests\data\demo_project\game.py`

```py

## game.py

# 导入 random 模块和 List、Tuple 类型
import random
from typing import List, Tuple

# 定义 Game 类
class Game:
    # 初始化方法
    def __init__(self):
        # 初始化游戏网格为 4x4 的二维数组，初始值为 0
        self.grid: List[List[int]] = [[0 for _ in range(4)] for _ in range(4)]
        # 初始化分数为 0
        self.score: int = 0
        # 初始化游戏结束标志为 False
        self.game_over: bool = False

    # 重置游戏方法
    def reset_game(self):
        # 重置游戏网格为 4x4 的二维数组，初始值为 0
        self.grid = [[0 for _ in range(4)] for _ in range(4)]
        # 重置分数为 0
        self.score = 0
        # 重置游戏结束标志为 False
        self.game_over = False
        # 添加两个新的方块
        self.add_new_tile()
        self.add_new_tile()

    # 移动方法
    def move(self, direction: str):
        # 根据方向移动网格中的方块
        if direction == "up":
            self._move_up()
        elif direction == "down":
            self._move_down()
        elif direction == "left":
            self._move_left()
        elif direction == "right":
            self._move_right()

    # 判断游戏是否结束方法
    def is_game_over(self) -> bool:
        # 遍历整个网格，判断游戏是否结束
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 0:
                    return False
                if j < 3 and self.grid[i][j] == self.grid[i][j + 1]:
                    return False
                if i < 3 and self.grid[i][j] == self.grid[i + 1][j]:
                    return False
        return True

    # 获取空白单元格位置方法
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        # 遍历整个网格，获取空白单元格的位置
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 0:
                    empty_cells.append((i, j))
        return empty_cells

    # 添加新方块方法
    def add_new_tile(self):
        # 获取空白单元格的位置
        empty_cells = self.get_empty_cells()
        # 如果有空白单元格
        if empty_cells:
            # 在空白单元格中随机选择一个位置
            x, y = random.choice(empty_cells)
            # 在选定位置生成一个新的方块，90% 的概率为 2，10% 的概率为 4
            self.grid[x][y] = 2 if random.random() < 0.9 else 4

    # 获取分数方法
    def get_score(self) -> int:
        return self.score

    # 向上移动方法
    def _move_up(self):
        # 遍历每一列
        for j in range(4):
            # 从第二行开始遍历到最后一行
            for i in range(1, 4):
                # 如果当前位置有方块
                if self.grid[i][j] != 0:
                    # 向上移动方块
                    for k in range(i, 0, -1):
                        if self.grid[k - 1][j] == 0:
                            self.grid[k - 1][j] = self.grid[k][j]
                            self.grid[k][j] = 0

    # 向下移动方法
    def _move_down(self):
        # 遍历每一列
        for j in range(4):
            # 从倒数第二行开始遍历到第一行
            for i in range(2, -1, -1):
                # 如果当前位置有方块
                if self.grid[i][j] != 0:
                    # 向下移动方块
                    for k in range(i, 3):
                        if self.grid[k + 1][j] == 0:
                            self.grid[k + 1][j] = self.grid[k][j]
                            self.grid[k][j] = 0

    # 向左移动方法
    def _move_left(self):
        # 遍历每一行
        for i in range(4):
            # 从第二列开始遍历到最后一列
            for j in range(1, 4):
                # 如果当前位置有方块
                if self.grid[i][j] != 0:
                    # 向左移动方块
                    for k in range(j, 0, -1):
                        if self.grid[i][k - 1] == 0:
                            self.grid[i][k - 1] = self.grid[i][k]
                            self.grid[i][k] = 0

    # 向右移动方法
    def _move_right(self):
        # 遍历每一行
        for i in range(4):
            # 从倒数第二列开始遍历到第一列
            for j in range(2, -1, -1):
                # 如果当前位置有方块
                if self.grid[i][j] != 0:
                    # 向右移动方块
                    for k in range(j, 3):
                        if self.grid[i][k + 1] == 0:
                            self.grid[i][k + 1] = self.grid[i][k]
                            self.grid[i][k] = 0

```