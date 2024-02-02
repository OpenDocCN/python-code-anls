# `basic-computer-games\88_3-D_Tic-Tac-Toe\python\qubit.py`

```py
#!/usr/bin/env python3

# 从 BASIC 计算机游戏中的 3D 井字棋源代码移植而来
# 作者为 David H. Ahl
# 代码起源于 Dartmouth College

from enum import Enum
from typing import Optional, Tuple, Union

# 定义移动的枚举类型，包括游戏状态和机器移动的类型
class Move(Enum):
    HUMAN_WIN = 0
    MACHINE_WIN = 1
    DRAW = 2
    MOVES = 3
    LIKES = 4
    TAKES = 5
    GET_OUT = 6
    YOU_FOX = 7
    NICE_TRY = 8
    CONCEDES = 9

# 定义玩家的枚举类型，包括空、人类和机器
class Player(Enum):
    EMPTY = 0
    HUMAN = 1
    MACHINE = 2

# 3D 井字棋游戏逻辑和机器对手
class TicTacToe3D:
    """The game logic for 3D Tic Tac Toe and the machine opponent"""

    # 获取指定位置的玩家
    def get(self, x, y, z) -> Player:
        m = self.board[4 * (4 * z + y) + x]
        if m == 40:
            return Player.MACHINE
        elif m == 8:
            return Player.HUMAN
        else:
            return Player.EMPTY

    # 在 3D 空间中进行移动
    def move_3d(self, x, y, z, player) -> bool:
        m = 4 * (4 * z + y) + x
        return self.move(m, player)

    # 在指定位置进行移动
    def move(self, m, player) -> bool:
        if self.board[m] > 1:
            return False

        if player == Player.MACHINE:
            self.board[m] = 40
        else:
            self.board[m] = 8
        return True

    # 获取 3D 位置对应的坐标
    def get_3d_position(self, m) -> Tuple[int, int, int]:
        x = m % 4
        y = (m // 4) % 4
        z = m // 16
        return x, y, z

    # 评估线的价值
    def evaluate_lines(self) -> None:
        self.lineValues = [0] * 76
        for j in range(76):
            value = 0
            for k in range(4):
                value += self.board[self.lines[j][k]]
            self.lineValues[j] = value

    # 标记线的策略
    def strategy_mark_line(self, i) -> None:
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                self.board[m] = 1

    # 清除策略标记
    def clear_strategy_marks(self) -> None:
        for i in range(64):
            if self.board[i] == 1:
                self.board[i] = 0
    # 标记并移动，根据给定的值范围和移动值选择最佳位置
    def mark_and_move(self, vlow, vhigh, vmove) -> Optional[Tuple[Move, int]]:
        """
        标记可以潜在地赢得游戏的人类或机器的线，并选择最佳的下棋位置
        """
        # 遍历76条线
        for i in range(76):
            value = 0
            # 计算每条线上的值
            for j in range(4):
                value += self.board[self.lines[i][j]]
            self.lineValues[i] = value
            # 如果值在给定范围内，则进行相应操作
            if vlow <= value < vhigh:
                if value > vlow:
                    # 返回移动三元组
                    return self.move_triple(i)
                # 标记该线
                self.strategy_mark_line(i)
        # 评估所有线
        self.evaluate_lines()

        # 再次遍历76条线
        for i in range(76):
            value = self.lineValues[i]
            # 如果值为4或者等于移动值，则返回对角线移动
            if value == 4 or value == vmove:
                return self.move_diagonals(i, 1)
        # 如果没有合适的移动，则返回空
        return None
    # 机器决定下一步棋的移动位置和结果
    def machine_move(self) -> Union[None, Tuple[Move, int], Tuple[Move, int, int]]:
        """machine works out what move to play"""
        # 清除策略标记
        self.clear_strategy_marks()

        # 评估棋盘上的线
        self.evaluate_lines()
        # 遍历不同情况下的事件
        for value, event in [
            (32, self.human_win),  # 人类获胜
            (120, self.machine_win),  # 机器获胜
            (24, self.block_human_win),  # 阻止人类获胜
        ]:
            for i in range(76):
                # 如果某条线的值等于特定值，则执行相应的事件
                if self.lineValues[i] == value:
                    return event(i)

        # 标记并移动
        m = self.mark_and_move(80, 88, 43)
        if m is not None:
            return m

        # 清除策略标记
        self.clear_strategy_marks()

        # 标记并移动
        m = self.mark_and_move(16, 24, 11)
        if m is not None:
            return m

        # 遍历不同情况下的事件
        for k in range(18):
            value = 0
            for i in range(4 * k, 4 * k + 4):
                for j in range(4):
                    value += self.board[self.lines[i][j]]
            # 如果值在特定范围内，则执行相应的事件
            if (32 <= value < 40) or (72 <= value < 80):
                for s in [1, 0]:
                    for i in range(4 * k, 4 * k + 4):
                        m = self.move_diagonals(i, s)
                        if m is not None:
                            return m

        # 清除策略标记
        self.clear_strategy_marks()

        # 遍历角落位置
        for y in self.corners:
            if self.board[y] == 0:
                return (Move.MOVES, y)

        # 遍历棋盘上的位置
        for i in range(64):
            if self.board[i] == 0:
                return (Move.LIKES, i)

        # 返回平局
        return (Move.DRAW, -1)

    # 人类获胜事件
    def human_win(self, i) -> Tuple[Move, int, int]:
        return (Move.HUMAN_WIN, -1, i)

    # 机器获胜事件
    def machine_win(self, i) -> Optional[Tuple[Move, int, int]]:
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                return (Move.MACHINE_WIN, m, i)
        return None

    # 阻止人类获胜事件
    def block_human_win(self, i) -> Optional[Tuple[Move, int]]:
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                return (Move.NICE_TRY, m)
        return None
    # 移动三个棋子，或者阻止玩家这样做
    def move_triple(self, i) -> Tuple[Move, int]:
        """make two lines-of-3 or prevent human from doing this"""
        # 遍历四个位置
        for j in range(4):
            # 获取当前位置的索引
            m = self.lines[i][j]
            # 如果该位置上有玩家的棋子
            if self.board[m] == 1:
                # 如果当前行的价值小于40
                if self.lineValues[i] < 40:
                    # 返回移动类型为 YOU_FOX，位置为 m
                    return (Move.YOU_FOX, m)
                else:
                    # 返回移动类型为 GET_OUT，位置为 m
                    return (Move.GET_OUT, m)
        # 如果没有符合条件的移动，返回 CONCEDES 类型和位置为 -1
        return (Move.CONCEDES, -1)

    # 在4x4方格的角落或中心盒子中选择移动
    def move_diagonals(self, i, s) -> Optional[Tuple[Move, int]]:
        if 0 < (i % 4) < 3:
            # 如果 i 除以 4 的余数在 1 和 2 之间，jrange 为 [1, 2]
            jrange = [1, 2]
        else:
            # 否则 jrange 为 [0, 3]
            jrange = [0, 3]
        # 遍历 jrange 中的位置
        for j in jrange:
            # 获取当前位置的索引
            m = self.lines[i][j]
            # 如果该位置上有玩家的棋子
            if self.board[m] == s:
                # 返回移动类型为 TAKES，位置为 m
                return (Move.TAKES, m)
        # 如果没有符合条件的移动，返回 None
        return None
class Qubit:
    # 将棋盘上的三维位置转换为字符串表示
    def move_code(self, board, m) -> str:
        x, y, z = board.get_3d_position(m)
        return f"{z + 1:d}{y + 1:d}{x + 1:d}"

    # 显示获胜的走法
    def show_win(self, board, i) -> None:
        for m in board.lines[i]:
            print(self.move_code(board, m))

    # 显示整个棋盘
    def show_board(self, board) -> None:
        c = " YM"
        for z in range(4):
            for y in range(4):
                print("   " * y, end="")
                for x in range(4):
                    p = board.get(x, y, z)
                    print(f"({c[p.value]})      ", end="")
                print("\n")
            print("\n")

    # 人类玩家的走法
    def human_move(self, board) -> bool:
        print()
        c = "1234"
        while True:
            h = input("Your move?\n")
            if h == "1":
                return False
            if h == "0":
                self.show_board(board)
                continue
            if (len(h) == 3) and (h[0] in c) and (h[1] in c) and (h[2] in c):
                x = c.find(h[2])
                y = c.find(h[1])
                z = c.find(h[0])
                if board.move_3d(x, y, z, Player.HUMAN):
                    break

                print("That square is used. Try again.")
            else:
                print("Incorrect move. Retype it--")

        return True

if __name__ == "__main__":
    game = Qubit()
    game.play()
```