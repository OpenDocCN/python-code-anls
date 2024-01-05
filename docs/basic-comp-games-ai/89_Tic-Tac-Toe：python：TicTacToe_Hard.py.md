# `89_Tic-Tac-Toe\python\TicTacToe_Hard.py`

```
from typing import List, Tuple, Union  # 导入类型提示模块

class TicTacToe:
    def __init__(self, pick, sz=3) -> None:  # 初始化函数，接受玩家选择和棋盘大小作为参数
        self.pick = pick  # 设置玩家选择
        self.dim_sz = sz  # 设置棋盘大小
        self.board = self.clear_board()  # 调用clear_board函数初始化棋盘

    def clear_board(self) -> List[List[str]]:  # 清空棋盘函数，返回一个二维列表
        board = [["blur" for i in range(self.dim_sz)] for j in range(self.dim_sz)]  # 创建一个3x3的二维列表并初始化为"blur"
        return board  # 返回初始化后的棋盘

    def move_record(self, r, c) -> Union[str, bool]:  # 记录玩家移动的函数，接受行和列作为参数，返回字符串或布尔值
        if r > self.dim_sz or c > self.dim_sz:  # 如果行或列超出棋盘范围
            return "Out of Bounds"  # 返回"Out of Bounds"
        if self.board[r][c] != "blur":  # 如果指定位置已经被占据
            return "Spot Pre-Occupied"  # 返回"Spot Pre-Occupied"
        self.board[r][c] = self.pick  # 将玩家选择填入指定位置
        return True  # 返回 True

    def check_win(self) -> int:  # 1 you won, 0 computer won, -1 tie  # 检查是否获胜，返回值为 1 表示玩家获胜，0 表示计算机获胜，-1 表示平局
        # Flag syntax -> first player no. ,
        # User is Player#1 ;
        # Check set 1 -> row and '\' diagonal & Check set 2 -> col and '/' diagonal
        # 标志语法 -> 第一个玩家编号，
        # 用户是玩家＃1；
        # 检查集1 -> 行和'\'对角线和检查集2 -> 列和'/'对角线

        for i in range(0, self.dim_sz):  # Rows  # 遍历行
            flag11 = True  # 初始化标志
            flag21 = True  # 初始化标志

            flag12 = True  # 初始化标志
            flag22 = True  # 初始化标志
            for j in range(0, self.dim_sz):  # 遍历列

                ch2 = self.board[i][j]  # 获取当前位置的值
                ch1 = self.board[j][i]  # 获取当前位置的值
                # Row
                # 行
                if ch1 == self.pick:  # if it's mine, computer didn't make it  # 如果是我的，计算机没有做
                    flag21 = False  # 更新标志
                elif ch1 == "blur":  # 如果是空白，则表示没有人放置棋子
                    flag11 = False
                    flag21 = False
                else:
                    flag11 = False  # 否则表示我没有放置棋子

                if ch2 == self.pick:  # 同样的逻辑，针对列
                    flag22 = False
                elif ch2 == "blur":
                    flag12 = False
                    flag22 = False
                else:
                    flag12 = False

            if flag11 is True or flag12 is True:  # 如果我赢了
                return 1
            if flag21 is True or flag22 is True:  # 电脑赢了
                return 0

        # 对角线#
        flag11 = True  # 初始化标志位flag11为True
        flag21 = True  # 初始化标志位flag21为True

        flag12 = True  # 初始化标志位flag12为True
        flag22 = True  # 初始化标志位flag22为True
        for i in range(0, self.dim_sz):  # 遍历范围为0到self.dim_sz的循环

            ch2 = self.board[i][i]  # 获取self.board[i][i]位置的字符赋值给ch2
            ch1 = self.board[i][self.dim_sz - 1 - i]  # 获取self.board[i][self.dim_sz - 1 - i]位置的字符赋值给ch1

            if ch1 == self.pick:  # 如果ch1等于self.pick
                flag21 = False  # 将flag21置为False
            elif ch1 == "blur":  # 否则如果ch1等于"blur"
                flag11 = False  # 将flag11置为False
                flag21 = False  # 将flag21置为False
            else:  # 否则
                flag11 = False  # 将flag11置为False

            if ch2 == self.pick:  # 如果ch2等于self.pick
                flag22 = False  # 将flag22置为False
            elif ch2 == "blur":
                # 如果 ch2 等于 "blur"，则将 flag12 和 flag22 设置为 False
                flag12 = False
                flag22 = False
            else:
                # 如果 ch2 不等于 "blur"，则将 flag12 设置为 False
                flag12 = False

        if flag11 or flag12:
            # 如果 flag11 或 flag12 为 True，则返回 1
            return 1
        if flag21 or flag22:
            # 如果 flag21 或 flag22 为 True，则返回 0
            return 0

        # 如果以上条件都不满足，则返回 -1
        return -1

    def next_move(self) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
        available_moves = []  # 将存储所有可用的移动
        player_win_spot = []  # 如果玩家获胜
        comp_pick = "O"
        if self.pick == "O":
            comp_pick = "X"
        for i in range(0, self.dim_sz):
            for j in range(0, self.dim_sz):  # 遍历每一列
                if self.board[i][j] == "blur":  # 如果当前位置为空
                    t = (i, j)  # 记录当前位置
                    available_moves.append(t)  # 将当前位置添加到可用移动列表中
                    self.board[i][j] = comp_pick  # 让计算机在当前位置下子
                    if self.check_win() == 0:  # 检查计算机是否获胜
                        return i, j  # 如果计算机获胜，则返回当前位置
                    self.board[i][j] = self.pick  # 让玩家在当前位置下子
                    if self.check_win() == 1:  # 如果玩家没有获胜
                        player_win_spot.append(t)  # 将当前位置添加到玩家可获胜位置列表中
                    self.board[i][j] = "blur"  # 恢复当前位置为空状态

        if len(player_win_spot) != 0:  # 如果玩家可获胜位置列表不为空
            self.board[player_win_spot[0][0]][player_win_spot[0][1]] = comp_pick  # 让计算机在玩家可获胜位置下子
            return player_win_spot[0][0], player_win_spot[0][1]  # 返回玩家可获胜位置

        if len(available_moves) == 1:  # 如果可用移动列表中只有一个位置
            self.board[available_moves[0][0]][available_moves[0][1]] = comp_pick  # 让计算机在该位置下子
        return [available_moves[0][0]], [available_moves[0][1]]  # 如果只有一个可用移动，返回该移动的坐标
        if len(available_moves) == 0:  # 如果没有可用移动
            return -1, -1  # 返回-1，-1表示没有可用移动
        c1, c2 = self.dim_sz // 2, self.dim_sz // 2  # 计算棋盘中心的坐标
        if (c1, c2) in available_moves:  # 如果中心坐标在可用移动中
            self.board[c1][c2] = comp_pick  # 在中心坐标处放置计算机选定的棋子
            return c1, c2  # 返回中心坐标
        for i in range(c1 - 1, -1, -1):  # 从中心向外遍历
            gap = c1 - i  # 计算距离中心的距离
            # 检查四个可能的位置
            # 边缘位置
            if (c1 - gap, c2 - gap) in available_moves:  # 如果左上角位置在可用移动中
                self.board[c1 - gap][c2 - gap] = comp_pick  # 在左上角位置放置计算机选定的棋子
                return c1 - gap, c2 - gap  # 返回左上角位置
            if (c1 - gap, c2 + gap) in available_moves:  # 如果右上角位置在可用移动中
                self.board[c1 - gap][c2 + gap] = comp_pick  # 在右上角位置放置计算机选定的棋子
                return c1 - gap, c2 + gap  # 返回右上角位置
            if (c1 + gap, c2 - gap) in available_moves:  # 如果左下角位置在可用移动中
                self.board[c1 + gap][c2 - gap] = comp_pick  # 在左下角位置放置计算机选定的棋子
            # 如果(c1 - gap, c2 - gap)在可用移动中
            if (c1 - gap, c2 - gap) in available_moves:
                # 在棋盘上放置计算机选择的棋子
                self.board[c1 - gap][c2 - gap] = comp_pick
                # 返回新的位置坐标
                return c1 - gap, c2 - gap
            # 如果(c1 - gap, c2 + gap)在可用移动中
            if (c1 - gap, c2 + gap) in available_moves:
                # 在棋盘上放置计算机选择的棋子
                self.board[c1 - gap][c2 + gap] = comp_pick
                # 返回新的位置坐标
                return c1 - gap, c2 + gap

            # 四条线

            # 遍历范围为0到gap
            for i in range(0, gap):
                # 如果(c1 - gap, c2 - gap + i)在可用移动中，表示从左上到右上
                if (c1 - gap, c2 - gap + i) in available_moves:
                    # 在棋盘上放置计算机选择的棋子
                    self.board[c1 - gap][c2 - gap + i] = comp_pick
                    # 返回新的位置坐标
                    return c1 - gap, c2 - gap + i
                # 如果(c1 + gap, c2 - gap + i)在可用移动中，表示从左下到右下
                if (c1 + gap, c2 - gap + i) in available_moves:
                    # 在棋盘上放置计算机选择的棋子
                    self.board[c1 + gap][c2 - gap + i] = comp_pick
                    # 返回新的位置坐标
                    return c1 + gap, c2 - gap + i
                # 如果(c1 - gap, c2 - gap)在可用移动中，表示从左上到左下
                if (c1 - gap, c2 - gap) in available_moves:
                    # 在棋盘上放置计算机选择的棋子
                    self.board[c1 - gap + i][c2 - gap] = comp_pick
                    # 返回新的位置坐标
                    return c1 - gap + i, c2 - gap
                if (
                    c1 - gap + i,
                    c2 + gap,
                ) in available_moves:  # RIGHT TOP TO RIGHT BOTTOM
                    self.board[c1 - gap + i][c2 + gap] = comp_pick
                    return c1 - gap + i, c2 + gap
```
这段代码是一个条件语句，判断是否在可用移动中存在指定的位置。如果存在，就在棋盘上放置计算机选择的棋子，并返回该位置的坐标。

```
        raise RuntimeError("No moves available")
```
如果没有可用的移动，就会引发一个运行时错误。

```
def display(game: TicTacToe) -> None:
    line1 = ""
    for i in range(0, game.dim_sz):
        for j in range(0, game.dim_sz - 1):
            if game.board[i][j] == "blur":
                line1 = line1 + "    |"
            else:
                line1 = line1 + "  " + game.board[i][j] + " |"
        if game.board[i][game.dim_sz - 1] == "blur":
            line1 = line1 + "    \n"
        else:
```
这段代码定义了一个名为`display`的函数，用于在控制台上显示游戏的当前状态。它通过遍历游戏棋盘的行和列来构建一个字符串`line1`，并根据棋盘上的情况添加空格、棋子或者空白。
# 创建一个空字符串 line1，用于存储游戏板的第一行内容
line1 = ""

# 遍历游戏板的第一行，将每个格子的内容添加到 line1 中
for i in range(game.dim_sz):
    line1 = line1 + "  " + game.board[0][i] + " "

# 将 line1 添加到 line1 末尾，并换行
line1 = line1 + " \n"
# 打印 line1
print(line1, "\n\n")
            # 调用游戏对象的 move_record 方法，记录玩家的移动，并将返回值赋给 temp
            temp = game.move_record(move[0], move[1])
            # 如果 temp 为假（即移动无效），则打印 temp
            if not temp:
                print(temp)

        # 如果游戏检查到玩家获胜，打印 "You Won!" 并结束循环
        if game.check_win() == 1:
            print("You Won!")
            break
        # 打印提示信息
        print("Your Move:- ")
        # 显示游戏状态
        display(game)
        # 获取计算机的下一步移动
        C1, C2 = game.next_move()
        # 如果 C1 和 C2 均为 -1，表示游戏平局，打印 "Game Tie!" 并结束循环
        if C1 == -1 and C2 == -1:
            print("Game Tie!")
            break
        # 如果游戏检查到玩家失败，打印 "You lost!" 并结束循环
        if game.check_win() == 0:
            print("You lost!")
            break
        # 打印提示信息
        print("Computer's Move :-")
        # 显示游戏状态
        display(game)
# 如果当前脚本被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```

这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以让脚本既可以作为独立的程序执行，也可以作为模块被其他程序引用。
```