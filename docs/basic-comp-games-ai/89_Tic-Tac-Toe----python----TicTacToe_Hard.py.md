# `basic-computer-games\89_Tic-Tac-Toe\python\TicTacToe_Hard.py`

```
from typing import List, Tuple, Union

# 定义一个井字游戏的类
class TicTacToe:
    # 初始化方法，设置玩家选择和棋盘大小
    def __init__(self, pick, sz=3) -> None:
        self.pick = pick
        self.dim_sz = sz
        self.board = self.clear_board()

    # 清空棋盘的方法
    def clear_board(self) -> List[List[str]]:
        # 创建一个默认大小的棋盘
        board = [["blur" for i in range(self.dim_sz)] for j in range(self.dim_sz)]
        return board

    # 记录玩家的移动
    def move_record(self, r, c) -> Union[str, bool]:
        if r > self.dim_sz or c > self.dim_sz:
            return "Out of Bounds"
        if self.board[r][c] != "blur":
            return "Spot Pre-Occupied"
        self.board[r][c] = self.pick
        return True

# 显示当前棋盘状态
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
            line1 = line1 + "  " + game.board[i][game.dim_sz - 1] + " \n"
    print(line1, "\n\n")

# 主函数
def main() -> None:
    pick = input("Pick 'X' or 'O' ").strip().upper()
    if pick == "O":
        game = TicTacToe("O")
    else:
        game = TicTacToe("X")
    display(game=game)
    # 无限循环，直到游戏结束
    while True:
        # 初始化临时变量为布尔值或字符串类型
        temp: Union[bool, str] = False
        # 内层循环，直到输入有效的移动
        while not temp:
            # 从用户输入中获取移动坐标，并转换为整数列表
            move = list(
                map(
                    int,
                    input("Make A Move in Grid System from (0,0) to (2,2) ").split(),
                )
            )
            # 调用游戏对象的移动记录方法，记录玩家的移动
            temp = game.move_record(move[0], move[1])
            # 如果移动无效，则打印错误信息
            if not temp:
                print(temp)

        # 如果玩家获胜，打印胜利信息并结束游戏
        if game.check_win() == 1:
            print("You Won!")
            break
        # 打印玩家的移动后的游戏状态
        print("Your Move:- ")
        display(game)
        # 获取计算机的下一步移动
        C1, C2 = game.next_move()
        # 如果计算机无法移动，打印平局信息并结束游戏
        if C1 == -1 and C2 == -1:
            print("Game Tie!")
            break
        # 如果玩家输掉游戏，打印失败信息并结束游戏
        if game.check_win() == 0:
            print("You lost!")
            break
        # 打印计算机的移动后的游戏状态
        print("Computer's Move :-")
        display(game)
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```