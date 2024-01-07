# `basic-computer-games\89_Tic-Tac-Toe\python\TicTacToe_Hard.py`

```

# 导入类型提示模块
from typing import List, Tuple, Union

# 定义显示游戏界面的函数
def display(game: TicTacToe) -> None:
    # 初始化第一行
    line1 = ""
    # 遍历游戏棋盘的行
    for i in range(0, game.dim_sz):
        # 遍历游戏棋盘的列
        for j in range(0, game.dim_sz - 1):
            # 如果当前格子为空
            if game.board[i][j] == "blur":
                line1 = line1 + "    |"
            else:
                line1 = line1 + "  " + game.board[i][j] + " |"
        # 处理最后一列
        if game.board[i][game.dim_sz - 1] == "blur":
            line1 = line1 + "    \n"
        else:
            line1 = line1 + "  " + game.board[i][game.dim_sz - 1] + " \n"
    # 打印游戏界面
    print(line1, "\n\n")

# 主函数
def main() -> None:
    # 获取玩家选择的棋子
    pick = input("Pick 'X' or 'O' ").strip().upper()
    # 根据玩家选择的棋子创建游戏对象
    if pick == "O":
        game = TicTacToe("O")
    else:
        game = TicTacToe("X")
    # 显示初始游戏界面
    display(game=game)
    # 游戏循环
    while True:
        temp: Union[bool, str] = False
        # 玩家输入下棋位置，直到输入合法位置
        while not temp:
            move = list(
                map(
                    int,
                    input("Make A Move in Grid System from (0,0) to (2,2) ").split(),
                )
            )
            temp = game.move_record(move[0], move[1])
            if not temp:
                print(temp)

        # 判断玩家是否获胜
        if game.check_win() == 1:
            print("You Won!")
            break
        print("Your Move:- ")
        display(game)
        # 计算电脑下棋位置
        C1, C2 = game.next_move()
        # 判断游戏是否结束
        if C1 == -1 and C2 == -1:
            print("Game Tie!")
            break
        if game.check_win() == 0:
            print("You lost!")
            break
        print("Computer's Move :-")
        display(game)

# 程序入口
if __name__ == "__main__":
    main()

```