# `d:/src/tocomm/basic-computer-games\67_One_Check\python\onecheck.py`

```
"""
ONE CHECK

Port to Python by imiro
"""

from typing import Tuple


def main() -> None:
    # Initial instructions
    print(" " * 30 + "ONE CHECK")  # 打印初始指令
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印初始指令
    print("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n")  # 打印初始指令
    print("48 CHECKERS ARE PLACED ON THE 2 OUTSIDE SPACES OF A")  # 打印初始指令
    print("STANDARD 64-SQUARE CHECKERBOARD.  THE OBJECT IS TO")  # 打印初始指令
    print("REMOVE AS MANY CHECKERS AS POSSIBLE BY DIAGONAL JUMPS")  # 打印初始指令
    print("(AS IN STANDARD CHECKERS).  USE THE NUMBERED BOARD TO")  # 打印初始指令
    print("INDICATE THE SQUARE YOU WISH TO JUMP FROM AND TO.  ON")  # 打印初始指令
    print("THE BOARD PRINTED OUT ON EACH TURN '1' INDICATES A")  # 打印初始指令
    # 打印提示信息
    print("CHECKER AND '0' AN EMPTY SQUARE.  WHEN YOU HAVE NO")
    print("POSSIBLE JUMPS REMAINING, INPUT A '0' IN RESPONSE TO")
    print("QUESTION 'JUMP FROM ?'\n")
    print("HERE IS THE NUMERICAL BOARD:\n")

    # 无限循环，直到用户选择退出游戏
    while True:
        # 打印棋盘的数字表示
        for j in range(1, 64, 8):
            for i in range(j, j + 7):
                print(i, end=(" " * (3 if i < 10 else 2)))
            print(j + 7)
        print("\nAND HERE IS THE OPENING POSITION OF THE CHECKERS.\n")

        # 调用 play_game() 函数进行游戏，并获取跳跃次数和剩余棋子数
        (jumps, left) = play_game()

        # 打印玩家的跳跃次数和剩余棋子数
        print()
        print(f"YOU MADE {jumps} JUMPS AND HAD {left} PIECES")
        print("REMAINING ON THE BOARD.\n")

        # 如果用户选择不再玩游戏，则退出循环
        if not (try_again()):
            break
    print("\nO.K.  HOPE YOU HAD FUN!!")  # 打印消息到控制台，提示用户游戏结束

def play_game() -> Tuple[str, str]:
    # Initialize board
    # Give more than 64 elements to accomodate 1-based indexing
    board = [1] * 70  # 创建一个长度为70的列表，用于表示游戏棋盘，初始值为1
    for j in range(19, 44, 8):  # 循环遍历棋盘的特定位置
        for i in range(j, j + 4):  # 在特定位置上将值设为0
            board[i] = 0
    jumps = 0  # 初始化跳跃次数为0
    while True:  # 无限循环
        # print board
        for j in range(1, 64, 8):  # 遍历棋盘的特定位置
            for i in range(j, j + 7):  # 在特定位置上打印棋盘的值
                print(board[i], end=" ")  # 打印棋盘的值并以空格结尾
            print(board[j + 7])  # 打印特定位置上的值并换行
        print()  # 打印空行
                or t2 > 8
                or f < 1
                or t < 1
                or f > 64
                or t > 64
                or (f1 + f2) % 2 == (t1 + t2) % 2
            ):
                print("ILLEGAL MOVE")
                continue

            # Make the move
            print("MOVE FROM", f_str, "TO", t_str)
            # Update the board with the new move
            update_board(f, t)
                or t2 > 8  # 如果目标位置的行数大于8
                or abs(f1 - t1) != 2  # 如果水平方向的移动距离不为2
                or abs(f2 - t2) != 2  # 如果垂直方向的移动距离不为2
                or board[(t + f) // 2] == 0  # 如果跳跃位置为空
                or board[f] == 0  # 如果起始位置为空
                or board[t] == 1  # 如果目标位置已经有棋子
            ):
                print("ILLEGAL MOVE.  TRY AGAIN...")  # 打印非法移动的提示信息
                continue  # 继续下一次循环
            break  # 跳出循环

        if f == 0:  # 如果起始位置为0
            break  # 跳出循环
        board[t] = 1  # 将目标位置标记为有棋子
        board[f] = 0  # 将起始位置标记为空
        board[(t + f) // 2] = 0  # 将跳跃位置标记为空
        jumps = jumps + 1  # 跳跃次数加一

    left = 0  # 初始化left为0
    for i in range(1, 64 + 1):  # 遍历1到64的范围
        left = left + board[i]  # 将列表 board 中索引为 i 的元素的值加到变量 left 上
    return (str(jumps), str(left))  # 返回 jumps 和 left 转换为字符串后的元组


def try_again() -> bool:
    print("TRY AGAIN", end=" ")  # 打印提示信息 "TRY AGAIN"，并以空格结尾
    answer = input().upper()  # 获取用户输入并转换为大写
    if answer == "YES":  # 如果用户输入为 "YES"
        return True  # 返回 True
    elif answer == "NO":  # 如果用户输入为 "NO"
        return False  # 返回 False
    print("PLEASE ANSWER 'YES' OR 'NO'.")  # 打印提示信息 "PLEASE ANSWER 'YES' OR 'NO'."
    return try_again()  # 递归调用 try_again 函数


if __name__ == "__main__":
    main()  # 调用主函数 main()
```