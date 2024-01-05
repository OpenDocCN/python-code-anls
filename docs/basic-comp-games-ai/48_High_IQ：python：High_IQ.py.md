# `48_High_IQ\python\High_IQ.py`

```
from typing import Dict  # 导入 Dict 类型，用于声明返回值类型为字典

def new_board() -> Dict[int, str]:  # 定义函数 new_board，返回类型为字典，键为整数，值为字符串
    """
    Using a dictionary in python to store the board,
    since we are not including all numbers within a given range.
    """  # 函数的文档字符串，解释了函数的作用和原因

    return {  # 返回一个字典
        13: "!",  # 键为 13，值为 "!"
        14: "!",  # 键为 14，值为 "!"
        15: "!",  # 键为 15，值为 "!"
        22: "!",  # 键为 22，值为 "!"
        23: "!",  # 键为 23，值为 "!"
        24: "!",  # 键为 24，值为 "!"
        29: "!",  # 键为 29，值为 "!"
        30: "!",  # 键为 30，值为 "!"
        31: "!",  # 键为 31，值为 "!"
        32: "!",  # 键为 32，值为 "!"
        33: "!",  # 键为 33，值为 "!"
34: "!",  # 创建一个键为34，值为"!"的字典项
35: "!",  # 创建一个键为35，值为"!"的字典项
38: "!",  # 创建一个键为38，值为"!"的字典项
39: "!",  # 创建一个键为39，值为"!"的字典项
40: "!",  # 创建一个键为40，值为"!"的字典项
42: "!",  # 创建一个键为42，值为"!"的字典项
43: "!",  # 创建一个键为43，值为"!"的字典项
44: "!",  # 创建一个键为44，值为"!"的字典项
47: "!",  # 创建一个键为47，值为"!"的字典项
48: "!",  # 创建一个键为48，值为"!"的字典项
49: "!",  # 创建一个键为49，值为"!"的字典项
50: "!",  # 创建一个键为50，值为"!"的字典项
51: "!",  # 创建一个键为51，值为"!"的字典项
52: "!",  # 创建一个键为52，值为"!"的字典项
53: "!",  # 创建一个键为53，值为"!"的字典项
58: "!",  # 创建一个键为58，值为"!"的字典项
59: "!",  # 创建一个键为59，值为"!"的字典项
60: "!",  # 创建一个键为60，值为"!"的字典项
67: "!",  # 创建一个键为67，值为"!"的字典项
68: "!",  # 创建一个键为68，值为"!"的字典项
        69: "!",  # 创建一个键为69，值为"!"的字典项
        41: "O",   # 创建一个键为41，值为"O"的字典项
    }


def print_instructions() -> None:  # 定义一个名为print_instructions的函数，不返回任何值
    print(  # 打印以下内容
        """
HERE IS THE BOARD:

          !    !    !
         13   14   15

          !    !    !
         22   23   24

!    !    !    !    !    !    !
29   30   31   32   33   34   35

!    !    !    !    !    !    !
```  # 打印游戏板的布局
38   # 根据 ZIP 文件名读取其二进制，封装成字节流
39   bio = BytesIO(open(fname, 'rb').read())
40   # 使用字节流里面内容创建 ZIP 对象
41   zip = zipfile.ZipFile(bio, 'r')
42   # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
43   fdict = {n:zip.read(n) for n in zip.namelist()}
44   # 关闭 ZIP 对象
47   zip.close()
48   # 返回结果字典
49   return fdict
50   
51   
52   
53   
58   TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD
59   WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG
60   NUMBERS.  OK, LET'S BEGIN.
67   """
68   )
69   
    # 打印第一行的棋盘格局
    print(" " * 2 + board[13] + board[14] + board[15])
    # 打印第二行的棋盘格局
    print(" " * 2 + board[22] + board[23] + board[24])
    # 打印第三行的棋盘格局
    print(
        board[29]
        + board[30]
        + board[31]
        + board[32]
        + board[33]
        + board[34]
        + board[35]
    )
    # 打印第四行的棋盘格局
    print(
        board[38]
        + board[39]
        + board[40]
        + board[41]
        + board[42]
        + board[43]
        + board[44]
    )
    print(
        board[47]  # 打印棋盘上第47个位置的内容
        + board[48]  # 打印棋盘上第48个位置的内容
        + board[49]  # 打印棋盘上第49个位置的内容
        + board[50]  # 打印棋盘上第50个位置的内容
        + board[51]  # 打印棋盘上第51个位置的内容
        + board[52]  # 打印棋盘上第52个位置的内容
        + board[53]  # 打印棋盘上第53个位置的内容
    )
    print(" " * 2 + board[58] + board[59] + board[60])  # 打印空格和棋盘上第58、59、60个位置的内容
    print(" " * 2 + board[67] + board[68] + board[69])  # 打印空格和棋盘上第67、68、69个位置的内容


def play_game() -> None:
    # 创建新的棋盘
    board = new_board()

    # 主游戏循环
    while not is_game_finished(board):  # 当游戏未结束时
        print_board(board)  # 打印当前棋盘状态
        while not move(board):
            print("ILLEGAL MOVE! TRY AGAIN")  # 如果移动不合法，打印提示信息

    # Check peg count and print the user's score
    peg_count = 0  # 初始化计数器
    for key in board.keys():  # 遍历棋盘上的位置
        if board[key] == "!":  # 如果该位置上有棋子
            peg_count += 1  # 计数器加一

    print("YOU HAD " + str(peg_count) + " PEGS REMAINING")  # 打印剩余的棋子数量

    if peg_count == 1:  # 如果只剩下一个棋子
        print("BRAVO! YOU MADE A PERFECT SCORE!")  # 打印祝贺信息
        print("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!")  # 提示用户保存记录


def move(board: Dict[int, str]) -> bool:
    """Queries the user to move. Returns false if the user puts in an invalid input or move, returns true if the move was successful"""
    start_input = input("MOVE WHICH PIECE? ")  # 提示用户输入要移动的棋子位置
    if not start_input.isdigit():  # 检查输入的起始位置是否为数字，如果不是则返回 False
        return False

    start = int(start_input)  # 将起始位置转换为整数类型

    if start not in board or board[start] != "!":  # 检查起始位置是否在棋盘上，且是否为"!"，如果不是则返回 False
        return False

    end_input = input("TO WHERE? ")  # 获取输入的目标位置

    if not end_input.isdigit():  # 检查输入的目标位置是否为数字，如果不是则返回 False
        return False

    end = int(end_input)  # 将目标位置转换为整数类型

    if end not in board or board[end] != "O":  # 检查目标位置是否在棋盘上，且是否为"O"，如果不是则返回 False
        return False

    difference = abs(start - end)  # 计算起始位置和目标位置的差值的绝对值
    center = int((end + start) / 2)  # 计算起始位置和目标位置的中间位置
    if (
        (difference == 2 or difference == 18)  # 检查 difference 是否等于 2 或 18
        and board[end] == "O"  # 检查 board[end] 是否为 "O"
        and board[center] == "!"  # 检查 board[center] 是否为 "!"
    ):
        board[start] = "O"  # 将 board[start] 设置为 "O"
        board[center] = "O"  # 将 board[center] 设置为 "O"
        board[end] = "!"  # 将 board[end] 设置为 "!"
        return True  # 返回 True
    else:
        return False  # 返回 False


def main() -> None:
    print(" " * 33 + "H-I-Q")  # 打印游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印创意计算的位置
    print_instructions()  # 调用打印游戏说明的函数
    play_game()  # 调用开始游戏的函数
def is_game_finished(board) -> bool:
    """Check all locations and whether or not a move is possible at that location."""
    # 遍历棋盘上的所有位置
    for pos in board.keys():
        # 如果当前位置有一个棋子
        if board[pos] == "!":
            # 遍历相邻的位置，分别为1和9
            for space in [1, 9]:
                # 检查下一个位置是否有一个棋子
                next_to_peg = ((pos + space) in board) and board[pos + space] == "!"
                # 检查向前（+位置）或向后（-位置）移动是否有可移动的空间
                has_movable_space = (
                    not ((pos - space) in board and board[pos - space] == "!")
                ) or (
                    not ((pos + space * 2) in board and board[pos + space * 2] == "!")
                )
                # 如果下一个位置有一个棋子并且有可移动的空间，则游戏未结束
                if next_to_peg and has_movable_space:
                    return False
    # 如果没有找到可移动的位置，则游戏结束
    return True


if __name__ == "__main__":
    main()
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```