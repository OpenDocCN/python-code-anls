# `26_Chomp\python\chomp.py`

```
#!/usr/bin/env python3  # 指定脚本解释器为 Python3

"""
CHOMP

Converted from BASIC to Python by Trevor Hobson
"""


class Canvas:  # 定义一个名为 Canvas 的类，用于绘制饼干

    def __init__(self, width=9, height=9, fill="*") -> None:  # 初始化方法，设置默认宽度、高度和填充字符
        self._buffer = []  # 创建一个空列表，用于存储绘制的图形
        for _ in range(height):  # 循环遍历高度
            line = []  # 创建一个空列表，用于存储每一行的字符
            for _ in range(width):  # 循环遍历宽度
                line.append(fill)  # 将填充字符添加到当前行
            self._buffer.append(line)  # 将当前行添加到图形缓冲区
        self._buffer[0][0] = "P"  # 将缓冲区中第一行第一列的字符设置为 "P"
    def render(self) -> str:
        # 创建一个包含初始行的列表
        lines = ["       1 2 3 4 5 6 7 8 9"]
        # 遍历缓冲区中的每一行，将其格式化后添加到行列表中
        for row, line in enumerate(self._buffer, start=1):
            lines.append(" " + str(row) + " " * 5 + " ".join(line))
        # 将行列表中的内容用换行符连接成一个字符串并返回
        return "\n".join(lines)

    def chomp(self, r, c) -> str:
        # 检查给定的行和列是否在缓冲区的范围内
        if not 1 <= r <= len(self._buffer) or not 1 <= c <= len(self._buffer[0]):
            return "Empty"
        # 如果给定位置为空格，则返回"Empty"
        elif self._buffer[r - 1][c - 1] == " ":
            return "Empty"
        # 如果给定位置是"P"，则返回"Poison"
        elif self._buffer[r - 1][c - 1] == "P":
            return "Poison"
        # 否则，将给定位置及其右下方的所有位置设置为空格，并返回"Chomp"
        else:
            for row in range(r - 1, len(self._buffer)):
                for column in range(c - 1, len(self._buffer[row])):
                    self._buffer[row][column] = " "
            return "Chomp"
def play_game() -> None:  # 定义一个名为play_game的函数，不返回任何值
    """Play one round of the game"""  # 函数的文档字符串，解释函数的作用
    players = 0  # 初始化变量players为0
    while players == 0:  # 当players为0时执行循环
        try:  # 尝试执行以下代码
            players = int(input("How many players "))  # 获取用户输入的玩家数量并转换为整数赋值给players变量
        except ValueError:  # 如果出现值错误
            print("Please enter a number.")  # 打印提示信息
    rows = 0  # 初始化变量rows为0
    while rows == 0:  # 当rows为0时执行循环
        try:  # 尝试执行以下代码
            rows = int(input("How many rows "))  # 获取用户输入的行数并转换为整数赋值给rows变量
            if rows > 9 or rows < 1:  # 如果行数大于9或小于1
                rows = 0  # 将rows重置为0
                print("Too many rows (9 is maximum).")  # 打印提示信息
        except ValueError:  # 如果出现值错误
            print("Please enter a number.")  # 打印提示信息
    # 初始化变量 columns 为 0
    columns = 0
    # 当 columns 为 0 时，循环执行以下代码
    while columns == 0:
        # 尝试获取用户输入的列数并转换为整数
        try:
            columns = int(input("How many columns "))
            # 如果列数大于 9 或小于 1，则将 columns 设为 0，并打印错误信息
            if columns > 9 or columns < 1:
                columns = 0
                print("Too many columns (9 is maximum).")

        # 如果用户输入的不是数字，则打印错误信息
        except ValueError:
            print("Please enter a number.")
    # 创建一个宽度为 columns，高度为 rows 的画布对象
    cookie = Canvas(width=columns, height=rows)
    # 初始化玩家变量为 0
    player = 0
    # 初始化 alive 变量为 True
    alive = True
    # 当 alive 为 True 时，循环执行以下代码
    while alive:
        # 打印空行
        print()
        # 打印画布对象的渲染结果
        print(cookie.render())
        # 打印空行
        print()
        # 玩家变量加一
        player += 1
        # 如果玩家变量大于玩家总数，则将玩家变量设为 1
        if player > players:
            player = 1
        while True:  # 无限循环，直到条件不满足时退出循环
            print("Player", player)  # 打印当前玩家信息
            player_row = -1  # 初始化玩家所选行为-1
            player_column = -1  # 初始化玩家所选列为-1
            while player_row == -1 or player_column == -1:  # 当玩家所选行或列为-1时循环
                try:  # 尝试执行以下代码
                    coordinates = [  # 创建一个包含玩家输入坐标的列表
                        int(item)  # 将输入的坐标转换为整数
                        for item in input("Coordinates of chomp (Row, Column) ").split(",")  # 从用户输入中获取坐标并以逗号分隔
                    ]
                    player_row = coordinates[0]  # 将玩家输入的行赋值给player_row
                    player_column = coordinates[1]  # 将玩家输入的列赋值给player_column

                except (ValueError, IndexError):  # 如果出现值错误或索引错误
                    print("Please enter valid coordinates.")  # 提示用户输入有效的坐标
            result = cookie.chomp(player_row, player_column)  # 调用cookie对象的chomp方法，并将结果赋值给result
            if result == "Empty":  # 如果结果为"Empty"
                print("No fair. You're trying to chomp on empty space!")  # 打印提示信息
            elif result == "Poison":
                # 如果结果是“Poison”，打印出玩家失败的消息，并将alive标记为False
                print("\nYou lose player", player)
                alive = False
                break
            else:
                # 否则，结束游戏循环
                break


def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "CHOMP")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)")
    # 如果玩家选择查看游戏规则
    if input("Do you want the rules (1=Yes, 0=No!) ") != "0":
        # 打印游戏规则
        print("Chomp is for 1 or more players (Humans only).\n")
        print("Here's how a board looks (This one is 5 by 7):")
        example = Canvas(width=7, height=5)
        print(example.render())
        print("\nThe board is a big cookie - R rows high and C columns")
        print("wide. You input R and C at the start. In the upper left")
        print("corner of the cookie is a poison square (P). The one who")
        print("chomps the poison square loses. To take a chomp, type the")
        # 打印游戏规则提示信息
        print("row and column of one of the squares on the cookie.")
        # 打印游戏规则提示信息
        print("All of the squares below and to the right of that square")
        # 打印游戏规则提示信息
        print("(Including that square, too) disappear -- CHOMP!!")
        # 打印游戏规则提示信息
        print("No fair chomping squares that have already been chomped,")
        # 打印游戏规则提示信息
        print("or that are outside the original dimensions of the cookie.\n")
        # 打印游戏规则提示信息
        print("Here we go...")
        # 打印游戏开始提示信息

    keep_playing = True
    # 初始化变量 keep_playing 为 True
    while keep_playing:
        # 当 keep_playing 为 True 时执行循环
        play_game()
        # 调用 play_game() 函数进行游戏
        keep_playing = input("\nAgain (1=Yes, 0=No!) ") == "1"
        # 根据用户输入判断是否继续游戏

if __name__ == "__main__":
    # 如果当前脚本为主程序时执行以下代码
    main()
    # 调用 main() 函数开始游戏
```