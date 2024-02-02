# `basic-computer-games\26_Chomp\python\chomp.py`

```py
#!/usr/bin/env python3
"""
CHOMP

Converted from BASIC to Python by Trevor Hobson
"""

class Canvas:
    """For drawing the cookie"""

    def __init__(self, width=9, height=9, fill="*") -> None:
        # 初始化画布，创建一个二维数组作为画布，填充为指定字符
        self._buffer = []
        for _ in range(height):
            line = []
            for _ in range(width):
                line.append(fill)
            self._buffer.append(line)
        # 将左上角的字符设置为 "P"
        self._buffer[0][0] = "P"

    def render(self) -> str:
        # 渲染画布，将画布内容转换为字符串形式
        lines = ["       1 2 3 4 5 6 7 8 9"]
        for row, line in enumerate(self._buffer, start=1):
            lines.append(" " + str(row) + " " * 5 + " ".join(line))
        return "\n".join(lines)

    def chomp(self, r, c) -> str:
        # 根据给定的行和列进行 "CHOMP" 操作
        if not 1 <= r <= len(self._buffer) or not 1 <= c <= len(self._buffer[0]):
            return "Empty"
        elif self._buffer[r - 1][c - 1] == " ":
            return "Empty"
        elif self._buffer[r - 1][c - 1] == "P":
            return "Poison"
        else:
            for row in range(r - 1, len(self._buffer)):
                for column in range(c - 1, len(self._buffer[row])):
                    self._buffer[row][column] = " "
            return "Chomp"

def play_game() -> None:
    """Play one round of the game"""
    players = 0
    while players == 0:
        try:
            players = int(input("How many players "))
        except ValueError:
            print("Please enter a number.")
    rows = 0
    while rows == 0:
        try:
            rows = int(input("How many rows "))
            if rows > 9 or rows < 1:
                rows = 0
                print("Too many rows (9 is maximum).")
        except ValueError:
            print("Please enter a number.")
    columns = 0
    while columns == 0:
        try:
            columns = int(input("How many columns "))
            if columns > 9 or columns < 1:
                columns = 0
                print("Too many columns (9 is maximum).")
        except ValueError:
            print("Please enter a number.")
    # 创建一个指定宽度和高度的画布对象
    cookie = Canvas(width=columns, height=rows)
    # 初始化玩家编号为0
    player = 0
    # 初始化玩家存活状态为True
    alive = True
    # 当玩家存活时执行循环
    while alive:
        # 打印空行
        print()
        # 打印画布对象的渲染结果
        print(cookie.render())
        # 打印空行
        print()
        # 玩家编号加1
        player += 1
        # 如果玩家编号超过玩家总数，则重置为1
        if player > players:
            player = 1
        # 无限循环，直到玩家输入有效坐标
        while True:
            # 打印当前玩家编号
            print("Player", player)
            # 初始化玩家行和列为-1
            player_row = -1
            player_column = -1
            # 当玩家行或列为-1时，继续循环
            while player_row == -1 or player_column == -1:
                try:
                    # 获取玩家输入的坐标，并转换为整数列表
                    coordinates = [
                        int(item)
                        for item in input("Coordinates of chomp (Row, Column) ").split(
                            ","
                        )
                    ]
                    # 将玩家输入的行和列分别赋值给对应变量
                    player_row = coordinates[0]
                    player_column = coordinates[1]
                # 捕获值错误和索引错误异常
                except (ValueError, IndexError):
                    # 提示玩家输入有效坐标
                    print("Please enter valid coordinates.")
            # 调用画布对象的chomp方法，传入玩家行和列，获取结果
            result = cookie.chomp(player_row, player_column)
            # 如果结果为"Empty"，提示玩家不能吃空格
            if result == "Empty":
                print("No fair. You're trying to chomp on empty space!")
            # 如果结果为"Poison"，提示玩家输掉游戏，并将存活状态设为False
            elif result == "Poison":
                print("\nYou lose player", player)
                alive = False
                break
            # 如果结果不是"Empty"或"Poison"，跳出内层循环
            else:
                break
# 定义主函数，没有返回值
def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "CHOMP")
    # 打印游戏信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)")
    # 如果用户需要规则，则打印游戏规则
    if input("Do you want the rules (1=Yes, 0=No!) ") != "0":
        print("Chomp is for 1 or more players (Humans only).\n")
        print("Here's how a board looks (This one is 5 by 7):")
        # 创建一个示例游戏板
        example = Canvas(width=7, height=5)
        # 打印示例游戏板
        print(example.render())
        print("\nThe board is a big cookie - R rows high and C columns")
        print("wide. You input R and C at the start. In the upper left")
        print("corner of the cookie is a poison square (P). The one who")
        print("chomps the poison square loses. To take a chomp, type the")
        print("row and column of one of the squares on the cookie.")
        print("All of the squares below and to the right of that square")
        print("(Including that square, too) disappear -- CHOMP!!")
        print("No fair chomping squares that have already been chomped,")
        print("or that are outside the original dimensions of the cookie.\n")
        print("Here we go...")

    # 初始化游戏继续标志
    keep_playing = True
    # 循环进行游戏
    while keep_playing:
        play_game()
        # 根据用户输入判断是否继续游戏
        keep_playing = input("\nAgain (1=Yes, 0=No!) ") == "1"


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```