# `basic-computer-games\26_Chomp\python\chomp.py`

```

#!/usr/bin/env python3
# 指定解释器为 Python3

"""
CHOMP

Converted from BASIC to Python by Trevor Hobson
"""
# 程序的简要介绍

class Canvas:
    """For drawing the cookie"""
    # 用于绘制饼干的类

    def __init__(self, width=9, height=9, fill="*") -> None:
        # 初始化方法，设置默认宽度、高度和填充字符
        self._buffer = []
        # 创建一个空列表
        for _ in range(height):
            # 循环遍历高度
            line = []
            for _ in range(width):
                # 循环遍历宽度
                line.append(fill)
                # 将填充字符添加到行中
            self._buffer.append(line)
            # 将行添加到缓冲区中
        self._buffer[0][0] = "P"
        # 将缓冲区的第一个元素设置为"P"

    def render(self) -> str:
        # 渲染方法，返回字符串
        lines = ["       1 2 3 4 5 6 7 8 9"]
        # 创建包含列号的列表
        for row, line in enumerate(self._buffer, start=1):
            # 遍历缓冲区中的行
            lines.append(" " + str(row) + " " * 5 + " ".join(line))
            # 将行号和行内容添加到列表中
        return "\n".join(lines)
        # 返回包含所有行的字符串

    def chomp(self, r, c) -> str:
        # chomp 方法，接受行和列参数，返回字符串
        if not 1 <= r <= len(self._buffer) or not 1 <= c <= len(self._buffer[0]):
            # 如果行或列超出范围
            return "Empty"
            # 返回"Empty"
        elif self._buffer[r - 1][c - 1] == " ":
            # 如果指定位置为空
            return "Empty"
            # 返回"Empty"
        elif self._buffer[r - 1][c - 1] == "P":
            # 如果指定位置为"P"
            return "Poison"
            # 返回"Poison"
        else:
            # 否则
            for row in range(r - 1, len(self._buffer)):
                # 遍历行
                for column in range(c - 1, len(self._buffer[row])):
                    # 遍历列
                    self._buffer[row][column] = " "
                    # 将指定位置及其右下方的位置设置为空
            return "Chomp"
            # 返回"Chomp"

def play_game() -> None:
    """Play one round of the game"""
    # 玩游戏的方法

def main() -> None:
    # 主方法
    print(" " * 33 + "CHOMP")
    # 打印游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    # 打印创意计算公司信息
    print("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)")
    # 打印游戏介绍
    if input("Do you want the rules (1=Yes, 0=No!) ") != "0":
        # 如果用户需要游戏规则
        print("Chomp is for 1 or more players (Humans only).\n")
        # 打印游戏玩家数量限制
        print("Here's how a board looks (This one is 5 by 7):")
        # 打印游戏板示例
        example = Canvas(width=7, height=5)
        # 创建一个示例游戏板
        print(example.render())
        # 打印示例游戏板
        print("\nThe board is a big cookie - R rows high and C columns")
        # 打印游戏板说明
        print("wide. You input R and C at the start. In the upper left")
        # 打印游戏板说明
        print("corner of the cookie is a poison square (P). The one who")
        # 打印游戏板说明
        print("chomps the poison square loses. To take a chomp, type the")
        # 打印游戏板说明
        print("row and column of one of the squares on the cookie.")
        # 打印游戏板说明
        print("All of the squares below and to the right of that square")
        # 打印游戏板说明
        print("(Including that square, too) disappear -- CHOMP!!")
        # 打印游戏板说明
        print("No fair chomping squares that have already been chomped,")
        # 打印游戏板说明
        print("or that are outside the original dimensions of the cookie.\n")
        # 打印游戏板说明
        print("Here we go...")

    keep_playing = True
    # 设置继续游戏标志为True
    while keep_playing:
        # 当继续游戏标志为True时
        play_game()
        # 进行游戏
        keep_playing = input("\nAgain (1=Yes, 0=No!) ") == "1"
        # 根据用户输入更新继续游戏标志

if __name__ == "__main__":
    main()
    # 如果是主程序，则执行主方法

```