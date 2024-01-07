# `basic-computer-games\32_Diamond\python\diamond.py`

```

"""
DIAMOND

Prints pretty diamond patterns to the screen.

Ported by Dave LeCompte
"""


# 打印漂亮的菱形图案到屏幕上
def print_diamond(begin_width, end_width, step, width, count) -> None:
    edge_string = "CC"  # 边缘字符
    fill = "!"  # 填充字符

    n = begin_width  # 菱形宽度起始值
    while True:
        line_buffer = " " * ((width - n) // 2)  # 行缓冲，用于存储每行的字符
        for across in range(count):  # 遍历每行
            for a in range(n):  # 遍历当前行的字符
                if a >= len(edge_string):  # 如果超出边缘字符的长度
                    line_buffer += fill  # 使用填充字符
                else:
                    line_buffer += edge_string[a]  # 使用边缘字符
            line_buffer += " " * (
                (width * (across + 1) + (width - n) // 2) - len(line_buffer)
            )  # 补充空格，使每行长度一致
        print(line_buffer)  # 打印当前行
        if n == end_width:  # 如果宽度达到结束宽度
            return  # 结束循环
        n += step  # 更新宽度值


# 主函数
def main() -> None:
    print(" " * 33, "DIAMOND")  # 打印标题
    print(" " * 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印信息
    print("FOR A PRETTY DIAMOND PATTERN,")  # 提示输入
    print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21")
    width = int(input())  # 获取输入的宽度
    print()

    PAGE_WIDTH = 60  # 页面宽度

    count = int(PAGE_WIDTH / width)  # 计算每行打印的菱形数量

    for _down in range(count):  # 循环打印菱形
        print_diamond(1, width, 2, width, count)  # 打印上半部分菱形
        print_diamond(width - 2, 1, -2, width, count)  # 打印下半部分菱形

    print()
    print()


if __name__ == "__main__":
    main()

```