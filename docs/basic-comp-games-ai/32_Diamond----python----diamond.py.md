# `basic-computer-games\32_Diamond\python\diamond.py`

```
"""
DIAMOND

Prints pretty diamond patterns to the screen.

Ported by Dave LeCompte
"""


# 打印菱形图案
def print_diamond(begin_width, end_width, step, width, count) -> None:
    # 边缘字符
    edge_string = "CC"
    # 填充字符
    fill = "!"

    # 初始化宽度
    n = begin_width
    while True:
        # 初始化行缓冲
        line_buffer = " " * ((width - n) // 2)
        for across in range(count):
            for a in range(n):
                if a >= len(edge_string):
                    line_buffer += fill
                else:
                    line_buffer += edge_string[a]
            line_buffer += " " * (
                (width * (across + 1) + (width - n) // 2) - len(line_buffer)
            )
        # 打印行缓冲
        print(line_buffer)
        # 判断是否达到结束宽度
        if n == end_width:
            return
        # 更新宽度
        n += step


# 主函数
def main() -> None:
    print(" " * 33, "DIAMOND")
    print(" " * 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("FOR A PRETTY DIAMOND PATTERN,")
    print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21")
    # 获取用户输入的宽度
    width = int(input())
    print()

    PAGE_WIDTH = 60

    # 计算每行打印的菱形数量
    count = int(PAGE_WIDTH / width)

    # 打印菱形图案
    for _down in range(count):
        print_diamond(1, width, 2, width, count)
        print_diamond(width - 2, 1, -2, width, count)

    print()
    print()


if __name__ == "__main__":
    main()
```