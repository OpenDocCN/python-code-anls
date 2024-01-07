# `basic-computer-games\19_Bunny\python\bunny.py`

```

#!/usr/bin/env python3

# 导入json模块
import json

# 这些数据是只读的，所以我们将其存储在一个元组中
with open("data.json") as f:
    DATA = tuple(json.load(f))

# 打印介绍信息
def print_intro() -> None:
    print(" " * 33 + "BUNNY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")

# 主函数
def main() -> None:
    print_intro()

    # 使用迭代器会给我们一个类似BASIC的READ命令的接口。我们将调用'next(data)'来获取下一个元素。
    data = iter(DATA)

    # 读取前5个数字。这些数字对应于字母表中的字母。
    # B=2, U=21, N=14, N=14, Y=25
    # 通常，列表推导对于转换序列中的每个元素都很有用。
    # 在这种情况下，我们使用range来重复调用next(data) 5次。下划线(_)表示range的值被丢弃。
    bunny = [next(data) for _ in range(5)]
    L = 64

    # 解释数据流是一个非常常见的软件任务。我们已经将前5个数字解释为字母表中的字母（A为1）。现在，我们将结合这个解释和后续数据的不同解释来在屏幕上绘制。
    # 绘图数据本质上是一系列以开始和结束偏移量给出的水平线段。
    while True:
        command = next(data)

        if command < 0:
            print()
            continue

        if command > 128:
            break

        # 如果我们到达了代码的这一部分，'command'表示线段的'开始'位置。
        start = command
        # 在开始位置放置光标
        print(" " * start, end="")

        # 接下来的数字表示线段的结束。
        end = next(data)
        # 与FOR I=X TO Y不同，'range'的'停止'参数是不包含的，所以我们必须加1
        for i in range(start, end + 1, 1):
            # 在绘制线段时循环遍历“BUNNY”中的字母
            j = i - 5 * int(i / 5)
            print(chr(L + bunny[j]), end="")


if __name__ == "__main__":
    main()

```