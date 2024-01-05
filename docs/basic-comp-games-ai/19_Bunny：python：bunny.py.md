# `19_Bunny\python\bunny.py`

```
#!/usr/bin/env python3  # 指定脚本的解释器为 Python 3

import json  # 导入 json 模块

# This data is meant to be read-only, so we are storing it in a tuple
# 打开名为 "data.json" 的文件，并将其内容加载为 JSON 格式，然后存储在一个元组中
with open("data.json") as f:
    DATA = tuple(json.load(f))  # 将 JSON 数据转换为元组并赋值给变量 DATA


def print_intro() -> None:  # 定义一个函数 print_intro，返回类型为 None
    print(" " * 33 + "BUNNY")  # 打印空格和字符串 "BUNNY"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印空格和字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n\n")  # 打印两个换行符


def main() -> None:  # 定义一个函数 main，返回类型为 None
    print_intro()  # 调用函数 print_intro

    # Using an iterator will give us a similar interface to BASIC's READ
    # 使用迭代器将给我们一个类似于 BASIC 的 READ 的接口
    # command. Instead of READ, we will call 'next(data)' to fetch the next element.
    # 将数据转换为迭代器，以便使用next(data)来获取下一个元素
    data = iter(DATA)

    # Read the first 5 numbers. These correspond to letters of the alphabet.
    # B=2, U=21, N=14, N=14, Y=25
    # 读取前5个数字，它们对应于字母表中的字母
    bunny = [next(data) for _ in range(5)]
    L = 64

    # Interpretting a stream of data is a very common software task. We've already intepretted
    # the first 5 numbers as letters of the alphabet (with A being 1). Now, we are going to
    # combine this with a different interpretation of the following data to draw on the screen.
    # The drawing data is essentially a series of horizontal line segments given as begin and end
    # offsets.
    # 解释数据流是一个非常常见的软件任务。我们已经将前5个数字解释为字母表中的字母（A为1）。现在，我们将结合以下数据的不同解释来在屏幕上绘制。
    # 绘图数据基本上是一系列以开始和结束偏移量给出的水平线段。
    while True:
        command = next(data)
        if command < 0:  # 如果命令小于0
            print()  # 打印空行
            continue  # 继续下一次循环

        if command > 128:  # 如果命令大于128
            break  # 跳出循环

        # 如果程序执行到这部分代码，'command'表示线段的起始位置。
        start = command  # 将起始位置赋值给start
        # 将光标移动到起始位置
        print(" " * start, end="")

        # 接下来的数字表示线段的结束位置。
        end = next(data)  # 从数据中获取下一个值作为结束位置
        # 与FOR I=X TO Y不同，'range'的'stop'参数是不包含的，所以我们必须加1
        for i in range(start, end + 1, 1):  # 循环从起始位置到结束位置
            # 在绘制线段时循环遍历“BUNNY”中的字母
            j = i - 5 * int(i / 5)  # 计算'BUNNY'中的索引
            print(chr(L + bunny[j]), end="")  # 打印对应的字符
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
```

这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以使代码更具可重用性和模块化。
```