# `basic-computer-games\19_Bunny\python\bunny.py`

```py
#!/usr/bin/env python3

# 导入 json 模块
import json

# 这些数据是只读的，所以我们将其存储在一个元组中
with open("data.json") as f:
    # 从文件中加载 JSON 数据并转换为元组
    DATA = tuple(json.load(f))

# 打印游戏介绍
def print_intro() -> None:
    print(" " * 33 + "BUNNY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")

# 主函数
def main() -> None:
    print_intro()

    # 使用迭代器会给我们一个类似于 BASIC 的 READ 命令的接口。我们将调用 'next(data)' 来获取下一个元素。
    data = iter(DATA)

    # 读取前5个数字。这些数字对应于字母表中的字母。
    # B=2, U=21, N=14, N=14, Y=25

    # 通常，列表推导式适用于转换序列中的每个元素。
    # 在这种情况下，我们使用 range 来重复调用 next(data) 5 次。下划线(_)表示忽略 range 中的值。
    bunny = [next(data) for _ in range(5)]
    L = 64

    # 解释数据流是一个非常常见的软件任务。我们已经将前5个数字解释为字母表中的字母（A 为 1）。
    # 现在，我们将结合这个解释和后续数据的不同解释来在屏幕上绘制。
    # 绘图数据本质上是一系列以起始和结束偏移量给出的水平线段。
    # 无限循环，不断获取下一个命令
    while True:
        # 获取下一个命令
        command = next(data)

        # 如果命令小于0，打印空行并继续下一次循环
        if command < 0:
            print()
            continue

        # 如果命令大于128，跳出循环
        if command > 128:
            break

        # 如果程序执行到这里，'command' 表示线段的起始位置
        start = command
        # 将光标移动到起始位置
        print(" " * start, end="")

        # 下一个数字表示线段的结束位置
        end = next(data)
        # 由于 'range' 的 'stop' 参数是不包含在内的，所以必须加1
        for i in range(start, end + 1, 1):
            # 在绘制线段时循环遍历"BUNNY"中的字母
            j = i - 5 * int(i / 5)
            print(chr(L + bunny[j]), end="")
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```