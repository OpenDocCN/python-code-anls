# `d:/src/tocomm/basic-computer-games\55_Life\python\life.py`

```
"""
LIFE

An implementation of John Conway's popular cellular automaton

Ported by Dave LeCompte
"""

from typing import Dict  # 导入 Dict 类型提示

PAGE_WIDTH = 64  # 设置页面宽度为 64

MAX_WIDTH = 70  # 设置最大宽度为 70
MAX_HEIGHT = 24  # 设置最大高度为 24


def print_centered(msg) -> None:  # 定义一个函数，用于打印居中的消息
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得消息居中
    print(spaces + msg)  # 打印居中的消息
# 定义一个打印标题的函数，接受一个字符串参数，并没有返回值
def print_header(title) -> None:
    # 调用打印居中函数，打印标题
    print_centered(title)
    # 打印固定的字符串
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print()
    print()
    print()


# 定义一个获取用户输入模式的函数，返回一个整数到字符串的字典
def get_pattern() -> Dict[int, str]:
    # 打印提示信息
    print("ENTER YOUR PATTERN:")
    # 初始化计数器
    c = 0

    # 初始化一个空字典
    pattern: Dict[int, str] = {}
    # 无限循环，直到用户输入"DONE"时结束
    while True:
        # 获取用户输入的一行
        line = input()
        # 如果用户输入"DONE"，则返回当前的模式字典
        if line == "DONE":
            return pattern

        # BASIC 输入会去掉前导空格。
        # Python input does not. The following allows you to start a
        # line with a dot to disable the whitespace stripping. This is
        # unnecessary for Python, but for historical accuracy, it's
        # staying in.
        # 检查行首是否为点号，如果是则将点号替换为空格
        if line[0] == ".":
            line = " " + line[1:]
        # 将处理后的行添加到模式字典中
        pattern[c] = line
        # 更新计数器
        c += 1


def main() -> None:
    # 打印游戏标题
    print_header("LIFE")

    # 获取用户输入的模式
    pattern = get_pattern()

    # 计算模式的高度和宽度
    pattern_height = len(pattern)
    pattern_width = 0
    for _line_num, line in pattern.items():
        pattern_width = max(pattern_width, len(line))
    # 设置最小 x 坐标，使输入图案居中
    min_x = 11 - pattern_height // 2
    # 设置最小 y 坐标，使输入图案居中
    min_y = 33 - pattern_width // 2
    # 设置最大 x 坐标
    max_x = MAX_HEIGHT - 1
    # 设置最大 y 坐标
    max_y = MAX_WIDTH - 1

    # 创建一个二维数组，用于表示活动区域的状态
    a = [[0 for y in range(MAX_WIDTH)] for x in range(MAX_HEIGHT)]
    # 初始化变量 p 和 g
    p = 0
    g = 0
    # 初始化变量 invalid
    invalid = False

    # line 140
    # 将输入图案转录到活动数组中
    for x in range(0, pattern_height):
        for y in range(0, len(pattern[x])):
            # 如果图案中的某个位置不为空格，则在活动数组中对应位置设置为 1
            if pattern[x][y] != " ":
                a[min_x + x][min_y + y] = 1
                p += 1

    # 打印空行
    print()
    print()  # 打印空行
    print()  # 打印空行
    while True:  # 进入无限循环
        if invalid:  # 如果条件 invalid 为真
            inv_str = "INVALID!"  # 将 inv_str 设置为 "INVALID!"
        else:  # 否则
            inv_str = ""  # 将 inv_str 设置为空字符串

        print(f"GENERATION: {g}\tPOPULATION: {p} {inv_str}")  # 打印 GENERATION 和 POPULATION 的值，以及 inv_str 的值

        next_min_x = MAX_HEIGHT - 1  # 设置 next_min_x 的值为 MAX_HEIGHT - 1
        next_min_y = MAX_WIDTH - 1  # 设置 next_min_y 的值为 MAX_WIDTH - 1
        next_max_x = 0  # 设置 next_max_x 的值为 0
        next_max_y = 0  # 设置 next_max_y 的值为 0

        p = 0  # 将 p 设置为 0
        g += 1  # 将 g 增加 1
        for _ in range(min_x):  # 循环 min_x 次，每次打印空行
            print()
        for x in range(min_x, max_x + 1):  # 遍历 x 坐标范围内的值
            print()  # 打印空行
            line_list = [" "] * MAX_WIDTH  # 创建一个长度为 MAX_WIDTH 的空格列表
            for y in range(min_y, max_y + 1):  # 遍历 y 坐标范围内的值
                if a[x][y] == 2:  # 如果数组中的值为 2
                    a[x][y] = 0  # 将其修改为 0
                    continue  # 继续下一次循环
                elif a[x][y] == 3:  # 如果数组中的值为 3
                    a[x][y] = 1  # 将其修改为 1
                elif a[x][y] != 1:  # 如果数组中的值不为 1
                    continue  # 继续下一次循环

                line_list[y] = "*"  # 将列表中对应位置的值修改为 "*"

                next_min_x = min(x, next_min_x)  # 更新 next_min_x 的值为 x 和当前 next_min_x 中的最小值
                next_max_x = max(x, next_max_x)  # 更新 next_max_x 的值为 x 和当前 next_max_x 中的最大值
                next_min_y = min(y, next_min_y)  # 更新 next_min_y 的值为 y 和当前 next_min_y 中的最小值
                next_max_y = max(y, next_max_y)  # 更新 next_max_y 的值为 y 和当前 next_max_y 中的最大值

            print("".join(line_list))  # 打印将列表中的值连接成字符串后的结果
        # line 295
        # 循环打印空行，直到达到最大高度
        for _ in range(max_x + 1, MAX_HEIGHT):
            print()

        # 打印空行
        print()

        # 更新最小和最大的 x 和 y 值
        min_x = next_min_x
        max_x = next_max_x
        min_y = next_min_y
        max_y = next_max_y

        # 如果最小 x 值小于 3，则将其设置为 3，并将 invalid 标记为 True
        if min_x < 3:
            min_x = 3
            invalid = True
        # 如果最大 x 值大于 22，则将其设置为 22，并将 invalid 标记为 True
        if max_x > 22:
            max_x = 22
            invalid = True
        # 如果最小 y 值小于 3，则将其设置为 3
        if min_y < 3:
            min_y = 3
            invalid = True  # 设置 invalid 变量为 True
        if max_y > 68:  # 如果 max_y 大于 68
            max_y = 68  # 将 max_y 设置为 68
            invalid = True  # 设置 invalid 变量为 True

        # line 309
        p = 0  # 初始化变量 p 为 0

        for x in range(min_x - 1, max_x + 2):  # 遍历 x 范围从 min_x - 1 到 max_x + 2
            for y in range(min_y - 1, max_y + 2):  # 遍历 y 范围从 min_y - 1 到 max_y + 2
                count = 0  # 初始化变量 count 为 0
                for i in range(x - 1, x + 2):  # 遍历 i 范围从 x - 1 到 x + 2
                    for j in range(y - 1, y + 2):  # 遍历 j 范围从 y - 1 到 y + 2
                        if a[i][j] == 1 or a[i][j] == 2:  # 如果 a[i][j] 等于 1 或者等于 2
                            count += 1  # count 加 1
                if a[x][y] == 0:  # 如果 a[x][y] 等于 0
                    if count == 3:  # 如果 count 等于 3
                        a[x][y] = 3  # 将 a[x][y] 设置为 3
                        p += 1  # p 加 1
                elif (count < 3) or (count > 4):  # 否则如果 count 小于 3 或者大于 4
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```