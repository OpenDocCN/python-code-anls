# `basic-computer-games\06_Banner\python\banner.py`

```py
#!/usr/bin/env python3
# 指定脚本解释器为 Python3

"""
BANNER

Converted from BASIC to Python by Trevor Hobson
"""
# 多行注释，用于说明脚本的来源和作者

letters = {
    " ": [0, 0, 0, 0, 0, 0, 0],  # 空格的ASCII艺术字对应的像素值
    "A": [505, 37, 35, 34, 35, 37, 505],  # 字母A的ASCII艺术字对应的像素值
    # ... 其他字母和符号的ASCII艺术字对应的像素值
}

def print_banner() -> None:
    f = [0] * 7  # 创建一个长度为7的列表，元素都为0
    j = [0] * 9  # 创建一个长度为9的列表，元素都为0
    # 无限循环，直到用户输入一个整数
    while True:
        try:
            # 获取用户输入的水平值并转换为整数
            horizontal = int(input("Horizontal "))
            # 如果水平值小于1，则抛出数值错误异常
            if horizontal < 1:
                raise ValueError("Horizontal must be greater than zero")
            # 跳出循环
            break

        # 捕获数值错误异常
        except ValueError:
            # 提示用户输入大于零的数字
            print("Please enter a number greater than zero")
    
    # 无限循环，直到用户输入一个整数
    while True:
        try:
            # 获取用户输入的垂直值并转换为整数
            vertical = int(input("Vertical "))
            # 如果垂直值小于1，则抛出数值错误异常
            if vertical < 1:
                raise ValueError("Vertical must be greater than zero")
            # 跳出循环
            break

        # 捕获数值错误异常
        except ValueError:
            # 提示用户输入大于零的数字
            print("Please enter a number greater than zero")
    
    # 初始化变量 g1
    g1 = 0
    # 如果用户输入的内容以字母 'y' 开头，则将 g1 设置为 1
    if input("Centered ").lower().startswith("y"):
        g1 = 1
    # 获取用户输入的字符并转换为大写
    character = input(
        "Character (type 'ALL' if you want character being printed) "
    ).upper()
    # 获取用户输入的语句
    statement = input("Statement ")

    # 设置页面，这意味着准备打印机，只需按回车键
    input("Set page ")
    # 遍历输入的语句中的每个字符
    for statement_char in statement:
        # 复制字母表中对应字符的数据
        s = letters[statement_char].copy()
        # 设置 x_str 为字符，如果 character 为 "ALL" 则设置为当前语句字符
        x_str = character
        # 如果 x_str 为 " "，则打印换行符
        if character == "ALL":
            x_str = statement_char
        # 如果 x_str 不为 " "，则进行字符打印
        if x_str == " ":
            print("\n" * (7 * horizontal))
        else:
            # 遍历每个字符的每一行
            for u in range(0, 7):
                # 遍历每个字符的每一列
                for k in range(8, -1, -1):
                    # 根据字符数据计算打印字符的位置
                    if 2**k >= s[u]:
                        j[8 - k] = 0
                    else:
                        j[8 - k] = 1
                        s[u] = s[u] - 2**k
                        # 如果字符数据为 1，则记录当前列的位置
                        if s[u] == 1:
                            f[u] = 8 - k
                            break
                # 打印字符
                for _t1 in range(1, horizontal + 1):
                    line_str = " " * int((63 - 4.5 * vertical) * g1 / len(x_str) + 1)
                    for b in range(0, f[u] + 1):
                        if j[b] == 0:
                            for _ in range(1, vertical + 1):
                                line_str = line_str + " " * len(x_str)
                        else:
                            line_str = line_str + x_str * vertical
                    print(line_str)
            # 打印字符之间的间隔
            print("\n" * (2 * horizontal - 1))
    # 打印更多的换行符
    # print("\n" * 75)  # Feed some more paper from the printer
# 如果当前模块被直接执行，而非被导入到其他模块中
if __name__ == "__main__":
    # 调用打印横幅的函数
    print_banner()
```