# `d:/src/tocomm/basic-computer-games\32_Diamond\python\diamond.py`

```
# 定义函数print_diamond，接受begin_width, end_width, step, width, count五个参数，返回空值
def print_diamond(begin_width, end_width, step, width, count) -> None:
    # 初始化边缘字符串
    edge_string = "CC"
    # 初始化填充字符
    fill = "!"

    # 初始化n为begin_width
    n = begin_width
    # 进入循环
    while True:
        # 初始化行缓冲，空格数为(width - n)除以2
        line_buffer = " " * ((width - n) // 2)
        # 遍历count次
        for across in range(count):
            # 遍历n次
            for a in range(n):
                # 如果a大于等于边缘字符串的长度
                if a >= len(edge_string):
                    # 将填充字符添加到行缓冲中
                    line_buffer += fill
                else:  # 如果不是边缘字符，将边缘字符添加到行缓冲区
                    line_buffer += edge_string[a]
            line_buffer += " " * (  # 在行缓冲区末尾添加空格，使得行缓冲区长度符合要求
                (width * (across + 1) + (width - n) // 2) - len(line_buffer)
            )
        print(line_buffer)  # 打印行缓冲区内容
        if n == end_width:  # 如果n等于结束宽度，结束函数
            return
        n += step  # 增加n的值

def main() -> None:
    print(" " * 33, "DIAMOND")  # 打印空格和"Diamond"
    print(" " * 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印空格和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("FOR A PRETTY DIAMOND PATTERN,")  # 打印提示信息
    print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21")  # 打印提示信息
    width = int(input())  # 获取用户输入的宽度并转换为整数
    print()

    PAGE_WIDTH = 60  # 设置页面宽度为60
    count = int(PAGE_WIDTH / width)  # 计算页面宽度除以给定宽度的商，并转换为整数，存储在变量count中

    for _down in range(count):  # 循环count次，下划线变量名表示在循环中不会使用该变量
        print_diamond(1, width, 2, width, count)  # 调用print_diamond函数，传入参数1, width, 2, width, count
        print_diamond(width - 2, 1, -2, width, count)  # 调用print_diamond函数，传入参数width - 2, 1, -2, width, count

    print()  # 打印空行
    print()  # 打印空行


if __name__ == "__main__":  # 如果当前脚本被直接执行，则执行以下代码
    main()  # 调用main函数
```