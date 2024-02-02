# `basic-computer-games\87_3-D_Plot\python\3dplot.py`

```py
#!/usr/bin/env python3
# 指定脚本解释器为 Python 3

# 3D PLOT
#
# Converted from BASIC to Python by Trevor Hobson
# 3D PLOT
# 由 Trevor Hobson 将 BASIC 转换为 Python

from math import exp, floor, sqrt
# 导入数学模块中的 exp, floor, sqrt 函数

def equation(x: float) -> float:
    # 定义一个函数，输入为浮点数 x，返回值为浮点数
    return 30 * exp(-x * x / 100)
    # 返回 30 乘以 e 的 x 平方除以 100 的负指数

def main() -> None:
    # 定义主函数，无返回值
    print(" " * 32 + "3D PLOT")
    # 打印空格乘以 32 加上 "3D PLOT"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n\n")
    # 打印空格乘以 15 加上 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" 和多个换行符

    for x in range(-300, 315, 15):
        # 遍历范围为 -300 到 315，步长为 15
        x1 = x / 10
        # 将 x 除以 10 赋值给 x1
        max_column = 0
        # 初始化最大列为 0
        y1 = 5 * floor(sqrt(900 - x1 * x1) / 5)
        # 计算 y1 的值
        y_plot = [" "] * 80
        # 创建长度为 80 的空格列表

        for y in range(y1, -(y1 + 5), -5):
            # 遍历范围为 y1 到 -(y1 + 5)，步长为 -5
            column = floor(25 + equation(sqrt(x1 * x1 + y * y)) - 0.7 * y)
            # 计算列的值
            if column > max_column:
                # 如果列的值大于最大列
                max_column = column
                # 更新最大列的值
                y_plot[column] = "*"
                # 将 "*" 赋值给 y_plot 列表的对应位置
        print("".join(y_plot))
        # 打印 y_plot 列表转换为字符串后的值


if __name__ == "__main__":
    main()
    # 如果脚本作为主程序执行，则调用 main 函数
```