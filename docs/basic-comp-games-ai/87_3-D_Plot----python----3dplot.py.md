# `basic-computer-games\87_3-D_Plot\python\3dplot.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

# 3D PLOT
#
# Converted from BASIC to Python by Trevor Hobson
# 3D PLOT 标题，由Trevor Hobson将BASIC转换为Python

from math import exp, floor, sqrt
# 导入数学模块中的exp、floor和sqrt函数

def equation(x: float) -> float:
    # 定义一个函数，输入为浮点数x，返回值为浮点数
    return 30 * exp(-x * x / 100)
    # 返回一个数学表达式的计算结果

def main() -> None:
    # 主函数，无返回值
    print(" " * 32 + "3D PLOT")
    # 打印输出标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n\n")
    # 打印输出信息

    for x in range(-300, 315, 15):
        # 循环遍历x的取值范围
        x1 = x / 10
        # 对x进行处理
        max_column = 0
        # 初始化最大列数为0
        y1 = 5 * floor(sqrt(900 - x1 * x1) / 5)
        # 对y进行处理
        y_plot = [" "] * 80
        # 初始化y_plot列表

        for y in range(y1, -(y1 + 5), -5):
            # 循环遍历y的取值范围
            column = floor(25 + equation(sqrt(x1 * x1 + y * y)) - 0.7 * y)
            # 对列数进行处理
            if column > max_column:
                # 如果列数大于最大列数
                max_column = column
                # 更新最大列数
                y_plot[column] = "*"
                # 在y_plot列表中更新对应位置的值为*
        print("".join(y_plot))
        # 打印输出y_plot列表的值


if __name__ == "__main__":
    # 如果当前脚本被直接执行
    main()
    # 调用主函数

```