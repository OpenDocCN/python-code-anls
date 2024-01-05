# `d:/src/tocomm/basic-computer-games\87_3-D_Plot\python\3dplot.py`

```
#!/usr/bin/env python3

# 3D PLOT
#
# Converted from BASIC to Python by Trevor Hobson

from math import exp, floor, sqrt


def equation(x: float) -> float:
    return 30 * exp(-x * x / 100)


def main() -> None:
    print(" " * 32 + "3D PLOT")  # 打印标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n\n")  # 打印信息

    for x in range(-300, 315, 15):  # 循环遍历 x 值
        x1 = x / 10  # 计算 x1 值
        max_column = 0  # 初始化最大列数
        y1 = 5 * floor(sqrt(900 - x1 * x1) / 5)  # 计算 y1 的值，根据给定的公式
        y_plot = [" "] * 80  # 创建一个长度为 80 的空列表 y_plot

        for y in range(y1, -(y1 + 5), -5):  # 遍历 y 的取值范围
            column = floor(25 + equation(sqrt(x1 * x1 + y * y)) - 0.7 * y)  # 计算 column 的值，根据给定的公式
            if column > max_column:  # 如果 column 大于 max_column
                max_column = column  # 更新 max_column 的值为 column
                y_plot[column] = "*"  # 在 y_plot 列表中的对应位置添加 "*"
        print("".join(y_plot))  # 打印 y_plot 列表转换为字符串后的结果


if __name__ == "__main__":
    main()  # 调用 main 函数
```