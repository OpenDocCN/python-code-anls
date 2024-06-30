# `D:\src\scipysrc\sympy\doc\src\guides\physics\generate_plots.py`

```
# 导入所需的库和模块
from sympy import Matrix, laplace_transform, inverse_laplace_transform, exp, cos, sqrt, sin
from sympy.abc import s, t
from sympy.physics.control import *

# 定义主函数 main_q3，用于计算给定的矩阵转换到 Laplace 域中的传递函数矩阵
def main_q3():
    # 定义一个包含符号表达式的 2x2 矩阵 g
    g =  Matrix([[exp(-t)*(1 - t), exp(-2*t)], [5*exp((-2*t))-exp((-t)), (cos((sqrt(3)*t)/2) - 3*sqrt(3)*sin((sqrt(3)*t)/2))*exp(-t/2)]])
    # 将矩阵 g 中每个元素进行 Laplace 变换，构成新的矩阵 G
    G = g.applyfunc(lambda a: laplace_transform(a, t, s)[0])
    # 将转换后的矩阵 G 转换为传递函数矩阵对象
    G = TransferFunctionMatrix.from_Matrix(G, s)
    return G

# 定义函数 q3_3，生成主函数 main_q3 返回的传递函数矩阵 G 的第一个元素的极点零点图
def q3_3():
    # 调用 main_q3 函数获取传递函数矩阵 G
    G = main_q3()
    # 绘制 G 的第一个元素的极点零点图
    pole_zero_plot(G[0, 0])

# 定义函数 q3_4，生成主函数 main_q3 返回的传递函数矩阵 G 的第一个元素的阶跃响应图
def q3_4():
    # 调用 main_q3 函数获取传递函数矩阵 G
    G = main_q3()
    # 获取 G 的第一个元素作为单输入单输出系统的传递函数
    tf1 = G[0, 0]
    # 绘制 tf1 的阶跃响应图
    step_response_plot(tf1)

# 定义函数 q3_5_1，生成主函数 main_q3 返回的传递函数矩阵 G 的第一个元素的波特图的振幅图
def q3_5_1():
    # 调用 main_q3 函数获取传递函数矩阵 G
    G = main_q3()
    # 获取 G 的第二个元素作为单输入单输出系统的传递函数
    tf2 = G[0, 1]
    # 绘制 tf2 的波特图振幅图
    bode_magnitude_plot(tf2)

# 定义函数 q3_5_2，生成主函数 main_q3 返回的传递函数矩阵 G 的第一个元素的波特图的振幅图（与 q3_5_1 重复，可能是重复或错误）
def q3_5_2():
    # 调用 main_q3 函数获取传递函数矩阵 G
    G = main_q3()
    # 获取 G 的第二个元素作为单输入单输出系统的传递函数
    tf2 = G[0, 1]
    # 绘制 tf2 的波特图振幅图
    bode_magnitude_plot(tf2)

# 定义函数 q5，用于构建一系列传递函数并展示其极点零点图
def q5():
    # 定义多个传递函数 G1, G2, G3, G4, H1, H2, H3
    G1 = TransferFunction(1, 10 + s, s)
    G2 = TransferFunction(1, 1 + s, s)
    G3 = TransferFunction(1 + s**2, 4 + 4*s + s**2, s)
    G4 = TransferFunction(1 + s, 6 + s, s)
    H1 = TransferFunction(1 + s, 2 + s, s)
    H2 = TransferFunction(2*(6 + s), 1 + s, s)
    H3 = TransferFunction(1, 1, s)
    # 构建多个串联和反馈系统
    sys1 = Series(G3, G4)
    sys2 = Feedback(sys1, H1, 1).doit()
    sys3 = Series(G2, sys2)
    sys4 = Feedback(sys3, H2).doit()
    sys5 = Series(G1, sys4)
    sys6 = Feedback(sys5, H3)
    # 对最终系统 sys6 进行化简和展开，并绘制其极点零点图
    sys6 = sys6.doit(cancel=True, expand=True)
    pole_zero_plot(sys6)
```