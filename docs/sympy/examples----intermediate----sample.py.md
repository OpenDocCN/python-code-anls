# `D:\src\scipysrc\sympy\examples\intermediate\sample.py`

```
"""
Utility functions for plotting sympy functions.

See examples\mplot2d.py and examples\mplot3d.py for usable 2d and 3d
graphing functions using matplotlib.
"""

# 导入必要的库
from sympy.core.sympify import sympify, SympifyError  # 导入 sympify 函数和异常类 SympifyError
from sympy.external import import_module  # 导入 import_module 函数
np = import_module('numpy')  # 导入 NumPy 库并赋值给 np

def sample2d(f, x_args):
    """
    Samples a 2d function f over specified intervals and returns two
    arrays (X, Y) suitable for plotting with matlab (matplotlib)
    syntax. See examples\mplot2d.py.

    f is a function of one variable, such as x**2.
    x_args is an interval given in the form (var, min, max, n)
    """
    try:
        f = sympify(f)  # 将输入的函数 f 转换为 SymPy 表达式
    except SympifyError:
        raise ValueError("f could not be interpreted as a SymPy function")  # 如果转换失败，抛出异常

    try:
        x, x_min, x_max, x_n = x_args  # 解包 x_args 元组，获取变量名、最小值、最大值、间隔数
    except (TypeError, IndexError):
        raise ValueError("x_args must be a tuple of the form (var, min, max, n)")  # 如果解包失败，抛出异常

    # 计算 X 轴的数值范围和步长
    x_l = float(x_max - x_min)
    x_d = x_l / float(x_n)
    X = np.arange(float(x_min), float(x_max) + x_d, x_d)  # 生成 X 轴的数据点

    Y = np.empty(len(X))  # 创建一个空的数组 Y，用于存储计算后的函数值
    for i in range(len(X)):
        try:
            Y[i] = float(f.subs(x, X[i]))  # 计算函数 f 在 X[i] 处的值并存储到 Y[i] 中
        except TypeError:
            Y[i] = None  # 如果计算失败，将 Y[i] 置为 None
    return X, Y  # 返回 X 和 Y 数组


def sample3d(f, x_args, y_args):
    """
    Samples a 3d function f over specified intervals and returns three
    2d arrays (X, Y, Z) suitable for plotting with matlab (matplotlib)
    syntax. See examples\mplot3d.py.

    f is a function of two variables, such as x**2 + y**2.
    x_args and y_args are intervals given in the form (var, min, max, n)
    """
    x, x_min, x_max, x_n = None, None, None, None
    y, y_min, y_max, y_n = None, None, None, None
    try:
        f = sympify(f)  # 将输入的函数 f 转换为 SymPy 表达式
    except SympifyError:
        raise ValueError("f could not be interpreted as a SymPy function")  # 如果转换失败，抛出异常

    try:
        x, x_min, x_max, x_n = x_args  # 解包 x_args 元组，获取 x 的相关参数
        y, y_min, y_max, y_n = y_args  # 解包 y_args 元组，获取 y 的相关参数
    except (TypeError, IndexError):
        raise ValueError("x_args and y_args must be tuples of the form (var, min, max, intervals)")  # 如果解包失败，抛出异常

    # 计算 X 轴和 Y 轴的数值范围和步长
    x_l = float(x_max - x_min)
    x_d = x_l / float(x_n)
    x_a = np.arange(float(x_min), float(x_max) + x_d, x_d)

    y_l = float(y_max - y_min)
    y_d = y_l / float(y_n)
    y_a = np.arange(float(y_min), float(y_max) + y_d, y_d)

    def meshgrid(x, y):
        """
        Taken from matplotlib.mlab.meshgrid.
        """
        x = np.array(x)
        y = np.array(y)
        numRows, numCols = len(y), len(x)
        x.shape = 1, numCols
        X = np.repeat(x, numRows, 0)

        y.shape = numRows, 1
        Y = np.repeat(y, numCols, 1)
        return X, Y  # 返回 meshgrid 后的 X 和 Y 数组

    X, Y = np.meshgrid(x_a, y_a)  # 生成 X, Y 的网格数据

    Z = np.ndarray((len(X), len(X[0])))  # 创建一个二维数组 Z，用于存储计算后的函数值
    for j in range(len(X)):
        for k in range(len(X[0])):
            try:
                Z[j][k] = float(f.subs(x, X[j][k]).subs(y, Y[j][k]))  # 计算函数 f 在 (X[j][k], Y[j][k]) 处的值并存储到 Z[j][k] 中
            except (TypeError, NotImplementedError):
                Z[j][k] = 0  # 如果计算失败，将 Z[j][k] 置为 0
    return X, Y, Z  # 返回 X, Y, Z 数组


def sample(f, *var_args):
    """
    Placeholder for a function that samples an arbitrary function f.
    """
    Samples a 2d or 3d function over specified intervals and returns
    a dataset suitable for plotting with matlab (matplotlib) syntax.
    Wrapper for sample2d and sample3d.

    f is a function of one or two variables, such as x**2.
    var_args are intervals for each variable given in the form (var, min, max, n)
    """
    # 如果变量参数列表 var_args 中只有一个元素
    if len(var_args) == 1:
        # 调用 sample2d 函数，将 f 和 var_args[0] 作为参数，返回结果
        return sample2d(f, var_args[0])
    # 如果变量参数列表 var_args 中有两个元素
    elif len(var_args) == 2:
        # 调用 sample3d 函数，将 f、var_args[0] 和 var_args[1] 作为参数，返回结果
        return sample3d(f, var_args[0], var_args[1])
    else:
        # 如果 var_args 中元素个数不是 1 或 2，则抛出 ValueError 异常
        raise ValueError("Only 2d and 3d sampling are supported at this time.")
```