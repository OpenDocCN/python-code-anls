# `.\pandas-ta\pandas_ta\utils\_math.py`

```
# 设置文件编码为 UTF-8
# 导入 functools 模块中的 reduce 函数
# 从 math 模块中导入 floor 函数并将其命名为 mfloor
# 从 operator 模块中导入 mul 函数
# 从 sys 模块中导入 float_info 对象并将其命名为 sflt
# 从 typing 模块中导入 List、Optional 和 Tuple 类型
from functools import reduce
from math import floor as mfloor
from operator import mul
from sys import float_info as sflt
from typing import List, Optional, Tuple

# 从 numpy 模块中导入 ones、triu、all、append、array、corrcoef、dot、fabs、exp、log、nan、ndarray、seterr、sqrt 和 sum 函数
from numpy import ones, triu
from numpy import all as npAll
from numpy import append as npAppend
from numpy import array as npArray
from numpy import corrcoef as npCorrcoef
from numpy import dot as npDot
from numpy import fabs as npFabs
from numpy import exp as npExp
from numpy import log as npLog
from numpy import nan as npNaN
from numpy import ndarray as npNdArray
from numpy import seterr
from numpy import sqrt as npSqrt
from numpy import sum as npSum

# 从 pandas 模块中导入 DataFrame 和 Series 类
from pandas import DataFrame, Series

# 从 pandas_ta 包中导入 Imports 模块
from pandas_ta import Imports
# 从 ._core 模块中导入 verify_series 函数
from ._core import verify_series

# 定义 combination 函数，接收任意关键字参数并返回整型值
def combination(**kwargs: dict) -> int:
    """https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python"""
    # 从 kwargs 中取出键为 "n" 的值，若不存在则默认为 1，并转换为整型
    n = int(npFabs(kwargs.pop("n", 1)))
    # 从 kwargs 中取出键为 "r" 的值，若不存在则默认为 0，并转换为整型
    r = int(npFabs(kwargs.pop("r", 0)))

    # 如果参数中存在 "repetition" 或 "multichoose" 键，则执行以下操作
    if kwargs.pop("repetition", False) or kwargs.pop("multichoose", False):
        # 计算修正后的 n 值
        n = n + r - 1

    # 若 r 小于 0，则返回 None
    # 如果 r 大于 n，则令 r 等于 n
    r = min(n, n - r)
    # 若 r 为 0，则返回 1
    if r == 0:
        return 1

    # 计算组合数的分子部分
    numerator = reduce(mul, range(n, n - r, -1), 1)
    # 计算组合数的分母部分
    denominator = reduce(mul, range(1, r + 1), 1)
    # 返回组合数结果
    return numerator // denominator


# 定义错误函数 erf(x)，接收一个参数 x，返回 erf(x) 的值
def erf(x):
    """Error Function erf(x)
    The algorithm comes from Handbook of Mathematical Functions, formula 7.1.26.
    Source: https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
    """
    # 保存 x 的符号
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # 定义常数
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # 使用 A&S 公式 7.1.26 计算 erf(x) 的值
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * npExp(-x * x)
    # 返回 erf(x) 的值，若 x 为负数，则取相反数
    return sign * y # erf(-x) = -erf(x)


# 定义斐波那契数列函数 fibonacci，接收一个整型参数 n 和任意关键字参数，返回一个 numpy 数组
def fibonacci(n: int = 2, **kwargs: dict) -> npNdArray:
    """Fibonacci Sequence as a numpy array"""
    # 将 n 转换为非负整数
    n = int(npFabs(n)) if n >= 0 else 2

    # 从 kwargs 中取出键为 "zero" 的值，若存在且为 True，则斐波那契数列以 0 开头
    zero = kwargs.pop("zero", False)
    if zero:
        a, b = 0, 1
    else:
        # 若不以 0 开头，则斐波那契数列从 1 开始，n 减 1
        n -= 1
        a, b = 1, 1

    # 初始化结果数组，包含斐波那契数列的第一个元素
    result = npArray([a])
    # 循环生成斐波那契数列
    for _ in range(0, n):
        a, b = b, a + b
        result = npAppend(result, a)

    # 从 kwargs 中取出键为 "weighted" 的值，若存在且为 True，则返回加权后的斐波那契数列
    weighted = kwargs.pop("weighted", False)
    if weighted:
        # 计算斐波那契数列的总和
        fib_sum = npSum(result)
        # 若总和大于 0，则返回斐波那契数列的每个元素除以总和的结果
        if fib_sum > 0:
            return result / fib_sum
        else:
            # 若总和小于等于 0，则直接返回斐波那契数列
            return result
    else:
        # 若不加权，则直接返回斐波那契数
    """Classic Linear Regression in Numpy or Scikit-Learn"""
    # 确保 x 和 y 是 Series 类型的数据
    x, y = verify_series(x), verify_series(y)
    # 获取 x 和 y 的大小
    m, n = x.size, y.size

    # 如果 x 和 y 的大小不相等，则打印错误信息并返回空字典
    if m != n:
        print(f"[X] Linear Regression X and y have unequal total observations: {m} != {n}")
        return {}

    # 如果导入了 sklearn 模块，则使用 sklearn 进行线性回归
    if Imports["sklearn"]:
        return _linear_regression_sklearn(x, y)
    # 否则使用 numpy 进行线性回归
    else:
        return _linear_regression_np(x, y)
# 返回给定序列的对数几何平均值
def log_geometric_mean(series: Series) -> float:
    n = series.size  # 获取序列的大小
    if n < 2: return 0  # 如果序列大小小于2，则返回0
    else:
        series = series.fillna(0) + 1  # 将序列中的空值填充为0，并加1
        if npAll(series > 0):  # 检查序列中的所有值是否大于0
            return npExp(npLog(series).sum() / n) - 1  # 计算序列的对数和的均值的指数，然后减去1
        return 0  # 如果序列中存在小于等于0的值，则返回0


# 返回帕斯卡三角形的第n行
def pascals_triangle(n: int = None, **kwargs: dict) -> npNdArray:
    n = int(npFabs(n)) if n is not None else 0  # 将n转换为整数并取绝对值，如果n为None则设为0

    # 计算
    triangle = npArray([combination(n=n, r=i) for i in range(0, n + 1)])  # 创建帕斯卡三角形的第n行
    triangle_sum = npSum(triangle)  # 计算三角形的总和
    triangle_weights = triangle / triangle_sum  # 计算每个元素的权重
    inverse_weights = 1 - triangle_weights  # 计算逆权重

    weighted = kwargs.pop("weighted", False)  # 获取weighted参数，默认为False
    inverse = kwargs.pop("inverse", False)  # 获取inverse参数，默认为False
    if weighted and inverse:  # 如果weighted和inverse都为True
        return inverse_weights  # 返回逆权重
    if weighted:  # 如果weighted为True
        return triangle_weights  # 返回权重
    if inverse:  # 如果inverse为True
        return None  # 返回None

    return triangle  # 返回帕斯卡三角形的第n行


# 返回对称三角形的第n行
def symmetric_triangle(n: int = None, **kwargs: dict) -> Optional[List[int]]:
    n = int(npFabs(n)) if n is not None else 2  # 将n转换为整数并取绝对值，如果n为None则设为2

    triangle = None
    if n == 2:  # 如果n为2
        triangle = [1, 1]  # 返回固定的列表

    if n > 2:  # 如果n大于2
        if n % 2 == 0:  # 如果n为偶数
            front = [i + 1 for i in range(0, mfloor(n / 2))]  # 创建前半部分列表
            triangle = front + front[::-1]  # 创建对称三角形
        else:
            front = [i + 1 for i in range(0, mfloor(0.5 * (n + 1)))]  # 创建前半部分列表
            triangle = front.copy()  # 复制前半部分列表
            front.pop()  # 移除最后一个元素
            triangle += front[::-1]  # 创建对称三角形

    if kwargs.pop("weighted", False) and isinstance(triangle, list):  # 如果weighted为True且triangle是列表类型
        triangle_sum = npSum(triangle)  # 计算三角形的总和
        triangle_weights = triangle / triangle_sum  # 计算每个元素的权重
        return triangle_weights  # 返回权重

    return triangle  # 返回对称三角形的第n行


# 返回权重与值x的点积
def weights(w: npNdArray):
    def _dot(x):
        return npDot(w, x)
    return _dot


# 如果值接近于零，则返回零，否则返回自身
def zero(x: Tuple[int, float]) -> Tuple[int, float]:
    return 0 if abs(x) < sflt.epsilon else x


# DataFrame相关性分析辅助函数
def df_error_analysis(dfA: DataFrame, dfB: DataFrame, **kwargs: dict) -> DataFrame:
    corr_method = kwargs.pop("corr_method", "pearson")  # 获取相关性计算方法，默认为pearson

    # 计算它们的差异和相关性
    diff = dfA - dfB  # 计算DataFrame的差异
    corr = dfA.corr(dfB, method=corr_method)  # 计算DataFrame的相关性

    # 用于绘图
    if kwargs.pop("plot", False):  # 如果plot为True
        diff.hist()  # 绘制差异的直方图
        if diff[diff > 0].any():  # 如果差异中存在大于0的值
            diff.plot(kind="kde")  # 绘制密度曲线图

    if kwargs.pop("triangular", False):  # 如果triangular为True
        return corr.where(triu(ones(corr.shape)).astype(bool))  # 返回上三角部分的相关性矩阵

    return corr  # 返回相关性矩阵


# 私有函数
# 使用 Numpy 实现简单的线性回归，适用于没有安装 sklearn 包的环境，接受两个一维数组作为输入
def _linear_regression_np(x: Series, y: Series) -> dict:
    # 初始化结果字典，所有值设为 NaN
    result = {"a": npNaN, "b": npNaN, "r": npNaN, "t": npNaN, "line": npNaN}
    # 计算 x 和 y 的总和
    x_sum = x.sum()
    y_sum = y.sum()

    # 如果 x 的总和不为 0
    if int(x_sum) != 0:
        # 计算 x 和 y 之间的相关系数
        r = npCorrcoef(x, y)[0, 1]

        m = x.size
        # 计算回归系数 b
        r_mix = m * (x * y).sum() - x_sum * y_sum
        b = r_mix // (m * (x * x).sum() - x_sum * x_sum)
        # 计算截距 a 和回归线
        a = y.mean() - b * x.mean()
        line = a + b * x

        # 临时保存 Numpy 的错误设置
        _np_err = seterr()
        # 忽略除零和无效值的错误
        seterr(divide="ignore", invalid="ignore")
        # 更新结果字典
        result = {
            "a": a, "b": b, "r": r,
            "t": r / npSqrt((1 - r * r) / (m - 2)),
            "line": line,
        }
        # 恢复 Numpy 的错误设置
        seterr(divide=_np_err["divide"], invalid=_np_err["invalid"])

    return result

# 使用 Scikit Learn 实现简单的线性回归，适用于安装了 sklearn 包的环境，接受两个一维数组作为输入
def _linear_regression_sklearn(x: Series, y: Series) -> dict:
    # 导入 LinearRegression 类
    from sklearn.linear_model import LinearRegression

    # 将 x 转换为 DataFrame，创建 LinearRegression 模型并拟合数据
    X = DataFrame(x)
    lr = LinearRegression().fit(X, y=y)
    # 计算决定系数
    r = lr.score(X, y=y)
    # 获取截距和斜率
    a, b = lr.intercept_, lr.coef_[0]

    # 更新结果字典
    result = {
        "a": a, "b": b, "r": r,
        "t": r / npSqrt((1 - r * r) / (x.size - 2)),
        "line": a + b * x
    }
    return result
```