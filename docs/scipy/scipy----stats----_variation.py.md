# `D:\src\scipysrc\scipy\scipy\stats\_variation.py`

```
# 导入NumPy库，简称为np
import numpy as np

# 导入Scipy库中的_nan函数，用于处理NaN值
from scipy._lib._util import _get_nan

# 导入Scipy库中的array_namespace和xp_copysign函数
from scipy._lib._array_api import array_namespace, xp_copysign

# 导入本地模块中的_axis_nan_policy_factory函数，用于生成带有NaN策略的函数装饰器
from ._axis_nan_policy import _axis_nan_policy_factory

# 使用_axis_nan_policy_factory装饰variation函数，配置参数如下：
# lambda x: x 是输入函数，返回原始值
# n_outputs=1 表示函数返回一个值
# result_to_tuple=lambda x: (x,) 将结果转换为元组
@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
# 定义variation函数，计算变异系数
def variation(a, axis=0, nan_policy='propagate', ddof=0, *, keepdims=False):
    """
    计算变异系数（Coefficient of Variation）。

    变异系数是标准差除以均值的结果。该函数等价于：

        np.std(x, axis=axis, ddof=ddof) / np.mean(x)

    ddof的默认值为0，但是很多定义中使用无偏样本方差的平方根来计算样本标准差，对应于ddof=1。

    函数不会对数据的均值取绝对值，因此如果均值为负数，返回值也可能为负数。

    Parameters
    ----------
    a : array_like
        输入数组。
    axis : int or None, optional
        计算变异系数的轴向。默认为0。如果为None，则在整个数组`a`上计算。
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        定义如何处理输入中包含`nan`的情况。可选的选项包括：

          * 'propagate': 返回`nan`
          * 'raise': 抛出异常
          * 'omit': 在计算中忽略`nan`值

        默认为'propagate'。
    ddof : int, optional
        在计算标准差时使用的“Delta Degrees Of Freedom”。标准差计算中的除数为“N - ddof”，其中“N”是元素数目。
        `ddof`必须小于“N”；如果不是，则结果将为`nan`或`inf`，取决于`N`和数组中的值。
        为了保持向后兼容性，默认情况下`ddof`为零，但建议使用`ddof=1`以确保计算样本标准差时使用无偏样本方差的平方根。

    Returns
    -------
    variation : ndarray
        沿指定轴向计算的变异系数。

    Notes
    -----
    有几种特殊情况处理而不会生成警告：

    * 如果均值和标准差都为零，则返回`nan`。
    * 如果均值为零且标准差不为零，则返回`inf`。
    * 如果输入长度为零（数组长度为零，或所有输入值为`nan`且`nan_policy`为`'omit'`），则返回`nan`。
    * 如果输入包含`inf`，则返回`nan`。

    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.

    Examples
    --------
    ```
    # 省略示例部分
    ```
    """
    # 实现函数的具体计算逻辑将在装饰器提供的功能基础上进行
    pass
    # 导入 NumPy 库，并使用 scipy.stats 中的 variation 函数计算给定数据的变异系数
    >>> import numpy as np
    >>> from scipy.stats import variation
    >>> variation([1, 2, 3, 4, 5], ddof=1)
    0.5270462766947299
    
    # 计算包含一些 `nan` 值的数组在指定维度上的变异系数：
    >>> x = np.array([[  10.0, np.nan, 11.0, 19.0, 23.0, 29.0, 98.0],
    ...               [  29.0,   30.0, 32.0, 33.0, 35.0, 56.0, 57.0],
    ...               [np.nan, np.nan, 12.0, 13.0, 16.0, 16.0, 17.0]])
    >>> variation(x, axis=1, ddof=1, nan_policy='omit')
    array([1.05109361, 0.31428986, 0.146483  ])
    
    """
    # 使用给定的数组 a 来创建适当的数组命名空间 xp
    xp = array_namespace(a)
    # 将 a 转换为 NumPy 数组
    a = xp.asarray(a)
    # 如果 axis 为 None，则需要将 a 重塑为一维数组并将 axis 设置为 0
    if axis is None:
        a = xp.reshape(a, (-1,))
        axis = 0
    
    # 获取数组 a 在指定维度上的大小
    n = a.shape[axis]
    # 获取数组 a 中的 NaN 值
    NaN = _get_nan(a)
    
    # 处理特殊情况：当数组 a 为空或者自由度 ddof 大于数组大小 n 时
    if a.size == 0 or ddof > n:
        # 返回一个填充值为 NaN 的数组，形状与 a.shape 去除指定 axis 后的形状一致
        shp = list(a.shape)
        shp.pop(axis)
        result = xp.full(shp, fill_value=NaN)
        return result[()] if result.ndim == 0 else result
    
    # 计算数组 a 在指定维度上的均值
    mean_a = xp.mean(a, axis=axis)
    
    # 处理特殊情况：当自由度 ddof 等于数组大小 n 时
    if ddof == n:
        # 另一种特殊情况：结果要么是 inf，要么是 NaN
        std_a = xp.std(a, axis=axis, correction=0)
        result = xp.where(std_a > 0, xp_copysign(xp.asarray(xp.inf), mean_a), NaN)
        return result[()] if result.ndim == 0 else result
    
    # 忽略除法相关的错误（如除以零或 NaN），计算数组 a 在指定维度上的标准差并除以均值得到变异系数
    with np.errstate(divide='ignore', invalid='ignore'):
        std_a = xp.std(a, axis=axis, correction=ddof)
        result = std_a / mean_a
    
    # 返回结果，如果结果是标量，则返回标量；如果结果是数组，则返回数组
    return result[()] if result.ndim == 0 else result
```