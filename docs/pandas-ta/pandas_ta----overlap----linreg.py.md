# `.\pandas-ta\pandas_ta\overlap\linreg.py`

```
# -*- coding: utf-8 -*-
# 从 numpy 库导入 array 并重命名为 npArray，导入 arctan 并重命名为 npAtan，导入 nan 并重命名为 npNaN，导入 pi 并重命名为 npPi，从 numpy 版本导入 version 并重命名为 npVersion
from numpy import array as npArray
from numpy import arctan as npAtan
from numpy import nan as npNaN
from numpy import pi as npPi
from numpy.version import version as npVersion
# 从 pandas 库导入 Series，从 pandas_ta.utils 导入 get_offset 和 verify_series 函数
from pandas import Series
from pandas_ta.utils import get_offset, verify_series


def linreg(close, length=None, offset=None, **kwargs):
    """Indicator: Linear Regression"""
    # 验证参数
    # 如果 length 存在且大于 0，则将其转换为整数，否则设为默认值 14
    length = int(length) if length and length > 0 else 14
    # 验证 close 是否为有效的 Series，并设定长度
    close = verify_series(close, length)
    # 获取偏移量
    offset = get_offset(offset)
    # 获取其他参数：angle、intercept、degrees、r、slope、tsf
    angle = kwargs.pop("angle", False)
    intercept = kwargs.pop("intercept", False)
    degrees = kwargs.pop("degrees", False)
    r = kwargs.pop("r", False)
    slope = kwargs.pop("slope", False)
    tsf = kwargs.pop("tsf", False)

    # 如果 close 为空，则返回 None
    if close is None: return

    # 计算结果
    # 生成 x 轴的值，范围从 1 到 length
    x = range(1, length + 1)  # [1, 2, ..., n] from 1 to n keeps Sum(xy) low
    # 计算 x 的和
    x_sum = 0.5 * length * (length + 1)
    # 计算 x 的平方和
    x2_sum = x_sum * (2 * length + 1) / 3
    # 计算除数
    divisor = length * x2_sum - x_sum * x_sum

    # 定义线性回归函数
    def linear_regression(series):
        # 计算 y 的和
        y_sum = series.sum()
        # 计算 x*y 的和
        xy_sum = (x * series).sum()

        # 计算斜率
        m = (length * xy_sum - x_sum * y_sum) / divisor
        # 如果 slope 为 True，则返回斜率
        if slope:
            return m
        # 计算截距
        b = (y_sum * x2_sum - x_sum * xy_sum) / divisor
        # 如果 intercept 为 True，则返回截距
        if intercept:
            return b

        # 如果 angle 为 True，则计算角度
        if angle:
            theta = npAtan(m)
            # 如果 degrees 为 True，则将角度转换为度
            if degrees:
                theta *= 180 / npPi
            return theta

        # 如果 r 为 True，则计算相关系数
        if r:
            # 计算 y^2 的和
            y2_sum = (series * series).sum()
            # 计算相关系数的分子
            rn = length * xy_sum - x_sum * y_sum
            # 计算相关系数的分母
            rd = (divisor * (length * y2_sum - y_sum * y_sum)) ** 0.5
            return rn / rd

        # 如果 tsf 为 True，则进行时间序列调整
        return m * length + b if tsf else m * (length - 1) + b

    # 定义滚动窗口函数
    def rolling_window(array, length):
        """https://github.com/twopirllc/pandas-ta/issues/285"""
        strides = array.strides + (array.strides[-1],)
        shape = array.shape[:-1] + (array.shape[-1] - length + 1, length)
        return as_strided(array, shape=shape, strides=strides)

    # 如果 numpy 版本大于等于 1.20.0，则使用 sliding_window_view 函数
    if npVersion >= "1.20.0":
        from numpy.lib.stride_tricks import sliding_window_view
        # 对于滑动窗口内的每个窗口，应用线性回归函数
        linreg_ = [linear_regression(_) for _ in sliding_window_view(npArray(close), length)]
    else:
        # 否则，使用 rolling_window 函数
        from numpy.lib.stride_tricks import as_strided
        # 对于滚动窗口内的每个窗口，应用线性回归函数
        linreg_ = [linear_regression(_) for _ in rolling_window(npArray(close), length)]

    # 创建 Series 对象，索引为 close 的索引，值为线性回归的结果
    linreg = Series([npNaN] * (length - 1) + linreg_, index=close.index)

    # 偏移结果
    if offset != 0:
        linreg = linreg.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        linreg.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        linreg.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    linreg.name = f"LR"
    if slope: linreg.name += "m"
    if intercept: linreg.name += "b"
    if angle: linreg.name += "a"
    if r: linreg.name += "r"

    linreg.name += f"_{length}"
    linreg.category = "overlap"

    return linreg
# 将文档字符串赋值给线性回归移动平均函数的__doc__属性，用于函数说明和文档生成
linreg.__doc__ = \
"""Linear Regression Moving Average (linreg)

Linear Regression Moving Average (LINREG). This is a simplified version of a
Standard Linear Regression. LINREG is a rolling regression of one variable. A
Standard Linear Regression is between two or more variables.

Source: TA Lib

Calculation:
    Default Inputs:
        length=14
    x = [1, 2, ..., n]
    x_sum = 0.5 * length * (length + 1)
    x2_sum = length * (length + 1) * (2 * length + 1) / 6
    divisor = length * x2_sum - x_sum * x_sum

    # 定义线性回归函数lr(series)，用于计算移动平均
    lr(series):
        # 计算系列的总和、平方和以及x和y的乘积和
        y_sum = series.sum()
        y2_sum = (series* series).sum()
        xy_sum = (x * series).sum()

        # 计算回归线的斜率m和截距b
        m = (length * xy_sum - x_sum * y_sum) / divisor
        b = (y_sum * x2_sum - x_sum * xy_sum) / divisor
        return m * (length - 1) + b

    # 使用rolling函数对close进行移动窗口处理，并应用lr函数计算移动平均
    linreg = close.rolling(length).apply(lr)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    angle (bool, optional): If True, returns the angle of the slope in radians.
        Default: False.
    degrees (bool, optional): If True, returns the angle of the slope in
        degrees. Default: False.
    intercept (bool, optional): If True, returns the angle of the slope in
        radians. Default: False.
    r (bool, optional): If True, returns it's correlation 'r'. Default: False.
    slope (bool, optional): If True, returns the slope. Default: False.
    tsf (bool, optional): If True, returns the Time Series Forecast value.
        Default: False.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
```