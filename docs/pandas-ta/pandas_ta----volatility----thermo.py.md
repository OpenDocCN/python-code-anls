# `.\pandas-ta\pandas_ta\volatility\thermo.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从 pandas_ta 库中的 overlap 模块导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta 库中的 utils 模块导入 get_offset、verify_series、get_drift 函数
from pandas_ta.utils import get_offset, verify_series, get_drift

# 定义函数 thermo，计算 Elder's Thermometer 指标
def thermo(high, low, length=None, long=None, short=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Elders Thermometer (THERMO)"""
    # 验证参数
    # 如果 length 存在且大于 0，则转换为整数，否则默认为 20
    length = int(length) if length and length > 0 else 20
    # 如果 long 存在且大于 0，则转换为浮点数，否则默认为 2
    long = float(long) if long and long > 0 else 2
    # 如果 short 存在且大于 0，则转换为浮点数，否则默认为 0.5
    short = float(short) if short and short > 0 else 0.5
    # 如果 mamode 是字符串，则保持不变，否则默认为 "ema"
    mamode = mamode if isinstance(mamode, str) else "ema"
    # 验证 high 和 low 系列数据，长度为 length
    high = verify_series(high, length)
    low = verify_series(low, length)
    # 获取 drift 和 offset
    drift = get_drift(drift)
    offset = get_offset(offset)
    # 从 kwargs 中弹出 asint 参数，默认为 True
    asint = kwargs.pop("asint", True)

    # 如果 high 或 low 为 None，则返回
    if high is None or low is None: return

    # 计算结果
    # 计算 Elder's Thermometer 的下限
    thermoL = (low.shift(drift) - low).abs()
    # 计算 Elder's Thermometer 的上限
    thermoH = (high - high.shift(drift)).abs()

    # 取较小的值作为 Elder's Thermometer
    thermo = thermoL
    thermo = thermo.where(thermoH < thermoL, thermoH)
    # 索引设置为 high 的索引
    thermo.index = high.index

    # 计算 Elder's Thermometer 的移动平均线
    thermo_ma = ma(mamode, thermo, length=length)

    # 生成信号
    thermo_long = thermo < (thermo_ma * long)
    thermo_short = thermo > (thermo_ma * short)

    # 二进制输出，用于信号
    if asint:
        thermo_long = thermo_long.astype(int)
        thermo_short = thermo_short.astype(int)

    # 调整偏移量
    if offset != 0:
        thermo = thermo.shift(offset)
        thermo_ma = thermo_ma.shift(offset)
        thermo_long = thermo_long.shift(offset)
        thermo_short = thermo_short.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        thermo.fillna(kwargs["fillna"], inplace=True)
        thermo_ma.fillna(kwargs["fillna"], inplace=True)
        thermo_long.fillna(kwargs["fillna"], inplace=True)
        thermo_short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        thermo.fillna(method=kwargs["fill_method"], inplace=True)
        thermo_ma.fillna(method=kwargs["fill_method"], inplace=True)
        thermo_long.fillna(method=kwargs["fill_method"], inplace=True)
        thermo_short.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    _props = f"_{length}_{long}_{short}"
    thermo.name = f"THERMO{_props}"
    thermo_ma.name = f"THERMOma{_props}"
    thermo_long.name = f"THERMOl{_props}"
    thermo_short.name = f"THERMOs{_props}"

    thermo.category = thermo_ma.category = thermo_long.category = thermo_short.category = "volatility"

    # 准备返回的 DataFrame
    data = {
        thermo.name: thermo,
        thermo_ma.name: thermo_ma,
        thermo_long.name: thermo_long,
        thermo_short.name: thermo_short
    }
    df = DataFrame(data)
    df.name = f"THERMO{_props}"
    df.category = thermo.category

    return df

# 为 thermo 函数添加文档字符串
thermo.__doc__ = \
"""Elders Thermometer (THERMO)

Elder's Thermometer measures price volatility.

Sources:
    https://www.motivewave.com/studies/elders_thermometer.htm
    # 导入所需的库
    import requests
    
    # 定义函数`get_tradingview_script`用于获取TradingView上的脚本内容
    def get_tradingview_script(url):
        # 发送GET请求获取指定URL的页面内容
        response = requests.get(url)
        # 返回页面内容的文本
        return response.text
    
    # 定义变量`script_url`，存储TradingView脚本的URL
    script_url = "https://www.tradingview.com/script/HqvTuEMW-Elder-s-Market-Thermometer-LazyBear/"
    
    # 调用`get_tradingview_script`函数，获取指定URL的脚本内容
    script_content = get_tradingview_script(script_url)
# 计算热力指标（thermo）和相关指标
Calculation:
    # 默认输入参数
    length=20, drift=1, mamode=EMA, long=2, short=0.5
    # EMA为指数移动平均

    # 计算低价的漂移
    thermoL = (low.shift(drift) - low).abs()
    # 计算高价的漂移
    thermoH = (high - high.shift(drift)).abs()

    # 选择较大的漂移值
    thermo = np.where(thermoH > thermoL, thermoH, thermoL)
    # 对漂移值进行指数移动平均
    thermo_ma = ema(thermo, length)

    # 判断是否满足买入条件
    thermo_long = thermo < (thermo_ma * long)
    # 判断是否满足卖出条件
    thermo_short = thermo > (thermo_ma * short)
    # 将布尔值转换为整数
    thermo_long = thermo_long.astype(int)
    thermo_short = thermo_short.astype(int)

Args:
    high (pd.Series): 'high' 的序列
    low (pd.Series): 'low' 的序列
    long(int): 买入因子
    short(float): 卖出因子
    length (int): 周期。默认值：20
    mamode (str): 参见 ```help(ta.ma)```。默认值：'ema'
    drift (int): 漂移周期。默认值：1
    offset (int): 结果的偏移周期数。默认值：0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): 填充方法的类型

Returns:
    pd.DataFrame: 包含 thermo, thermo_ma, thermo_long, thermo_short 列的数据框
```