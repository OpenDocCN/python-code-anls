# `.\pandas-ta\pandas_ta\overlap\ma.py`

```py
# 设置文件编码为 UTF-8
# 导入 Series 类
from pandas import Series

# 导入不同的移动平均方法
# 从模块中导入指定的函数
from .dema import dema
from .ema import ema
from .fwma import fwma
from .hma import hma
from .linreg import linreg
from .midpoint import midpoint
from .pwma import pwma
from .rma import rma
from .sinwma import sinwma
from .sma import sma
from .swma import swma
from .t3 import t3
from .tema import tema
from .trima import trima
from .vidya import vidya
from .wma import wma
from .zlma import zlma

# 定义移动平均函数，用于简化移动平均的选择
def ma(name:str = None, source:Series = None, **kwargs) -> Series:
    """Simple MA Utility for easier MA selection

    Available MAs:
        dema, ema, fwma, hma, linreg, midpoint, pwma, rma,
        sinwma, sma, swma, t3, tema, trima, vidya, wma, zlma

    Examples:
        ema8 = ta.ma("ema", df.close, length=8)
        sma50 = ta.ma("sma", df.close, length=50)
        pwma10 = ta.ma("pwma", df.close, length=10, asc=False)

    Args:
        name (str): One of the Available MAs. Default: "ema"
        source (pd.Series): The 'source' Series.

    Kwargs:
        Any additional kwargs the MA may require.

    Returns:
        pd.Series: New feature generated.
    """

    # 支持的移动平均方法列表
    _mas = [
        "dema", "ema", "fwma", "hma", "linreg", "midpoint", "pwma", "rma",
        "sinwma", "sma", "swma", "t3", "tema", "trima", "vidya", "wma", "zlma"
    ]
    # 如果没有指定移动平均方法和数据源，则返回支持的移动平均方法列表
    if name is None and source is None:
        return _mas
    # 如果指定的移动平均方法是字符串，并且在支持的方法列表中，则转换为小写
    elif isinstance(name, str) and name.lower() in _mas:
        name = name.lower()
    else: # "ema"
        # 如果未指定移动平均方法，则默认选择 EMA
        name = _mas[1]

    # 根据选择的移动平均方法调用相应的函数
    if   name == "dema": return dema(source, **kwargs)
    elif name == "fwma": return fwma(source, **kwargs)
    elif name == "hma": return hma(source, **kwargs)
    elif name == "linreg": return linreg(source, **kwargs)
    elif name == "midpoint": return midpoint(source, **kwargs)
    elif name == "pwma": return pwma(source, **kwargs)
    elif name == "rma": return rma(source, **kwargs)
    elif name == "sinwma": return sinwma(source, **kwargs)
    elif name == "sma": return sma(source, **kwargs)
    elif name == "swma": return swma(source, **kwargs)
    elif name == "t3": return t3(source, **kwargs)
    elif name == "tema": return tema(source, **kwargs)
    elif name == "trima": return trima(source, **kwargs)
    elif name == "vidya": return vidya(source, **kwargs)
    elif name == "wma": return wma(source, **kwargs)
    elif name == "zlma": return zlma(source, **kwargs)
    else: return ema(source, **kwargs)
```