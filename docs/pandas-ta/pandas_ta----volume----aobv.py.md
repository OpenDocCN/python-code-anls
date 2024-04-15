# `.\pandas-ta\pandas_ta\volume\aobv.py`

```
# -*- coding: utf-8 -*-
# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 从当前目录下的 obv 模块中导入 obv 函数
from .obv import obv
# 从 pandas_ta.overlap 模块中导入 ma 函数
from pandas_ta.overlap import ma
# 从 pandas_ta.trend 模块中导入 long_run 和 short_run 函数
from pandas_ta.trend import long_run, short_run
# 从 pandas_ta.utils 模块中导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series

# 定义名为 aobv 的函数，计算 Archer On Balance Volume (AOBV) 指标
def aobv(close, volume, fast=None, slow=None, max_lookback=None, min_lookback=None, mamode=None, offset=None, **kwargs):
    """Indicator: Archer On Balance Volume (AOBV)"""
    # 验证参数
    # 如果 fast 存在且大于 0，则将其转换为整数，否则设为默认值 4
    fast = int(fast) if fast and fast > 0 else 4
    # 如果 slow 存在且大于 0，则将其转换为整数，否则设为默认值 12
    slow = int(slow) if slow and slow > 0 else 12
    # 如果 max_lookback 存在且大于 0，则将其转换为整数，否则设为默认值 2
    max_lookback = int(max_lookback) if max_lookback and max_lookback > 0 else 2
    # 如果 min_lookback 存在且大于 0，则将其转换为整数，否则设为默认值 2
    min_lookback = int(min_lookback) if min_lookback and min_lookback > 0 else 2
    # 如果 slow 小于 fast，则交换它们的值
    if slow < fast:
        fast, slow = slow, fast
    # 如果 mamode 不是字符串类型，则设为默认值 "ema"
    mamode = mamode if isinstance(mamode, str) else "ema"
    # 计算需要处理的数据长度
    _length = max(fast, slow, max_lookback, min_lookback)
    # 验证 close 和 volume 是否为有效的数据序列，长度为 _length
    close = verify_series(close, _length)
    volume = verify_series(volume, _length)
    # 获取偏移量
    offset = get_offset(offset)
    # 如果 kwargs 中存在 "length" 键，则将其移除
    if "length" in kwargs: kwargs.pop("length")
    # 从 kwargs 中获取 "run_length" 键的值，如果不存在则设为默认值 2
    run_length = kwargs.pop("run_length", 2)

    # 如果 close 或 volume 为 None，则返回空
    if close is None or volume is None: return

    # 计算结果
    # 计算 On Balance Volume（OBV）
    obv_ = obv(close=close, volume=volume, **kwargs)
    # 计算 OBV 的快速移动平均线
    maf = ma(mamode, obv_, length=fast, **kwargs)
    # 计算 OBV 的慢速移动平均线
    mas = ma(mamode, obv_, length=slow, **kwargs)

    # 当快速和慢速移动平均线长度为指定长度时
    obv_long = long_run(maf, mas, length=run_length)
    obv_short = short_run(maf, mas, length=run_length)

    # 考虑偏移量
    if offset != 0:
        obv_ = obv_.shift(offset)
        maf = maf.shift(offset)
        mas = mas.shift(offset)
        obv_long = obv_long.shift(offset)
        obv_short = obv_short.shift(offset)

    # 处理填充值
    if "fillna" in kwargs:
        obv_.fillna(kwargs["fillna"], inplace=True)
        maf.fillna(kwargs["fillna"], inplace=True)
        mas.fillna(kwargs["fillna"], inplace=True)
        obv_long.fillna(kwargs["fillna"], inplace=True)
        obv_short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        obv_.fillna(method=kwargs["fill_method"], inplace=True)
        maf.fillna(method=kwargs["fill_method"], inplace=True)
        mas.fillna(method=kwargs["fill_method"], inplace=True)
        obv_long.fillna(method=kwargs["fill_method"], inplace=True)
        obv_short.fillna(method=kwargs["fill_method"], inplace=True)

    # 准备返回的 DataFrame
    _mode = mamode.lower()[0] if len(mamode) else ""
    data = {
        obv_.name: obv_,
        f"OBV_min_{min_lookback}": obv_.rolling(min_lookback).min(),
        f"OBV_max_{max_lookback}": obv_.rolling(max_lookback).max(),
        f"OBV{_mode}_{fast}": maf,
        f"OBV{_mode}_{slow}": mas,
        f"AOBV_LR_{run_length}": obv_long,
        f"AOBV_SR_{run_length}": obv_short,
    }
    # 创建 DataFrame
    aobvdf = DataFrame(data)

    # 给 DataFrame 命名并分类
    aobvdf.name = f"AOBV{_mode}_{fast}_{slow}_{min_lookback}_{max_lookback}_{run_length}"
    aobvdf.category = "volume"

    return aobvdf
```