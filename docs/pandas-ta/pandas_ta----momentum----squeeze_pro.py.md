# `.\pandas-ta\pandas_ta\momentum\squeeze_pro.py`

```py
# -*- coding: utf-8 -*-
# 导入所需模块和对象
from numpy import NaN as npNaN
from pandas import DataFrame
from pandas_ta.momentum import mom
from pandas_ta.overlap import ema, sma
from pandas_ta.trend import decreasing, increasing
from pandas_ta.volatility import bbands, kc
from pandas_ta.utils import get_offset
from pandas_ta.utils import unsigned_differences, verify_series

# 定义函数：Squeeze Momentum (SQZ) PRO，计算挤压动量指标
def squeeze_pro(high, low, close, bb_length=None, bb_std=None, kc_length=None, kc_scalar_wide=None, kc_scalar_normal=None, kc_scalar_narrow=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs):
    """Indicator: Squeeze Momentum (SQZ) PRO"""
    # 验证参数有效性，并设置默认值
    bb_length = int(bb_length) if bb_length and bb_length > 0 else 20
    bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0
    kc_length = int(kc_length) if kc_length and kc_length > 0 else 20
    kc_scalar_wide = float(kc_scalar_wide) if kc_scalar_wide and kc_scalar_wide > 0 else 2
    kc_scalar_normal = float(kc_scalar_normal) if kc_scalar_normal and kc_scalar_normal > 0 else 1.5
    kc_scalar_narrow = float(kc_scalar_narrow) if kc_scalar_narrow and kc_scalar_narrow > 0 else 1
    mom_length = int(mom_length) if mom_length and mom_length > 0 else 12
    mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6

    _length = max(bb_length, kc_length, mom_length, mom_smooth)
    # 确保输入数据的有效性
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    # 验证 KC 系数的有效性
    valid_kc_scaler = kc_scalar_wide > kc_scalar_normal and kc_scalar_normal > kc_scalar_narrow
    if not valid_kc_scaler: return
    if high is None or low is None or close is None: return

    # 获取参数中的 True Range 数据，默认为真
    use_tr = kwargs.setdefault("tr", True)
    asint = kwargs.pop("asint", True)
    detailed = kwargs.pop("detailed", False)
    mamode = mamode if isinstance(mamode, str) else "sma"

    # 函数：简化数据框列名
    def simplify_columns(df, n=3):
        df.columns = df.columns.str.lower()
        return [c.split("_")[0][n - 1:n] for c in df.columns]

    # 计算结果
    bbd = bbands(close, length=bb_length, std=bb_std, mamode=mamode)
    kch_wide = kc(high, low, close, length=kc_length, scalar=kc_scalar_wide, mamode=mamode, tr=use_tr)
    kch_normal = kc(high, low, close, length=kc_length, scalar=kc_scalar_normal, mamode=mamode, tr=use_tr)
    kch_narrow = kc(high, low, close, length=kc_length, scalar=kc_scalar_narrow, mamode=mamode, tr=use_tr)

    # 简化 KC 和 BBAND 列名以便动态访问
    bbd.columns = simplify_columns(bbd)
    kch_wide.columns = simplify_columns(kch_wide)
    kch_normal.columns = simplify_columns(kch_normal)
    kch_narrow.columns = simplify_columns(kch_narrow)

    # 计算动量
    momo = mom(close, length=mom_length)
    # 根据参数选择使用 EMA 还是 SMA
    if mamode.lower() == "ema":
        squeeze = ema(momo, length=mom_smooth)
    else: # "sma"
        squeeze = sma(momo, length=mom_smooth)

    # 分类挤压状态
    squeeze_on_wide = (bbd.l > kch_wide.l) & (bbd.u < kch_wide.u)
    # 计算是否在正常宽度 Keltner 通道内挤压
    squeeze_on_normal = (bbd.l > kch_normal.l) & (bbd.u < kch_normal.u)
    # 计算是否在狭窄 Keltner 通道内挤压
    squeeze_on_narrow = (bbd.l > kch_narrow.l) & (bbd.u < kch_narrow.u)
    # 计算是否在宽幅 Keltner 通道外挤压
    squeeze_off_wide = (bbd.l < kch_wide.l) & (bbd.u > kch_wide.u)
    # 计算未挤压的情况
    no_squeeze = ~squeeze_on_wide & ~squeeze_off_wide

    # 偏移处理
    if offset != 0:
        # 将挤压标志位移
        squeeze = squeeze.shift(offset)
        squeeze_on_wide = squeeze_on_wide.shift(offset)
        squeeze_on_normal = squeeze_on_normal.shift(offset)
        squeeze_on_narrow = squeeze_on_narrow.shift(offset)
        squeeze_off_wide = squeeze_off_wide.shift(offset)
        no_squeeze = no_squeeze.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        # 使用指定值填充 NaN 值
        squeeze.fillna(kwargs["fillna"], inplace=True)
        squeeze_on_wide.fillna(kwargs["fillna"], inplace=True)
        squeeze_on_normal.fillna(kwargs["fillna"], inplace=True)
        squeeze_on_narrow.fillna(kwargs["fillna"], inplace=True)
        squeeze_off_wide.fillna(kwargs["fillna"], inplace=True)
        no_squeeze.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        # 使用指定的填充方法填充 NaN 值
        squeeze.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_on_wide.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_on_normal.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_on_narrow.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_off_wide.fillna(method=kwargs["fill_method"], inplace=True)
        no_squeeze.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar_wide}_{kc_scalar_normal}_{kc_scalar_narrow}"
    # 设置挤压系列的名称
    squeeze.name = f"SQZPRO{_props}"

    # 创建数据字典
    data = {
        squeeze.name: squeeze,
        # 基于不同条件的挤压情况创建数据列
        f"SQZPRO_ON_WIDE": squeeze_on_wide.astype(int) if asint else squeeze_on_wide,
        f"SQZPRO_ON_NORMAL": squeeze_on_normal.astype(int) if asint else squeeze_on_normal,
        f"SQZPRO_ON_NARROW": squeeze_on_narrow.astype(int) if asint else squeeze_on_narrow,
        f"SQZPRO_OFF": squeeze_off_wide.astype(int) if asint else squeeze_off_wide,
        f"SQZPRO_NO": no_squeeze.astype(int) if asint else no_squeeze,
    }
    # 创建 DataFrame 对象
    df = DataFrame(data)
    # 设置 DataFrame 和挤压系列的名称和分类
    df.name = squeeze.name
    df.category = squeeze.category = "momentum"

    # 详细的挤压系列
    # 如果 detailed 参数为 True，则执行以下逻辑
    if detailed:
        # 从 squeeze 中提取非负值
        pos_squeeze = squeeze[squeeze >= 0]
        # 从 squeeze 中提取负值
        neg_squeeze = squeeze[squeeze < 0]

        # 计算非负值序列的无符号差异，转换为整数
        pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
        # 计算负值序列的无符号差异，转换为整数
        neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

        # 将差异乘以 squeeze，得到调整后的差异值
        pos_inc *= squeeze
        pos_dec *= squeeze
        neg_dec *= squeeze
        neg_inc *= squeeze

        # 将调整后的差异中为零的值替换为 NaN
        pos_inc.replace(0, np.NaN, inplace=True)
        pos_dec.replace(0, np.NaN, inplace=True)
        neg_dec.replace(0, np.NaN, inplace=True)
        neg_inc.replace(0, np.NaN, inplace=True)

        # 计算 squeeze 乘以增加函数的结果，并将为零的值替换为 NaN
        sqz_inc = squeeze * increasing(squeeze)
        sqz_dec = squeeze * decreasing(squeeze)
        sqz_inc.replace(0, np.NaN, inplace=True)
        sqz_dec.replace(0, np.NaN, inplace=True)

        # 处理填充值
        if "fillna" in kwargs:
            sqz_inc.fillna(kwargs["fillna"], inplace=True)
            sqz_dec.fillna(kwargs["fillna"], inplace=True)
            pos_inc.fillna(kwargs["fillna"], inplace=True)
            pos_dec.fillna(kwargs["fillna"], inplace=True)
            neg_dec.fillna(kwargs["fillna"], inplace=True)
            neg_inc.fillna(kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            sqz_inc.fillna(method=kwargs["fill_method"], inplace=True)
            sqz_dec.fillna(method=kwargs["fill_method"], inplace=True)
            pos_inc.fillna(method=kwargs["fill_method"], inplace=True)
            pos_dec.fillna(method=kwargs["fill_method"], inplace=True)
            neg_dec.fillna(method=kwargs["fill_method"], inplace=True)
            neg_inc.fillna(method=kwargs["fill_method"], inplace=True)

        # 将调整后的差异值和 squeeze 乘以增加函数的结果添加到 DataFrame 中
        df[f"SQZPRO_INC"] = sqz_inc
        df[f"SQZPRO_DEC"] = sqz_dec
        df[f"SQZPRO_PINC"] = pos_inc
        df[f"SQZPRO_PDEC"] = pos_dec
        df[f"SQZPRO_NDEC"] = neg_dec
        df[f"SQZPRO_NINC"] = neg_inc

    # 返回处理后的 DataFrame
    return df
# 设置Squeeze PRO指标的文档字符串，描述了该指标的作用、来源以及计算方法等信息
squeeze_pro.__doc__ = \
"""Squeeze PRO(SQZPRO)

This indicator is an extended version of "TTM Squeeze" from John Carter.
The default is based on John Carter's "TTM Squeeze" indicator, as discussed
in his book "Mastering the Trade" (chapter 11). The Squeeze indicator attempts
to capture the relationship between two studies: Bollinger Bands® and Keltner's
Channels. When the volatility increases, so does the distance between the bands,
conversely, when the volatility declines, the distance also decreases. It finds
sections of the Bollinger Bands® study which fall inside the Keltner's Channels.

# 给出指标来源的链接
Sources:
    https://usethinkscript.com/threads/john-carters-squeeze-pro-indicator-for-thinkorswim-free.4021/
    https://www.tradingview.com/script/TAAt6eRX-Squeeze-PRO-Indicator-Makit0/

# 指标的计算方法，包括默认输入参数、所使用的指标、以及计算过程
Calculation:
    Default Inputs:
        bb_length=20, bb_std=2, kc_length=20, kc_scalar_wide=2,
        kc_scalar_normal=1.5, kc_scalar_narrow=1, mom_length=12,
        mom_smooth=6, tr=True,
    BB = Bollinger Bands
    KC = Keltner Channels
    MOM = Momentum
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    TR = True Range

# 计算所需的变量
    RANGE = TR(high, low, close) if using_tr else high - low
    BB_LOW, BB_MID, BB_HIGH = BB(close, bb_length, std=bb_std)
    KC_LOW_WIDE, KC_MID_WIDE, KC_HIGH_WIDE = KC(high, low, close, kc_length, kc_scalar_wide, TR)
    KC_LOW_NORMAL, KC_MID_NORMAL, KC_HIGH_NORMAL = KC(high, low, close, kc_length, kc_scalar_normal, TR)
    KC_LOW_NARROW, KC_MID_NARROW, KC_HIGH_NARROW = KC(high, low, close, kc_length, kc_scalar_narrow, TR)

# 计算动量指标
    MOMO = MOM(close, mom_length)
    if mamode == "ema":
        SQZPRO = EMA(MOMO, mom_smooth)
    else:
        SQZPRO = EMA(momo, mom_smooth)

# 判断是否处于Squeeze状态
    SQZPRO_ON_WIDE  = (BB_LOW > KC_LOW_WIDE) and (BB_HIGH < KC_HIGH_WIDE)
    SQZPRO_ON_NORMAL  = (BB_LOW > KC_LOW_NORMAL) and (BB_HIGH < KC_HIGH_NORMAL)
    SQZPRO_ON_NARROW  = (BB_LOW > KC_LOW_NARROW) and (BB_HIGH < KC_HIGH_NARROW)
    SQZPRO_OFF_WIDE = (BB_LOW < KC_LOW_WIDE) and (BB_HIGH > KC_HIGH_WIDE)
    SQZPRO_NO = !SQZ_ON_WIDE and !SQZ_OFF_WIDE

# 定义函数参数及参数说明
Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    bb_length (int): Bollinger Bands period. Default: 20
    bb_std (float): Bollinger Bands Std. Dev. Default: 2
    kc_length (int): Keltner Channel period. Default: 20
    kc_scalar_wide (float): Keltner Channel scalar for wider channel. Default: 2
    kc_scalar_normal (float): Keltner Channel scalar for normal channel. Default: 1.5
    kc_scalar_narrow (float): Keltner Channel scalar for narrow channel. Default: 1
    mom_length (int): Momentum Period. Default: 12
    mom_smooth (int): Smoothing Period of Momentum. Default: 6
    mamode (str): Only "ema" or "sma". Default: "sma"
    offset (int): How many periods to offset the result. Default: 0

# 定义可选关键字参数及其默认值
Kwargs:
    tr (value, optional): Use True Range for Keltner Channels. Default: True
"""
    # asint (value, optional): 是否使用整数而不是布尔值。默认值为True
    # mamode (value, optional): 使用哪种移动平均线。默认值为"sma"
    # detailed (value, optional): 返回用于可视化的SQZ的额外变体。默认值为False
    # fillna (value, optional): pd.DataFrame.fillna(value)的填充值
    # fill_method (value, optional): 填充方法的类型
# 返回一个 pandas DataFrame 对象，包含默认的 SQZPRO、SQZPRO_ON_WIDE、SQZPRO_ON_NORMAL、SQZPRO_ON_NARROW、SQZPRO_OFF_WIDE、SQZPRO_NO 列。如果 'detailed' 参数为 True，则包含更详细的列。
```