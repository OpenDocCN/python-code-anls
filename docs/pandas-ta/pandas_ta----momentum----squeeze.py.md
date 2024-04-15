# `.\pandas-ta\pandas_ta\momentum\squeeze.py`

```
# -*- coding: utf-8 -*-  # 指定文件编码格式为 UTF-8
from numpy import nan as npNaN  # 导入 numpy 库中的 nan 并起别名为 npNaN
from pandas import DataFrame  # 导入 pandas 库中的 DataFrame 类
from pandas_ta.momentum import mom  # 导入 pandas_ta 库中的 momentum 模块，并导入其中的 mom 函数
from pandas_ta.overlap import ema, linreg, sma  # 导入 pandas_ta 库中的 overlap 模块，并导入其中的 ema、linreg、sma 函数
from pandas_ta.trend import decreasing, increasing  # 导入 pandas_ta 库中的 trend 模块，并导入其中的 decreasing、increasing 函数
from pandas_ta.volatility import bbands, kc  # 导入 pandas_ta 库中的 volatility 模块，并导入其中的 bbands、kc 函数
from pandas_ta.utils import get_offset  # 导入 pandas_ta 库中的 utils 模块，并导入其中的 get_offset 函数
from pandas_ta.utils import unsigned_differences, verify_series  # 导入 pandas_ta 库中的 utils 模块，并导入其中的 unsigned_differences、verify_series 函数

# 定义函数：Squeeze Momentum (SQZ)
def squeeze(high, low, close, bb_length=None, bb_std=None, kc_length=None, kc_scalar=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs):
    """Indicator: Squeeze Momentum (SQZ)"""  # 函数文档字符串，指示 SQZ 指标的作用
    # 验证参数
    bb_length = int(bb_length) if bb_length and bb_length > 0 else 20  # 如果 bb_length 存在且大于 0，则转换为整数，否则设为默认值 20
    bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0  # 如果 bb_std 存在且大于 0，则转换为浮点数，否则设为默认值 2.0
    kc_length = int(kc_length) if kc_length and kc_length > 0 else 20  # 如果 kc_length 存在且大于 0，则转换为整数，否则设为默认值 20
    kc_scalar = float(kc_scalar) if kc_scalar and kc_scalar > 0 else 1.5  # 如果 kc_scalar 存在且大于 0，则转换为浮点数，否则设为默认值 1.5
    mom_length = int(mom_length) if mom_length and mom_length > 0 else 12  # 如果 mom_length 存在且大于 0，则转换为整数，否则设为默认值 12
    mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6  # 如果 mom_smooth 存在且大于 0，则转换为整数，否则设为默认值 6
    _length = max(bb_length, kc_length, mom_length, mom_smooth)  # 计算参数的最大长度
    high = verify_series(high, _length)  # 验证 high 序列的长度
    low = verify_series(low, _length)  # 验证 low 序列的长度
    close = verify_series(close, _length)  # 验证 close 序列的长度
    offset = get_offset(offset)  # 获取偏移量

    if high is None or low is None or close is None: return  # 如果 high、low、close 中有任何一个为 None，则返回

    use_tr = kwargs.setdefault("tr", True)  # 设置参数 tr，默认为 True
    asint = kwargs.pop("asint", True)  # 弹出参数 asint，默认为 True
    detailed = kwargs.pop("detailed", False)  # 弹出参数 detailed，默认为 False
    lazybear = kwargs.pop("lazybear", False)  # 弹出参数 lazybear，默认为 False
    mamode = mamode if isinstance(mamode, str) else "sma"  # 如果 mamode 是字符串类型，则保持不变，否则设为默认值 "sma"

    # 定义函数：简化列名
    def simplify_columns(df, n=3):
        df.columns = df.columns.str.lower()  # 将列名转换为小写
        return [c.split("_")[0][n - 1:n] for c in df.columns]  # 返回简化后的列名列表

    # 计算结果
    bbd = bbands(close, length=bb_length, std=bb_std, mamode=mamode)  # 计算布林带指标
    kch = kc(high, low, close, length=kc_length, scalar=kc_scalar, mamode=mamode, tr=use_tr)  # 计算 Keltner 通道指标

    # 简化 KC 和 BBAND 列名以便动态访问
    bbd.columns = simplify_columns(bbd)  # 简化布林带指标列名
    kch.columns = simplify_columns(kch)  # 简化 Keltner 通道指标列名

    if lazybear:  # 如果 lazybear 参数为真
        highest_high = high.rolling(kc_length).max()  # 计算最高高度
        lowest_low = low.rolling(kc_length).min()  # 计算最低低度
        avg_ = 0.25 * (highest_high + lowest_low) + 0.5 * kch.b  # 计算平均值

        squeeze = linreg(close - avg_, length=kc_length)  # 计算线性回归

    else:  # 如果 lazybear 参数为假
        momo = mom(close, length=mom_length)  # 计算动量
        if mamode.lower() == "ema":  # 如果 mamode 参数为 "ema"
            squeeze = ema(momo, length=mom_smooth)  # 计算指数移动平均
        else:  # 否则（mamode 参数为 "sma"）
            squeeze = sma(momo, length=mom_smooth)  # 计算简单移动平均

    # 分类 Squeeze
    squeeze_on = (bbd.l > kch.l) & (bbd.u < kch.u)  # 计算 Squeeze on
    squeeze_off = (bbd.l < kch.l) & (bbd.u > kch.u) 
    # 如果参数中包含 "fillna"，则使用指定值填充缺失值
    if "fillna" in kwargs:
        # 使用指定值填充缺失值，inplace=True 表示在原数据上进行修改
        squeeze.fillna(kwargs["fillna"], inplace=True)
        squeeze_on.fillna(kwargs["fillna"], inplace=True)
        squeeze_off.fillna(kwargs["fillna"], inplace=True)
        no_squeeze.fillna(kwargs["fillna"], inplace=True)
    
    # 如果参数中包含 "fill_method"，则使用指定方法填充缺失值
    if "fill_method" in kwargs:
        # 使用指定方法填充缺失值，inplace=True 表示在原数据上进行修改
        squeeze.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_on.fillna(method=kwargs["fill_method"], inplace=True)
        squeeze_off.fillna(method=kwargs["fill_method"], inplace=True)
        no_squeeze.fillna(method=kwargs["fill_method"], inplace=True)

    # 设置名称和分类
    _props = "" if use_tr else "hlr"
    _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar}"
    _props += "_LB" if lazybear else ""
    squeeze.name = f"SQZ{_props}"

    # 创建数据字典
    data = {
        squeeze.name: squeeze,
        f"SQZ_ON": squeeze_on.astype(int) if asint else squeeze_on,
        f"SQZ_OFF": squeeze_off.astype(int) if asint else squeeze_off,
        f"SQZ_NO": no_squeeze.astype(int) if asint else no_squeeze,
    }
    # 创建 DataFrame 对象
    df = DataFrame(data)
    df.name = squeeze.name
    df.category = squeeze.category = "momentum"

    # 如果需要详细信息
    if detailed:
        # 分别获取正数和负数的数据
        pos_squeeze = squeeze[squeeze >= 0]
        neg_squeeze = squeeze[squeeze < 0]

        # 计算正数和负数的增量和减量
        pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
        neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

        # 计算正数和负数的增量和减量乘以原数据
        pos_inc *= squeeze
        pos_dec *= squeeze
        neg_dec *= squeeze
        neg_inc *= squeeze

        # 将值为 0 的数据替换为 NaN
        pos_inc.replace(0, np.NaN, inplace=True)
        pos_dec.replace(0, np.NaN, inplace=True)
        neg_dec.replace(0, np.NaN, inplace=True)
        neg_inc.replace(0, np.NaN, inplace=True)

        # 计算正数和负数的增量和减量乘以原数据
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

        # 添加详细信息列
        df[f"SQZ_INC"] = sqz_inc
        df[f"SQZ_DEC"] = sqz_dec
        df[f"SQZ_PINC"] = pos_inc
        df[f"SQZ_PDEC"] = pos_dec
        df[f"SQZ_NDEC"] = neg_dec
        df[f"SQZ_NINC"] = neg_inc

    # 返回 DataFrame 对象
    return df
# 设置 squeeze 函数的文档字符串，解释了 squeeze 指标的作用和计算方法
squeeze.__doc__ = \
"""Squeeze (SQZ)

The default is based on John Carter's "TTM Squeeze" indicator, as discussed
in his book "Mastering the Trade" (chapter 11). The Squeeze indicator attempts
to capture the relationship between two studies: Bollinger Bands® and Keltner's
Channels. When the volatility increases, so does the distance between the bands,
conversely, when the volatility declines, the distance also decreases. It finds
sections of the Bollinger Bands® study which fall inside the Keltner's Channels.

Sources:
    https://tradestation.tradingappstore.com/products/TTMSqueeze
    https://www.tradingview.com/scripts/lazybear/
    https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/T-U/TTM-Squeeze

Calculation:
    Default Inputs:
        bb_length=20, bb_std=2, kc_length=20, kc_scalar=1.5, mom_length=12,
        mom_smooth=12, tr=True, lazybear=False,
    BB = Bollinger Bands
    KC = Keltner Channels
    MOM = Momentum
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    TR = True Range

    RANGE = TR(high, low, close) if using_tr else high - low
    BB_LOW, BB_MID, BB_HIGH = BB(close, bb_length, std=bb_std)
    KC_LOW, KC_MID, KC_HIGH = KC(high, low, close, kc_length, kc_scalar, TR)

    if lazybear:
        HH = high.rolling(kc_length).max()
        LL = low.rolling(kc_length).min()
        AVG  = 0.25 * (HH + LL) + 0.5 * KC_MID
        SQZ = linreg(close - AVG, kc_length)
    else:
        MOMO = MOM(close, mom_length)
        if mamode == "ema":
            SQZ = EMA(MOMO, mom_smooth)
        else:
            SQZ = EMA(momo, mom_smooth)

    SQZ_ON  = (BB_LOW > KC_LOW) and (BB_HIGH < KC_HIGH)
    SQZ_OFF = (BB_LOW < KC_LOW) and (BB_HIGH > KC_HIGH)
    NO_SQZ = !SQZ_ON and !SQZ_OFF

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    bb_length (int): Bollinger Bands period. Default: 20
    bb_std (float): Bollinger Bands Std. Dev. Default: 2
    kc_length (int): Keltner Channel period. Default: 20
    kc_scalar (float): Keltner Channel scalar. Default: 1.5
    mom_length (int): Momentum Period. Default: 12
    mom_smooth (int): Smoothing Period of Momentum. Default: 6
    mamode (str): Only "ema" or "sma". Default: "sma"
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    tr (value, optional): Use True Range for Keltner Channels. Default: True
    asint (value, optional): Use integers instead of bool. Default: True
    mamode (value, optional): Which MA to use. Default: "sma"
    lazybear (value, optional): Use LazyBear's TradingView implementation.
        Default: False
    detailed (value, optional): Return additional variations of SQZ for
        visualization. Default: False
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
"""
    # 创建一个 pandas DataFrame，默认包含 SQZ、SQZ_ON、SQZ_OFF、NO_SQZ 列。如果 'detailed' 参数为 True，则包含更详细的列。
"""
# 导入必要的模块：字符串 I/O 以及 zip 文件处理
from io import BytesIO
import zipfile

# 定义函数 unzip_data，接受文件名参数
def unzip_data(zip_file):
    # 打开 zip 文件并读取其内容
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # 解压缩文件到指定路径
        zip_ref.extractall('unzipped_data')
```