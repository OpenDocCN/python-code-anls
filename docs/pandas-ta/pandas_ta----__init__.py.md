# `.\pandas-ta\pandas_ta\__init__.py`

```
# 定义模块名称
name = "pandas_ta"
"""
.. moduleauthor:: Kevin Johnson
"""

# 导入模块
from importlib.util import find_spec
from pathlib import Path
from pkg_resources import get_distribution, DistributionNotFound

# 获取已安装的模块分布对象
_dist = get_distribution("pandas_ta")
try:
    # 获取模块所在路径
    here = Path(_dist.location) / __file__
    # 检查文件是否存在
    if not here.exists():
        # 如果未安装，但存在其他已安装版本
        raise DistributionNotFound
except DistributionNotFound:
    # 如果未找到分布对象，则提示安装该项目
    __version__ = "Please install this project with setup.py"

# 获取模块版本号
version = __version__ = _dist.version

# 检查导入的模块是否存在
Imports = {
    "alphaVantage-api": find_spec("alphaVantageAPI") is not None,
    "matplotlib": find_spec("matplotlib") is not None,
    "mplfinance": find_spec("mplfinance") is not None,
    "numba": find_spec("numba") is not None,
    "yaml": find_spec("yaml") is not None,
    "scipy": find_spec("scipy") is not None,
    "sklearn": find_spec("sklearn") is not None,
    "statsmodels": find_spec("statsmodels") is not None,
    "stochastic": find_spec("stochastic") is not None,
    "talib": find_spec("talib") is not None,
    "tqdm": find_spec("tqdm") is not None,
    "vectorbt": find_spec("vectorbt") is not None,
    "yfinance": find_spec("yfinance") is not None,
}

# 不是最理想的，也不是动态的，但它可以工作。
# 之后会找到一个动态的解决方案。
Category = {
    # 蜡烛图
    "candles": [
        "cdl_pattern", "cdl_z", "ha"
    ],
    # 周期
    "cycles": ["ebsw"],
    # 动量
    "momentum": [
        "ao", "apo", "bias", "bop", "brar", "cci", "cfo", "cg", "cmo",
        "coppock", "cti", "er", "eri", "fisher", "inertia", "kdj", "kst", "macd",
        "mom", "pgo", "ppo", "psl", "pvo", "qqe", "roc", "rsi", "rsx", "rvgi",
        "slope", "smi", "squeeze", "squeeze_pro", "stc", "stoch", "stochrsi", "td_seq", "trix",
        "tsi", "uo", "willr"
    ],
    # 重叠
    "overlap": [
        "alma", "dema", "ema", "fwma", "hilo", "hl2", "hlc3", "hma", "ichimoku",
        "jma", "kama", "linreg", "mcgd", "midpoint", "midprice", "ohlc4",
        "pwma", "rma", "sinwma", "sma", "ssf", "supertrend", "swma", "t3",
        "tema", "trima", "vidya", "vwap", "vwma", "wcp", "wma", "zlma"
    ],
    # 性能
    "performance": ["log_return", "percent_return"],
    # 统计
    "statistics": [
        "entropy", "kurtosis", "mad", "median", "quantile", "skew", "stdev",
        "tos_stdevall", "variance", "zscore"
    ],
    # 趋势
    "trend": [
        "adx", "amat", "aroon", "chop", "cksp", "decay", "decreasing", "dpo",
        "increasing", "long_run", "psar", "qstick", "short_run", "tsignals",
        "ttm_trend", "vhf", "vortex", "xsignals"
    ],
    # 波动性
    "volatility": [
        "aberration", "accbands", "atr", "bbands", "donchian", "hwc", "kc", "massi",
        "natr", "pdist", "rvi", "thermo", "true_range", "ui"
    ],

    # 交易量，"vp" 或 "Volume Profile" 是独特的
}
    # "volume" 键对应的值是一个列表，包含了各种技术指标的名称
    "volume": [
        "ad",    # AD 指标，积累/分配线
        "adosc", # AD 指标，震荡指标
        "aobv",  # 指标，累积/派发线
        "cmf",   # CMF 指标，资金流量指标
        "efi",   # EFI 指标，振荡器
        "eom",   # EOM 指标，指标
        "kvo",   # Klinger Oscillator（克林格震荡器）指标
        "mfi",   # MFI 指标，资金流指标
        "nvi",   # NVI 指标，价值线
        "obv",   # OBV 指标，累积/分配线
        "pvi",   # PVI 指标，价值线
        "pvol",  # PVO 指标，价值线
        "pvr",   # PVR 指标，价值线
        "pvt"    # PVT 指标，价值线
    ],
# 字典，用于指定聚合函数的方式，对于开盘价、最高价、最低价、收盘价和成交量分别指定了不同的聚合方式
CANGLE_AGG = {
    "open": "first",    # 开盘价取第一个值
    "high": "max",      # 最高价取最大值
    "low": "min",       # 最低价取最小值
    "close": "last",    # 收盘价取最后一个值
    "volume": "sum"     # 成交量取总和
}

# 字典，用于记录各个交易所的时区偏移
EXCHANGE_TZ = {
    "NZSX": 12,         # 新西兰股票交易所，时区偏移为+12
    "ASX": 11,          # 澳大利亚证券交易所，时区偏移为+11
    "TSE": 9,           # 东京证券交易所，时区偏移为+9
    "HKE": 8,           # 香港证券交易所，时区偏移为+8
    "SSE": 8,           # 上海证券交易所，时区偏移为+8
    "SGX": 8,           # 新加坡证券交易所，时区偏移为+8
    "NSE": 5.5,         # 印度证券交易所，时区偏移为+5.5
    "DIFX": 4,          # 迪拜金融市场，时区偏移为+4
    "RTS": 3,           # 莫斯科证券交易所，时区偏移为+3
    "JSE": 2,           # 南非证券交易所，时区偏移为+2
    "FWB": 1,           # 法兰克福证券交易所，时区偏移为+1
    "LSE": 1,           # 伦敦证券交易所，时区偏移为+1
    "BMF": -2,          # 巴西商品与期货交易所，时区偏移为-2
    "NYSE": -4,         # 纽约证券交易所，时区偏移为-4
    "TSX": -4           # 多伦多证券交易所，时区偏移为-4
}

# 字典，用于定义一些时间相关的常量
RATE = {
    "DAYS_PER_MONTH": 21,              # 每月交易日数
    "MINUTES_PER_HOUR": 60,            # 每小时分钟数
    "MONTHS_PER_YEAR": 12,             # 每年月份数
    "QUARTERS_PER_YEAR": 4,            # 每年季度数
    "TRADING_DAYS_PER_YEAR": 252,      # 每年交易日数，保持为偶数
    "TRADING_HOURS_PER_DAY": 6.5,      # 每日交易小时数
    "WEEKS_PER_YEAR": 52,              # 每年周数
    "YEARLY": 1                         # 年度
}

# 从 pandas_ta.core 模块导入所有内容
from pandas_ta.core import *
```