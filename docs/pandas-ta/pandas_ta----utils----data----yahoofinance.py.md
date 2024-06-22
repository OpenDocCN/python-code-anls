# `.\pandas-ta\pandas_ta\utils\data\yahoofinance.py`

```py
# -*- coding: utf-8 -*-
# 导入 DataFrame 类
from pandas import DataFrame
# 导入 Imports、RATE、version 变量
from pandas_ta import Imports, RATE, version
# 导入 _camelCase2Title 函数和 ytd 函数
from .._core import _camelCase2Title
from .._time import ytd

# 定义函数 yf，用于包装 yfinance
def yf(ticker: str, **kwargs):
    """yf - yfinance wrapper

    It retrieves market data (ohlcv) from Yahoo Finance using yfinance.
    To install yfinance. (pip install yfinance) This method can also pull
    additional data using the 'kind' kwarg. By default kind=None and retrieves
    Historical Chart Data.

    Other options of 'kind' include:
    * All: "all"
        - Prints everything below but only returns Chart History to Pandas TA
    * Company Information: "info"
    * Institutional Holders: "institutional_holders" or "ih"
    * Major Holders: "major_holders" or "mh"
    * Mutual Fund Holders: "mutualfund_holders" or "mfh"
    * Recommendations (YTD): "recommendations" or "rec"
    * Earnings Calendar: "calendar" or "cal"
    * Earnings: "earnings" or "earn"
    * Sustainability/ESG Scores: "sustainability", "sus" or "esg"
    * Financials: "financials" or "fin"
        - Returns in order: Income Statement, Balance Sheet and Cash Flow
    * Option Chain: "option_chain" or "oc"
        - Uses the nearest expiration date by default
        - Change the expiration date using kwarg "exp"
        - Show ITM options, set kwarg "itm" to True. Or OTM options, set
        kwarg "itm" to False.
    * Chart History:
        - The only data returned to Pandas TA.

    Args:
        ticker (str): Any string for a ticker you would use with yfinance.
            Default: "SPY"
    Kwargs:
        calls (bool): When True, prints only Option Calls for the Option Chain.
            Default: None
        desc (bool): Will print Company Description when printing Company
            Information. Default: False
        exp (str): Used to print other Option Chains for the given Expiration
            Date. Default: Nearest Expiration Date for the Option Chains
        interval (str): A yfinance argument. Default: "1d"
        itm (bool): When printing Option Chains, shows ITM Options when True.
            When False, it shows OTM Options: Default: None
        kind (str): Options see above. Default: None
        period (str): A yfinance argument. Default: "max"
        proxy (dict): Proxy for yfinance to use. Default: {}
        puts (bool): When True, prints only Option Puts for the Option Chain.
            Default: None
        show (int > 0): How many last rows of Chart History to show.
            Default: None
        snd (int): How many recent Splits and Dividends to show in Company
            Information. Default: 5
        verbose (bool): Prints Company Information "info" and a Chart History
            header to the screen. Default: False

    Returns:
        Exits if the DataFrame is empty or None
        Otherwise it returns a DataFrame of the Chart History
    """
    # 从 kwargs 中获取 verbose 参数，默认为 False
    verbose = kwargs.pop("verbose", False)
    # 如果 ticker 不为空且为字符串类型且长度大于0，则将 ticker 转换为大写
    if ticker is not None and isinstance(ticker, str) and len(ticker):
        ticker = ticker.upper()
    else:
        # 如果 ticker 为空或不是字符串类型或长度为0，则将 ticker 设置为 "SPY"
        ticker = "SPY"

    # 从 kwargs 中弹出 "kind" 键对应的值，如果不存在则为 None
    kind = kwargs.pop("kind", None)
    # 如果 kind 不为空且为字符串类型且长度大于0，则将 kind 转换为小写
    if kind is not None and isinstance(kind, str) and len(kind):
        kind = kind.lower()

    # 从 kwargs 中弹出 "period" 键对应的值，如果不存在则为 "max"
    period = kwargs.pop("period", "max")
    # 从 kwargs 中弹出 "interval" 键对应的值，如果不存在则为 "1d"
    interval = kwargs.pop("interval", "1d")
    # 从 kwargs 中弹出 "proxy" 键对应的值，如果不存在则为一个空字典
    proxy = kwargs.pop("proxy", {})
    # 从 kwargs 中弹出 "show" 键对应的值，如果不存在则为 None
    show = kwargs.pop("show", None)

    # 如果 Imports 中没有 yfinance 模块，则打印提示信息并返回
    if not Imports["yfinance"]:
        print(f"[X] Please install yfinance to use this method. (pip install yfinance)")
        return
    else:
        # 如果有 yfinance 模块，则返回一个空的 DataFrame 对象
        return DataFrame()
```