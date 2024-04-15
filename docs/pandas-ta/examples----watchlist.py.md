# `.\pandas-ta\examples\watchlist.py`

```
# 设置文件编码为 utf-8
# 导入 datetime 模块并重命名为 dt
import datetime as dt

# 从 pathlib 模块中导入 Path 类
# 从 random 模块中导入 random 函数
# 从 typing 模块中导入 Tuple 类型
from pathlib import Path
from random import random
from typing import Tuple

# 导入 pandas 模块并重命名为 pd
# 从 pandas_datareader 库中导入 data 模块并重命名为 pdr
import pandas as pd  # pip install pandas
from pandas_datareader import data as pdr
# 导入 yfinance 模块并重命名为 yf
import yfinance as yf

# 使用 yfinance 的 pdr_override 函数来覆盖默认数据源
yf.pdr_override() # <== that's all it takes :-)

# 从 numpy 模块中导入 arange、append、array 函数并重命名
from numpy import arange as npArange
from numpy import append as npAppend
from numpy import array as npArray

# 导入 alphaVantageAPI 模块并重命名为 AV
# 导入 pandas_ta 模块
import alphaVantageAPI as AV # pip install alphaVantage-api
import pandas_ta as ta # pip install pandas_ta

# 定义一个函数 colors，用于返回颜色组合
def colors(colors: str = None, default: str = "GrRd"):
    # 颜色别名映射
    aliases = {
        # Pairs
        "BkGy": ["black", "gray"],
        "BkSv": ["black", "silver"],
        "BkPr": ["black", "purple"],
        "BkBl": ["black", "blue"],
        "FcLi": ["fuchsia", "lime"],
        "GrRd": ["green", "red"],
        "GyBk": ["gray", "black"],
        "GyBl": ["gray", "blue"],
        "GyOr": ["gray", "orange"],
        "GyPr": ["gray", "purple"],
        "GySv": ["gray", "silver"],
        "RdGr": ["red", "green"],
        "SvGy": ["silver", "gray"],
        # Triples
        "BkGrRd": ["black", "green", "red"],
        "BkBlPr": ["black", "blue", "purple"],
        "GrOrRd": ["green", "orange", "red"],
        "RdOrGr": ["red", "orange", "green"],
        # Quads
        "BkGrOrRd": ["black", "green", "orange", "red"],
        # Quints
        "BkGrOrRdMr": ["black", "green", "orange", "red", "maroon"],
        # Indicators
        "bbands": ["blue", "navy", "blue"],
        "kc": ["purple", "fuchsia", "purple"],
    }
    # 设置默认颜色组合
    aliases["default"] = aliases[default]
    # 如果输入的颜色在别名映射中，则返回对应颜色组合，否则返回默认颜色组合
    if colors in aliases.keys():
        return aliases[colors]
    return aliases["default"]

# 定义 Watchlist 类
class Watchlist(object):
    """
    # Watchlist Class (** This is subject to change! **)
    A simple Class to load/download financial market data and automatically
    apply Technical Analysis indicators with a Pandas TA Strategy.

    Default Strategy: pandas_ta.CommonStrategy

    ## Package Support:
    ### Data Source (Default: AlphaVantage)
    - AlphaVantage (pip install alphaVantage-api).
    - Python Binance (pip install python-binance). # Future Support
    - Yahoo Finance (pip install yfinance). # Almost Supported

    # Technical Analysis:
    - Pandas TA (pip install pandas_ta)

    ## Required Arguments:
    - tickers: A list of strings containing tickers. Example: ["SPY", "AAPL"]
    """

    # 初始化 Watchlist 类
    def __init__(self,
        tickers: list, tf: str = None, name: str = None,
        strategy: ta.Strategy = None, ds_name: str = "av", **kwargs,
    ):
        # 设置属性
        self.verbose = kwargs.pop("verbose", False)
        self.debug = kwargs.pop("debug", False)
        self.timed = kwargs.pop("timed", False)

        self.tickers = tickers
        self.tf = tf
        self.name = name if isinstance(name, str) else f"Watch: {', '.join(tickers)}"
        self.data = None
        self.kwargs = kwargs
        self.strategy = strategy

        # 初始化数据源
        self._init_data_source(ds_name)
    # 初始化数据源
    def _init_data_source(self, ds: str) -> None:
        # 将数据源名称转换为小写，并将其设置为实例属性，如果输入不是字符串则默认为 "av"
        self.ds_name = ds.lower() if isinstance(ds, str) else "av"

        # 默认情况下使用 AlphaVantage 数据源
        AVkwargs = {"api_key": "YOUR API KEY", "clean": True, "export": True, "output_size": "full", "premium": False}
        # 从传入参数中取出 AlphaVantage 的参数设置，如果不存在则使用默认设置
        self.av_kwargs = self.kwargs.pop("av_kwargs", AVkwargs)
        # 根据参数创建 AlphaVantage 数据源对象
        self.ds = AV.AlphaVantage(**self.av_kwargs)
        # 设置文件路径为数据源的导出路径
        self.file_path = self.ds.export_path

        # 如果数据源名称为 "yahoo"，则将数据源更改为 Yahoo Finance
        if self.ds_name == "yahoo":
            self.ds = yf

    # 删除 DataFrame 中的指定列
    def _drop_columns(self, df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
        if cols is None or not isinstance(cols, list):
            # 如果未指定列或者列不是列表，则使用默认列名
            cols = ["Unnamed: 0", "date", "split", "split_coefficient", "dividend", "dividends"]
        else:
            # 否则保持列不变
            cols
        """Helper methods to drop columns silently."""
        # 获取 DataFrame 的列名列表
        df_columns = list(df.columns)
        # 如果 DataFrame 存在指定的列名，则删除这些列
        if any(_ in df_columns for _ in cols):
            # 如果处于调试模式，则打印可能被删除的列名
            if self.debug:
                print(f"[i] Possible columns dropped: {', '.join(cols)}")
            # 删除指定的列，并忽略可能出现的错误
            df = df.drop(cols, axis=1, errors="ignore")
        return df

    # 加载所有指定股票的数据
    def _load_all(self, **kwargs) -> dict:
        """Updates the Watchlist's data property with a dictionary of DataFrames
        keyed by ticker."""
        # 如果指定了股票列表且列表不为空，则加载每个股票的数据并存储在字典中
        if (self.tickers is not None and isinstance(self.tickers, list) and
                len(self.tickers)):
            self.data = {ticker: self.load(ticker, **kwargs) for ticker in self.tickers}
            return self.data

    # 绘制图表
    def _plot(self, df, mas:bool = True, constants:bool = False, **kwargs) -> None:

        if constants:
            # 定义图表的常数线
            chart_lines = npAppend(npArange(-5, 6, 1), npArange(-100, 110, 10))
            # 添加图表的常数线
            df.ta.constants(True, chart_lines) # Adding the constants for the charts
            # 从 DataFrame 中删除指定的常数线
            df.ta.constants(False, npArray([-60, -40, 40, 60])) # Removing some constants from the DataFrame
            # 如果 verbose 为真，则打印常数线添加完成的消息
            if self.verbose: print(f"[i] {df.ticker} constants added.")

        if ta.Imports["matplotlib"]:
            # 从 kwargs 中获取绘图参数
            _exchange = kwargs.pop("exchange", "NYSE")
            _time = ta.get_time(_exchange, to_string=True)
            _kind = kwargs.pop("plot_kind", None)
            _figsize = kwargs.pop("figsize", (16, 10))
            _colors = kwargs.pop("figsize", ["black", "green", "orange", "red", "maroon"])
            _grid = kwargs.pop("grid", True)
            _alpha = kwargs.pop("alpha", 1)
            _last = kwargs.pop("last", 252)
            _title = kwargs.pop("title", f"{df.ticker}   {_time}   [{self.ds_name}]")

            col = kwargs.pop("close", "close")
            if mas:
                # 如果 mas 为真，则绘制均线
                # df.ta.strategy(self.strategy, append=True)
                # 从 DataFrame 中获取价格和均线数据
                price = df[[col, "SMA_10", "SMA_20", "SMA_50", "SMA_200"]]
            else:
                # 否则只获取价格数据
                price = df[col]

            if _kind is None:
                # 如果未指定绘图类型，则绘制线图
                price.tail(_last).plot(figsize=_figsize, color=_colors, linewidth=2, title=_title, grid=_grid, alpha=_alpha)
            else:
                # 否则打印未实现绘图类型的消息，并返回
                print(f"[X] Plot kind not implemented")
                return
    def load(self,
        ticker: str = None, tf: str = None, index: str = "date",
        drop: list = [], plot: bool = False, **kwargs
    ) -> pd.DataFrame:
        """Loads or Downloads (if a local csv does not exist) the data from the
        Data Source. When successful, it returns a Data Frame for the requested
        ticker. If no tickers are given, it loads all the tickers."""

        # 设置时间框架（Time Frame），如果未指定则使用默认值，并将其转换为大写
        tf = self.tf if tf is None else tf.upper()
        # 如果 ticker 参数不为 None，并且是字符串类型，则将其转换为大写
        if ticker is not None and isinstance(ticker, str):
            ticker = str(ticker).upper()
        else:
            # 如果没有指定 ticker，则输出正在加载所有 ticker 的消息，并加载所有 ticker 数据
            print(f"[!] Loading All: {', '.join(self.tickers)}")
            self._load_all(**kwargs)
            return

        # 构建文件名
        filename_ = f"{ticker}_{tf}.csv"
        # 构建当前文件路径
        current_file = Path(self.file_path) / filename_

        # 从本地加载或从数据源下载数据
        if current_file.exists():
            # 如果本地文件存在，则加载本地文件
            file_loaded = f"[i] Loaded {ticker}[{tf}]: {filename_}"
            # 如果数据源名称为 "av" 或 "yahoo"，则按照特定方式读取数据
            if self.ds_name in ["av", "yahoo"]:
                # 读取本地 CSV 文件到 DataFrame
                df = pd.read_csv(current_file, index_col=0)
                # 如果 DataFrame 不是按照日期时间顺序排列，则重新设置索引
                if not df.ta.datetime_ordered:
                    df = df.set_index(pd.DatetimeIndex(df.index))
                # 输出已加载文件的消息
                print(file_loaded)
            else:
                # 如果数据源名称不为 "av" 或 "yahoo"，则输出文件未找到的消息
                print(f"[X] {filename_} not found in {Path(self.file_path)}")
                return
        else:
            # 如果本地文件不存在，则从数据源下载数据
            print(f"[+] Downloading[{self.ds_name}]: {ticker}[{tf}]")
            if self.ds_name == "av":
                # 使用 Alpha Vantage 数据源获取数据
                df = self.ds.data(ticker, tf)
                # 如果 DataFrame 不是按照日期时间顺序排列，则重新设置索引
                if not df.ta.datetime_ordered:
                    df = df.set_index(pd.DatetimeIndex(df[index]))
            if self.ds_name == "yahoo":
                # 使用 Yahoo 数据源获取历史数据
                yf_data = self.ds.Ticker(ticker)
                df = yf_data.history(period="max")
                # 保存下载的数据到本地 CSV 文件
                to_save = f"{self.file_path}/{ticker}_{tf}.csv"
                print(f"[+] Saving: {to_save}")
                df.to_csv(to_save)

        # 移除指定列
        df = self._drop_columns(df, drop)

        # 如果设置了 analyze 参数为 True（默认为 True），则执行技术分析
        if kwargs.pop("analyze", True):
            if self.debug: print(f"[+] TA[{len(self.strategy.ta)}]: {self.strategy.name}")
            # 执行技术分析
            df.ta.strategy(self.strategy, timed=self.timed, **kwargs)

        # 将 ticker 和 tf 属性附加到 DataFrame
        df.ticker = ticker 
        df.tf = tf

        # 如果设置了 plot 参数为 True，则绘制 DataFrame
        if plot: self._plot(df, **kwargs)
        return df

    @property
    def data(self) -> dict:
        """When not None, it contains a dictionary of DataFrames keyed by ticker. data = {"SPY": pd.DataFrame, ...}"""
        # 返回数据字典属性
        return self._data

    @data.setter
    def data(self, value: dict) -> None:
        # 设置数据字典属性，并在后续检查其键值对的类型
        if value is not None and isinstance(value, dict):
            if self.verbose:
                print(f"[+] New data")
            self._data = value
        else:
            self._data = None

    @property
    def name(self) -> str:
        """The name of the Watchlist. Default: "Watchlist: {Watchlist.tickers}"."""
        # 返回观察列表的名称属性
        return self._name
    # 设置属性 name 的 setter 方法，用于设置 Watchlist 的名称
    def name(self, value: str) -> None:
        # 检查传入的值是否为字符串类型
        if isinstance(value, str):
            # 如果是字符串类型，则将其赋值给 _name 属性
            self._name = str(value)
        else:
            # 如果不是字符串类型，则将 Watchlist 的 tickers 联合起来作为名称
            self._name = f"Watchlist: {', '.join(self.tickers)}"

    # 获取属性 strategy 的 getter 方法，返回当前的策略对象
    def strategy(self) -> ta.Strategy:
        """Sets a valid Strategy. Default: pandas_ta.CommonStrategy"""
        return self._strategy

    # 设置属性 strategy 的 setter 方法，用于设置 Watchlist 的策略
    def strategy(self, value: ta.Strategy) -> None:
        # 检查传入的值是否为有效的策略对象
        if value is not None and isinstance(value, ta.Strategy):
            # 如果是有效的策略对象，则将其赋值给 _strategy 属性
            self._strategy = value
        else:
            # 如果不是有效的策略对象，则将默认的 CommonStrategy 赋值给 _strategy 属性
            self._strategy = ta.CommonStrategy

    # 获取属性 tf 的 getter 方法，返回当前的时间框架
    def tf(self) -> str:
        """Alias for timeframe. Default: 'D'"""
        return self._tf

    # 设置属性 tf 的 setter 方法，用于设置 Watchlist 的时间框架
    def tf(self, value: str) -> None:
        # 检查传入的值是否为字符串类型
        if isinstance(value, str):
            # 如果是字符串类型，则将其赋值给 _tf 属性
            value = str(value)
            self._tf = value
        else:
            # 如果不是字符串类型，则将默认值 'D' 赋值给 _tf 属性
            self._tf = "D"

    # 获取属性 tickers 的 getter 方法，返回当前的股票列表
    def tickers(self) -> list:
        """tickers

        If a string, it it converted to a list. Example: "AAPL" -> ["AAPL"]
            * Does not accept, comma seperated strings.
        If a list, checks if it is a list of strings.
        """
        return self._tickers

    # 设置属性 tickers 的 setter 方法，用于设置 Watchlist 的股票列表
    def tickers(self, value: Tuple[list, str]) -> None:
        # 检查传入的值是否为有效值
        if value is None:
            print(f"[X] {value} is not a value in Watchlist ticker.")
            return
        # 检查传入的值是否为列表且��表中的元素都是字符串类型
        elif isinstance(value, list) and [isinstance(_, str) for _ in value]:
            # 如果是列表且元素都是字符串类型，则将列表中的元素转换为大写后赋值给 _tickers 属性
            self._tickers = list(map(str.upper, value))
        # 检查传入的值是否为字符串类型
        elif isinstance(value, str):
            # 如果是字符串类型，则将其转换为大写后作为单个元素的列表赋值给 _tickers 属性
            self._tickers = [value.upper()]
        # 将 _tickers 属性的值作为名称
        self.name = self._tickers

    # 获取属性 verbose 的 getter 方法，返回当前的详细输出设置
    def verbose(self) -> bool:
        """Toggle the verbose property. Default: False"""
        return self._verbose

    # 设置属性 verbose 的 setter 方法，用于设置 Watchlist 的详细输出设置
    def verbose(self, value: bool) -> None:
        # 检查传入的值是否为布尔类型
        if isinstance(value, bool):
            # 如果是布尔类型，则将其赋值给 _verbose 属性
            self._verbose = bool(value)
        else:
            # 如果不是布尔类型，则将默认值 False 赋值给 _verbose 属性
            self._verbose = False

    # 返回 Pandas Ta 中可用指标的列表
    def indicators(self, *args, **kwargs) -> any:
        """Returns the list of indicators that are available with Pandas Ta."""
        pd.DataFrame().ta.indicators(*args, **kwargs)

    # 返回 Watchlist 对象的字符串表示形式
    def __repr__(self) -> str:
        # 构建 Watchlist 对象的字符串表示形式
        s = f"Watch(name='{self.name}', ds_name='{self.ds_name}', tickers[{len(self.tickers)}]='{', '.join(self.tickers)}', tf='{self.tf}', strategy[{self.strategy.total_ta()}]='{self.strategy.name}'"
        # 如果数据不为空，则添加数据的信息
        if self.data is not None:
            s += f", data[{len(self.data.keys())}])"
            return s
        return s + ")"
```