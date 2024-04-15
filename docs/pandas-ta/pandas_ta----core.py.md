# `.\pandas-ta\pandas_ta\core.py`

```
# -*- coding: utf-8 -*-

# 导入所需的模块
from dataclasses import dataclass, field
from multiprocessing import cpu_count, Pool
from pathlib import Path
from time import perf_counter
from typing import List, Tuple
from warnings import simplefilter

# 导入第三方库
import pandas as pd
from numpy import log10 as npLog10
from numpy import ndarray as npNdarray
from pandas.core.base import PandasObject

# 导入自定义模块
from pandas_ta import Category, Imports, version
from pandas_ta.candles.cdl_pattern import ALL_PATTERNS
from pandas_ta.candles import *
from pandas_ta.cycles import *
from pandas_ta.momentum import *
from pandas_ta.overlap import *
from pandas_ta.performance import *
from pandas_ta.statistics import *
from pandas_ta.trend import *
from pandas_ta.volatility import *
from pandas_ta.volume import *
from pandas_ta.utils import *

# 创建一个空的 DataFrame
df = pd.DataFrame()

# Strategy DataClass
@dataclass
class Strategy:
    """Strategy DataClass
    A way to name and group your favorite indicators

    Args:
        name (str): Some short memorable string.  Note: Case-insensitive "All" is reserved.
        ta (list of dicts): A list of dicts containing keyword arguments where "kind" is the indicator.
        description (str): A more detailed description of what the Strategy tries to capture. Default: None
        created (str): At datetime string of when it was created. Default: Automatically generated. *Subject to change*

    Example TA:
    ta = [
        {"kind": "sma", "length": 200},
        {"kind": "sma", "close": "volume", "length": 50},
        {"kind": "bbands", "length": 20},
        {"kind": "rsi"},
        {"kind": "macd", "fast": 8, "slow": 21},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
    ]
    """

    name: str  # = None # Required.
    ta: List = field(default_factory=list)  # Required.
    # Helpful. More descriptive version or notes or w/e.
    description: str = "TA Description"
    # Optional. Gets Exchange Time and Local Time execution time
    created: str = get_time(to_string=True)
    # 在初始化对象后执行的方法，用于完成一些初始化操作
    def __post_init__(self):
        # 判断是否有名称，默认为True
        has_name = True
        # 判断是否为TA（Technical Analyst），默认为False
        is_ta = False
        # 存储需要的参数列表的初始化
        required_args = ["[X] Strategy requires the following argument(s):"]

        # 检查名称是否为字符串类型
        name_is_str = isinstance(self.name, str)
        # 检查TA是否为列表
        ta_is_list = isinstance(self.ta, list)

        # 如果名称为None或者不是字符串类型
        if self.name is None or not name_is_str:
            # 添加名称参数的错误信息到列表
            required_args.append(' - name. Must be a string. Example: "My TA". Note: "all" is reserved.')
            # 更新has_name的值，但实际上这里似乎有一个逻辑错误，应该是 has_name = not has_name
            has_name != has_name

        # 如果TA为None
        if self.ta is None:
            # 将TA设置为None
            self.ta = None
        # 如果TA不为None且为列表类型且总的TA数量大于0
        elif self.ta is not None and ta_is_list and self.total_ta() > 0:
            # 检查TA列表中的所有元素是否都为字典类型且字典键数量大于0
            # 不检查字典值是否为有效的指标参数，用户需要查阅指标文档以获取所有指标参数信息
            is_ta = all([isinstance(_, dict) and len(_.keys()) > 0 for _ in self.ta])
        # 如果TA不符合预期
        else:
            # 添加TA参数的错误信息到列表
            s = " - ta. Format is a list of dicts. Example: [{'kind': 'sma', 'length': 10}]"
            s += "\n       Check the indicator for the correct arguments if you receive this error."
            required_args.append(s)

        # 如果需要的参数列表中的项数量大于1
        if len(required_args) > 1:
            # 打印出每个错误信息
            [print(_) for _ in required_args]
            # 返回None
            return None

    # 返回TA数量
    def total_ta(self):
        # 如果TA不为None，则返回TA列表的长度，否则返回0
        return len(self.ta) if self.ta is not None else 0
# 定义名为 "All" 的默认策略，不包含任何技术指标
AllStrategy = Strategy(
    name="All",
    description="All the indicators with their default settings. Pandas TA default.",
    ta=None,
)

# 定义名为 "Common Price and Volume SMAs" 的默认策略，包含常见的价格和成交量简单移动平均线
CommonStrategy = Strategy(
    name="Common Price and Volume SMAs",
    description="Common Price SMAs: 10, 20, 50, 200 and Volume SMA: 20.",
    ta=[
        {"kind": "sma", "length": 10},
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOL"}
    ]
)

# 用于扩展 Pandas DataFrame 的基类
class BasePandasObject(PandasObject):
    """Simple PandasObject Extension

    Ensures the DataFrame is not empty and has columns.
    It would be a sad Panda otherwise.

    Args:
        df (pd.DataFrame): Extends Pandas DataFrame
    """

    def __init__(self, df, **kwargs):
        # 如果 DataFrame 为空，则直接返回
        if df.empty: return
        # 如果 DataFrame 至少有一个列
        if len(df.columns) > 0:
            # 将常见列名映射到统一的小写形式
            common_names = {
                "Date": "date",
                "Time": "time",
                "Timestamp": "timestamp",
                "Datetime": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "split",
            }
            # 预先删除所有值均为 NaN 的行
            # 可能需要移动到 AnalysisIndicators.__call__() 中以通过 kwargs 进行切换
            # df.dropna(axis=0, inplace=True)
            # 预先将列名重命名为小写形式
            df.rename(columns=common_names, errors="ignore", inplace=True)

            # 预先将索引名转换为小写形式
            index_name = df.index.name
            if index_name is not None:
                df.index.rename(index_name.lower(), inplace=True)

            self._df = df
        else:
            # 如果没有列，则引发错误
            raise AttributeError(f"[X] No columns!")

    def __call__(self, kind, *args, **kwargs):
        raise NotImplementedError()


# Pandas TA - DataFrame Analysis Indicators
@pd.api.extensions.register_dataframe_accessor("ta")
class AnalysisIndicators(BasePandasObject):
    """
    This Pandas Extension is named 'ta' for Technical Analysis. In other words,
    it is a Numerical Time Series Feature Generator where the Time Series data
    is biased towards Financial Market data; typical data includes columns
    named :"open", "high", "low", "close", "volume".

    This TA Library hopefully allows you to apply familiar and unique Technical
    Analysis Indicators easily with the DataFrame Extension named 'ta'. Even
    though 'ta' is a Pandas DataFrame Extension, you can still call Technical
    Analysis indicators individually if you are more comfortable with that
    approach or it allows you to easily and automatically apply the indicators

"""
    # 通过策略方法调用指标。参见：help(ta.strategy)。
    # 默认情况下，'ta'扩展使用小写列名：open, high, low, close和volume。您可以在调用指标时提供替代名称来覆盖默认设置。例如，调用指标hl2()。
    # 使用默认列：open, high, low, close和volume。
    >>> df.ta.hl2()
    >>> df.ta(kind="hl2")

    # 使用DataFrame列：Open, High, Low, Close和Volume。
    >>> df.ta.hl2(high="High", low="Low")
    >>> df.ta(kind="hl2", high="High", low="Low")

    # 如果您不想使用DataFrame扩展，只需正常调用。
    >>> sma10 = ta.sma(df["Close"]) # 默认长度=10
    >>> sma50 = ta.sma(df["Close"], length=50)
    >>> ichimoku, span = ta.ichimoku(df["High"], df["Low"], df["Close"])

    # 参数:
    # kind (str, optional): 默认值：None。Kind是指标的'名称'。在调用之前将kind转换为小写。
    # timed (bool, optional): 默认值：False。对执行速度感兴趣吗？
    # kwargs: 扩展特定的修饰符。
    # append (bool, optional): 默认值：False。当为True时，将结果列附加到DataFrame中。

    # 返回:
    # 大多数指标将返回一个Pandas Series。像MACD、BBANDS、KC等其他指标将返回一个Pandas DataFrame。另一方面，Ichimoku将返回两个DataFrame，已知期间的Ichimoku DataFrame和Span值的未来的Span DataFrame。

    # 让我们开始吧！

    # 1. 加载'ta'模块:
    >>> import pandas as pd
    >>> import ta as ta

    # 2. 加载一些数据:
    >>> df = pd.read_csv("AAPL.csv", index_col="date", parse_dates=True)

    # 3. 帮助！
    # 3a. 一般帮助:
    >>> help(df.ta)
    >>> df.ta()
    # 3b. 指标帮助:
    >>> help(ta.apo)
    # 3c. 指标扩展帮助:
    >>> help(df.ta.apo)

    # 4. 调用指标的方式。
    # 4a. 标准：仅调用APO指标，不使用"ta" DataFrame扩展。
    >>> ta.apo(df["close"])
    # 4b. DataFrame扩展：使用"ta" DataFrame扩展仅调用APO指标。
    >>> df.ta.apo()
    # 4c. DataFrame扩展（kind）：使用'kind'调用APO。
    >>> df.ta(kind="apo")
    # 4d. 策略:
    >>> df.ta.strategy("All") # 默认
    >>> df.ta.strategy(ta.Strategy("My Strat", ta=[{"kind": "apo"}])) # 自定义

    # 5. 使用kwargs
    # 5a. 将结果附加到工作df中。
    >>> df.ta.apo(append=True)
    # 5b. 计时一个指标。
    >>> apo = df.ta(kind="apo", timed=True)
    >>> print(apo.timed)
    """

    # 初始化变量
    _adjusted = None
    _cores = cpu_count()
    _df = DataFrame()
    _exchange = "NYSE"
    _time_range = "years"
    _last_run = get_time(_exchange, to_string=True)

    # 初始化方法
    def __init__(self, pandas_obj):
        # 验证传入的pandas对象
        self._validate(pandas_obj)
        self._df = pandas_obj
        self._last_run = get_time(self._exchange, to_string=True)

    @staticmethod
    # 验证输入对象是否为 Pandas 的 DataFrame 或 Series 类型
    def _validate(obj: Tuple[pd.DataFrame, pd.Series]):
        if not isinstance(obj, pd.DataFrame) and not isinstance(obj, pd.Series):
            raise AttributeError("[X] Must be either a Pandas Series or DataFrame.")

    # DataFrame 行为方法
    def __call__(
            self, kind: str = None,
            timed: bool = False, version: bool = False, **kwargs
        ):
        if version: print(f"Pandas TA - Technical Analysis Indicators - v{self.version}")
        try:
            if isinstance(kind, str):
                kind = kind.lower()
                fn = getattr(self, kind)

                if timed:
                    stime = perf_counter()

                # 运行指标
                result = fn(**kwargs)  # = getattr(self, kind)(**kwargs)
                self._last_run = get_time(self.exchange, to_string=True) # 保存运行完成的时间

                if timed:
                    result.timed = final_time(stime)
                    print(f"[+] {kind}: {result.timed}")

                return result
            else:
                self.help()

        except BaseException:
            pass

    # 公共获取/设置 DataFrame 属性
    @property
    def adjusted(self) -> str:
        """property: df.ta.adjusted"""
        return self._adjusted

    @adjusted.setter
    def adjusted(self, value: str) -> None:
        """property: df.ta.adjusted = 'adj_close'"""
        if value is not None and isinstance(value, str):
            self._adjusted = value
        else:
            self._adjusted = None

    @property
    def cores(self) -> str:
        """返回类别。"""
        return self._cores

    @cores.setter
    def cores(self, value: int) -> None:
        """property: df.ta.cores = integer"""
        cpus = cpu_count()
        if value is not None and isinstance(value, int):
            self._cores = int(value) if 0 <= value <= cpus else cpus
        else:
            self._cores = cpus

    @property
    def exchange(self) -> str:
        """返回当前交易所。默认值为 "NYSE"。"""
        return self._exchange

    @exchange.setter
    def exchange(self, value: str) -> None:
        """property: df.ta.exchange = "LSE" """
        if value is not None and isinstance(value, str) and value in EXCHANGE_TZ.keys():
            self._exchange = value

    @property
    def last_run(self) -> str:
        """返回 DataFrame 上次运行的时间。"""
        return self._last_run

    # 公共获取 DataFrame 属性
    @property
    def categories(self) -> str:
        """返回类别。"""
        return list(Category.keys())

    @property
    def datetime_ordered(self) -> bool:
        """如果索引是日期时间且有序，则返回 True。"""
        hasdf = hasattr(self, "_df")
        if hasdf:
            return is_datetime_ordered(self._df)
        return hasdf
    # 反转 DataFrame，相当于 df.iloc[::-1]
    def reverse(self) -> pd.DataFrame:
        """Reverses the DataFrame. Simply: df.iloc[::-1]"""
        return self._df.iloc[::-1]

    # 返回 DataFrame 的时间范围，以浮点数表示，默认单位为 "years"。调用 help(ta.toal_time) 查看更多信息
    @property
    def time_range(self) -> float:
        """Returns the time ranges of the DataFrame as a float. Default is in "years". help(ta.toal_time)"""
        return total_time(self._df, self._time_range)

    # 设置 DataFrame 的时间范围为给定值，默认为 "years"
    @time_range.setter
    def time_range(self, value: str) -> None:
        """property: df.ta.time_range = "years" (Default)"""
        if value is not None and isinstance(value, str):
            self._time_range = value
        else:
            self._time_range = "years"

    # 将 DataFrame 的索引设置为 UTC 格式
    @property
    def to_utc(self) -> None:
        """Sets the DataFrame index to UTC format"""
        self._df = to_utc(self._df)

    # 返回版本信息
    @property
    def version(self) -> str:
        """Returns the version."""
        return version

    # 私有 DataFrame 方法：给结果列添加前缀和/或后缀
    def _add_prefix_suffix(self, result=None, **kwargs) -> None:
        """Add prefix and/or suffix to the result columns"""
        if result is None:
            return
        else:
            prefix = suffix = ""
            delimiter = kwargs.setdefault("delimiter", "_")

            # 检查是否有指定的前缀和后缀
            if "prefix" in kwargs:
                prefix = f"{kwargs['prefix']}{delimiter}"
            if "suffix" in kwargs:
                suffix = f"{delimiter}{kwargs['suffix']}"

            # 如果结果是 Series，则修改名称；否则修改列名
            if isinstance(result, pd.Series):
                result.name = prefix + result.name + suffix
            else:
                result.columns = [prefix + column + suffix for column in result.columns]
    # 将 Pandas Series 或 DataFrame 的列追加到 self._df 中
    def _append(self, result=None, **kwargs) -> None:
        """Appends a Pandas Series or DataFrame columns to self._df."""
        # 如果 kwargs 中包含 "append" 并且为 True
        if "append" in kwargs and kwargs["append"]:
            # 获取 self._df 和 result
            df = self._df
            # 如果 df 或 result 为 None，则返回
            if df is None or result is None: return
            else:
                # 忽略性能警告
                simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
                # 如果 kwargs 中包含 "col_names" 并且不是元组
                if "col_names" in kwargs and not isinstance(kwargs["col_names"], tuple):
                    # 将 kwargs["col_names"] 转换为元组
                    kwargs["col_names"] = (kwargs["col_names"],) # 注意：tuple(kwargs["col_names"]) 不起作用

                # 如果 result 是 DataFrame
                if isinstance(result, pd.DataFrame):
                    # 如果在 kwargs 中指定了列名，则重命名列
                    # 否则使用默认列名
                    if "col_names" in kwargs and isinstance(kwargs["col_names"], tuple):
                        if len(kwargs["col_names"]) >= len(result.columns):
                            # 遍历 result 的列和 kwargs["col_names"]，将数据追加到 df 中
                            for col, ind_name in zip(result.columns, kwargs["col_names"]):
                                df[ind_name] = result.loc[:, col]
                        else:
                            print(f"Not enough col_names were specified : got {len(kwargs['col_names'])}, expected {len(result.columns)}.")
                            return
                    else:
                        # 使用默认列名将 result 的列追加到 df 中
                        for i, column in enumerate(result.columns):
                            df[column] = result.iloc[:, i]
                else:
                    # 如果指定了列名，则使用列名，否则使用 result 的名称
                    ind_name = (
                        kwargs["col_names"][0] if "col_names" in kwargs and
                        isinstance(kwargs["col_names"], tuple) else result.name
                    )
                    # 将 result 追加到 df 中
                    df[ind_name] = result

    # 检查所有值都为缺失值的列，并返回这些列
    def _check_na_columns(self, stdout: bool = True):
        """Returns the columns in which all it's values are na."""
        return [x for x in self._df.columns if all(self._df[x].isna())]
    def _get_column(self, series):
        """Attempts to get the correct series or 'column' and return it."""
        # 获取 DataFrame
        df = self._df
        # 如果 DataFrame 为空则返回空
        if df is None: return

        # 如果传入的 series 是一个 pandas Series 对象，则直接返回
        if isinstance(series, pd.Series):
            return series
        # 如果 series 为 None，则根据条件返回相应的 DataFrame 列
        elif series is None:
            return df[self.adjusted] if self.adjusted is not None else None
        # 如果 series 是一个字符串
        elif isinstance(series, str):
            # 如果该字符串是 DataFrame 的列名，则返回该列
            if series in df.columns:
                return df[series]
            else:
                # 否则尝试匹配类似的列名，以防输错
                matches = df.columns.str.match(series, case=False)
                match = [i for i, x in enumerate(matches) if x]
                # 如果找到匹配的列，则返回该列，否则输出未找到的提示信息
                cols = ", ".join(list(df.columns))
                NOT_FOUND = f"[X] Ooops!!! It's {series not in df.columns}, the series '{series}' was not found in {cols}"
                return df.iloc[:, match[0]] if len(match) else print(NOT_FOUND)

    def _indicators_by_category(self, name: str) -> list:
        """Returns indicators by Categorical name."""
        # 根据分类名称返回指标列表
        return Category[name] if name in self.categories else None

    def _mp_worker(self, arguments: tuple):
        """Multiprocessing Worker to handle different Methods."""
        # 多进程工作函数，用于处理不同的方法
        method, args, kwargs = arguments

        # 如果方法不是 "ichimoku"，则调用对应的方法并返回结果
        if method != "ichimoku":
            return getattr(self, method)(*args, **kwargs)
        else:
            # 如果方法是 "ichimoku"，则调用对应的方法并返回结果的第一个元素
            return getattr(self, method)(*args, **kwargs)[0]

    def _post_process(self, result, **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
        """Applies any additional modifications to the DataFrame
        * Applies prefixes and/or suffixes
        * Appends the result to main DataFrame
        """
        # 对 DataFrame 进行任何额外修改
        # * 添加前缀和/或后缀
        # * 将结果附加到主 DataFrame

        # 获取是否显示详细信息的标志
        verbose = kwargs.pop("verbose", False)
        # 如果结果不是 Series 或 DataFrame，则输出错误信息（如果 verbose 为 True）
        if not isinstance(result, (pd.Series, pd.DataFrame)):
            if verbose:
                print(f"[X] Oops! The result was not a Series or DataFrame.")
            return self._df
        else:
            # 如果结果是 DataFrame 并且传入了列号，则只保留指定列
            result = (result.iloc[:, [int(n) for n in kwargs["col_numbers"]]]
                      if isinstance(result, pd.DataFrame) and
                      "col_numbers" in kwargs and
                      kwargs["col_numbers"] is not None else result)
            # 添加前缀/后缀并将结果附加到 DataFrame 中
            self._add_prefix_suffix(result=result, **kwargs)
            self._append(result=result, **kwargs)
        return result
    # 辅助方法，确定策略的模式和名称。返回元组：(name:str, mode:dict)
    def _strategy_mode(self, *args) -> tuple:
        name = "All"  # 默认策略名称为"All"
        mode = {"all": False, "category": False, "custom": False}  # 初始化模式字典

        if len(args) == 0:  # 如果参数为空
            mode["all"] = True  # 设置模式为"all"
        else:
            if isinstance(args[0], str):  # 如果参数是字符串
                if args[0].lower() == "all":  # 如果参数为"all"
                    name, mode["all"] = name, True  # 设置名称和模式为"all"
                if args[0].lower() in self.categories:  # 如果参数在类别中
                    name, mode["category"] = args[0], True  # 设置名称和模式为类别

            if isinstance(args[0], Strategy):  # 如果参数是策略对象
                strategy_ = args[0]
                if strategy_.ta is None or strategy_.name.lower() == "all":  # 如果策略对象为空或名称为"all"
                    name, mode["all"] = name, True  # 设置名称和模式为"all"
                elif strategy_.name.lower() in self.categories:  # 如果策略名称在类别中
                    name, mode["category"] = strategy_.name, True  # 设置名称和模式为类别
                else:
                    name, mode["custom"] = strategy_.name, True  # 设置名称和模式为自定义

        return name, mode  # 返回名称和模式的元组

    # 公共 DataFrame 方法
    def constants(self, append: bool, values: list):
        """Constants

        Add or remove constants to the DataFrame easily with Numpy's arrays or
        lists. Useful when you need easily accessible horizontal lines for
        charting.

        Add constant '1' to the DataFrame
        >>> df.ta.constants(True, [1])
        Remove constant '1' to the DataFrame
        >>> df.ta.constants(False, [1])

        Adding constants for charting
        >>> import numpy as np
        >>> chart_lines = np.append(np.arange(-4, 5, 1), np.arange(-100, 110, 10))
        >>> df.ta.constants(True, chart_lines)
        Removing some constants from the DataFrame
        >>> df.ta.constants(False, np.array([-60, -40, 40, 60]))

        Args:
            append (bool): If True, appends a Numpy range of constants to the
                working DataFrame.  If False, it removes the constant range from
                the working DataFrame. Default: None.

        Returns:
            Returns the appended constants
            Returns nothing to the user.  Either adds or removes constant ranges
            from the working DataFrame.
        """
        if isinstance(values, npNdarray) or isinstance(values, list):  # 如果值是 Numpy 数组或列表
            if append:  # 如果是追加操作
                for x in values:  # 遍历值
                    self._df[f"{x}"] = x  # 在 DataFrame 中添加常量列
                return self._df[self._df.columns[-len(values):]]  # 返回添加的常量列
            else:  # 如果是删除操作
                for x in values:  # 遍历值
                    del self._df[f"{x}"]  # 从 DataFrame 中删除常量列
    def indicators(self, **kwargs):
        """List of Indicators

        kwargs:
            as_list (bool, optional): When True, it returns a list of the
                indicators. Default: False.
            exclude (list, optional): The passed in list will be excluded
                from the indicators list. Default: None.

        Returns:
            Prints the list of indicators. If as_list=True, then a list.
        """
        # 设置默认参数as_list为False，如果用户没有提供as_list参数，则默认为False
        as_list = kwargs.setdefault("as_list", False)
        # 公共非指标方法
        helper_methods = ["constants", "indicators", "strategy"]
        # 公共df.ta属性
        ta_properties = [
            "adjusted",
            "categories",
            "cores",
            "datetime_ordered",
            "exchange",
            "last_run",
            "reverse",
            "ticker",
            "time_range",
            "to_utc",
            "version",
        ]

        # 获取DataFrame.ta下的所有公共非指标方法并生成列表
        ta_indicators = list((x for x in dir(pd.DataFrame().ta) if not x.startswith("_") and not x.endswith("_")))

        # 添加要删除的Pandas TA方法和属性
        removed = helper_methods + ta_properties

        # 添加要从指标列表中排除的用户指定方法
        user_excluded = kwargs.setdefault("exclude", [])
        # 如果用户指定了排除列表并且列表长度大于0，则将其加入removed列表
        if isinstance(user_excluded, list) and len(user_excluded) > 0:
            removed += user_excluded

        # 从指标列表中移除不需要的指标
        [ta_indicators.remove(x) for x in removed]

        # 如果as_list为True，则立即返回指标列表
        if as_list:
            return ta_indicators

        # 获取指标的总数
        total_indicators = len(ta_indicators)
        header = f"Pandas TA - Technical Analysis Indicators - v{self.version}"
        s = f"{header}\nTotal Indicators & Utilities: {total_indicators + len(ALL_PATTERNS)}\n"
        # 如果存在指标，则打印指标列表和蜡烛图案列表
        if total_indicators > 0:
            print(f"{s}Abbreviations:\n    {', '.join(ta_indicators)}\n\nCandle Patterns:\n    {', '.join(ALL_PATTERNS)}")
        else:
            # 如果不存在指标，则仅打印标题和总数
            print(s)
    def ticker(self, ticker: str, **kwargs):
        """ticker

        This method downloads Historical Data if the package yfinance is installed.
        Additionally it can run a ta.Strategy; Builtin or Custom. It returns a
        DataFrame if there the DataFrame is not empty, otherwise it exits. For
        additional yfinance arguments, use help(ta.yf).

        Historical Data
        >>> df = df.ta.ticker("aapl")
        More specifically
        >>> df = df.ta.ticker("aapl", period="max", interval="1d", kind=None)

        Changing the period of Historical Data
        Period is used instead of start/end
        >>> df = df.ta.ticker("aapl", period="1y")

        Changing the period and interval of Historical Data
        Retrieves the past year in weeks
        >>> df = df.ta.ticker("aapl", period="1y", interval="1wk")
        Retrieves the past month in hours
        >>> df = df.ta.ticker("aapl", period="1mo", interval="1h")

        Show everything
        >>> df = df.ta.ticker("aapl", kind="all")

        Args:
            ticker (str): Any string for a ticker you would use with yfinance.
                Default: "SPY"
        Kwargs:
            kind (str): Options see above. Default: "history"
            ds (str): Data Source to use. Default: "yahoo"
            strategy (str | ta.Strategy): Which strategy to apply after
                downloading chart history. Default: None

            See help(ta.yf) for additional kwargs

        Returns:
            Exits if the DataFrame is empty or None
            Otherwise it returns a DataFrame
        """
        # 获取数据源，默认为 Yahoo
        ds = kwargs.pop("ds", "yahoo")
        # 获取策略，默认为 None
        strategy = kwargs.pop("strategy", None)

        # Fetch the Data
        # 小写化数据源字符串并检查是否为字符串类型
        ds = ds.lower() is not None and isinstance(ds, str)
        # 使用 yfinance 获取数据
        df = yf(ticker, **kwargs)

        # 如果 DataFrame 为空，返回 None
        if df is None: return
        # 如果 DataFrame 为空，打印信息并返回 None
        elif df.empty:
            print(f"[X] DataFrame is empty: {df.shape}")
            return
        else:
            # 如果 lc_cols 参数为 True，则将 DataFrame 的索引和列名转换为小写
            if kwargs.pop("lc_cols", False):
                df.index.name = df.index.name.lower()
                df.columns = df.columns.str.lower()
            # 将数据保存到对象的属性中
            self._df = df

        # 如果有策略参数，则执行策略
        if strategy is not None: self.strategy(strategy, **kwargs)
        # 返回 DataFrame
        return df


    # Public DataFrame Methods: Indicators and Utilities
    # Candles
    def cdl_pattern(self, name="all", offset=None, **kwargs):
        # 获取开盘价，默认为 "open" 列
        open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取最高价，默认为 "high" 列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取最低价，默认为 "low" 列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 cdl_pattern 函数获取结果
        result = cdl_pattern(open_=open_, high=high, low=low, close=close, name=name, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)
    # 计算并返回 CDL_Z 指标
    def cdl_z(self, full=None, offset=None, **kwargs):
        # 获取 'open' 参数对应的列
        open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取 'high' 参数对应的列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 'low' 参数对应的列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取 'close' 参数对应的列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 cdl_z 函数计算 CDL_Z 指标
        result = cdl_z(open_=open_, high=high, low=low, close=close, full=full, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算并返回 HA 指标
    def ha(self, offset=None, **kwargs):
        # 获取 'open' 参数对应的列
        open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取 'high' 参数对应的列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 'low' 参数对应的列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取 'close' 参数对应的列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 ha 函数计算 HA 指标
        result = ha(open_=open_, high=high, low=low, close=close, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算并返回 EBSW 指标
    # 参数：close - 收盘价列；length - 长度；bars - 柱子数量；offset - 偏移量
    def ebsw(self, close=None, length=None, bars=None, offset=None, **kwargs):
        # 获取 'close' 参数对应的列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 ebsw 函数计算 EBSW 指标
        result = ebsw(close=close, length=length, bars=bars, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算并返回 AO 指标
    def ao(self, fast=None, slow=None, offset=None, **kwargs):
        # 获取 'high' 参数对应的列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 'low' 参数对应的列
        low = self._get_column(kwargs.pop("low", "low"))
        # 调用 ao 函数计算 AO 指标
        result = ao(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算并返回 APO 指标
    def apo(self, fast=None, slow=None, mamode=None, offset=None, **kwargs):
        # 获取 'close' 参数对应的列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 apo 函数计算 APO 指标
        result = apo(close=close, fast=fast, slow=slow, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算并返回 BIAS 指标
    def bias(self, length=None, mamode=None, offset=None, **kwargs):
        # 获取 'close' 参数对应的列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 bias 函数计算 BIAS 指标
        result = bias(close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算并返回 BOP 指标
    def bop(self, percentage=False, offset=None, **kwargs):
        # 获取 'open' 参数对应的列
        open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取 'high' 参数对应的列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 'low' 参数对应的列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取 'close' 参数对应的列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 bop 函数计算 BOP 指标
        result = bop(open_=open_, high=high, low=low, close=close, percentage=percentage, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算 BRAR 指标
    def brar(self, length=None, scalar=None, drift=None, offset=None, **kwargs):
        # 获取“open”列，默认为"open"
        open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取“high”列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取“low”列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 BRAR 指标
        result = brar(open_=open_, high=high, low=low, close=close, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 CCI 指标
    def cci(self, length=None, c=None, offset=None, **kwargs):
        # 获取“high”列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取“low”列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 CCI 指标
        result = cci(high=high, low=low, close=close, length=length, c=c, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 CFO 指标
    def cfo(self, length=None, offset=None, **kwargs):
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 CFO 指标
        result = cfo(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 CG 指标
    def cg(self, length=None, offset=None, **kwargs):
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 CG 指标
        result = cg(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 CMO 指标
    def cmo(self, length=None, scalar=None, drift=None, offset=None, **kwargs):
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 CMO 指标
        result = cmo(close=close, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Coppock 指标
    def coppock(self, length=None, fast=None, slow=None, offset=None, **kwargs):
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 Coppock 指标
        result = coppock(close=close, length=length, fast=fast, slow=slow, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 CTI 指标
    def cti(self, length=None, offset=None, **kwargs):
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 CTI 指标
        result = cti(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 DM 指标
    def dm(self, drift=None, offset=None, mamode=None, **kwargs):
        # 获取“high”列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取“low”列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 计算 DM 指标
        result = dm(high=high, low=low, drift=drift, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 ER 指标
    def er(self, length=None, drift=None, offset=None, **kwargs):
        # 获取“close”列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 ER 指标
        result = er(close=close, length=length, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算 ERI 指标并返回结果
    def eri(self, length=None, offset=None, **kwargs):
        # 获取高价、低价、收盘价列数据
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 ERI 指标
        result = eri(high=high, low=low, close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)

    # 计算 Fisher 变换指标并返回结果
    def fisher(self, length=None, signal=None, offset=None, **kwargs):
        # 获取高价、低价列数据
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        # 计算 Fisher 变换指标
        result = fisher(high=high, low=low, length=length, signal=signal, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)

    # 计算 Inertia 指标并返回结果
    def inertia(self, length=None, rvi_length=None, scalar=None, refined=None, thirds=None, mamode=None, drift=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        # 如果 refined 或 thirds 不为空，则获取高价、低价列数据
        if refined is not None or thirds is not None:
            high = self._get_column(kwargs.pop("high", "high"))
            low = self._get_column(kwargs.pop("low", "low"))
            # 计算 Inertia 指标
            result = inertia(close=close, high=high, low=low, length=length, rvi_length=rvi_length, scalar=scalar, refined=refined, thirds=thirds, mamode=mamode, drift=drift, offset=offset, **kwargs)
        else:
            # 计算 Inertia 指标
            result = inertia(close=close, length=length, rvi_length=rvi_length, scalar=scalar, refined=refined, thirds=thirds, mamode=mamode, drift=drift, offset=offset, **kwargs)

        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)

    # 计算 KDJ 指标并返回结果
    def kdj(self, length=None, signal=None, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 KDJ 指标
        result = kdj(high=high, low=low, close=close, length=length, signal=signal, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)

    # 计算 KST 指标并返回结果
    def kst(self, roc1=None, roc2=None, roc3=None, roc4=None, sma1=None, sma2=None, sma3=None, sma4=None, signal=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 KST 指标
        result = kst(close=close, roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4, sma1=sma1, sma2=sma2, sma3=sma3, sma4=sma4, signal=signal, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)

    # 计算 MACD 指标并返回结果
    def macd(self, fast=None, slow=None, signal=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 MACD 指标
        result = macd(close=close, fast=fast, slow=slow, signal=signal, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)

    # 计算动量指标并返回结果
    def mom(self, length=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算动量指标
        result = mom(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)
    # 计算 PGO（Price Gravity Oscillator），价格引力振荡器
    def pgo(self, length=None, offset=None, **kwargs):
        # 获取 high 列，默认为 high
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 low 列，默认为 low
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取 close 列，默认为 close
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 pgo 函数计算 PGO 指标
        result = pgo(high=high, low=low, close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 PPO（Percentage Price Oscillator），百分比价格振荡器
    def ppo(self, fast=None, slow=None, scalar=None, mamode=None, offset=None, **kwargs):
        # 获取 close 列，默认为 close
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 ppo 函数计算 PPO 指标
        result = ppo(close=close, fast=fast, slow=slow, scalar=scalar, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 PSL（Price Series Length），价格序列长度
    def psl(self, open_=None, length=None, scalar=None, drift=None, offset=None, **kwargs):
        # 如果传入了 open_ 参数，则获取 open_ 列，默认为 open
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))

        # 获取 close 列，默认为 close
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 psl 函数计算 PSL 指标
        result = psl(close=close, open_=open_, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 PVO（Percentage Volume Oscillator），百分比成交量振荡器
    def pvo(self, fast=None, slow=None, signal=None, scalar=None, offset=None, **kwargs):
        # 获取 volume 列，默认为 volume
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 调用 pvo 函数计算 PVO 指标
        result = pvo(volume=volume, fast=fast, slow=slow, signal=signal, scalar=scalar, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 QQE（Quantitative Qualitative Estimation），定量定性估计
    def qqe(self, length=None, smooth=None, factor=None, mamode=None, offset=None, **kwargs):
        # 获取 close 列，默认为 close
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 qqe 函数计算 QQE 指标
        result = qqe(close=close, length=length, smooth=smooth, factor=factor, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 ROC（Rate of Change），变化率
    def roc(self, length=None, offset=None, **kwargs):
        # 获取 close 列，默认为 close
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 roc 函数计算 ROC 指标
        result = roc(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 RSI（Relative Strength Index），相对强弱指数
    def rsi(self, length=None, scalar=None, drift=None, offset=None, **kwargs):
        # 获取 close 列，默认为 close
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 rsi 函数计算 RSI 指标
        result = rsi(close=close, length=length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 RSX（Relative Strength Index），相对强弱指数
    def rsx(self, length=None, drift=None, offset=None, **kwargs):
        # 获取 close 列，默认为 close
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 rsx 函数计算 RSX 指标
        result = rsx(close=close, length=length, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算 RVGI 指标
    def rvgi(self, length=None, swma_length=None, offset=None, **kwargs):
        # 获取指定列的数据
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 RVGI 指标
        result = rvgi(open_=open_, high=high, low=low, close=close, length=length, swma_length=swma_length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算斜率指标
    def slope(self, length=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算斜率指标
        result = slope(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 SMI 指标
    def smi(self, fast=None, slow=None, signal=None, scalar=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 SMI 指标
        result = smi(close=close, fast=fast, slow=slow, signal=signal, scalar=scalar, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Squeeze 指标
    def squeeze(self, bb_length=None, bb_std=None, kc_length=None, kc_scalar=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 Squeeze 指标
        result = squeeze(high=high, low=low, close=close, bb_length=bb_length, bb_std=bb_std, kc_length=kc_length, kc_scalar=kc_scalar, mom_length=mom_length, mom_smooth=mom_smooth, use_tr=use_tr, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Squeeze Pro 指标
    def squeeze_pro(self, bb_length=None, bb_std=None, kc_length=None, kc_scalar_wide=None, kc_scalar_normal=None, kc_scalar_narrow=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 Squeeze Pro 指标
        result = squeeze_pro(high=high, low=low, close=close, bb_length=bb_length, bb_std=bb_std, kc_length=kc_length, kc_scalar_wide=kc_scalar_wide, kc_scalar_normal=kc_scalar_normal, kc_scalar_narrow=kc_scalar_narrow, mom_length=mom_length, mom_smooth=mom_smooth, use_tr=use_tr, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 STC 指标
    def stc(self, ma1=None, ma2=None, osc=None, tclength=None, fast=None, slow=None, factor=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 STC 指标
        result = stc(close=close, ma1=ma1, ma2=ma2, osc=osc, tclength=tclength, fast=fast, slow=slow, factor=factor, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算随机指标（Stochastic Oscillator）
    def stoch(self, fast_k=None, slow_k=None, slow_d=None, mamode=None, offset=None, **kwargs):
        # 获取数据列中的最高价，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取数据列中的最低价，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取数据列中的收盘价，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 使用stoch函数计算随机指标，结果存储在result变量中
        result = stoch(high=high, low=low, close=close, fast_k=fast_k, slow_k=slow_k, slow_d=slow_d, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理并返回
        return self._post_process(result, **kwargs)

    # 计算随机相对强弱指数（Stochastic RSI）
    def stochrsi(self, length=None, rsi_length=None, k=None, d=None, mamode=None, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = stochrsi(high=high, low=low, close=close, length=length, rsi_length=rsi_length, k=k, d=d, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # 计算Tom DeMark序列（Tom DeMark Sequential）
    def td_seq(self, asint=None, offset=None, show_all=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        result = td_seq(close=close, asint=asint, offset=offset, show_all=show_all, **kwargs)
        return self._post_process(result, **kwargs)

    # 计算三重指数平滑移动平均线（Triple Exponential Moving Average）
    def trix(self, length=None, signal=None, scalar=None, drift=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        result = trix(close=close, length=length, signal=signal, scalar=scalar, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # 计算真实强度指数（True Strength Index）
    def tsi(self, fast=None, slow=None, drift=None, mamode=None, offset=None, **kwargs):
        close = self._get_column(kwargs.pop("close", "close"))
        result = tsi(close=close, fast=fast, slow=slow, drift=drift, mamode=mamode, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # 计算终极指标（Ultimate Oscillator）
    def uo(self, fast=None, medium=None, slow=None, fast_w=None, medium_w=None, slow_w=None, drift=None, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = uo(high=high, low=low, close=close, fast=fast, medium=medium, slow=slow, fast_w=fast_w, medium_w=medium_w, slow_w=slow_w, drift=drift, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # 计算威廉指标（Williams %R）
    def willr(self, length=None, percentage=True, offset=None, **kwargs):
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        result = willr(high=high, low=low, close=close, length=length, percentage=percentage, offset=offset, **kwargs)
        return self._post_process(result, **kwargs)

    # Overlap
    # 计算 Arnaud Legoux 移动平均线（ALMA）
    def alma(self, length=None, sigma=None, distribution_offset=None, offset=None, **kwargs):
        # 获取要计算 ALMA 的列，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 ALMA 函数计算结果
        result = alma(close=close, length=length, sigma=sigma, distribution_offset=distribution_offset, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算双指数移动平均线（DEMA）
    def dema(self, length=None, offset=None, **kwargs):
        # 获取要计算 DEMA 的列，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 DEMA 函数计算结果
        result = dema(close=close, length=length, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算指数移动平均线（EMA）
    def ema(self, length=None, offset=None, **kwargs):
        # 获取要计算 EMA 的列，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 EMA 函数计算结果
        result = ema(close=close, length=length, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算前加权移动平均线（FWMA）
    def fwma(self, length=None, offset=None, **kwargs):
        # 获取要计算 FWMA 的列，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 FWMA 函数计算结果
        result = fwma(close=close, length=length, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算最高价最低价波动范围（High-Low Range）
    def hilo(self, high_length=None, low_length=None, mamode=None, offset=None, **kwargs):
        # 获取高价、低价和收盘价列，默认为 "high"、"low" 和 "close" 列
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 High-Low Range 函数计算结果
        result = hilo(high=high, low=low, close=close, high_length=high_length, low_length=low_length, mamode=mamode, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算最高价与最低价的平均值（High-Low Average 2）
    def hl2(self, offset=None, **kwargs):
        # 获取高价和低价列，默认为 "high" 和 "low" 列
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        # 调用 High-Low Average 2 函数计算结果
        result = hl2(high=high, low=low, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算最高价、最低价和收盘价的均值（High-Low-Close Average 3）
    def hlc3(self, offset=None, **kwargs):
        # 获取高价、低价和收盘价列，默认为 "high"、"low" 和 "close" 列
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 High-Low-Close Average 3 函数计算结果
        result = hlc3(high=high, low=low, close=close, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 HULL 移动平均线（HMA）
    def hma(self, length=None, offset=None, **kwargs):
        # 获取要计算 HMA 的列，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 HMA 函数计算结果
        result = hma(close=close, length=length, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算带权移动平均线（HWMA）
    def hwma(self, na=None, nb=None, nc=None, offset=None, **kwargs):
        # 获取要计算 HWMA 的列，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 HWMA 函数计算结果
        result = hwma(close=close, na=na, nb=nb, nc=nc, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 JURIK Moving Average（JMA）
    def jma(self, length=None, phase=None, offset=None, **kwargs):
        # 获取要计算 JMA 的列，默认为 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 JMA 函数计算结果
        result = jma(close=close, length=length, phase=phase, offset=offset, **kwargs)
        # 对计算结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算Kaufman自适应移动平均线（KAMA），返回处理后的结果
    def kama(self, length=None, fast=None, slow=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用kama函数计算KAMA指标
        result = kama(close=close, length=length, fast=fast, slow=slow, offset=offset, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)

    # 计算一目均衡表（Ichimoku Cloud）指标，返回处理后的结果和云图
    def ichimoku(self, tenkan=None, kijun=None, senkou=None, include_chikou=True, offset=None, **kwargs):
        # 获取最高价、最低价和收盘价列数据
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用ichimoku函数计算一目均衡表指标和云图
        result, span = ichimoku(high=high, low=low, close=close, tenkan=tenkan, kijun=kijun, senkou=senkou, include_chikou=include_chikou, offset=offset, **kwargs)
        # 添加前缀后缀
        self._add_prefix_suffix(result, **kwargs)
        self._add_prefix_suffix(span, **kwargs)
        # 添加结果
        self._append(result, **kwargs)
        # 返回处理后的结果和云图
        return result, span

    # 计算线性回归线指标，返回处理后的结果
    def linreg(self, length=None, offset=None, adjust=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用linreg函数计算线性回归线指标
        result = linreg(close=close, length=length, offset=offset, adjust=adjust, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)

    # 计算McClellan Divergence指标，返回处理后的结果
    def mcgd(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用mcgd函数计算McClellan Divergence指标
        result = mcgd(close=close, length=length, offset=offset, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)

    # 计算中点价格指标，返回处理后的结果
    def midpoint(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用midpoint函数计算中点价格指标
        result = midpoint(close=close, length=length, offset=offset, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)

    # 计算中间价格指标，返回处理后的结果
    def midprice(self, length=None, offset=None, **kwargs):
        # 获取最高价和最低价列数据
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        # 调用midprice函数计算中间价格指标
        result = midprice(high=high, low=low, length=length, offset=offset, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)

    # 计算OHLC4指标，返回处理后的结果
    def ohlc4(self, offset=None, **kwargs):
        # 获取开盘价、最高价、最低价和收盘价列数据
        open_ = self._get_column(kwargs.pop("open", "open"))
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用ohlc4函数计算OHLC4指标
        result = ohlc4(open_=open_, high=high, low=low, close=close, offset=offset, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)

    # 计算平均加权移动平均线（PWMA），返回处理后的结果
    def pwma(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用pwma函数计算PWMA指标
        result = pwma(close=close, length=length, offset=offset, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)

    # 计算指数加权移动平均线（RMA），返回处理后的结果
    def rma(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用rma函数计算RMA指标
        result = rma(close=close, length=length, offset=offset, **kwargs)
        # 处理结果后返回
        return self._post_process(result, **kwargs)
    # 计算简单指数加权移动平均值
    def sinwma(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算简单指数加权移动平均值
        result = sinwma(close=close, length=length, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算简单移动平均值
    def sma(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算简单移动平均值
        result = sma(close=close, length=length, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算平滑移动平均值
    def ssf(self, length=None, poles=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算平滑移动平均值
        result = ssf(close=close, length=length, poles=poles, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算超级趋势指标
    def supertrend(self, length=None, multiplier=None, offset=None, **kwargs):
        # 获取最高价、最低价和收盘价列数据
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算超级趋势指标
        result = supertrend(high=high, low=low, close=close, length=length, multiplier=multiplier, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算带权移动平均值
    def swma(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算带权移动平均值
        result = swma(close=close, length=length, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算三重指数移动平均值
    def t3(self, length=None, a=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算三重指数移动平均值
        result = t3(close=close, length=length, a=a, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算三重指数移动平均值
    def tema(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算三重指数移动平均值
        result = tema(close=close, length=length, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算三角形移动平均值
    def trima(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算三角形移动平均值
        result = trima(close=close, length=length, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算变动强度平均值
    def vidya(self, length=None, offset=None, **kwargs):
        # 获取收盘价列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算变动强度平均值
        result = vidya(close=close, length=length, offset=offset, **kwargs)
        # 后处理���果并返回
        return self._post_process(result, **kwargs)

    # 计算成交量加权平均价
    def vwap(self, anchor=None, offset=None, **kwargs):
        # 获取最高价、最低价、收盘价和成交量列数据
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))

        # 如果数据不是按日期排序的，则重新设置索引
        if not self.datetime_ordered:
            volume.index = self._df.index

        # 计算成交量加权平均价
        result = vwap(high=high, low=low, close=close, volume=volume, anchor=anchor, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)
    # VWMA (Volume Weighted Moving Average)
    def vwma(self, volume=None, length=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为"volume"
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 VWMA，并返回结果
        result = vwma(close=close, volume=volume, length=length, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    # WCP (Weighted Close Price)
    def wcp(self, offset=None, **kwargs):
        # 获取最高价列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取最低价列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 WCP，并返回结果
        result = wcp(high=high, low=low, close=close, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    # WMA (Weighted Moving Average)
    def wma(self, length=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 WMA，并返回结果
        result = wma(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    # ZLMA (Zero Lag Moving Average)
    def zlma(self, length=None, mamode=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 ZLMA，并返回结果
        result = zlma(close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    # Performance Metrics
    def log_return(self, length=None, cumulative=False, percent=False, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算对数收益率，并返回结果
        result = log_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    def percent_return(self, length=None, cumulative=False, percent=False, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算百分比收益率，并返回结果
        result = percent_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    # Statistical Metrics
    def entropy(self, length=None, base=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算信息熵，并返回结果
        result = entropy(close=close, length=length, base=base, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    def kurtosis(self, length=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算峰度，并返回结果
        result = kurtosis(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    def mad(self, length=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算绝对中位差，并返回结果
        result = mad(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)

    def median(self, length=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算中位数，并返回结果
        result = median(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理，并返回
        return self._post_process(result, **kwargs)
    # 计算给定数据列的分位数
    def quantile(self, length=None, q=None, offset=None, **kwargs):
        # 获取数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 quantile 函数计算分位数
        result = quantile(close=close, length=length, q=q, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算给定数据列的偏度
    def skew(self, length=None, offset=None, **kwargs):
        # 获取数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 skew 函数计算偏度
        result = skew(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算给定数据列的标准差
    def stdev(self, length=None, offset=None, **kwargs):
        # 获取数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 stdev 函数计算标准差
        result = stdev(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算给定数据列的累积标准差
    def tos_stdevall(self, length=None, stds=None, offset=None, **kwargs):
        # 获取数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 tos_stdevall 函数计算累积标准差
        result = tos_stdevall(close=close, length=length, stds=stds, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算给定数据列的方差
    def variance(self, length=None, offset=None, **kwargs):
        # 获取数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 variance 函数计算方差
        result = variance(close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算给定数据列的 Z 分数
    def zscore(self, length=None, std=None, offset=None, **kwargs):
        # 获取数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 zscore 函数计算 Z 分数
        result = zscore(close=close, length=length, std=std, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 趋势指标：计算 ADX（平均趋向指数）
    def adx(self, length=None, lensig=None, mamode=None, scalar=None, drift=None, offset=None, **kwargs):
        # 获取高价数据列，默认为“high”
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价数据列，默认为“low”
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 adx 函数计算 ADX
        result = adx(high=high, low=low, close=close, length=length, lensig=lensig, mamode=mamode, scalar=scalar, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 趋势指标：计算 AMAT（自适应移动平均趋势）
    def amat(self, fast=None, slow=None, mamode=None, lookback=None, offset=None, **kwargs):
        # 获取收盘价数据列，默认为“close”
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 amat 函数计算 AMAT
        result = amat(close=close, fast=fast, slow=slow, mamode=mamode, lookback=lookback, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 趋势指标：计算 Aroon 指标
    def aroon(self, length=None, scalar=None, offset=None, **kwargs):
        # 获取高价数据列，默认为“high”
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价数据列，默认为“low”
        low = self._get_column(kwargs.pop("low", "low"))
        # 调用 aroon 函数计算 Aroon 指标
        result = aroon(high=high, low=low, length=length, scalar=scalar, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 对数据进行 chop 操作，返回处理后的结果
    def chop(self, length=None, atr_length=None, scalar=None, drift=None, offset=None, **kwargs):
        # 获取 high 列数据
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 low 列数据
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取 close 列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 chop 函数处理数据
        result = chop(high=high, low=low, close=close, length=length, atr_length=atr_length, scalar=scalar, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 对数据进行 cksp 操作，返回处理后的结果
    def cksp(self, p=None, x=None, q=None, mamode=None, offset=None, **kwargs):
        # 获取 high 列数据
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 low 列数据
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取 close 列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 cksp 函数处理数据
        result = cksp(high=high, low=low, close=close, p=p, x=x, q=q, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 对数据进行 decay 操作，返回处理后的结果
    def decay(self, length=None, mode=None, offset=None, **kwargs):
        # 获取 close 列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 decay 函数处理数据
        result = decay(close=close, length=length, mode=mode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 对数据进行 decreasing 操作，返回处理后的结果
    def decreasing(self, length=None, strict=None, asint=None, offset=None, **kwargs):
        # 获取 close 列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 decreasing 函数处理数据
        result = decreasing(close=close, length=length, strict=strict, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 对数据进行 dpo 操作，返回处理后的结果
    def dpo(self, length=None, centered=True, offset=None, **kwargs):
        # 获取 close 列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 dpo 函数处理数据
        result = dpo(close=close, length=length, centered=centered, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 对数据进行 increasing 操作，返回处理后的结果
    def increasing(self, length=None, strict=None, asint=None, offset=None, **kwargs):
        # 获取 close 列数据
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 increasing 函数处理数据
        result = increasing(close=close, length=length, strict=strict, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 对数据进行 long_run 操作，返回处理后的结果
    def long_run(self, fast=None, slow=None, length=None, offset=None, **kwargs):
        # 如果 fast 和 slow 都为 None，则直接返回原始数据
        if fast is None and slow is None:
            return self._df
        else:
            # 调用 long_run 函数处理数据
            result = long_run(fast=fast, slow=slow, length=length, offset=offset, **kwargs)
            # 对结果进行后处理
            return self._post_process(result, **kwargs)

    # 对数据进行 psar 操作，返回处理后的结果
    def psar(self, af0=None, af=None, max_af=None, offset=None, **kwargs):
        # 获取 high 列数据
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取 low 列数据
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取 close 列数据
        close = self._get_column(kwargs.pop("close", None))
        # 调用 psar 函数处理数据
        result = psar(high=high, low=low, close=close, af0=af0, af=af, max_af=max_af, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算 Qstick 指标
    def qstick(self, length=None, offset=None, **kwargs):
        # 获取 "open" 列
        open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 Qstick 指标
        result = qstick(open_=open_, close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Short Run 指标
    def short_run(self, fast=None, slow=None, length=None, offset=None, **kwargs):
        # 如果 fast 和 slow 都为 None，则返回原始数据
        if fast is None and slow is None:
            return self._df
        else:
            # 计算 Short Run 指标
            result = short_run(fast=fast, slow=slow, length=length, offset=offset, **kwargs)
            # 对结果进行后处理
            return self._post_process(result, **kwargs)

    # 计算 Supertrend 指标
    def supertrend(self, period=None, multiplier=None, mamode=None, drift=None, offset=None, **kwargs):
        # 获取 "high"、"low"、"close" 列
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 Supertrend 指标
        result = supertrend(high=high, low=low, close=close, period=period, multiplier=multiplier, mamode=mamode, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Tsignals 指标
    def tsignals(self, trend=None, asbool=None, trend_reset=None, trend_offset=None, offset=None, **kwargs):
        # 如果 trend 为 None，则返回原始数据
        if trend is None:
            return self._df
        else:
            # 计算 Tsignals 指标
            result = tsignals(trend, asbool=asbool, trend_offset=trend_offset, trend_reset=trend_reset, offset=offset, **kwargs)
            # 对结果进行后处理
            return self._post_process(result, **kwargs)

    # 计算 TTM Trend 指标
    def ttm_trend(self, length=None, offset=None, **kwargs):
        # 获取 "high"、"low"、"close" 列
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 TTM Trend 指标
        result = ttm_trend(high=high, low=low, close=close, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 VHF 指标
    def vhf(self, length=None, drift=None, offset=None, **kwargs):
        # 获取 "close" 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 VHF 指标
        result = vhf(close=close, length=length, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Vortex 指标
    def vortex(self, drift=None, offset=None, **kwargs):
        # 获取 "high"、"low"、"close" 列
        high = self._get_column(kwargs.pop("high", "high"))
        low = self._get_column(kwargs.pop("low", "low"))
        close = self._get_column(kwargs.pop("close", "close"))
        # 计算 Vortex 指标
        result = vortex(high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Xsignals 指标
    def xsignals(self, signal=None, xa=None, xb=None, above=None, long=None, asbool=None, trend_reset=None, trend_offset=None, offset=None, **kwargs):
        # 如果 signal 为 None，则返回原始数据
        if signal is None:
            return self._df
        else:
            # 计算 Xsignals 指标
            result = xsignals(signal=signal, xa=xa, xb=xb, above=above, long=long, asbool=asbool, trend_offset=trend_offset, trend_reset=trend_reset, offset=offset, **kwargs)
            # 对结果进行后处理
            return self._post_process(result, **kwargs)
    # 判断两个时间序列中的元素是否在指定的位置关系上
    def above(self, asint=True, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'a' 列
        a = self._get_column(kwargs.pop("close", "a"))
        # 获取参数中指定的列，默认为 'b' 列
        b = self._get_column(kwargs.pop("close", "b"))
        # 调用 above 函数判断 a 是否在 b 之上
        result = above(series_a=a, series_b=b, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 判断时间序列中的元素是否在给定值之上
    def above_value(self, value=None, asint=True, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'a' 列
        a = self._get_column(kwargs.pop("close", "a"))
        # 调用 above_value 函数判断 a 是否在给定值之上
        result = above_value(series_a=a, value=value, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 判断两个时间序列中的元素是否在指定的位置关系下
    def below(self, asint=True, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'a' 列
        a = self._get_column(kwargs.pop("close", "a"))
        # 获取参数中指定的列，默认为 'b' 列
        b = self._get_column(kwargs.pop("close", "b"))
        # 调用 below 函数判断 a 是否在 b 之下
        result = below(series_a=a, series_b=b, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 判断时间序列中的元素是否在给定值之下
    def below_value(self, value=None, asint=True, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'a' 列
        a = self._get_column(kwargs.pop("close", "a"))
        # 调用 below_value 函数判断 a 是否在给定值之下
        result = below_value(series_a=a, value=value, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 判断两个时间序列是否在指定位置交叉
    def cross(self, above=True, asint=True, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'a' 列
        a = self._get_column(kwargs.pop("close", "a"))
        # 获取参数中指定的列，默认为 'b' 列
        b = self._get_column(kwargs.pop("close", "b"))
        # 调用 cross 函数判断两个时间序列是否交叉
        result = cross(series_a=a, series_b=b, above=above, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 判断时间序列是否与给定值交叉
    def cross_value(self, value=None, above=True, asint=True, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'a' 列
        a = self._get_column(kwargs.pop("close", "a"))
        # 调用 cross_value 函数判断时间序列是否与给定值交叉
        result = cross_value(series_a=a, value=value, above=above, asint=asint, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算异常偏差
    def aberration(self, length=None, atr_length=None, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'high' 列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取参数中指定的列，默认为 'low' 列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取参数中指定的列，默认为 'close' 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 aberration 函数计算异常偏差
        result = aberration(high=high, low=low, close=close, length=length, atr_length=atr_length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算累积/分配线
    def accbands(self, length=None, c=None, mamode=None, offset=None, **kwargs):
        # 获取参数中指定的列，默认为 'high' 列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取参数中指定的列，默认为 'low' 列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取参数中指定的列，默认为 'close' 列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 accbands 函数计算累积/分配线
        result = accbands(high=high, low=low, close=close, length=length, c=c, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算 ATR（Average True Range）指标
    def atr(self, length=None, mamode=None, offset=None, **kwargs):
        # 获取高价列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 atr 函数计算 ATR 指标
        result = atr(high=high, low=low, close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Bollinger Bands（布林带）指标
    def bbands(self, length=None, std=None, mamode=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close  = self._get_column(kwargs.pop("close", "close"))
        # 调用 bbands 函数计算布林带指标
        result = bbands(close=close, length=length, std=std, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Donchian Channel（唐奇安通道）指标
    def donchian(self, lower_length=None, upper_length=None, offset=None, **kwargs):
        # 获取高价列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 调用 donchian 函数计算唐奇安通道指标
        result = donchian(high=high, low=low, lower_length=lower_length, upper_length=upper_length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Hull Moving Average（HMA）指标
    def hwc(self, na=None, nb=None, nc=None, nd=None, scalar=None, offset=None, **kwargs):
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 hwc 函数计算 HMA 指标
        result = hwc(close=close, na=na, nb=nb, nc=nc, nd=nd, scalar=scalar, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Keltner Channel（凯尔特纳通道）指标
    def kc(self, length=None, scalar=None, mamode=None, offset=None, **kwargs):
        # 获取高价列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 kc 函数计算凯尔特纳通道指标
        result = kc(high=high, low=low, close=close, length=length, scalar=scalar, mamode=mamode, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Mass Index（质量指数）指标
    def massi(self, fast=None, slow=None, offset=None, **kwargs):
        # 获取高价列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 调用 massi 函数计算质量指数指标
        result = massi(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算 Normalized Average True Range（NATR）指标
    def natr(self, length=None, mamode=None, scalar=None, offset=None, **kwargs):
        # 获取��价列，默认为"high"
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列，默认为"low"
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列，默认为"close"
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用 natr 函数计算 NATR 指标
        result = natr(high=high, low=low, close=close, length=length, mamode=mamode, scalar=scalar, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算概率分布函数（Probability Distribution Function）
    def pdist(self, drift=None, offset=None, **kwargs):
        # 获取开盘价列
        open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取最高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取最低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用pdist函数计算概率分布函数
        result = pdist(open_=open_, high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算相对波动指数（Relative Volatility Index）
    def rvi(self, length=None, scalar=None, refined=None, thirds=None, mamode=None, drift=None, offset=None, **kwargs):
        # 获取最高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取最低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用rvi函数计算相对波动指数
        result = rvi(high=high, low=low, close=close, length=length, scalar=scalar, refined=refined, thirds=thirds, mamode=mamode, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算热力指标（Thermo）
    def thermo(self, long=None, short= None, length=None, mamode=None, drift=None, offset=None, **kwargs):
        # 获取最高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取最低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 调用thermo函数计算热力指标
        result = thermo(high=high, low=low, long=long, short=short, length=length, mamode=mamode, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算真实波幅（True Range）
    def true_range(self, drift=None, offset=None, **kwargs):
        # 获取最高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取最低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用true_range函数计算真实波幅
        result = true_range(high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算Ulcer Index
    def ui(self, length=None, scalar=None, offset=None, **kwargs):
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 调用ui函数计算Ulcer Index
        result = ui(close=close, length=length, scalar=scalar, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    #  Accumulation/Distribution指标
    def ad(self, open_=None, signed=True, offset=None, **kwargs):
        # 如果指定了开盘价列，则获取开盘价列
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取最高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取最低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 调用ad函数计算Accumulation/Distribution指标
        result = ad(high=high, low=low, close=close, volume=volume, open_=open_, signed=signed, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算累积/派发线指标（Accumulation/Distribution Oscillator，ADOSC）
    def adosc(self, open_=None, fast=None, slow=None, signed=True, offset=None, **kwargs):
        # 如果提供了开盘价，则获取开盘价列；否则使用默认列名
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算ADOSC指标
        result = adosc(high=high, low=low, close=close, volume=volume, open_=open_, fast=fast, slow=slow, signed=signed, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算累积/派发成本指标（Accumulation/Distribution Cost，AOBV）
    def aobv(self, fast=None, slow=None, mamode=None, max_lookback=None, min_lookback=None, offset=None, **kwargs):
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算AOBV指标
        result = aobv(close=close, volume=volume, fast=fast, slow=slow, mamode=mamode, max_lookback=max_lookback, min_lookback=min_lookback, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算资金流量指标（Chaikin Money Flow，CMF）
    def cmf(self, open_=None, length=None, offset=None, **kwargs):
        # 如果提供了开盘价，则获取开盘价列；否则使用默认列名
        if open_ is not None:
            open_ = self._get_column(kwargs.pop("open", "open"))
        # 获取高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算CMF指标
        result = cmf(high=high, low=low, close=close, volume=volume, open_=open_, length=length, offset=offset, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算能量潮指标（Elder Force Index，EFI）
    def efi(self, length=None, mamode=None, offset=None, drift=None, **kwargs):
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算EFI指标
        result = efi(close=close, volume=volume, length=length, offset=offset, mamode=mamode, drift=drift, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)

    # 计算简单动量指标（Ease of Movement，EOM）
    def eom(self, length=None, divisor=None, offset=None, drift=None, **kwargs):
        # 获取高价列
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算EOM指标
        result = eom(high=high, low=low, close=close, volume=volume, length=length, divisor=divisor, offset=offset, drift=drift, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
    # 计算 KVO（Volume Oscillator）指标
    def kvo(self, fast=None, slow=None, length_sig=None, mamode=None, offset=None, drift=None, **kwargs):
        # 获取高价列，默认为 'high'
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列，默认为 'low'
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 KVO 指标
        result = kvo(high=high, low=low, close=close, volume=volume, fast=fast, slow=slow, length_sig=length_sig, mamode=mamode, offset=offset, drift=drift, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算 MFI（Money Flow Index）指标
    def mfi(self, length=None, drift=None, offset=None, **kwargs):
        # 获取高价列，默认为 'high'
        high = self._get_column(kwargs.pop("high", "high"))
        # 获取低价列，默认为 'low'
        low = self._get_column(kwargs.pop("low", "low"))
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 MFI 指标
        result = mfi(high=high, low=low, close=close, volume=volume, length=length, drift=drift, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算 NVI（Negative Volume Index）指标
    def nvi(self, length=None, initial=None, signed=True, offset=None, **kwargs):
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 NVI 指标
        result = nvi(close=close, volume=volume, length=length, initial=initial, signed=signed, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算 OBV（On Balance Volume）指标
    def obv(self, offset=None, **kwargs):
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 OBV 指标
        result = obv(close=close, volume=volume, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算 PVI（Positive Volume Index）指标
    def pvi(self, length=None, initial=None, signed=True, offset=None, **kwargs):
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 PVI 指标
        result = pvi(close=close, volume=volume, length=length, initial=initial, signed=signed, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算 PVol（Price Volume Trend）指标
    def pvol(self, volume=None, offset=None, **kwargs):
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 PVol 指标
        result = pvol(close=close, volume=volume, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算 PVR（Price Volume Rank）指标
    def pvr(self, **kwargs):
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 PVR 指标
        result = pvr(close=close, volume=volume)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)

    # 计算 PVT（Price Volume Trend）指标
    def pvt(self, offset=None, **kwargs):
        # 获取收盘价列，默认为 'close'
        close = self._get_column(kwargs.pop("close", "close"))
        # 获取成交量列，默认为 'volume'
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 计算 PVT 指标
        result = pvt(close=close, volume=volume, offset=offset, **kwargs)
        # 后处理结果并返回
        return self._post_process(result, **kwargs)
    # 定义一个方法 vp，用于计算成交量价比
    def vp(self, width=None, percent=None, **kwargs):
        # 获取列名，如果未提供则使用默认列名
        close = self._get_column(kwargs.pop("close", "close"))
        volume = self._get_column(kwargs.pop("volume", "volume"))
        # 调用 vp 函数计算成交量价比
        result = vp(close=close, volume=volume, width=width, percent=percent, **kwargs)
        # 对结果进行后处理
        return self._post_process(result, **kwargs)
```