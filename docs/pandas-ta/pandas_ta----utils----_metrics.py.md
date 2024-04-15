# `.\pandas-ta\pandas_ta\utils\_metrics.py`

```
# -*- coding: utf-8 -*-
# 引入需要的类型提示模块
from typing import Tuple

# 引入 numpy 库的 log、nan、sqrt 函数
from numpy import log as npLog
from numpy import nan as npNaN
from numpy import sqrt as npSqrt

# 引入 pandas 库的 Series、Timedelta 类
from pandas import Series, Timedelta

# 引入自定义的核心模块中的 verify_series 函数
from ._core import verify_series

# 引入自定义的时间模块中的 total_time 函数
from ._time import total_time

# 引入自定义的数学模块中的 linear_regression、log_geometric_mean 函数
from ._math import linear_regression, log_geometric_mean

# 引入 pandas_ta 库中的 RATE 常量
from pandas_ta import RATE

# 引入 pandas_ta 库中的性能模块中的 drawdown、log_return、percent_return 函数
from pandas_ta.performance import drawdown, log_return, percent_return


def cagr(close: Series) -> float:
    """复合年增长率

    Args:
        close (pd.Series): 'close' 的序列

    >>> result = ta.cagr(df.close)
    """
    # 确保 close 是有效的 Series
    close = verify_series(close)
    # 获取序列的起始和结束值
    start, end = close.iloc[0], close.iloc[-1]
    # 计算并返回复合年增长率
    return ((end / start) ** (1 / total_time(close))) - 1


def calmar_ratio(close: Series, method: str = "percent", years: int = 3) -> float:
    """Calmar 比率通常是在过去三年内的最大回撤率的百分比。

    Args:
        close (pd.Series): 'close' 的序列
        method (str): 最大回撤计算选项：'dollar'、'percent'、'log'。默认值：'dollar'
        years (int): 使用的年数。默认值：3

    >>> result = ta.calmar_ratio(close, method="percent", years=3)
    """
    if years <= 0:
        # 如果年数参数小于等于 0，则打印错误消息并返回
        print(f"[!] calmar_ratio 'years' 参数必须大于零。")
        return
    # 确保 close 是有效的 Series
    close = verify_series(close)

    # 获取指定年数前的日期
    n_years_ago = close.index[-1] - Timedelta(days=365.25 * years)
    # 从指定日期开始截取序列
    close = close[close.index > n_years_ago]

    # 计算并返回 Calmar 比率
    return cagr(close) / max_drawdown(close, method=method)


def downside_deviation(returns: Series, benchmark_rate: float = 0.0, tf: str = "years") -> float:
    """Sortino 比率的下行偏差。假定基准利率是年化的。根据数据中每年的期数进行调整。

    Args:
        returns (pd.Series): 'returns' 的序列
        benchmark_rate (float): 要使用的基准利率。默认值：0.0
        tf (str): 时间范围选项：'days'、'weeks'、'months'、'years'。默认值：'years'

    >>> result = ta.downside_deviation(returns, benchmark_rate=0.0, tf="years")
    """
    # 用于去年化基准利率和年化结果的天数
    # 确保 returns 是有效的 Series
    returns = verify_series(returns)
    days_per_year = returns.shape[0] / total_time(returns, tf)

    # 调整后的基准利率
    adjusted_benchmark_rate = ((1 + benchmark_rate) ** (1 / days_per_year)) - 1

    # 计算下行偏差
    downside = adjusted_benchmark_rate - returns
    downside_sum_of_squares = (downside[downside > 0] ** 2).sum()
    downside_deviation = npSqrt(downside_sum_of_squares / (returns.shape[0] - 1))
    return downside_deviation * npSqrt(days_per_year)


def jensens_alpha(returns: Series, benchmark_returns: Series) -> float:
    """一系列与基准的 Jensen's 'Alpha'。

    Args:
        returns (pd.Series): 'returns' 的序列
        benchmark_returns (pd.Series): 'benchmark_returns' 的序列

    >>> result = ta.jensens_alpha(returns, benchmark_returns)
    """
    # 确保 returns 是有效的 Series
    returns = verify_series(returns)
    # 确保 benchmark_returns 是一个 Series 对象，并返回一个验证过的 Series 对象
    benchmark_returns = verify_series(benchmark_returns)
    
    # 对 benchmark_returns 进行插值处理，使得其中的缺失值被填充，原地修改（不创建新对象）
    benchmark_returns.interpolate(inplace=True)
    
    # 对 benchmark_returns 和 returns 进行线性回归分析，并返回其中的斜率参数（截距参数不返回）
    return linear_regression(benchmark_returns, returns)["a"]
def log_max_drawdown(close: Series) -> float:
    """Calculate the logarithmic maximum drawdown of a series.

    Args:
        close (pd.Series): Series of 'close' prices.

    >>> result = ta.log_max_drawdown(close)
    """
    # Ensure 'close' series is valid
    close = verify_series(close)
    # Calculate the log return from the beginning to the end of the series
    log_return = npLog(close.iloc[-1]) - npLog(close.iloc[0])
    # Return the log return minus the maximum drawdown of the series
    return log_return - max_drawdown(close, method="log")


def max_drawdown(close: Series, method:str = None, all:bool = False) -> float:
    """Calculate the maximum drawdown from a series of closing prices.

    Args:
        close (pd.Series): Series of 'close' prices.
        method (str): Options for calculating max drawdown: 'dollar', 'percent', 'log'.
            Default: 'dollar'.
        all (bool): If True, return all three methods as a dictionary.
            Default: False.

    >>> result = ta.max_drawdown(close, method="dollar", all=False)
    """
    # Ensure 'close' series is valid
    close = verify_series(close)
    # Calculate the maximum drawdown using the drawdown function
    max_dd = drawdown(close).max()

    # Dictionary containing maximum drawdown values for different methods
    max_dd_ = {
        "dollar": max_dd.iloc[0],
        "percent": max_dd.iloc[1],
        "log": max_dd.iloc[2]
    }
    # If 'all' is True, return all methods as a dictionary
    if all: return max_dd_

    # If 'method' is specified and valid, return the corresponding value
    if isinstance(method, str) and method in max_dd_.keys():
        return max_dd_[method]
    # Default to dollar method if 'method' is not specified or invalid
    return max_dd_["dollar"]


def optimal_leverage(
        close: Series, benchmark_rate: float = 0.0,
        period: Tuple[float, int] = RATE["TRADING_DAYS_PER_YEAR"],
        log: bool = False, capital: float = 1., **kwargs
    ) -> float:
    """Calculate the optimal leverage of a series. WARNING: Incomplete. Do NOT use.

    Args:
        close (pd.Series): Series of 'close' prices.
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0.
        period (int, float): Period to use to calculate Mean Annual Return and
            Annual Standard Deviation.
            Default: None or the default sharpe_ratio.period().
        log (bool): If True, calculates log_return. Otherwise, it returns
            percent_return. Default: False.

    >>> result = ta.optimal_leverage(close, benchmark_rate=0.0, log=False)
    """
    # Ensure 'close' series is valid
    close = verify_series(close)

    # Check if use_cagr is specified
    use_cagr = kwargs.pop("use_cagr", False)
    # Calculate returns based on whether log or percent return is specified
    returns = percent_return(close=close) if not log else log_return(close=close)

    # Calculate period mean and standard deviation
    period_mu = period * returns.mean()
    period_std = npSqrt(period) * returns.std()

    # Calculate mean excess return and optimal leverage
    mean_excess_return = period_mu - benchmark_rate
    opt_leverage = (period_std ** -2) * mean_excess_return

    # Calculate the amount based on capital and optimal leverage
    amount = int(capital * opt_leverage)
    return amount


def pure_profit_score(close: Series) -> Tuple[float, int]:
    """Calculate the pure profit score of a series.

    Args:
        close (pd.Series): Series of 'close' prices.

    >>> result = ta.pure_profit_score(df.close)
    """
    # Ensure 'close' series is valid
    close = verify_series(close)
    # Create a series of zeros with the same index as 'close'
    close_index = Series(0, index=close.reset_index().index)

    # Calculate the linear regression 'r' value
    r = linear_regression(close_index, close)["r"]
    # If 'r' value is not NaN, return 'r' multiplied by CAGR of 'close'
    if r is not npNaN:
        return r * cagr(close)
    # Otherwise, return 0
    return 0
# 计算夏普比率的函数
def sharpe_ratio(close: Series, benchmark_rate: float = 0.0, log: bool = False, use_cagr: bool = False, period: int = RATE["TRADING_DAYS_PER_YEAR"]) -> float:
    """Sharpe Ratio of a series.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False
        use_cagr (bool): Use cagr - benchmark_rate instead. Default: False
        period (int, float): Period to use to calculate Mean Annual Return and
            Annual Standard Deviation.
            Default: RATE["TRADING_DAYS_PER_YEAR"] (currently 252)

    >>> result = ta.sharpe_ratio(close, benchmark_rate=0.0, log=False)
    """
    # 验证输入的数据是否为Series类型
    close = verify_series(close)
    # 根据log参数选择计算百分比收益率或对数收益率
    returns = percent_return(close=close) if not log else log_return(close=close)

    # 如果使用cagr参数，则返回年复合增长率与波动率的比率
    if use_cagr:
        return cagr(close) / volatility(close, returns, log=log)
    else:
        # 计算期望收益率和标准差
        period_mu = period * returns.mean()
        period_std = npSqrt(period) * returns.std()
        return (period_mu - benchmark_rate) / period_std


# 计算Sortino比率的函数
def sortino_ratio(close: Series, benchmark_rate: float = 0.0, log: bool = False) -> float:
    """Sortino Ratio of a series.

    Args:
        close (pd.Series): Series of 'close's
        benchmark_rate (float): Benchmark Rate to use. Default: 0.0
        log (bool): If True, calculates log_return. Otherwise it returns
            percent_return. Default: False

    >>> result = ta.sortino_ratio(close, benchmark_rate=0.0, log=False)
    """
    # 验证输入的数据是否为Series类型
    close = verify_series(close)
    # 根据log参数选择计算百分比收益率或对数收益率
    returns = percent_return(close=close) if not log else log_return(close=close)

    # 计算Sortino比率
    result  = cagr(close) - benchmark_rate
    result /= downside_deviation(returns)
    return result


# 计算波动率的函数
def volatility(close: Series, tf: str = "years", returns: bool = False, log: bool = False, **kwargs) -> float:
    """Volatility of a series. Default: 'years'

    Args:
        close (pd.Series): Series of 'close's
        tf (str): Time Frame options: 'days', 'weeks', 'months', and 'years'.
            Default: 'years'
        returns (bool): If True, then it replace the close Series with the user
            defined Series; typically user generated returns or percent returns
            or log returns. Default: False
        log (bool): If True, calculates log_return. Otherwise it calculates
            percent_return. Default: False

    >>> result = ta.volatility(close, tf="years", returns=False, log=False, **kwargs)
    """
    # 验证输入的数据是否为Series类型
    close = verify_series(close)

    # 如果returns参数为False，则计算百分比收益率或对数收益率
    if not returns:
        returns = percent_return(close=close) if not log else log_return(close=close)
    else:
        returns = close

    # 计算对数几何平均值的标准差作为波动率
    returns = log_geometric_mean(returns).std()
    return returns
```