# `.\pandas-ta\pandas_ta\candles\ha.py`

```
# 设置文件编码为 UTF-8
# 导入 DataFrame 类
from pandas import DataFrame
# 导入 get_offset 和 verify_series 函数
from pandas_ta.utils import get_offset, verify_series


# 定义 Heikin Ashi 函数
def ha(open_, high, low, close, offset=None, **kwargs):
    """Candle Type: Heikin Ashi"""
    # 验证参数，确保它们都是 pd.Series 类型
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    # 获取偏移量
    offset = get_offset(offset)

    # 计算结果
    m = close.size
    # 创建 DataFrame 对象，包含 HA_open、HA_high、HA_low 和 HA_close 列
    df = DataFrame({
        "HA_open": 0.5 * (open_.iloc[0] + close.iloc[0]),
        "HA_high": high,
        "HA_low": low,
        "HA_close": 0.25 * (open_ + high + low + close),
    })

    # 计算 HA_open 列
    for i in range(1, m):
        df["HA_open"][i] = 0.5 * (df["HA_open"][i - 1] + df["HA_close"][i - 1])

    # 计算 HA_high 和 HA_low 列
    df["HA_high"] = df[["HA_open", "HA_high", "HA_close"]].max(axis=1)
    df["HA_low"] = df[["HA_open", "HA_low", "HA_close"]].min(axis=1)

    # 处理偏移
    if offset != 0:
        df = df.shift(offset)

    # 处理填充
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # 命名和分类
    df.name = "Heikin-Ashi"
    df.category = "candles"

    return df


# 设置 Heikin Ashi 函数的文档字符串
ha.__doc__ = \
"""Heikin Ashi Candles (HA)

The Heikin-Ashi technique averages price data to create a Japanese
candlestick chart that filters out market noise. Heikin-Ashi charts,
developed by Munehisa Homma in the 1700s, share some characteristics
with standard candlestick charts but differ based on the values used
to create each candle. Instead of using the open, high, low, and close
like standard candlestick charts, the Heikin-Ashi technique uses a
modified formula based on two-period averages. This gives the chart a
smoother appearance, making it easier to spots trends and reversals,
but also obscures gaps and some price data.

Sources:
    https://www.investopedia.com/terms/h/heikinashi.asp

Calculation:
    HA_OPEN[0] = (open[0] + close[0]) / 2
    HA_CLOSE = (open[0] + high[0] + low[0] + close[0]) / 4

    for i > 1 in df.index:
        HA_OPEN = (HA_OPEN[i−1] + HA_CLOSE[i−1]) / 2

    HA_HIGH = MAX(HA_OPEN, HA_HIGH, HA_CLOSE)
    HA_LOW = MIN(HA_OPEN, HA_LOW, HA_CLOSE)

    How to Calculate Heikin-Ashi

    Use one period to create the first Heikin-Ashi (HA) candle, using
    the formulas. For example use the high, low, open, and close to
    create the first HA close price. Use the open and close to create
    the first HA open. The high of the period will be the first HA high,
    and the low will be the first HA low. With the first HA calculated,
    it is now possible to continue computing the HA candles per the formulas.
​​
Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:

"""
    # 创建一个 Pandas 数据帧，该数据帧包含 ha_open、ha_high、ha_low、ha_close 四列。
# 导入所需模块
import requests
import json

# 定义函数 `get_weather`，接收城市名作为参数，返回该城市的天气信息
def get_weather(city):
    # 构建 API 请求 URL，使用城市名作为查询参数
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=YOUR_API_KEY"
    # 发送 GET 请求获取天气数据
    response = requests.get(url)
    # 解析 JSON 格式的响应数据
    data = response.json()
    # 返回解析后的天气数据
    return data

# 定义函数 `main`，程序的入口点
def main():
    # 输入要查询的城市
    city = input("Enter city name: ")
    # 调用 `get_weather` 函数获取城市天气信息
    weather_data = get_weather(city)
    # 打印城市天气信息
    print(json.dumps(weather_data, indent=4))

# 如果该脚本直接运行，则执行 `main` 函数
if __name__ == "__main__":
    main()
```