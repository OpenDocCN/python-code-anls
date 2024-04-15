# `.\pandas-ta\pandas_ta\utils\data\alphavantage.py`

```
# -*- coding: utf-8 -*-
# 导入 DataFrame 类
from pandas import DataFrame
# 导入 Imports 对象，RATE 对象，version 对象
from pandas_ta import Imports, RATE, version

# 定义 av 函数，获取 alphaVantage 数据
def av(ticker: str, **kwargs):
    # 打印关键字参数 kwargs
    print(f"[!] kwargs: {kwargs}")
    # 从 kwargs 中弹出 verbose 参数，默认为 False
    verbose = kwargs.pop("verbose", False)
    # 从 kwargs 中弹出 kind 参数，默认为 "history"
    kind = kwargs.pop("kind", "history")
    # 将 kind 转换为小写
    kind = kind.lower()
    # 从 kwargs 中弹出 interval 参数，默认为 "D"
    interval = kwargs.pop("interval", "D")
    # 从 kwargs 中弹出 show 参数，默认为 None
    show = kwargs.pop("show", None)
    # 从 kwargs 中弹出 last 参数，但是没有使用到

    # 如果 ticker 不为空且是字符串类型，则将其转换为大写，否则为 None
    ticker = ticker.upper() if ticker is not None and isinstance(ticker, str) else None

    # 如果 alphaVantage-api 可用且 ticker 不为空
    if Imports["alphaVantage-api"] and ticker is not None:
        # 导入 alphaVantageAPI 模块并重命名为 AV
        import alphaVantageAPI as AV
        # 定义 AVC 字典，包含 API 密钥和其他参数
        AVC = {"api_key": "YOUR API KEY", "clean": True, "export": False, "output_size": "full", "premium": False}
        # 从 kwargs 中获取 av_kwargs 参数，如果不存在则使用 AVC
        _config = kwargs.pop("av_kwargs", AVC)
        # 创建 AlphaVantage 对象 av
        av = AV.AlphaVantage(**_config)

        # 从 kwargs 中获取 period 参数，默认为 av.output_size
        period = kwargs.pop("period", av.output_size)

        # 定义 _all 列表和 div 变量
        _all, div = ["all"], "=" * 53 # Max div width is 80

        # 如果 kind 在 _all 列表中或者 verbose 为真，则执行下面的代码
        if kind in _all or verbose: pass

        # 如果 kind 在 _all 列表或者 ["history", "h"] 列表中
        if kind in _all + ["history", "h"]:
            # 如果 verbose 为真
            if verbose:
                # 打印信息，显示 Pandas TA 版本和 alphaVantage-api
                print("\n====  Chart History       " + div + f"\n[*] Pandas TA v{version} & alphaVantage-api")
                # 打印下载信息，显示下载的股票信息和时间间隔
                print(f"[+] Downloading {ticker}[{interval}:{period}] from {av.API_NAME} (https://www.alphavantage.co/)")
            # 获取股票数据并保存到 df 变量中
            df = av.data(ticker, interval)
            # 设置 DataFrame 的名称为 ticker
            df.name = ticker
            # 如果 show 不为空且是正整数且大于 0
            if show is not None and isinstance(show, int) and show > 0:
                # 打印 DataFrame 最后几行数据
                print(f"\n{df.name}\n{df.tail(show)}\n")
            # 返回 DataFrame 对象
            return df

    # 如果上述条件都不满足，则返回一个空的 DataFrame 对象
    return DataFrame()
```  
```