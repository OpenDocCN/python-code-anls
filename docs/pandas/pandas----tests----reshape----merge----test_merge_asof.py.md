# `D:\src\scipysrc\pandas\pandas\tests\reshape\merge\test_merge_asof.py`

```
import datetime  # 导入datetime模块，用于处理日期和时间

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行单元测试

import pandas.util._test_decorators as td  # 导入pandas的测试装饰器模块

import pandas as pd  # 导入pandas库，用于数据处理和分析
from pandas import (  # 从pandas库中导入以下函数和类
    Index,  # 索引类，用于处理轴标签
    Timedelta,  # 时间增量类，表示两个时间点之间的差
    merge_asof,  # 按照时间序列合并数据
    option_context,  # 临时设置选项的上下文管理器
    to_datetime,  # 将输入转换为datetime类型
)
import pandas._testing as tm  # 导入pandas测试模块，用于编写测试函数

from pandas.core.reshape.merge import MergeError  # 导入pandas的合并错误类


class TestAsOfMerge:
    def prep_data(self, df, dedupe=False):
        if dedupe:
            df = df.drop_duplicates(["time", "ticker"], keep="last").reset_index(
                drop=True
            )
        df.time = to_datetime(df.time)  # 将df的time列转换为datetime类型
        return df

    @pytest.fixture  # 使用pytest的fixture装饰器，定义测试用例的前置条件
    # 定义一个方法用于生成交易数据的DataFrame
    def trades(self):
        # 创建包含交易数据的DataFrame，包括时间、股票代码、价格、数量和市场中心
        df = pd.DataFrame(
            [
                ["20160525 13:30:00.023", "MSFT", "51.9500", "75", "NASDAQ"],
                ["20160525 13:30:00.038", "MSFT", "51.9500", "155", "NASDAQ"],
                ["20160525 13:30:00.048", "GOOG", "720.7700", "100", "NASDAQ"],
                ["20160525 13:30:00.048", "GOOG", "720.9200", "100", "NASDAQ"],
                ["20160525 13:30:00.048", "GOOG", "720.9300", "200", "NASDAQ"],
                ["20160525 13:30:00.048", "GOOG", "720.9300", "300", "NASDAQ"],
                ["20160525 13:30:00.048", "GOOG", "720.9300", "600", "NASDAQ"],
                ["20160525 13:30:00.048", "GOOG", "720.9300", "44", "NASDAQ"],
                ["20160525 13:30:00.074", "AAPL", "98.6700", "478343", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.6700", "478343", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.6600", "6", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.6500", "30", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.6500", "75", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.6500", "20", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.6500", "35", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.6500", "10", "NASDAQ"],
                ["20160525 13:30:00.075", "AAPL", "98.5500", "6", "ARCA"],
                ["20160525 13:30:00.075", "AAPL", "98.5500", "6", "ARCA"],
                ["20160525 13:30:00.076", "AAPL", "98.5600", "1000", "ARCA"],
                ["20160525 13:30:00.076", "AAPL", "98.5600", "200", "ARCA"],
                ["20160525 13:30:00.076", "AAPL", "98.5600", "300", "ARCA"],
                ["20160525 13:30:00.076", "AAPL", "98.5600", "400", "ARCA"],
                ["20160525 13:30:00.076", "AAPL", "98.5600", "600", "ARCA"],
                ["20160525 13:30:00.076", "AAPL", "98.5600", "200", "ARCA"],
                ["20160525 13:30:00.078", "MSFT", "51.9500", "783", "NASDAQ"],
                ["20160525 13:30:00.078", "MSFT", "51.9500", "100", "NASDAQ"],
                ["20160525 13:30:00.078", "MSFT", "51.9500", "100", "NASDAQ"],
            ],
            columns="time,ticker,price,quantity,marketCenter".split(","),
        )
        # 将价格列转换为浮点数类型
        df["price"] = df["price"].astype("float64")
        # 将数量列转换为整数类型
        df["quantity"] = df["quantity"].astype("int64")
        # 调用类中的数据预处理方法，返回处理后的数据
        return self.prep_data(df)

    @pytest.fixture
    def quotes(self):
        # 创建包含交易数据的 pandas DataFrame 对象
        df = pd.DataFrame(
            [
                ["20160525 13:30:00.023", "GOOG", "720.50", "720.93"],
                ["20160525 13:30:00.023", "MSFT", "51.95", "51.95"],
                ["20160525 13:30:00.041", "MSFT", "51.95", "51.95"],
                ["20160525 13:30:00.048", "GOOG", "720.50", "720.93"],
                ["20160525 13:30:00.048", "GOOG", "720.50", "720.93"],
                ["20160525 13:30:00.048", "GOOG", "720.50", "720.93"],
                ["20160525 13:30:00.048", "GOOG", "720.50", "720.93"],
                ["20160525 13:30:00.072", "GOOG", "720.50", "720.88"],
                ["20160525 13:30:00.075", "AAPL", "98.55", "98.56"],
                ["20160525 13:30:00.076", "AAPL", "98.55", "98.56"],
                ["20160525 13:30:00.076", "AAPL", "98.55", "98.56"],
                ["20160525 13:30:00.076", "AAPL", "98.55", "98.56"],
                ["20160525 13:30:00.078", "MSFT", "51.95", "51.95"],
                ["20160525 13:30:00.078", "MSFT", "51.95", "51.95"],
                ["20160525 13:30:00.078", "MSFT", "51.95", "51.95"],
                ["20160525 13:30:00.078", "MSFT", "51.92", "51.95"],
            ],
            columns="time,ticker,bid,ask".split(","),
        )
        # 将 'bid' 列和 'ask' 列的数据类型转换为 float64
        df["bid"] = df["bid"].astype("float64")
        df["ask"] = df["ask"].astype("float64")
        # 调用 prep_data 方法处理数据，并返回结果
        return self.prep_data(df, dedupe=True)

    @pytest.fixture
    def test_examples1(self):
        """doc-string examples"""
        # 创建左右两个包含数据的 pandas DataFrame 对象
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})

        # 创建预期的合并后的结果 DataFrame 对象
        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [1, 3, 7]}
        )

        # 调用 merge_asof 方法执行左连接操作，并返回结果
        result = merge_asof(left, right, on="a")
        # 使用测试工具函数检查结果是否与预期一致
        tm.assert_frame_equal(result, expected)

    def test_examples3(self):
        """doc-string examples"""
        # GH14887

        # 创建左右两个包含数据的 pandas DataFrame 对象
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})

        # 创建预期的合并后的结果 DataFrame 对象
        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [1, 6, np.nan]}
        )

        # 调用 merge_asof 方法执行左连接操作，并返回结果
        result = merge_asof(left, right, on="a", direction="forward")
        # 使用测试工具函数检查结果是否与预期一致
        tm.assert_frame_equal(result, expected)

    def test_examples4(self):
        """doc-string examples"""
        # GH14887

        # 创建左右两个包含数据的 pandas DataFrame 对象
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})

        # 创建预期的合并后的结果 DataFrame 对象
        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [1, 6, 7]}
        )

        # 调用 merge_asof 方法执行左连接操作，并返回结果
        result = merge_asof(left, right, on="a", direction="nearest")
        # 使用测试工具函数检查结果是否与预期一致
        tm.assert_frame_equal(result, expected)
    # 定义测试方法，用于测试 merge_asof 函数的基本功能
    def test_basic(self, trades, asof, quotes):
        # 设置预期结果为 asof 数据框
        expected = asof

        # 调用 merge_asof 函数，按时间 ('time') 和股票代码 ('ticker') 进行合并
        result = merge_asof(trades, quotes, on="time", by="ticker")
        # 使用测试框架检查结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试 merge_asof 函数对分类数据的处理
    def test_basic_categorical(self, trades, asof, quotes):
        # 设置预期结果为 asof 数据框
        expected = asof
        # 将 trades、quotes 和 expected 数据框中的 ticker 列转换为分类类型
        trades.ticker = trades.ticker.astype("category")
        quotes.ticker = quotes.ticker.astype("category")
        expected.ticker = expected.ticker.astype("category")

        # 调用 merge_asof 函数，按时间 ('time') 和股票代码 ('ticker') 进行合并
        result = merge_asof(trades, quotes, on="time", by="ticker")
        # 使用测试框架检查结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试 merge_asof 函数左侧数据框索引的用例
    def test_basic_left_index(self, trades, asof, quotes):
        # GH14253
        # 设置预期结果为 asof 数据框
        expected = asof
        # 将 trades 数据框按时间 ('time') 列设置为索引
        trades = trades.set_index("time")

        # 调用 merge_asof 函数，将 trades 数据框和 quotes 数据框按左侧索引、右侧时间 ('time') 列，以及股票代码 ('ticker') 进行合并
        result = merge_asof(
            trades, quotes, left_index=True, right_on="time", by="ticker"
        )
        # 将预期结果的索引设置为结果数据框的索引
        expected.index = result.index
        # 调整预期结果的列顺序，使时间列出现在左侧数据框列后面
        expected = expected[result.columns]
        # 使用测试框架检查结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试 merge_asof 函数右侧数据框索引的用例
    def test_basic_right_index(self, trades, asof, quotes):
        # 设置预期结果为 asof 数据框
        expected = asof
        # 将 quotes 数据框按时间 ('time') 列设置为索引
        quotes = quotes.set_index("time")

        # 调用 merge_asof 函数，将 trades 数据框和 quotes 数据框按左侧时间 ('time') 列、右侧索引进行合并
        result = merge_asof(
            trades, quotes, left_on="time", right_index=True, by="ticker"
        )
        # 使用测试框架检查结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试 merge_asof 函数左右侧数据框都设置索引的用例
    def test_basic_left_index_right_index(self, trades, asof, quotes):
        # 将预期结果的时间列设置为索引
        expected = asof.set_index("time")
        # 将 trades 和 quotes 数据框的时间列设置为索引
        trades = trades.set_index("time")
        quotes = quotes.set_index("time")

        # 调用 merge_asof 函数，按左右侧时间索引和股票代码 ('ticker') 进行合并
        result = merge_asof(
            trades, quotes, left_index=True, right_index=True, by="ticker"
        )
        # 使用测试框架检查结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试 merge_asof 函数处理左侧数据框多级索引的用例
    def test_multi_index_left(self, trades, quotes):
        # MultiIndex 是不允许的
        trades = trades.set_index(["time", "price"])
        quotes = quotes.set_index("time")
        # 使用 pytest 框架验证合并时会抛出 MergeError 异常，匹配错误消息 "left can only have one index"
        with pytest.raises(MergeError, match="left can only have one index"):
            merge_asof(trades, quotes, left_index=True, right_index=True)

    # 定义测试方法，用于测试 merge_asof 函数处理右侧数据框多级索引的用例
    def test_multi_index_right(self, trades, quotes):
        # MultiIndex 是不允许的
        trades = trades.set_index("time")
        quotes = quotes.set_index(["time", "bid"])
        # 使用 pytest 框架验证合并时会抛出 MergeError 异常，匹配错误消息 "right can only have one index"
        with pytest.raises(MergeError, match="right can only have one index"):
            merge_asof(trades, quotes, left_index=True, right_index=True)

    # 定义测试方法，用于测试 merge_asof 函数同时使用 "on" 参数和左侧索引的用例
    def test_on_and_index_left_on(self, trades, quotes):
        # 禁止同时使用 "on" 参数和左侧索引
        trades = trades.set_index("time")
        quotes = quotes.set_index("time")
        msg = 'Can only pass argument "left_on" OR "left_index" not both.'
        # 使用 pytest 框架验证合并时会抛出 MergeError 异常，匹配错误消息 msg
        with pytest.raises(MergeError, match=msg):
            merge_asof(
                trades, quotes, left_on="price", left_index=True, right_index=True
            )
    # 测试函数：使用左表时间作为索引，右表时间作为索引，同时指定 right_on 和 right_index，预期会抛出 MergeError 异常并匹配指定错误信息
    def test_on_and_index_right_on(self, trades, quotes):
        # 将 trades DataFrame 的 "time" 列设为索引
        trades = trades.set_index("time")
        # 将 quotes DataFrame 的 "time" 列设为索引
        quotes = quotes.set_index("time")
        # 定义错误信息
        msg = 'Can only pass argument "right_on" OR "right_index" not both.'
        # 使用 pytest 检查是否会抛出 MergeError 异常，并匹配错误信息
        with pytest.raises(MergeError, match=msg):
            # 执行 merge_asof 函数，同时指定 right_on 和 right_index，期望抛出异常
            merge_asof(
                trades, quotes, right_on="bid", left_index=True, right_index=True
            )

    # 测试函数：基本测试，使用 left_by 和 right_by 进行合并
    def test_basic_left_by_right_by(self, trades, asof, quotes):
        # GH14253
        # 预期的合并结果是 asof
        expected = asof

        # 执行 merge_asof 函数，使用 trades 的 "time" 列和 quotes 的 "time" 列进行合并，
        # trades 使用 "ticker" 列，quotes 使用 "ticker" 列作为辅助键
        result = merge_asof(
            trades, quotes, on="time", left_by="ticker", right_by="ticker"
        )
        # 比较结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试 right_by 参数缺失的情况
    def test_missing_right_by(self, trades, asof, quotes):
        # 预期的合并结果是 asof
        expected = asof

        # 过滤 quotes DataFrame，排除 ticker 为 "MSFT" 的行
        q = quotes[quotes.ticker != "MSFT"]
        # 执行 merge_asof 函数，使用 trades 的 "time" 列和 quotes 的 "time" 列进行合并，
        # trades 使用 "ticker" 列作为辅助键，quotes 默认使用 "ticker" 列作为辅助键
        result = merge_asof(trades, q, on="time", by="ticker")
        # 将预期结果中 ticker 为 "MSFT" 的行的 "bid" 和 "ask" 列设为 NaN
        expected.loc[expected.ticker == "MSFT", ["bid", "ask"]] = np.nan
        # 比较结果和预期是否相等
        tm.assert_frame_equal(result, expected)
    def test_multiby(self):
        # GH13936
        # 创建包含交易数据的DataFrame，包括时间、股票代码、交易所、价格和数量
        trades = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.046",
                        "20160525 13:30:00.048",
                        "20160525 13:30:00.050",
                    ]
                ),
                "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
                "exch": ["ARCA", "NSDQ", "NSDQ", "BATS", "NSDQ"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.00],
                "quantity": [75, 155, 100, 100, 100],
            },
            columns=["time", "ticker", "exch", "price", "quantity"],
        )

        # 创建包含报价数据的DataFrame，包括时间、股票代码、交易所、买入价和卖出价
        quotes = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.030",
                        "20160525 13:30:00.041",
                        "20160525 13:30:00.045",
                        "20160525 13:30:00.049",
                    ]
                ),
                "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL"],
                "exch": ["BATS", "NSDQ", "ARCA", "ARCA", "NSDQ", "ARCA"],
                "bid": [720.51, 51.95, 51.97, 51.99, 720.50, 97.99],
                "ask": [720.92, 51.96, 51.98, 52.00, 720.93, 98.01],
            },
            columns=["time", "ticker", "exch", "bid", "ask"],
        )

        # 创建预期的合并后的DataFrame，包含时间、股票代码、交易所、价格、数量、买入价和卖出价
        expected = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.046",
                        "20160525 13:30:00.048",
                        "20160525 13:30:00.050",
                    ]
                ),
                "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
                "exch": ["ARCA", "NSDQ", "NSDQ", "BATS", "NSDQ"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.00],
                "quantity": [75, 155, 100, 100, 100],
                "bid": [np.nan, 51.95, 720.50, 720.51, np.nan],
                "ask": [np.nan, 51.96, 720.93, 720.92, np.nan],
            },
            columns=["time", "ticker", "exch", "price", "quantity", "bid", "ask"],
        )

        # 执行合并操作，按时间和股票代码、交易所进行合并
        result = merge_asof(trades, quotes, on="time", by=["ticker", "exch"])
        
        # 使用测试框架检查结果DataFrame是否与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["object", "string"])
    # 定义一个测试函数，用于测试合并异步数据的情况，特别是不同类型的数据
    def test_multiby_heterogeneous_types(self, dtype):
        # GH13936
        # 创建包含交易数据的DataFrame，包括时间、股票代码、交易所、价格和数量
        trades = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.046",
                        "20160525 13:30:00.048",
                        "20160525 13:30:00.050",
                    ]
                ),
                "ticker": [0, 0, 1, 1, 2],
                "exch": ["ARCA", "NSDQ", "NSDQ", "BATS", "NSDQ"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.00],
                "quantity": [75, 155, 100, 100, 100],
            },
            columns=["time", "ticker", "exch", "price", "quantity"],
        )
        # 将指定列转换为特定数据类型（如整数或字符串）
        trades = trades.astype({"ticker": dtype, "exch": dtype})

        # 创建包含报价数据的DataFrame，包括时间、股票代码、交易所、买入价和卖出价
        quotes = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.030",
                        "20160525 13:30:00.041",
                        "20160525 13:30:00.045",
                        "20160525 13:30:00.049",
                    ]
                ),
                "ticker": [1, 0, 0, 0, 1, 2],
                "exch": ["BATS", "NSDQ", "ARCA", "ARCA", "NSDQ", "ARCA"],
                "bid": [720.51, 51.95, 51.97, 51.99, 720.50, 97.99],
                "ask": [720.92, 51.96, 51.98, 52.00, 720.93, 98.01],
            },
            columns=["time", "ticker", "exch", "bid", "ask"],
        )
        # 将指定列转换为特定数据类型（如整数或字符串）
        quotes = quotes.astype({"ticker": dtype, "exch": dtype})

        # 创建期望的DataFrame，包括时间、股票代码、交易所、价格、数量、买入价和卖出价
        expected = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.023",
                        "20160525 13:30:00.046",
                        "20160525 13:30:00.048",
                        "20160525 13:30:00.050",
                    ]
                ),
                "ticker": [0, 0, 1, 1, 2],
                "exch": ["ARCA", "NSDQ", "NSDQ", "BATS", "NSDQ"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.00],
                "quantity": [75, 155, 100, 100, 100],
                "bid": [np.nan, 51.95, 720.50, 720.51, np.nan],
                "ask": [np.nan, 51.96, 720.93, 720.92, np.nan],
            },
            columns=["time", "ticker", "exch", "price", "quantity", "bid", "ask"],
        )
        # 将指定列转换为特定数据类型（如整数或字符串）
        expected = expected.astype({"ticker": dtype, "exch": dtype})

        # 执行合并操作，按照时间和股票代码、交易所作为键，合并交易和报价数据
        result = merge_asof(trades, quotes, on="time", by=["ticker", "exch"])
        # 使用测试框架检查合并结果是否与期望结果相等
        tm.assert_frame_equal(result, expected)
    def test_mismatched_index_dtype(self):
        # 定义测试函数，测试处理索引数据类型不匹配的情况
        left = pd.DataFrame(
            [
                [to_datetime("20160602"), 1, "a"],
                [to_datetime("20160602"), 2, "a"],
                [to_datetime("20160603"), 1, "b"],
                [to_datetime("20160603"), 2, "b"],
            ],
            columns=["time", "k1", "k2"],
        ).set_index("time")
        # 将左表的索引数据类型修改为与右表不同的类型
        left.index = left.index - pd.Timestamp(0)

        right = pd.DataFrame(
            [
                [to_datetime("20160502"), 1, "a", 1.0],
                [to_datetime("20160502"), 2, "a", 2.0],
                [to_datetime("20160503"), 1, "b", 3.0],
                [to_datetime("20160503"), 2, "b", 4.0],
            ],
            columns=["time", "k1", "k2", "value"],
        ).set_index("time")

        msg = "incompatible merge keys"
        # 使用 pytest 的断言，期望捕获 MergeError 异常，并检查异常消息
        with pytest.raises(MergeError, match=msg):
            # 调用 merge_asof 函数，左右表使用索引进行合并，合并键为 ["k1", "k2"]
            merge_asof(left, right, left_index=True, right_index=True, by=["k1", "k2"])

    def test_multiby_indexed(self):
        # GH15676 测试案例
        # 创建左表 DataFrame，设置 time 列为索引
        left = pd.DataFrame(
            [
                [to_datetime("20160602"), 1, "a"],
                [to_datetime("20160602"), 2, "a"],
                [to_datetime("20160603"), 1, "b"],
                [to_datetime("20160603"), 2, "b"],
            ],
            columns=["time", "k1", "k2"],
        ).set_index("time")

        # 创建右表 DataFrame，设置 time 列为索引
        right = pd.DataFrame(
            [
                [to_datetime("20160502"), 1, "a", 1.0],
                [to_datetime("20160502"), 2, "a", 2.0],
                [to_datetime("20160503"), 1, "b", 3.0],
                [to_datetime("20160503"), 2, "b", 4.0],
            ],
            columns=["time", "k1", "k2", "value"],
        ).set_index("time")

        # 期望的合并结果 DataFrame，设置 time 列为索引
        expected = pd.DataFrame(
            [
                [to_datetime("20160602"), 1, "a", 1.0],
                [to_datetime("20160602"), 2, "a", 2.0],
                [to_datetime("20160603"), 1, "b", 3.0],
                [to_datetime("20160603"), 2, "b", 4.0],
            ],
            columns=["time", "k1", "k2", "value"],
        ).set_index("time")

        # 调用 merge_asof 函数，左右表使用索引进行合并，合并键为 ["k1", "k2"]
        result = merge_asof(
            left, right, left_index=True, right_index=True, by=["k1", "k2"]
        )

        # 使用 pandas.testing.assert_frame_equal 函数断言预期结果与实际结果是否相等
        tm.assert_frame_equal(expected, result)

        # 使用 pytest 的断言，期望捕获 MergeError 异常，并检查异常消息
        with pytest.raises(
            MergeError, match="left_by and right_by must be the same length"
        ):
            # 调用 merge_asof 函数，左右表使用索引进行合并，但是左右表的合并键长度不一致
            merge_asof(
                left,
                right,
                left_index=True,
                right_index=True,
                left_by=["k1", "k2"],
                right_by=["k1"],
            )
    # 定义一个测试方法，测试在不使用"by"参数的基本情况下的合并操作
    def test_basic_no_by(self, trades, asof, quotes):
        # 创建一个匿名函数，从数据中选择 ticker 为 "MSFT" 的行，并丢弃 "ticker" 列，重新索引
        f = (
            lambda x: x[x.ticker == "MSFT"]
            .drop("ticker", axis=1)
            .reset_index(drop=True)
        )

        # 期望的结果是对 asof 数据应用函数 f
        expected = f(asof)
        # 对 trades 数据应用函数 f
        trades = f(trades)
        # 对 quotes 数据应用函数 f
        quotes = f(quotes)

        # 调用 merge_asof 函数，将 trades 和 quotes 按照 "time" 列进行合并
        result = merge_asof(trades, quotes, on="time")
        # 使用测试框架验证 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，验证合并时使用不兼容的合并键会抛出 MergeError 异常
    def test_valid_join_keys(self, trades, quotes):
        # 定义期望的错误消息正则表达式
        msg = r"incompatible merge keys \[1\] .* must be the same type"

        # 使用 pytest 来检查 merge_asof 函数调用时是否会抛出指定异常，并且匹配指定的错误消息
        with pytest.raises(MergeError, match=msg):
            merge_asof(trades, quotes, left_on="time", right_on="bid", by="ticker")

        # 同样使用 pytest 来检查是否会抛出指定异常，错误消息为 "can only asof on a key for left"
        with pytest.raises(MergeError, match="can only asof on a key for left"):
            merge_asof(trades, quotes, on=["time", "ticker"], by="ticker")

        # 再次使用 pytest 来检查是否会抛出指定异常，错误消息同样为 "can only asof on a key for left"
        with pytest.raises(MergeError, match="can only asof on a key for left"):
            merge_asof(trades, quotes, by="ticker")

    # 定义一个测试方法，测试在存在重复数据的情况下的合并操作
    def test_with_duplicates(self, datapath, trades, quotes, asof):
        # 将 quotes 数据框重复合并，按照 "time" 和 "ticker" 列排序，重新索引
        q = (
            pd.concat([quotes, quotes])
            .sort_values(["time", "ticker"])
            .reset_index(drop=True)
        )
        # 调用 merge_asof 函数，将 trades 和 q 按照 "time" 和 "ticker" 列进行合并
        result = merge_asof(trades, q, on="time", by="ticker")
        # 准备期望的数据结果，通过调用 prep_data 方法生成
        expected = self.prep_data(asof)
        # 使用测试框架验证 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，测试在不指定 "on" 参数的情况下的合并操作
    def test_with_duplicates_no_on(self):
        # 创建两个数据框 df1 和 df2，它们分别包含 "key" 和 "left_val" / "right_val" 列
        df1 = pd.DataFrame({"key": [1, 1, 3], "left_val": [1, 2, 3]})
        df2 = pd.DataFrame({"key": [1, 2, 2], "right_val": [1, 2, 3]})
        # 调用 merge_asof 函数，将 df1 和 df2 按照 "key" 列进行合并
        result = merge_asof(df1, df2, on="key")
        # 准备期望的数据结果，包含 "key"、"left_val" 和 "right_val" 列
        expected = pd.DataFrame(
            {"key": [1, 1, 3], "left_val": [1, 2, 3], "right_val": [1, 1, 3]}
        )
        # 使用测试框架验证 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，验证在传递非布尔类型参数给 "allow_exact_matches" 参数时会抛出异常
    def test_valid_allow_exact_matches(self, trades, quotes):
        # 定义期望的错误消息
        msg = "allow_exact_matches must be boolean, passed foo"

        # 使用 pytest 来检查 merge_asof 函数调用时是否会抛出指定异常，错误消息为 msg
        with pytest.raises(MergeError, match=msg):
            merge_asof(
                trades, quotes, on="time", by="ticker", allow_exact_matches="foo"
            )
    def test_valid_tolerance(self, trades, quotes):
        # 使用 merge_asof 函数将 trades 和 quotes 数据按照时间戳 "time" 和股票代码 "ticker" 进行合并，容忍时间差为 1 秒
        merge_asof(trades, quotes, on="time", by="ticker", tolerance=Timedelta("1s"))

        # 使用 merge_asof 函数将 trades 和 quotes 数据按照整数索引 "index" 和股票代码 "ticker" 进行合并，容忍时间差为 1
        merge_asof(
            trades.reset_index(),
            quotes.reset_index(),
            on="index",
            by="ticker",
            tolerance=1,
        )

        # 设置错误消息匹配模式
        msg = r"incompatible tolerance .*, must be compat with type .*"

        # 检查容差不兼容的情况是否引发 MergeError 异常，并匹配指定的错误消息
        with pytest.raises(MergeError, match=msg):
            merge_asof(trades, quotes, on="time", by="ticker", tolerance=1)

        # 检查容差为浮点数时是否引发 MergeError 异常，并匹配指定的错误消息
        with pytest.raises(MergeError, match=msg):
            merge_asof(
                trades.reset_index(),
                quotes.reset_index(),
                on="index",
                by="ticker",
                tolerance=1.0,
            )

        # 设置容差必须为正数的错误消息
        msg = "tolerance must be positive"

        # 检查容差为负数时间增量时是否引发 MergeError 异常，并匹配指定的错误消息
        with pytest.raises(MergeError, match=msg):
            merge_asof(
                trades, quotes, on="time", by="ticker", tolerance=-Timedelta("1s")
            )

        # 检查容差为负整数时是否引发 MergeError 异常，并匹配指定的错误消息
        with pytest.raises(MergeError, match=msg):
            merge_asof(
                trades.reset_index(),
                quotes.reset_index(),
                on="index",
                by="ticker",
                tolerance=-1,
            )

    def test_non_sorted(self, trades, quotes):
        # 按时间戳 "time" 降序对 trades 和 quotes 进行排序
        trades = trades.sort_values("time", ascending=False)
        quotes = quotes.sort_values("time", ascending=False)

        # 断言 trades 数据的时间戳 "time" 不是单调递增的，预期是假的
        assert not trades.time.is_monotonic_increasing
        # 断言 quotes 数据的时间戳 "time" 不是单调递增的，预期是假的
        assert not quotes.time.is_monotonic_increasing
        # 检查未排序的 keys 引发 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="left keys must be sorted"):
            merge_asof(trades, quotes, on="time", by="ticker")

        # 按时间戳 "time" 升序对 trades 进行排序
        trades = trades.sort_values("time")
        # 断言 trades 数据的时间戳 "time" 是单调递增的，预期是真的
        assert trades.time.is_monotonic_increasing
        # 断言 quotes 数据的时间戳 "time" 不是单调递增的，预期是假的
        assert not quotes.time.is_monotonic_increasing
        # 检查未排序的 keys 引发 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="right keys must be sorted"):
            merge_asof(trades, quotes, on="time", by="ticker")

        # 按时间戳 "time" 升序对 quotes 进行排序
        quotes = quotes.sort_values("time")
        # 断言 trades 数据的时间戳 "time" 是单调递增的，预期是真的
        assert trades.time.is_monotonic_increasing
        # 断言 quotes 数据的时间戳 "time" 是单调递增的，预期是真的

        # 使用 merge_asof 函数将按时间戳 "time" 和股票代码 "ticker" 进行合并，允许存在重复数据
        merge_asof(trades, quotes, on="time", by="ticker")

    @pytest.mark.parametrize(
        "tolerance_ts",
        [Timedelta("1day"), datetime.timedelta(days=1)],
        ids=["Timedelta", "datetime.timedelta"],
    )
    def test_tolerance(self, tolerance_ts, trades, quotes, tolerance):
        # 使用 merge_asof 函数将 trades 和 quotes 数据按时间戳 "time" 和股票代码 "ticker" 进行合并，
        # 允许指定的容差 tolerance_ts
        result = merge_asof(
            trades, quotes, on="time", by="ticker", tolerance=tolerance_ts
        )
        # 预期的合并结果
        expected = tolerance
        # 断言实际结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
    def test_tolerance_forward(self):
        # GH14887
        # 创建左侧DataFrame
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        # 创建右侧DataFrame
        right = pd.DataFrame({"a": [1, 2, 3, 7, 11], "right_val": [1, 2, 3, 7, 11]})
        
        # 预期的合并结果DataFrame
        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [1, np.nan, 11]}
        )
        
        # 执行合并操作，使用前向方向合并，设置容忍度为1
        result = merge_asof(left, right, on="a", direction="forward", tolerance=1)
        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_tolerance_nearest(self):
        # GH14887
        # 创建左侧DataFrame
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        # 创建右侧DataFrame
        right = pd.DataFrame({"a": [1, 2, 3, 7, 11], "right_val": [1, 2, 3, 7, 11]})
        
        # 预期的合并结果DataFrame
        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [1, np.nan, 11]}
        )
        
        # 执行合并操作，使用最近方向合并，设置容忍度为1
        result = merge_asof(left, right, on="a", direction="nearest", tolerance=1)
        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_tolerance_tz(self, unit):
        # GH 14844
        # 创建左侧DataFrame，包含带时区的日期列
        left = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=to_datetime("2016-01-02"),
                    freq="D",
                    periods=5,
                    tz=datetime.timezone.utc,
                    unit=unit,
                ),
                "value1": np.arange(5),
            }
        )
        # 创建右侧DataFrame，包含带时区的日期列
        right = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=to_datetime("2016-01-01"),
                    freq="D",
                    periods=5,
                    tz=datetime.timezone.utc,
                    unit=unit,
                ),
                "value2": list("ABCDE"),
            }
        )
        
        # 执行合并操作，按日期列合并，设置容忍度为1天的时间增量
        result = merge_asof(left, right, on="date", tolerance=Timedelta("1 day"))

        # 预期的合并结果DataFrame
        expected = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=to_datetime("2016-01-02"),
                    freq="D",
                    periods=5,
                    tz=datetime.timezone.utc,
                    unit=unit,
                ),
                "value1": np.arange(5),
                "value2": list("BCDEE"),
            }
        )
        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_tolerance_float(self):
        # GH22981
        # 创建左侧DataFrame，包含浮点数列
        left = pd.DataFrame({"a": [1.1, 3.5, 10.9], "left_val": ["a", "b", "c"]})
        # 创建右侧DataFrame，包含浮点数列
        right = pd.DataFrame(
            {"a": [1.0, 2.5, 3.3, 7.5, 11.5], "right_val": [1.0, 2.5, 3.3, 7.5, 11.5]}
        )
        
        # 预期的合并结果DataFrame
        expected = pd.DataFrame(
            {
                "a": [1.1, 3.5, 10.9],
                "left_val": ["a", "b", "c"],
                "right_val": [1, 3.3, np.nan],
            }
        )
        
        # 执行合并操作，使用最近方向合并，设置容忍度为0.5
        result = merge_asof(left, right, on="a", direction="nearest", tolerance=0.5)
        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)
    # 定义测试方法，用于测试在给定交易数据、报价数据和容差情况下的索引容忍性
    def test_index_tolerance(self, trades, quotes, tolerance):
        # GH 15135
        # 生成期望的DataFrame，将其索引设置为"time"
        expected = tolerance.set_index("time")
        # 将交易数据的索引设置为"time"
        trades = trades.set_index("time")
        # 将报价数据的索引设置为"time"
        quotes = quotes.set_index("time")

        # 执行asof合并操作，基于左右DataFrame的索引和ticker列，使用1天的时间容差
        result = merge_asof(
            trades,
            quotes,
            left_index=True,
            right_index=True,
            by="ticker",
            tolerance=Timedelta("1day"),
        )
        # 断言合并后的结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试在允许精确匹配但方向向前的情况下的合并操作
    def test_allow_exact_matches_forward(self):
        # GH14887

        # 创建左侧DataFrame
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        # 创建右侧DataFrame
        right = pd.DataFrame({"a": [1, 2, 3, 7, 11], "right_val": [1, 2, 3, 7, 11]})

        # 创建期望的DataFrame
        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [2, 7, 11]}
        )

        # 执行asof合并操作，基于"a"列，向前方向合并，不允许精确匹配
        result = merge_asof(
            left, right, on="a", direction="forward", allow_exact_matches=False
        )
        # 断言合并后的结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试在允许精确匹配但最近邻的情况下的合并操作
    def test_allow_exact_matches_nearest(self):
        # GH14887

        # 创建左侧DataFrame
        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        # 创建右侧DataFrame
        right = pd.DataFrame({"a": [1, 2, 3, 7, 11], "right_val": [1, 2, 3, 7, 11]})

        # 创建期望的DataFrame
        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [2, 3, 11]}
        )

        # 执行asof合并操作，基于"a"列，最近邻方向合并，不允许精确匹配
        result = merge_asof(
            left, right, on="a", direction="nearest", allow_exact_matches=False
        )
        # 断言合并后的结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)
    # 测试确保在合并数据帧时，允许精确匹配和容忍度的设置
    def test_allow_exact_matches_and_tolerance2(self):
        # GH 13695: GitHub issue reference

        # 创建第一个数据帧 df1，包含一个时间列和一个用户名列
        df1 = pd.DataFrame(
            {"time": to_datetime(["2016-07-15 13:30:00.030"]), "username": ["bob"]}
        )

        # 创建第二个数据帧 df2，包含时间列和版本号列
        df2 = pd.DataFrame(
            {
                "time": to_datetime(
                    ["2016-07-15 13:30:00.000", "2016-07-15 13:30:00.030"]
                ),
                "version": [1, 2],
            }
        )

        # 使用 merge_asof 函数将 df1 和 df2 按照时间列 'time' 合并
        result = merge_asof(df1, df2, on="time")

        # 创建预期的结果数据帧 expected，保证时间匹配时版本号为 2
        expected = pd.DataFrame(
            {
                "time": to_datetime(["2016-07-15 13:30:00.030"]),
                "username": ["bob"],
                "version": [2],
            }
        )

        # 断言合并的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)

        # 再次使用 merge_asof 函数，但禁用精确匹配，期望版本号为 1
        result = merge_asof(df1, df2, on="time", allow_exact_matches=False)
        expected = pd.DataFrame(
            {
                "time": to_datetime(["2016-07-15 13:30:00.030"]),
                "username": ["bob"],
                "version": [1],
            }
        )
        tm.assert_frame_equal(result, expected)

        # 第三次使用 merge_asof 函数，设置容忍度为 10 毫秒，期望版本号为 NaN
        result = merge_asof(
            df1,
            df2,
            on="time",
            allow_exact_matches=False,
            tolerance=Timedelta("10ms"),
        )
        expected = pd.DataFrame(
            {
                "time": to_datetime(["2016-07-15 13:30:00.030"]),
                "username": ["bob"],
                "version": [np.nan],
            }
        )
        tm.assert_frame_equal(result, expected)

    # 另一个测试函数，测试允许精确匹配和容忍度的设置
    def test_allow_exact_matches_and_tolerance3(self):
        # GH 13709: GitHub issue reference

        # 创建第一个数据帧 df1，包含两行，每行时间相同但用户名不同
        df1 = pd.DataFrame(
            {
                "time": to_datetime(
                    ["2016-07-15 13:30:00.030", "2016-07-15 13:30:00.030"]
                ),
                "username": ["bob", "charlie"],
            }
        )

        # 创建第二个数据帧 df2，包含时间列和版本号列
        df2 = pd.DataFrame(
            {
                "time": to_datetime(
                    ["2016-07-15 13:30:00.000", "2016-07-15 13:30:00.030"]
                ),
                "version": [1, 2],
            }
        )

        # 使用 merge_asof 函数，禁用精确匹配，并设置容忍度为 10 毫秒
        result = merge_asof(
            df1,
            df2,
            on="time",
            allow_exact_matches=False,
            tolerance=Timedelta("10ms"),
        )

        # 创建预期的结果数据帧 expected，时间匹配但版本号无法匹配，均为 NaN
        expected = pd.DataFrame(
            {
                "time": to_datetime(
                    ["2016-07-15 13:30:00.030", "2016-07-15 13:30:00.030"]
                ),
                "username": ["bob", "charlie"],
                "version": [np.nan, np.nan],
            }
        )

        # 断言合并的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)
    def test_allow_exact_matches_and_tolerance_forward(self):
        # 测试函数，验证在向前方向上允许精确匹配和容差的行为

        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        # 创建左侧数据框，包含列"a"和"left_val"

        right = pd.DataFrame({"a": [1, 3, 4, 6, 11], "right_val": [1, 3, 4, 6, 11]})
        # 创建右侧数据框，包含列"a"和"right_val"

        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [np.nan, 6, 11]}
        )
        # 创建期望的结果数据框，包含列"a"、"left_val"和"right_val"

        result = merge_asof(
            left,
            right,
            on="a",
            direction="forward",
            allow_exact_matches=False,
            tolerance=1,
        )
        # 使用merge_asof函数将左右数据框按照列"a"合并，向前方向合并，不允许精确匹配，容差为1

        tm.assert_frame_equal(result, expected)
        # 使用tm.assert_frame_equal断言函数检查result和expected是否相等

    def test_allow_exact_matches_and_tolerance_nearest(self):
        # 测试函数，验证在最近方向上允许精确匹配和容差的行为

        left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
        # 创建左侧数据框，包含列"a"和"left_val"

        right = pd.DataFrame({"a": [1, 3, 4, 6, 11], "right_val": [1, 3, 4, 7, 11]})
        # 创建右侧数据框，包含列"a"和"right_val"

        expected = pd.DataFrame(
            {"a": [1, 5, 10], "left_val": ["a", "b", "c"], "right_val": [np.nan, 4, 11]}
        )
        # 创建期望的结果数据框，包含列"a"、"left_val"和"right_val"

        result = merge_asof(
            left,
            right,
            on="a",
            direction="nearest",
            allow_exact_matches=False,
            tolerance=1,
        )
        # 使用merge_asof函数将左右数据框按照列"a"合并，最近方向合并，不允许精确匹配，容差为1

        tm.assert_frame_equal(result, expected)
        # 使用tm.assert_frame_equal断言函数检查result和expected是否相等

    def test_forward_by(self):
        # 测试函数，验证按照额外的列"b"在向前方向上的合并行为

        left = pd.DataFrame(
            {
                "a": [1, 5, 10, 12, 15],
                "b": ["X", "X", "Y", "Z", "Y"],
                "left_val": ["a", "b", "c", "d", "e"],
            }
        )
        # 创建左侧数据框，包含列"a"、"b"和"left_val"

        right = pd.DataFrame(
            {
                "a": [1, 6, 11, 15, 16],
                "b": ["X", "Z", "Y", "Z", "Y"],
                "right_val": [1, 6, 11, 15, 16],
            }
        )
        # 创建右侧数据框，包含列"a"、"b"和"right_val"

        expected = pd.DataFrame(
            {
                "a": [1, 5, 10, 12, 15],
                "b": ["X", "X", "Y", "Z", "Y"],
                "left_val": ["a", "b", "c", "d", "e"],
                "right_val": [1, np.nan, 11, 15, 16],
            }
        )
        # 创建期望的结果数据框，包含列"a"、"b"、"left_val"和"right_val"

        result = merge_asof(left, right, on="a", by="b", direction="forward")
        # 使用merge_asof函数将左右数据框按照列"a"和"b"合并，按照"b"分组，在向前方向上合并

        tm.assert_frame_equal(result, expected)
        # 使用tm.assert_frame_equal断言函数检查result和expected是否相等

    def test_nearest_by(self):
        # 测试函数，验证按照额外的列"b"在最近方向上的合并行为

        left = pd.DataFrame(
            {
                "a": [1, 5, 10, 12, 15],
                "b": ["X", "X", "Z", "Z", "Y"],
                "left_val": ["a", "b", "c", "d", "e"],
            }
        )
        # 创建左侧数据框，包含列"a"、"b"和"left_val"

        right = pd.DataFrame(
            {
                "a": [1, 6, 11, 15, 16],
                "b": ["X", "Z", "Z", "Z", "Y"],
                "right_val": [1, 6, 11, 15, 16],
            }
        )
        # 创建右侧数据框，包含列"a"、"b"和"right_val"

        expected = pd.DataFrame(
            {
                "a": [1, 5, 10, 12, 15],
                "b": ["X", "X", "Z", "Z", "Y"],
                "left_val": ["a", "b", "c", "d", "e"],
                "right_val": [1, 1, 11, 11, 16],
            }
        )
        # 创建期望的结果数据框，包含列"a"、"b"、"left_val"和"right_val"

        result = merge_asof(left, right, on="a", by="b", direction="nearest")
        # 使用merge_asof函数将左右数据框按照列"a"和"b"合并，按照"b"分组，在最近方向上合并

        tm.assert_frame_equal(result, expected)
        # 使用tm.assert_frame_equal断言函数检查result和expected是否相等
    def test_by_int(self):
        # 根据数据类型进行特殊化处理，验证其正确性
        df1 = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.020",
                        "20160525 13:30:00.030",
                        "20160525 13:30:00.040",
                        "20160525 13:30:00.050",
                        "20160525 13:30:00.060",
                    ]
                ),
                "key": [1, 2, 1, 3, 2],
                "value1": [1.1, 1.2, 1.3, 1.4, 1.5],
            },
            columns=["time", "key", "value1"],
        )

        df2 = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.015",
                        "20160525 13:30:00.020",
                        "20160525 13:30:00.025",
                        "20160525 13:30:00.035",
                        "20160525 13:30:00.040",
                        "20160525 13:30:00.055",
                        "20160525 13:30:00.060",
                        "20160525 13:30:00.065",
                    ]
                ),
                "key": [2, 1, 1, 3, 2, 1, 2, 3],
                "value2": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
            },
            columns=["time", "key", "value2"],
        )

        # 使用merge_asof函数将df1和df2按时间和键值合并
        result = merge_asof(df1, df2, on="time", by="key")

        expected = pd.DataFrame(
            {
                "time": to_datetime(
                    [
                        "20160525 13:30:00.020",
                        "20160525 13:30:00.030",
                        "20160525 13:30:00.040",
                        "20160525 13:30:00.050",
                        "20160525 13:30:00.060",
                    ]
                ),
                "key": [1, 2, 1, 3, 2],
                "value1": [1.1, 1.2, 1.3, 1.4, 1.5],
                "value2": [2.2, 2.1, 2.3, 2.4, 2.7],
            },
            columns=["time", "key", "value1", "value2"],
        )

        # 使用assert_frame_equal函数验证结果和期望的DataFrame是否相等
        tm.assert_frame_equal(result, expected)
    def test_on_float(self):
        # 测试处理浮点数的情况

        # 创建第一个数据框，包含价格和对应的股票代码，按价格排序
        df1 = pd.DataFrame(
            {
                "price": [5.01, 0.0023, 25.13, 340.05, 30.78, 1040.90, 0.0078],
                "symbol": list("ABCDEFG"),
            },
            columns=["symbol", "price"],
        )

        # 创建第二个数据框，包含价格和最小价格变动值
        df2 = pd.DataFrame(
            {"price": [0.0, 1.0, 100.0], "mpv": [0.0001, 0.01, 0.05]},
            columns=["price", "mpv"],
        )

        # 根据价格对第一个数据框排序，并重置索引
        df1 = df1.sort_values("price").reset_index(drop=True)

        # 调用 merge_asof 函数进行数据框的合并操作
        result = merge_asof(df1, df2, on="price")

        # 创建预期结果的数据框
        expected = pd.DataFrame(
            {
                "symbol": list("BGACEDF"),
                "price": [0.0023, 0.0078, 5.01, 25.13, 30.78, 340.05, 1040.90],
                "mpv": [0.0001, 0.0001, 0.01, 0.01, 0.01, 0.05, 0.05],
            },
            columns=["symbol", "price", "mpv"],
        )

        # 使用 pytest 框架提供的 assert_frame_equal 函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    def test_on_specialized_type(self, any_real_numpy_dtype):
        # 测试处理特定类型的情况，参考 gh-13936

        # 根据给定的 numpy 数据类型创建对应的数据类型
        dtype = np.dtype(any_real_numpy_dtype).type

        # 创建第一个数据框，包含值和对应的股票代码，按值排序
        df1 = pd.DataFrame(
            {"value": [5, 2, 25, 100, 78, 120, 79], "symbol": list("ABCDEFG")},
            columns=["symbol", "value"],
        )
        # 将第一个数据框的值列转换为指定的数据类型
        df1.value = dtype(df1.value)

        # 创建第二个数据框，包含值和对应的结果，按值排序
        df2 = pd.DataFrame(
            {"value": [0, 80, 120, 125], "result": list("xyzw")},
            columns=["value", "result"],
        )
        # 将第二个数据框的值列转换为指定的数据类型
        df2.value = dtype(df2.value)

        # 根据值对第一个数据框排序，并重置索引
        df1 = df1.sort_values("value").reset_index(drop=True)
        
        # 调用 merge_asof 函数进行数据框的合并操作
        result = merge_asof(df1, df2, on="value")

        # 创建预期结果的数据框
        expected = pd.DataFrame(
            {
                "symbol": list("BACEGDF"),
                "value": [2, 5, 25, 78, 79, 100, 120],
                "result": list("xxxxxyz"),
            },
            columns=["symbol", "value", "result"],
        )
        # 使用预期的数据类型转换第一个数据框的值列
        expected.value = dtype(expected.value)

        # 使用 pytest 框架提供的 assert_frame_equal 函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)
    def test_on_specialized_type_by_int(self, any_real_numpy_dtype):
        # 在特定类型（由参数 any_real_numpy_dtype 指定）上执行测试，这是一个针对特定问题（gh-13936）的测试函数

        # 将参数 any_real_numpy_dtype 转换为 NumPy dtype 并获取其类型
        dtype = np.dtype(any_real_numpy_dtype).type

        # 创建第一个数据框 df1，包含 symbol、key 和 value 列
        df1 = pd.DataFrame(
            {
                "value": [5, 2, 25, 100, 78, 120, 79],
                "key": [1, 2, 3, 2, 3, 1, 2],
                "symbol": list("ABCDEFG"),
            },
            columns=["symbol", "key", "value"],
        )
        # 将 df1 的 value 列数据类型转换为指定的 dtype

        df1.value = dtype(df1.value)

        # 创建第二个数据框 df2，包含 value、key 和 result 列
        df2 = pd.DataFrame(
            {"value": [0, 80, 120, 125], "key": [1, 2, 2, 3], "result": list("xyzw")},
            columns=["value", "key", "result"],
        )
        # 将 df2 的 value 列数据类型转换为指定的 dtype

        df2.value = dtype(df2.value)

        # 根据 value 列对 df1 进行排序，并重新设置索引
        df1 = df1.sort_values("value").reset_index(drop=True)

        # 使用 merge_asof 函数将 df1 和 df2 按照 value 列和 key 列合并
        result = merge_asof(df1, df2, on="value", by="key")

        # 创建期望结果的数据框 expected，包含 symbol、key、value 和 result 列
        expected = pd.DataFrame(
            {
                "symbol": list("BACEGDF"),
                "key": [2, 1, 3, 3, 2, 2, 1],
                "value": [2, 5, 25, 78, 79, 100, 120],
                "result": [np.nan, "x", np.nan, np.nan, np.nan, "y", "x"],
            },
            columns=["symbol", "key", "value", "result"],
        )
        # 将 expected 的 value 列数据类型转换为指定的 dtype

        expected.value = dtype(expected.value)

        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 数据框
        tm.assert_frame_equal(result, expected)

    def test_on_float_by_int(self):
        # 在 float 类型和 int 类型上同时进行类型专用测试

        # 创建第一个数据框 df1，包含 symbol、exch 和 price 列
        df1 = pd.DataFrame(
            {
                "symbol": list("AAABBBCCC"),
                "exch": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "price": [
                    3.26,
                    3.2599,
                    3.2598,
                    12.58,
                    12.59,
                    12.5,
                    378.15,
                    378.2,
                    378.25,
                ],
            },
            columns=["symbol", "exch", "price"],
        )

        # 创建第二个数据框 df2，包含 exch、price 和 mpv 列
        df2 = pd.DataFrame(
            {
                "exch": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "price": [0.0, 1.0, 100.0, 0.0, 5.0, 100.0, 0.0, 5.0, 1000.0],
                "mpv": [0.0001, 0.01, 0.05, 0.0001, 0.01, 0.1, 0.0001, 0.25, 1.0],
            },
            columns=["exch", "price", "mpv"],
        )

        # 根据 price 列对 df1 和 df2 进行排序，并重新设置索引
        df1 = df1.sort_values("price").reset_index(drop=True)
        df2 = df2.sort_values("price").reset_index(drop=True)

        # 使用 merge_asof 函数将 df1 和 df2 按照 price 列和 exch 列合并
        result = merge_asof(df1, df2, on="price", by="exch")

        # 创建期望结果的数据框 expected，包含 symbol、exch、price 和 mpv 列
        expected = pd.DataFrame(
            {
                "symbol": list("AAABBBCCC"),
                "exch": [3, 2, 1, 3, 1, 2, 1, 2, 3],
                "price": [
                    3.2598,
                    3.2599,
                    3.26,
                    12.5,
                    12.58,
                    12.59,
                    378.15,
                    378.2,
                    378.25,
                ],
                "mpv": [0.0001, 0.0001, 0.01, 0.25, 0.01, 0.01, 0.05, 0.1, 0.25],
            },
            columns=["symbol", "exch", "price", "mpv"],
        )

        # 使用 tm.assert_frame_equal 函数比较 result 和 expected 数据框
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试在推断字符串使用情况下是否会引发数据类型错误
    def test_merge_datatype_error_raises(self, using_infer_string):
        # 根据是否使用推断字符串，选择不同的错误消息
        if using_infer_string:
            msg = "incompatible merge keys"
        else:
            msg = r"Incompatible merge dtype, .*, both sides must have numeric dtype"

        # 创建左侧和右侧的 Pandas DataFrame
        left = pd.DataFrame({"left_val": [1, 5, 10], "a": ["a", "b", "c"]})
        right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7], "a": [1, 2, 3, 6, 7]})

        # 使用 pytest 检查是否会引发 MergeError，并匹配特定的错误消息
        with pytest.raises(MergeError, match=msg):
            merge_asof(left, right, on="a")

    # 定义一个测试函数，用于测试在分类数据类型不兼容时是否会引发错误
    def test_merge_datatype_categorical_error_raises(self):
        # 指定分类数据类型不兼容时的错误消息
        msg = (
            r"incompatible merge keys \[0\] .* both sides category, "
            "but not equal ones"
        )

        # 创建左侧和右侧的 Pandas DataFrame，其中包含分类数据类型
        left = pd.DataFrame(
            {"left_val": [1, 5, 10], "a": pd.Categorical(["a", "b", "c"])}
        )
        right = pd.DataFrame(
            {
                "right_val": [1, 2, 3, 6, 7],
                "a": pd.Categorical(["a", "X", "c", "X", "b"]),
            }
        )

        # 使用 pytest 检查是否会引发 MergeError，并匹配特定的错误消息
        with pytest.raises(MergeError, match=msg):
            merge_asof(left, right, on="a")

    # 定义一个测试函数，用于测试在多列分组和包含分类列时的合并操作
    def test_merge_groupby_multiple_column_with_categorical_column(self):
        # GH 16454：测试用例描述
        df = pd.DataFrame({"x": [0], "y": [0], "z": pd.Categorical([0])})
        # 执行合并操作，并将结果与预期结果进行比较
        result = merge_asof(df, df, on="x", by=["y", "z"])
        expected = pd.DataFrame({"x": [0], "y": [0], "z": pd.Categorical([0])})
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化装饰器定义一个测试函数，用于测试在合并操作中空值的处理
    @pytest.mark.parametrize(
        "func", [lambda x: x, to_datetime], ids=["numeric", "datetime"]
    )
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_merge_on_nans(self, func, side):
        # GH 23189：测试用例描述
        # 根据合并操作的一侧（left 或 right），设置相应的错误消息
        msg = f"Merge keys contain null values on {side} side"
        # 创建包含空值和非空值的 Pandas DataFrame
        nulls = func([1.0, 5.0, np.nan])
        non_nulls = func([1.0, 5.0, 10.0])
        df_null = pd.DataFrame({"a": nulls, "left_val": ["a", "b", "c"]})
        df = pd.DataFrame({"a": non_nulls, "right_val": [1, 6, 11]})

        # 使用 pytest 检查是否会引发 ValueError，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            if side == "left":
                merge_asof(df_null, df, on="a")
            else:
                merge_asof(df, df_null, on="a")
    # 定义一个测试方法，用于测试在给定任意数值类型和推断字符串类型下的行为
    def test_by_nullable(self, any_numeric_ea_dtype, using_infer_string):
        # 注：如果使用 np.array([np.nan, 1]) 而不是 pd.array，这个测试会通过。
        # 除此之外，我 (@jbrockmendel) 完全不清楚期望的行为是什么。
        # TODO(GH#32306): 这里可能与预期行为相关。

        # 创建一个包含 NA 值的 Pandas 数组，指定数据类型为 any_numeric_ea_dtype
        arr = pd.array([pd.NA, 0, 1], dtype=any_numeric_ea_dtype)
        # 根据数据类型的种类（整数或无符号整数），确定最大值
        if arr.dtype.kind in ["i", "u"]:
            max_val = np.iinfo(arr.dtype.numpy_dtype).max
        else:
            max_val = np.finfo(arr.dtype.numpy_dtype).max
        # 将数组中的第三个元素设置为最大值，以确保 arr._values_for_argsort
        # 不是一个单射（injective）函数
        arr[2] = max_val

        # 创建左侧 DataFrame
        left = pd.DataFrame(
            {
                "by_col1": arr,
                "by_col2": ["HELLO", "To", "You"],
                "on_col": [2, 4, 6],
                "value": ["a", "c", "e"],
            }
        )
        # 创建右侧 DataFrame
        right = pd.DataFrame(
            {
                "by_col1": arr,
                "by_col2": ["WORLD", "Wide", "Web"],
                "on_col": [1, 2, 6],
                "value": ["b", "d", "f"],
            }
        )

        # 执行 asof 合并操作，按照 ["by_col1", "by_col2"] 列合并，on_col 作为键
        result = merge_asof(left, right, by=["by_col1", "by_col2"], on="on_col")
        # 创建预期的 DataFrame
        expected = pd.DataFrame(
            {
                "by_col1": arr,
                "by_col2": ["HELLO", "To", "You"],
                "on_col": [2, 4, 6],
                "value_x": ["a", "c", "e"],  # 左侧 DataFrame 的 value 列
            }
        )
        # 添加 value_y 列到预期的 DataFrame，并初始化为全 NaN 数组
        expected["value_y"] = np.array([np.nan, np.nan, np.nan], dtype=object)
        # 如果 using_infer_string 为 True，则将 value_y 列转换为 string[pyarrow_numpy] 类型
        if using_infer_string:
            expected["value_y"] = expected["value_y"].astype("string[pyarrow_numpy]")
        # 断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试按照时区感知的列进行合并
    def test_merge_by_col_tz_aware(self):
        # GH 21184
        # 创建左侧 DataFrame，包含一个时区感知的 DatetimeIndex 列
        left = pd.DataFrame(
            {
                "by_col": pd.DatetimeIndex(["2018-01-01"]).tz_localize("UTC"),
                "on_col": [2],
                "values": ["a"],
            }
        )
        # 创建右侧 DataFrame，包含一个时区感知的 DatetimeIndex 列
        right = pd.DataFrame(
            {
                "by_col": pd.DatetimeIndex(["2018-01-01"]).tz_localize("UTC"),
                "on_col": [1],
                "values": ["b"],
            }
        )
        # 执行 asof 合并操作，按照 "by_col" 列合并，on_col 作为键
        result = merge_asof(left, right, by="by_col", on="on_col")
        # 创建预期的 DataFrame
        expected = pd.DataFrame(
            [[pd.Timestamp("2018-01-01", tz="UTC"), 2, "a", "b"]],
            columns=["by_col", "on_col", "values_x", "values_y"],
        )
        # 断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试 merge_asof 函数在混合时区感知情况下的行为
    def test_by_mixed_tz_aware(self, using_infer_string):
        # 创建左侧数据框，包含多列数据，其中 by_col1 列为带时区信息的日期索引
        left = pd.DataFrame(
            {
                "by_col1": pd.DatetimeIndex(["2018-01-01"]).tz_localize("UTC"),
                "by_col2": ["HELLO"],
                "on_col": [2],
                "value": ["a"],
            }
        )
        # 创建右侧数据框，也包含多列数据，by_col1 列同样为带时区信息的日期索引
        right = pd.DataFrame(
            {
                "by_col1": pd.DatetimeIndex(["2018-01-01"]).tz_localize("UTC"),
                "by_col2": ["WORLD"],
                "on_col": [1],
                "value": ["b"],
            }
        )
        # 使用 merge_asof 函数根据 by_col1 和 by_col2 列进行关联，使用 on_col 列作为排序键
        result = merge_asof(left, right, by=["by_col1", "by_col2"], on="on_col")
        # 创建预期结果数据框，包含一行数据，列名分别为 by_col1, by_col2, on_col, value_x
        expected = pd.DataFrame(
            [[pd.Timestamp("2018-01-01", tz="UTC"), "HELLO", 2, "a"]],
            columns=["by_col1", "by_col2", "on_col", "value_x"],
        )
        # 向预期结果数据框添加 value_y 列，并初始化为 NaN 值的 numpy 数组
        expected["value_y"] = np.array([np.nan], dtype=object)
        # 如果 using_infer_string 为 True，则将 value_y 列的数据类型转换为 string[pyarrow_numpy]
        if using_infer_string:
            expected["value_y"] = expected["value_y"].astype("string[pyarrow_numpy]")
        # 使用 assert_frame_equal 函数比较实际结果和预期结果的数据框
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 标记的参数化测试方法，用于测试不同 dtype 下的 merge_asof 函数行为
    @pytest.mark.parametrize("dtype", ["float64", "int16", "m8[ns]", "M8[us]"])
    def test_by_dtype(self, dtype):
        # 创建左侧数据框，包含 by_col 列，使用给定的 dtype 参数作为数据类型
        left = pd.DataFrame(
            {
                "by_col": np.array([1], dtype=dtype),
                "on_col": [2],
                "value": ["a"],
            }
        )
        # 创建右侧数据框，同样包含 by_col 列，使用给定的 dtype 参数作为数据类型
        right = pd.DataFrame(
            {
                "by_col": np.array([1], dtype=dtype),
                "on_col": [1],
                "value": ["b"],
            }
        )
        # 使用 merge_asof 函数根据 by_col 列进行关联，使用 on_col 列作为排序键
        result = merge_asof(left, right, by="by_col", on="on_col")
        # 创建预期结果数据框，包含 by_col, on_col, value_x, value_y 列
        expected = pd.DataFrame(
            {
                "by_col": np.array([1], dtype=dtype),
                "on_col": [2],
                "value_x": ["a"],
                "value_y": ["b"],
            }
        )
        # 使用 assert_frame_equal 函数比较实际结果和预期结果的数据框
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试时间增量容忍性，nearest 方向
    def test_timedelta_tolerance_nearest(self, unit):
        # GH 27642：GitHub 上的 issue 编号，说明这个测试的背景信息

        # 如果 unit 参数为 "s"，则跳过测试，因为这会使得 left['time'] 被舍入
        if unit == "s":
            pytest.skip(
                "This test is invalid with unit='s' because that would "
                "round left['time']"
            )

        # 创建一个名为 left 的 DataFrame，包含时间和左侧数据列
        left = pd.DataFrame(
            list(zip([0, 5, 10, 15, 20, 25], [0, 1, 2, 3, 4, 5])),
            columns=["time", "left"],
        )

        # 将 left 中的时间列转换为以毫秒为单位的时间增量，并根据 unit 参数指定的单位进行类型转换
        left["time"] = pd.to_timedelta(left["time"], "ms").astype(f"m8[{unit}]")

        # 创建一个名为 right 的 DataFrame，包含时间和右侧数据列
        right = pd.DataFrame(
            list(zip([0, 3, 9, 12, 15, 18], [0, 1, 2, 3, 4, 5])),
            columns=["time", "right"],
        )

        # 将 right 中的时间列转换为以毫秒为单位的时间增量，并根据 unit 参数指定的单位进行类型转换
        right["time"] = pd.to_timedelta(right["time"], "ms").astype(f"m8[{unit}]")

        # 创建一个名为 expected 的 DataFrame，包含时间、左侧数据和右侧数据列
        expected = pd.DataFrame(
            list(
                zip(
                    [0, 5, 10, 15, 20, 25],
                    [0, 1, 2, 3, 4, 5],
                    [0, np.nan, 2, 4, np.nan, np.nan],
                )
            ),
            columns=["time", "left", "right"],
        )

        # 将 expected 中的时间列转换为以毫秒为单位的时间增量，并根据 unit 参数指定的单位进行类型转换
        expected["time"] = pd.to_timedelta(expected["time"], "ms").astype(f"m8[{unit}]")

        # 使用 merge_asof 函数，按照时间列 "time" 进行最近方向的近似合并
        result = merge_asof(
            left, right, on="time", tolerance=Timedelta("1ms"), direction="nearest"
        )

        # 断言 result 和 expected 是否相等，以确保测试的准确性
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，用于测试整数类型的容忍性
    def test_int_type_tolerance(self, any_int_dtype):
        # GH #28870：GitHub 上的 issue 编号，说明这个测试的背景信息

        # 创建一个名为 left 的 DataFrame，包含整数列 'a' 和左侧值列
        left = pd.DataFrame({"a": [0, 10, 20], "left_val": [1, 2, 3]})
        
        # 创建一个名为 right 的 DataFrame，包含整数列 'a' 和右侧值列
        right = pd.DataFrame({"a": [5, 15, 25], "right_val": [1, 2, 3]})
        
        # 将 left 中的整数列 'a' 转换为指定的任意整数数据类型 any_int_dtype
        left["a"] = left["a"].astype(any_int_dtype)
        
        # 将 right 中的整数列 'a' 转换为指定的任意整数数据类型 any_int_dtype
        right["a"] = right["a"].astype(any_int_dtype)

        # 创建一个名为 expected 的 DataFrame，包含整数列 'a'、左侧值列和右侧值列
        expected = pd.DataFrame(
            {"a": [0, 10, 20], "left_val": [1, 2, 3], "right_val": [np.nan, 1.0, 2.0]}
        )

        # 将 expected 中的整数列 'a' 转换为指定的任意整数数据类型 any_int_dtype
        expected["a"] = expected["a"].astype(any_int_dtype)

        # 使用 merge_asof 函数，按照整数列 'a' 进行容忍性为 10 的合并
        result = merge_asof(left, right, on="a", tolerance=10)

        # 断言 result 和 expected 是否相等，以确保测试的准确性
        tm.assert_frame_equal(result, expected)
    def test_merge_index_column_tz(self):
        # GH 29864
        # 创建一个时间索引，从"2019-10-01"开始，频率为30分钟，共5个时间点，带有时区信息"UTC"
        index = pd.date_range("2019-10-01", freq="30min", periods=5, tz="UTC")
        
        # 创建左侧数据框，包含一列名为"xyz"，索引从index的第二个时间点开始
        left = pd.DataFrame([0.9, 0.8, 0.7, 0.6], columns=["xyz"], index=index[1:])
        
        # 创建右侧数据框，包含列"from_date"和"abc"，其中"from_date"列使用index作为数据，"abc"列为[2.46, 2.46, 2.46, 2.46, 2.19]
        right = pd.DataFrame({"from_date": index, "abc": [2.46] * 4 + [2.19]})
        
        # 使用merge_asof函数，将左右数据框按照指定条件合并，左侧使用索引，右侧使用"from_date"列
        result = merge_asof(
            left=left, right=right, left_index=True, right_on=["from_date"]
        )
        
        # 创建期望的数据框，包含"xyz"、"from_date"和"abc"列，索引从"2019-10-01 00:30:00"开始，频率为30分钟，共4个时间点，带有时区信息"UTC"
        expected = pd.DataFrame(
            {
                "xyz": [0.9, 0.8, 0.7, 0.6],
                "from_date": index[1:],
                "abc": [2.46] * 3 + [2.19],
            },
            index=pd.date_range(
                "2019-10-01 00:30:00", freq="30min", periods=4, tz="UTC"
            ),
        )
        
        # 断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)

        # 再次使用merge_asof函数，但这次左右数据框互换，左侧使用"from_date"列，右侧使用索引
        result = merge_asof(
            left=right, right=left, right_index=True, left_on=["from_date"]
        )
        
        # 创建另一组期望的数据框，包含"from_date"、"abc"和"xyz"列，索引为[0, 1, 2, 3, 4]
        expected = pd.DataFrame(
            {
                "from_date": index,
                "abc": [2.46] * 4 + [2.19],
                "xyz": [np.nan, 0.9, 0.8, 0.7, 0.6],
            },
            index=Index([0, 1, 2, 3, 4]),
        )
        
        # 断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    def test_left_index_right_index_tolerance(self, unit):
        # https://github.com/pandas-dev/pandas/issues/35558
        # 如果unit为"s"，跳过此测试，因为在单位为"s"时，会对dr1进行四舍五入，使得测试无效
        if unit == "s":
            pytest.skip(
                "This test is invalid with unit='s' because that would round dr1"
            )

        # 创建时间范围dr1，从"1/1/2020"到"1/20/2020"，频率为2天，单位为unit，加上0.4秒后转换为unit单位的时间
        dr1 = pd.date_range(
            start="1/1/2020", end="1/20/2020", freq="2D", unit=unit
        ) + Timedelta(seconds=0.4).as_unit(unit)
        
        # 创建时间范围dr2，从"1/1/2020"到"2/1/2020"，单位为unit
        dr2 = pd.date_range(start="1/1/2020", end="2/1/2020", unit=unit)

        # 创建数据框df1，包含"val1"列，索引为dr1的DatetimeIndex
        df1 = pd.DataFrame({"val1": "foo"}, index=pd.DatetimeIndex(dr1))
        
        # 创建数据框df2，包含"val2"列，索引为dr2的DatetimeIndex
        df2 = pd.DataFrame({"val2": "bar"}, index=pd.DatetimeIndex(dr2))

        # 创建期望的数据框，包含"val1"和"val2"列，索引为dr1的DatetimeIndex
        expected = pd.DataFrame(
            {"val1": "foo", "val2": "bar"}, index=pd.DatetimeIndex(dr1)
        )
        
        # 使用merge_asof函数，将df1和df2按照指定条件合并，左右数据框使用索引，合并容差为0.5秒
        result = merge_asof(
            df1,
            df2,
            left_index=True,
            right_index=True,
            tolerance=Timedelta(seconds=0.5),
        )
        
        # 断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器，为 infer_string 参数提供两个测试用例：False 和带有 pytest 标记的 True 测试用例
@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
# 使用 pytest 的 parametrize 装饰器，为 kwargs 参数提供三个测试用例字典
@pytest.mark.parametrize(
    "kwargs", [{"on": "x"}, {"left_index": True, "right_index": True}]
)
# 使用 pytest 的 parametrize 装饰器，为 data 参数提供两个测试用例列表
@pytest.mark.parametrize(
    "data",
    [["2019-06-01 00:09:12", "2019-06-01 00:10:29"], [1.0, "2019-06-01 00:10:29"]],
)
# 定义测试函数 test_merge_asof_non_numerical_dtype，用于测试非数值类型的数据合并行为
def test_merge_asof_non_numerical_dtype(kwargs, data, infer_string):
    # GH#29130: 标识 GitHub 问题号
    with option_context("future.infer_string", infer_string):
        # 创建左侧 DataFrame，以 data 列表作为值，data 列表作为索引
        left = pd.DataFrame({"x": data}, index=data)
        # 创建右侧 DataFrame，以 data 列表作为值，data 列表作为索引
        right = pd.DataFrame({"x": data}, index=data)
        # 断言合并过程中会抛出 MergeError 异常，并匹配指定的错误信息正则表达式
        with pytest.raises(
            MergeError,
            match=r"Incompatible merge dtype, .*, both sides must have numeric dtype",
        ):
            # 调用 merge_asof 函数进行合并操作，传入 kwargs 参数
            merge_asof(left, right, **kwargs)


# 定义测试函数 test_merge_asof_non_numerical_dtype_object，用于测试包含对象类型的数据合并行为
def test_merge_asof_non_numerical_dtype_object():
    # GH#29130: 标识 GitHub 问题号
    # 创建左侧 DataFrame，包含字符串列 "a" 和 "left_val1"
    left = pd.DataFrame({"a": ["12", "13", "15"], "left_val1": ["a", "b", "c"]})
    # 创建右侧 DataFrame，包含字符串列 "a" 和 "left_val"
    right = pd.DataFrame({"a": ["a", "b", "c"], "left_val": ["d", "e", "f"]})
    # 断言合并过程中会抛出 MergeError 异常，并匹配指定的错误信息正则表达式
    with pytest.raises(
        MergeError,
        match=r"Incompatible merge dtype, .*, both sides must have numeric dtype",
    ):
        # 调用 merge_asof 函数进行合并操作，传入指定的列名参数
        merge_asof(
            left,
            right,
            left_on="left_val1",
            right_on="a",
            left_by="a",
            right_by="left_val",
        )


# 使用 pytest 的 parametrize 装饰器，为 kwargs 参数提供三个测试用例字典
@pytest.mark.parametrize(
    "kwargs",
    [
        {"right_index": True, "left_index": True},
        {"left_on": "left_time", "right_index": True},
        {"left_index": True, "right_on": "right"},
    ],
)
# 定义测试函数 test_merge_asof_index_behavior，测试合并操作在不同索引条件下的行为
def test_merge_asof_index_behavior(kwargs):
    # GH 33463: 标识 GitHub 问题号
    # 创建索引为 [1, 5, 10]，名称为 "test" 的索引对象
    index = Index([1, 5, 10], name="test")
    # 创建左侧 DataFrame，包含 "left" 和 "left_time" 列，以 index 作为索引
    left = pd.DataFrame({"left": ["a", "b", "c"], "left_time": [1, 4, 10]}, index=index)
    # 创建右侧 DataFrame，包含 "right" 列，以列表 [1, 2, 3, 6, 7] 作为索引
    right = pd.DataFrame({"right": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    # 调用 merge_asof 函数进行合并操作，传入 kwargs 参数
    result = merge_asof(left, right, **kwargs)

    # 创建期望的 DataFrame，包含 "left"、"left_time" 和 "right" 列，以 index 作为索引
    expected = pd.DataFrame(
        {"left": ["a", "b", "c"], "left_time": [1, 4, 10], "right": [1, 3, 7]},
        index=index,
    )
    # 使用 assert_frame_equal 断言 result 和 expected 的内容一致
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_merge_asof_numeric_column_in_index，测试在索引中包含数值列的合并操作行为
def test_merge_asof_numeric_column_in_index():
    # GH#34488: 标识 GitHub 问题号
    # 创建左侧 DataFrame，包含 "b" 列，以 [1, 2, 3] 作为索引，索引名称为 "a"
    left = pd.DataFrame({"b": [10, 11, 12]}, index=Index([1, 2, 3], name="a"))
    # 创建右侧 DataFrame，包含 "c" 列，以 [0, 2, 3] 作为索引，索引名称为 "a"
    right = pd.DataFrame({"c": [20, 21, 22]}, index=Index([0, 2, 3], name="a"))

    # 调用 merge_asof 函数进行合并操作，传入 left_on 和 right_on 参数
    result = merge_asof(left, right, left_on="a", right_on="a")
    # 创建期望的 DataFrame，包含 "a"、"b" 和 "c" 列
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [10, 11, 12], "c": [20, 21, 22]})
    # 使用 assert_frame_equal 断言 result 和 expected 的内容一致
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_merge_asof_numeric_column_in_multiindex，测试在多级索引中包含数值列的合并操作行为
def test_merge_asof_numeric_column_in_multiindex():
    # GH#34488: 标识 GitHub 问题号
    # 创建左侧 DataFrame，包含 "b" 列，以两个数组作为多级索引，分别为 [1, 2, 3] 和 ["a", "b", "c"]，名称分别为 "a" 和 "z"
    left = pd.DataFrame(
        {"b": [10, 11, 12]},
        index=pd.MultiIndex.from_arrays([[1, 2, 3], ["a", "b", "c"]], names=["a", "z"]),
    )
    # 创建右侧 DataFrame，包含 "c" 列，以两个数组作为多级索引，分别为 [1, 2, 3] 和 ["x", "y", "z"]，名称分别为 "a" 和 "y"
    right = pd.DataFrame(
        {"c": [20, 21, 22]},
        index=pd.MultiIndex.from_arrays([[1, 2, 3], ["x", "y", "z"]], names=["a", "y"]),
    )

    # 调用 merge_asof 函数进行合并操作，传入 left_on 和 right_on 参数
    result = merge_asof(left, right, left_on="a", right_on="a")
    # 创建期望的 DataFrame，包含 "a"、"b" 和 "c" 列
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [10, 11, 12], "c": [20, 21, 22]})
    # 使用 assert_frame_equal 断言 result 和 expected 的内容一致
    tm.assert_frame_equal(result, expected)
    # 使用测试工具库中的 assert_frame_equal 函数比较 result 和 expected 两个数据框
    tm.assert_frame_equal(result, expected)
def test_merge_asof_numeri_column_in_index_object_dtype():
    # GH#34488
    # 创建包含整数列 'b' 和索引为字符串的DataFrame 'left'
    left = pd.DataFrame({"b": [10, 11, 12]}, index=Index(["1", "2", "3"], name="a"))
    # 创建包含整数列 'c' 和索引为字符串的DataFrame 'right'
    right = pd.DataFrame({"c": [20, 21, 22]}, index=Index(["m", "n", "o"], name="a"))

    # 使用 pytest 的断言检查是否抛出 MergeError 异常，匹配特定错误信息
    with pytest.raises(
        MergeError,
        match=r"Incompatible merge dtype, .*, both sides must have numeric dtype",
    ):
        # 调用 merge_asof 函数尝试合并 'left' 和 'right'，使用 'left_on' 和 'right_on' 参数指定合并键
        merge_asof(left, right, left_on="a", right_on="a")

    # 将 'left' DataFrame 重置索引并重新设置索引列，然后设置为多级索引
    left = left.reset_index().set_index(["a", "b"])
    # 将 'right' DataFrame 重置索引并重新设置索引列，然后设置为多级索引
    right = right.reset_index().set_index(["a", "c"])

    # 使用 pytest 的断言检查是否抛出 MergeError 异常，匹配特定错误信息
    with pytest.raises(
        MergeError,
        match=r"Incompatible merge dtype, .*, both sides must have numeric dtype",
    ):
        # 再次尝试合并 'left' 和 'right'，使用 'left_on' 和 'right_on' 参数指定合并键
        merge_asof(left, right, left_on="a", right_on="a")


def test_merge_asof_array_as_on(unit):
    # GH#42844
    # 创建一个具有指定单位的日期时间索引 'dti'
    dti = pd.DatetimeIndex(
        ["2021/01/01 00:37", "2021/01/01 01:40"], dtype=f"M8[{unit}]"
    )
    # 创建包含 'a' 和 'ts' 列的 'right' DataFrame
    right = pd.DataFrame(
        {
            "a": [2, 6],
            "ts": dti,
        }
    )
    # 创建时间序列 'ts_merge'，从指定时间开始，频率为每小时
    ts_merge = pd.date_range(
        start=pd.Timestamp("2021/01/01 00:00"), periods=3, freq="1h", unit=unit
    )
    # 创建 'left' DataFrame 包含整数列 'b'
    left = pd.DataFrame({"b": [4, 8, 7]})
    # 调用 merge_asof 函数尝试合并 'left' 和 'right'，使用 'left_on' 和 'right_on' 参数指定合并键
    result = merge_asof(
        left,
        right,
        left_on=ts_merge,
        right_on="ts",
        allow_exact_matches=False,
        direction="backward",
    )
    # 创建预期结果的 DataFrame 'expected'
    expected = pd.DataFrame({"b": [4, 8, 7], "a": [np.nan, 2, 6], "ts": ts_merge})
    # 使用 pytest 的断言检查 'result' 和 'expected' 是否相等
    tm.assert_frame_equal(result, expected)

    # 再次调用 merge_asof 函数，反向合并 'right' 和 'left'，使用 'left_on' 和 'right_on' 参数指定合并键
    result = merge_asof(
        right,
        left,
        left_on="ts",
        right_on=ts_merge,
        allow_exact_matches=False,
        direction="backward",
    )
    # 创建预期结果的 DataFrame 'expected'
    expected = pd.DataFrame(
        {
            "a": [2, 6],
            "ts": dti,
            "b": [4, 8],
        }
    )
    # 使用 pytest 的断言检查 'result' 和 'expected' 是否相等
    tm.assert_frame_equal(result, expected)


def test_merge_asof_raise_for_duplicate_columns():
    # GH#50102
    # 创建包含重复列标签 'a' 的 'left' DataFrame
    left = pd.DataFrame([[1, 2, "a"]], columns=["a", "a", "left_val"])
    # 创建包含重复列标签 'a' 的 'right' DataFrame
    right = pd.DataFrame([[1, 1, 1]], columns=["a", "a", "right_val"])

    # 使用 pytest 的断言检查是否抛出 ValueError 异常，匹配特定错误信息
    with pytest.raises(ValueError, match="column label 'a'"):
        # 调用 merge_asof 函数尝试合并 'left' 和 'right'，使用 'on' 参数指定合并键
        merge_asof(left, right, on="a")

    with pytest.raises(ValueError, match="column label 'a'"):
        # 再次调用 merge_asof 函数尝试合并 'left' 和 'right'，使用 'left_on' 和 'right_on' 参数指定合并键
        merge_asof(left, right, left_on="a", right_on="right_val")

    with pytest.raises(ValueError, match="column label 'a'"):
        # 再次调用 merge_asof 函数尝试合并 'left' 和 'right'，使用 'left_on' 和 'right_on' 参数指定合并键
        merge_asof(left, right, left_on="left_val", right_on="a")


@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        pytest.param("int64[pyarrow]", marks=td.skip_if_no("pyarrow")),
        pytest.param("timestamp[s][pyarrow]", marks=td.skip_if_no("pyarrow")),
    ],
)
def test_merge_asof_extension_dtype(dtype):
    # GH 52904
    # 创建包含 'join_col' 和 'left_val' 列的 'left' DataFrame
    left = pd.DataFrame(
        {
            "join_col": [1, 3, 5],
            "left_val": [1, 2, 3],
        }
    )
    # 创建包含 'join_col' 和 'right_val' 列的 'right' DataFrame
    right = pd.DataFrame(
        {
            "join_col": [2, 3, 4],
            "right_val": [1, 2, 3],
        }
    )
    # 将 'left' DataFrame 中的 'join_col' 列转换为指定的扩展数据类型 'dtype'
    left = left.astype({"join_col": dtype})
    # 将 'right' DataFrame 中的 'join_col' 列转换为指定的扩展数据类型 'dtype'
    right = right.astype({"join_col": dtype})
    # 使用 merge_asof 函数将 left 和 right DataFrames 按照 "join_col" 列进行模糊合并
    result = merge_asof(left, right, on="join_col")
    # 创建一个预期的 DataFrame，包含列 "join_col"、"left_val" 和 "right_val"
    expected = pd.DataFrame(
        {
            "join_col": [1, 3, 5],
            "left_val": [1, 2, 3],
            "right_val": [np.nan, 2.0, 3.0],
        }
    )
    # 将 "join_col" 列的数据类型转换为变量 dtype 指定的类型
    expected = expected.astype({"join_col": dtype})
    # 使用 assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
@td.skip_if_no("pyarrow")
# 使用测试装饰器，如果缺少 pyarrow 模块则跳过该测试
def test_merge_asof_pyarrow_td_tolerance():
    # GH 56486
    # 创建一个包含单个日期时间对象的 Series，数据类型为 pyarrow 支持的 timestamp[us, UTC]
    ser = pd.Series(
        [datetime.datetime(2023, 1, 1)], dtype="timestamp[us, UTC][pyarrow]"
    )
    # 创建一个包含 "timestamp" 和 "value" 列的 DataFrame
    df = pd.DataFrame(
        {
            "timestamp": ser,
            "value": [1],
        }
    )
    # 调用 merge_asof 函数进行数据合并，基于 "timestamp" 列，设置 1 秒的时间容差
    result = merge_asof(df, df, on="timestamp", tolerance=Timedelta("1s"))
    # 创建预期的 DataFrame，包含 "timestamp"、"value_x" 和 "value_y" 列
    expected = pd.DataFrame(
        {
            "timestamp": ser,
            "value_x": [1],
            "value_y": [1],
        }
    )
    # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_merge_asof_read_only_ndarray():
    # GH 53513
    # 创建包含单个元素的 Series，并设置索引为只读状态
    left = pd.Series([2], index=[2], name="left")
    right = pd.Series([1], index=[1], name="right")
    left.index.values.flags.writeable = False
    right.index.values.flags.writeable = False
    # 调用 merge_asof 函数，根据左右索引进行合并
    result = merge_asof(left, right, left_index=True, right_index=True)
    # 创建预期的 DataFrame，包含 "left" 和 "right" 列，并设置索引为 [2]
    expected = pd.DataFrame({"left": [2], "right": [1]}, index=[2])
    tm.assert_frame_equal(result, expected)


def test_merge_asof_multiby_with_categorical():
    # GH 43541
    # 创建包含多列数据的 DataFrame，其中 "c1" 使用分类数据类型
    left = pd.DataFrame(
        {
            "c1": pd.Categorical(["a", "a", "b", "b"], categories=["a", "b"]),
            "c2": ["x"] * 4,
            "t": [1] * 4,
            "v": range(4),
        }
    )
    # 创建包含多列数据的 DataFrame，其中 "c1" 使用分类数据类型
    right = pd.DataFrame(
        {
            "c1": pd.Categorical(["b", "b"], categories=["b", "a"]),
            "c2": ["x"] * 2,
            "t": [1, 2],
            "v": range(2),
        }
    )
    # 调用 merge_asof 函数，根据 ["c1", "c2"] 列进行合并，基于 "t" 列向前合并
    result = merge_asof(
        left,
        right,
        by=["c1", "c2"],
        on="t",
        direction="forward",
        suffixes=["_left", "_right"],
    )
    # 创建预期的 DataFrame，包含 "c1"、"c2"、"t"、"v_left" 和 "v_right" 列
    expected = pd.DataFrame(
        {
            "c1": pd.Categorical(["a", "a", "b", "b"], categories=["a", "b"]),
            "c2": ["x"] * 4,
            "t": [1] * 4,
            "v_left": range(4),
            "v_right": [np.nan, np.nan, 0.0, 0.0],
        }
    )
    # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
```