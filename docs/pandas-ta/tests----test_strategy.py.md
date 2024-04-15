# `.\pandas-ta\tests\test_strategy.py`

```
# 导入必要的模块
# 必须与其它测试分开运行，以确保成功运行
from multiprocessing import cpu_count  # 导入获取 CPU 核心数量的函数
from time import perf_counter  # 导入性能计数器

# 导入所需的配置和上下文
from .config import sample_data  # 从配置文件中导入示例数据
from .context import pandas_ta  # 导入上下文中的 pandas_ta 模块

from unittest import skip, skipUnless, TestCase  # 导入单元测试相关的函数和类
from pandas import DataFrame  # 导入 DataFrame 类

# 策略测试参数设置
cores = cpu_count()  # 获取 CPU 核心数量
cumulative = False  # 是否累积计算
speed_table = False  # 是否生成速度表
strategy_timed = False  # 是否记录策略的时间
timed = True  # 是否记录时间
verbose = False  # 是否冗长输出

# 测试策略方法的类
class TestStrategyMethods(TestCase):

    @classmethod
    def setUpClass(cls):
        # 设置测试类的共享数据
        cls.data = sample_data  # 使用示例数据
        cls.data.ta.cores = cores  # 设置数据的核心数量
        cls.speed_test = DataFrame()  # 创建一个空的 DataFrame 用于存储速度测试结果

    @classmethod
    def tearDownClass(cls):
        # 在测试类执行完成后的清理工作
        cls.speed_test = cls.speed_test.T  # 转置速度测试结果
        cls.speed_test.index.name = "Test"  # 设置索引名称为 "Test"
        cls.speed_test.columns = ["Columns", "Seconds"]  # 设置列名
        if cumulative:  # 如果设置了累积计算
            cls.speed_test["Cum. Seconds"] = cls.speed_test["Seconds"].cumsum()  # 计算累积秒数
        if speed_table:  # 如果设置了生成速度表
            cls.speed_test.to_csv("tests/speed_test.csv")  # 将速度测试结果保存为 CSV 文件
        if timed:  # 如果记录时间
            tca = cls.speed_test['Columns'].sum()  # 总列数
            tcs = cls.speed_test['Seconds'].sum()  # 总秒数
            cps = f"[i] Total Columns / Second for All Tests: { tca / tcs:.5f} "  # 计算每秒处理的列数
            print("=" * len(cps))  # 打印分隔线
            print(cls.speed_test)  # 打印速度测试结果
            print(f"[i] Cores: {cls.data.ta.cores}")  # 打印核心数量
            print(f"[i] Total Datapoints per run: {cls.data.shape[0]}")  # 打印每次运行的数据点总数
            print(f"[i] Total Columns added: {tca}")  # 打印添加的总列数
            print(f"[i] Total Seconds for All Tests: {tcs:.5f}")  # 打印所有测试的总秒数
            print(cps)  # 打印每秒处理的列数
            print("=" * len(cps))  # 打印分隔线
        del cls.data  # 删除示例数据，释放内存

    def setUp(self):
        # 每个测试方法执行前的设置工作
        self.added_cols = 0  # 添加的列数初始化为 0
        self.category = ""  # 测试类别初始化为空字符串
        self.init_cols = len(self.data.columns)  # 记录初始列数
        self.time_diff = 0  # 计算时间差初始化为 0
        self.result = None  # 测试结果初始化为 None
        if verbose:  # 如果设置了冗长输出
            print()  # 输出空行
        if timed:  # 如果记录时间
            self.stime = perf_counter()  # 记录开始时间

    def tearDown(self):
        # 每个测试方法执行后的清理工作
        if timed:  # 如果记录时间
            self.time_diff = perf_counter() - self.stime  # 计算时间差
        self.added_cols = len(self.data.columns) - self.init_cols  # 计算添加的列数
        self.assertGreaterEqual(self.added_cols, 1)  # 断言添加的列数大于等于 1

        self.result = self.data[self.data.columns[-self.added_cols:]]  # 获取测试结果
        self.assertIsInstance(self.result, DataFrame)  # 断言测试结果是 DataFrame 类型
        self.data.drop(columns=self.result.columns, axis=1, inplace=True)  # 在数据中删除测试结果的列

        self.speed_test[self.category] = [self.added_cols, self.time_diff]  # 记录测试结果到速度测试 DataFrame 中

    # 测试所有策略
    # @skip
    def test_all(self):
        self.category = "All"  # 设置测试类别为 "All"
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)  # 调用策略函数进行测试

    # 测试所有有序策略
    def test_all_ordered(self):
        self.category = "All"  # 设置测试类别为 "All"
        self.data.ta.strategy(ordered=True, verbose=verbose, timed=strategy_timed)  # 调用有序策略函数进行测试
        self.category = "All Ordered"  # 重命名测试类别为 "All Ordered"，用于速度表显示

    # 只在冗长输出模式下运行的测试方法
    @skipUnless(verbose, "verbose mode only")
    def test_all_strategy(self):
        self.data.ta.strategy(pandas_ta.AllStrategy, verbose=verbose, timed=strategy_timed)  # 调用特定策略进行测试
    # 仅在 verbose 模式下才运行该测试，用于测试所有名称策略
    @skipUnless(verbose, "verbose mode only")
    def test_all_name_strategy(self):
        self.category = "All"
        # 调用 ta.strategy() 方法执行指定类别的策略，传入 verbose 和 timed 参数
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # 测试所有多参数策略
    def test_all_multiparams_strategy(self):
        self.category = "All"
        # 调用 ta.strategy() 方法执行指定类别的策略，传入不同的参数长度
        self.data.ta.strategy(self.category, length=10, verbose=verbose, timed=strategy_timed)
        self.data.ta.strategy(self.category, length=50, verbose=verbose, timed=strategy_timed)
        self.data.ta.strategy(self.category, fast=5, slow=10, verbose=verbose, timed=strategy_timed)
        # 重命名类别以供速度表使用
        self.category = "All Multiruns with diff Args"

    # 测试 candles 类别策略
    def test_candles_category(self):
        self.category = "Candles"
        # 调用 ta.strategy() 方法执行指定类别的策略，传入 verbose 和 timed 参数
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # 测试 common 类别策略
    def test_common(self):
        self.category = "Common"
        # 调用 ta.strategy() 方法执行指定类别的策略，传入 verbose 和 timed 参数
        self.data.ta.strategy(pandas_ta.CommonStrategy, verbose=verbose, timed=strategy_timed)

    # 测试 cycles 类别策略
    def test_cycles_category(self):
        self.category = "Cycles"
        # 调用 ta.strategy() 方法执行指定类别的策略，传入 verbose 和 timed 参数
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)

    # 测试 custom A 类别策略
    def test_custom_a(self):
        self.category = "Custom A"
        print()
        print(self.category)

        # 自定义指标列表
        momo_bands_sma_ta = [
            {"kind": "cdl_pattern", "name": "tristar"},  # 1
            {"kind": "rsi"},  # 1
            {"kind": "macd"},  # 3
            {"kind": "sma", "length": 50},  # 1
            {"kind": "sma", "length": 200 },  # 1
            {"kind": "bbands", "length": 20},  # 3
            {"kind": "log_return", "cumulative": True},  # 1
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"} # 1
        ]

        # 创建自定义策略对象
        custom = pandas_ta.Strategy(
            "Commons with Cumulative Log Return EMA Chain",  # name
            momo_bands_sma_ta,  # ta
            "Common indicators with specific lengths and a chained indicator",  # description
        )
        # 调用 ta.strategy() 方法执行自定义策略，传入 verbose 和 timed 参数
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        # 断言数据列数为 15
        self.assertEqual(len(self.data.columns), 15)

    # 测试 custom B 类别策略
    def test_custom_args_tuple(self):
        self.category = "Custom B"

        # 自定义指标参数列表
        custom_args_ta = [
            {"kind": "ema", "params": (5,)},
            {"kind": "fisher", "params": (13, 7)}
        ]

        # 创建自定义策略对象
        custom = pandas_ta.Strategy(
            "Custom Args Tuple",
            custom_args_ta,
            "Allow for easy filling in indicator arguments by argument placement."
        )
        # 调用 ta.strategy() 方法执行自定义策略，传入 verbose 和 timed 参数
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
    # 测试自定义列名元组
    def test_custom_col_names_tuple(self):
        # 设置类别为 "Custom C"
        self.category = "Custom C"
    
        # 自定义参数列表，包含一个字典，指定了指标类型和列名元组
        custom_args_ta = [{"kind": "bbands", "col_names": ("LB", "MB", "UB", "BW", "BP")}]
    
        # 创建自定义策略对象
        custom = pandas_ta.Strategy(
            "Custom Col Numbers Tuple",  # 策略名称
            custom_args_ta,  # 自定义参数
            "Allow for easy renaming of resultant columns",  # 描述
        )
        # 应用策略到数据
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试自定义列数字元组
    def test_custom_col_numbers_tuple(self):
        # 设置类别为 "Custom D"
        self.category = "Custom D"
    
        # 自定义参数列表，包含一个字典，指定了指标类型和列数字元组
        custom_args_ta = [{"kind": "macd", "col_numbers": (1,)}]
    
        # 创建自定义策略对象
        custom = pandas_ta.Strategy(
            "Custom Col Numbers Tuple",  # 策略名称
            custom_args_ta,  # 自定义参数
            "Allow for easy selection of resultant columns",  # 描述
        )
        # 应用策略到数据
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试自定义指标
    def test_custom_a(self):
        # 设置类别为 "Custom E"
        self.category = "Custom E"
    
        # 自定义参数列表，包含多个字典，指定了不同的指标类型和参数
        amat_logret_ta = [
            {"kind": "amat", "fast": 20, "slow": 50 },  # AMAT指标
            {"kind": "log_return", "cumulative": True},  # 对数收益率
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5}  # 指数移动平均线
        ]
    
        # 创建自定义策略对象
        custom = pandas_ta.Strategy(
            "AMAT Log Returns",  # 策略名称
            amat_logret_ta,  # 自定义参数
            "AMAT Log Returns",  # 描述
        )
        # 应用策略到数据，设置ordered=True按照给定顺序执行指标计算
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed, ordered=True)
        # 添加信号列
        self.data.ta.tsignals(trend=self.data["AMATe_LR_20_50_2"], append=True)
        # 断言结果列数量为13
        self.assertEqual(len(self.data.columns), 13)
    
    # @skip
    # 测试动量类别指标
    def test_momentum_category(self):
        # 设置类别为 "Momentum"
        self.category = "Momentum"
        # 应用指定类别的策略到数据
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试重叠类别指标
    def test_overlap_category(self):
        # 设置类别为 "Overlap"
        self.category = "Overlap"
        # 应用指定类别的策略到数据
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试性能类别指标
    def test_performance_category(self):
        # 设置类别为 "Performance"
        self.category = "Performance"
        # 应用指定类别的策略到数据
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试统计类别指标
    def test_statistics_category(self):
        # 设置类别为 "Statistics"
        self.category = "Statistics"
        # 应用指定类别的策略到数据
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试趋势类别指标
    def test_trend_category(self):
        # 设置类别为 "Trend"
        self.category = "Trend"
        # 应用指定类别的策略到数据
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试波动率类别指标
    def test_volatility_category(self):
        # 设置类别为 "Volatility"
        self.category = "Volatility"
        # 应用指定类别的策略到数据
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)
    
    # @skip
    # 测试交易量类别指标
    def test_volume_category(self):
        # 设置类别为 "Volume"
        self.category = "Volume"
        # 应用指定类别的策略到数据
        self.data.ta.strategy(self.category, verbose=verbose, timed=strategy_timed)
    
    # @skipUnless(verbose, "verbose mode only")
    # 定义一个测试方法，用于测试不使用多进程的情况
    def test_all_no_multiprocessing(self):
        # 设置测试类别为"All with No Multiprocessing"
        self.category = "All with No Multiprocessing"

        # 保存当前核心数，并将核心数设置为0
        cores = self.data.ta.cores
        self.data.ta.cores = 0
        # 运行策略，设置参数verbose和timed
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)
        # 恢复原来的核心数
        self.data.ta.cores = cores

    # @skipUnless(verbose, "verbose mode only")
    # 定义一个测试方法，用于测试不使用多进程的自定义情况
    def test_custom_no_multiprocessing(self):
        # 设置测试类别为"Custom A with No Multiprocessing"
        self.category = "Custom A with No Multiprocessing"

        # 保存当前核心数，并将核心数设置为0
        cores = self.data.ta.cores
        self.data.ta.cores = 0

        # 定义一组包含不同指标的策略
        momo_bands_sma_ta = [
            {"kind": "rsi"},  # 1
            {"kind": "macd"},  # 3
            {"kind": "sma", "length": 50},  # 1
            {"kind": "sma", "length": 100, "col_names": "sma100"},  # 1
            {"kind": "sma", "length": 200 },  # 1
            {"kind": "bbands", "length": 20},  # 3
            {"kind": "log_return", "cumulative": True},  # 1
            {"kind": "ema", "close": "CUMLOGRET_1", "length": 5, "suffix": "CLR"}
        ]

        # 创建自定义策略对象
        custom = pandas_ta.Strategy(
            "Commons with Cumulative Log Return EMA Chain",  # name
            momo_bands_sma_ta,  # ta
            "Common indicators with specific lengths and a chained indicator",  # description
        )
        # 运行自定义策略，设置参数verbose和timed
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        # 恢复原来的核心数
        self.data.ta.cores = cores
```