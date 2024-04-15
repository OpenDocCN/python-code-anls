# `.\pandas-ta\tests\test_utils_metrics.py`

```py
# 导入 unittest 模块中的 skip 和 TestCase 类
from unittest import skip, TestCase
# 导入 pandas 模块中的 DataFrame 类
from pandas import DataFrame
# 从当前目录下的 config 模块中导入 sample_data 变量
from .config import sample_data
# 从当前目录下的 context 模块中导入 pandas_ta 模块
from .context import pandas_ta

# 定义测试类 TestUtilityMetrics，继承自 TestCase 类
class TestUtilityMetrics(TestCase):

    # 类方法，设置测试所需的数据
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data
        cls.close = cls.data["close"]
        # 计算价格序列的百分比收益率，不累积
        cls.pctret = pandas_ta.percent_return(cls.close, cumulative=False)
        # 计算价格序列的对数收益率，不累积
        cls.logret = pandas_ta.percent_return(cls.close, cumulative=False)

    # 类方法，清理测试所需的数据
    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.pctret
        del cls.logret

    # 设置测试用例的前置操作
    def setUp(self): pass
    # 设置测试用例的后置操作
    def tearDown(self): pass

    # 测试计算 CAGR 的函数
    def test_cagr(self):
        # 调用 utils 模块中的 cagr 函数计算 CAGR
        result = pandas_ta.utils.cagr(self.data.close)
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于 0
        self.assertGreater(result, 0)

    # 测试计算 Calmar 比率的函数
    def test_calmar_ratio(self):
        # 调用 calmar_ratio 函数计算 Calmar 比率
        result = pandas_ta.calmar_ratio(self.close)
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

        # 传入参数 years=0 调用 calmar_ratio 函数
        result = pandas_ta.calmar_ratio(self.close, years=0)
        # 断言返回结果为 None
        self.assertIsNone(result)

        # 传入参数 years=-2 调用 calmar_ratio 函数
        result = pandas_ta.calmar_ratio(self.close, years=-2)
        # 断言返回结果为 None
        self.assertIsNone(result)

    # 测试计算下行偏差的函数
    def test_downside_deviation(self):
        # 调用 downside_deviation 函数计算下行偏差
        result = pandas_ta.downside_deviation(self.pctret)
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

        # 调用 downside_deviation 函数计算下行偏差
        result = pandas_ta.downside_deviation(self.logret)
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

    # 测试计算回撤的函数
    def test_drawdown(self):
        # 调用 drawdown 函数计算回撤
        result = pandas_ta.drawdown(self.pctret)
        # 断言返回结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "DD"
        self.assertEqual(result.name, "DD")

        # 调用 drawdown 函数计算回撤
        result = pandas_ta.drawdown(self.logret)
        # 断言返回结果类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言结果的名称为 "DD"
        self.assertEqual(result.name, "DD")

    # 测试计算 Jensen's Alpha 的函数
    def test_jensens_alpha(self):
        # 从百分比收益率中随机抽取与收盘价序列长度相同的样本作为基准收益率
        bench_return = self.pctret.sample(n=self.close.shape[0], random_state=1)
        # 调用 jensens_alpha 函数计算 Jensen's Alpha
        result = pandas_ta.jensens_alpha(self.close, bench_return)
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

    # 测试计算对数最大回撤的函数
    def test_log_max_drawdown(self):
        # 调用 log_max_drawdown 函数计算对数最大回撤
        result = pandas_ta.log_max_drawdown(self.close)
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

    # 测试计算最大回撤的函数
    def test_max_drawdown(self):
        # 调用 max_drawdown 函数计算最大回撤
        result = pandas_ta.max_drawdown(self.close)
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

        # 传入参数 method="percent" 调用 max_drawdown 函数
        result = pandas_ta.max_drawdown(self.close, method="percent")
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

        # 传入参数 method="log" 调用 max_drawdown 函数
        result = pandas_ta.max_drawdown(self.close, method="log")
        # 断言返回结果类型为浮点数
        self.assertIsInstance(result, float)
        # 断言计算结果大于等于 0
        self.assertGreaterEqual(result, 0)

        # 传入参数 all=True 调用 max_drawdown 函数
        result = pandas_ta.max_drawdown(self.close, all=True)
        # 断言返回结果类型为字典
        self.assertIsInstance(result, dict)
        # 断言字
    # 测试 optimal_leverage 函数，计算最佳杠杆倍数
    def test_optimal_leverage(self):
        # 调用 optimal_leverage 函数，计算默认参数情况下的最佳杠杆倍数
        result = pandas_ta.optimal_leverage(self.close)
        # 断言结果为整数类型
        self.assertIsInstance(result, int)
        # 调用 optimal_leverage 函数，计算开启日志情况下的最佳杠杆倍数
        result = pandas_ta.optimal_leverage(self.close, log=True)
        # 断言结果为整数类型
        self.assertIsInstance(result, int)

    # 测试 pure_profit_score 函数，计算纯利得分
    def test_pure_profit_score(self):
        # 调用 pure_profit_score 函数，计算纯利得分
        result = pandas_ta.pure_profit_score(self.close)
        # 断言结果大于等于0
        self.assertGreaterEqual(result, 0)

    # 测试 sharpe_ratio 函数，计算夏普比率
    def test_sharpe_ratio(self):
        # 调用 sharpe_ratio 函数，计算夏普比率
        result = pandas_ta.sharpe_ratio(self.close)
        # 断言结果为浮点数类型
        self.assertIsInstance(result, float)
        # 断言结果大于等于0
        self.assertGreaterEqual(result, 0)

    # 测试 sortino_ratio 函数，计算索提诺比率
    def test_sortino_ratio(self):
        # 调用 sortino_ratio 函数，计算索提诺比率
        result = pandas_ta.sortino_ratio(self.close)
        # 断言结果为浮点数类型
        self.assertIsInstance(result, float)
        # 断言结果大于等于0
        self.assertGreaterEqual(result, 0)

    # 测试 volatility 函数，计算波动率
    def test_volatility(self):
        # 计算收益率
        returns_ = pandas_ta.percent_return(self.close)
        # 调用 volatility 函数，计算波动率，返回结果包含收益率信息
        result = pandas_ta.utils.volatility(returns_, returns=True)
        # 断言结果为浮点数类型
        self.assertIsInstance(result, float)
        # 断言结果大于等于0
        self.assertGreaterEqual(result, 0)

        # 遍历不同时间段
        for tf in ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]:
            # 调用 volatility 函数，计算指定时间段的波动率
            result = pandas_ta.utils.volatility(self.close, tf)
            # 使用子测试，传入时间段参数
            with self.subTest(tf=tf):
                # 断言结果为浮点数类型
                self.assertIsInstance(result, float)
                # 断言结果大于等于0
                self.assertGreaterEqual(result, 0)
```