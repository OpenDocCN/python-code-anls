# `.\pandas-ta\tests\test_ext_indicator_performance.py`

```py
# 从当前目录下的 config 模块导入 sample_data 对象
from .config import sample_data
# 从当前目录下的 context 模块导入 pandas_ta 扩展
from .context import pandas_ta
# 导入 TestCase 类
from unittest import TestCase
# 导入 DataFrame 类
from pandas import DataFrame

# 定义 TestPerformaceExtension 类，继承自 TestCase 类
class TestPerformaceExtension(TestCase):
    # 在测试类中所有测试方法执行前执行的类方法
    @classmethod
    def setUpClass(cls):
        # 初始化测试数据为 sample_data
        cls.data = sample_data
        # 检查是否收盘价大于其50日简单移动平均线
        cls.islong = cls.data["close"] > pandas_ta.sma(cls.data["close"], length=50)

    # 在测试类中所有测试方法执行后执行的类方法
    @classmethod
    def tearDownClass(cls):
        # 删除测试数据对象
        del cls.data
        # 删除 islong 对象

    # 在每个测试方法执行前执行的方法
    def setUp(self): pass
    # 在每个测试方法执行后执行的方法
    def tearDown(self): pass

    # 测试对数收益率扩展方法
    def test_log_return_ext(self):
        # 计算对数收益率，并将结果附加到数据框中
        self.data.ta.log_return(append=True)
        # 断言数据对象为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据框的最后一列为 "LOGRET_1"
        self.assertEqual(self.data.columns[-1], "LOGRET_1")

    # 测试累积对数收益率扩展方法
    def test_cum_log_return_ext(self):
        # 计算累积对数收益率，并将结果附加到数据框中
        self.data.ta.log_return(append=True, cumulative=True)
        # 断言数据对象为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据框的最后一列为 "CUMLOGRET_1"

    # 测试百分比收益率扩展方法
    def test_percent_return_ext(self):
        # 计算百分比收益率，并将结果附加到数据框中
        self.data.ta.percent_return(append=True)
        # 断言数据对象为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据框的最后一列为 "PCTRET_1"

    # 测试累积百分比收益率扩展方法
    def test_cum_percent_return_ext(self):
        # 计算累积百分比收益率，并将结果附加到数据框中
        self.data.ta.percent_return(append=True, cumulative=True)
        # 断言数据对象为 DataFrame 类型
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据框的最后一列为 "CUMPCTRET_1"
```