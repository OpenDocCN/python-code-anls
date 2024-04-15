# `.\pandas-ta\tests\test_indicator_performance.py`

```
# 导入所需的模块和函数
from .config import sample_data
from .context import pandas_ta
# 从 unittest 模块中导入 TestCase 类
from unittest import TestCase
# 从 pandas 模块中导入 Series 类
from pandas import Series

# 定义测试类 TestPerformace，继承自 TestCase 类
class TestPerformace(TestCase):
    # 在所有测试方法执行之前执行的设置方法
    @classmethod
    def setUpClass(cls):
        # 从配置文件中获取示例数据
        cls.data = sample_data
        # 获取示例数据中的 close 列作为测试数据
        cls.close = cls.data["close"]
        # 判断 close 列是否大于其简单移动平均值的 Series 对象，并转换为整数类型
        cls.islong = (cls.close > pandas_ta.sma(cls.close, length=8)).astype(int)
        # 计算 close 列的非累积百分比收益率的 Series 对象
        cls.pctret = pandas_ta.percent_return(cls.close, cumulative=False)
        # 计算 close 列的非累积对数收益率的 Series 对象
        cls.logret = pandas_ta.percent_return(cls.close, cumulative=False)

    # 在所有测试方法执行之后执行的清理方法
    @classmethod
    def tearDownClass(cls):
        # 清理示例数据、close 列、islong 列、pctret 列和 logret 列
        del cls.data
        del cls.close
        del cls.islong
        del cls.pctret
        del cls.logret

    # 设置测试方法前的准备工作，此处不需要执行任何操作，因此留空
    def setUp(self): pass

    # 设置测试方法后的清理工作，此处不需要执行任何操作，因此留空
    def tearDown(self): pass

    # 测试对数收益率计算的方法
    def test_log_return(self):
        # 调用对数收益率计算函数，获取结果
        result = pandas_ta.log_return(self.close)
        # 断言结果类型为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "LOGRET_1"
        self.assertEqual(result.name, "LOGRET_1")

    # 测试累积对数收益率计算的方法
    def test_cum_log_return(self):
        # 调用累积对数收益率计算函数，获取结果
        result = pandas_ta.log_return(self.close, cumulative=True)
        # 断言结果类型为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "CUMLOGRET_1"
        self.assertEqual(result.name, "CUMLOGRET_1")

    # 测试百分比收益率计算的方法
    def test_percent_return(self):
        # 调用百分比收益率计算函数，获取结果
        result = pandas_ta.percent_return(self.close, cumulative=False)
        # 断言结果类型为 Series 类型
        self.assertIsInstance(result, Series)
        # 断言结果的名称为 "PCTRET_1"
        self.assertEqual(result.name, "PCTRET_1")

    # 测试累积百分比收益率计算的方法
    def test_cum_percent_return(self):
        # 调用累积百分比收益率计算函数，获取结果
        result = pandas_ta.percent_return(self.close, cumulative=True)
        # 断言结果的名称为 "CUMPCTRET_1"
        self.assertEqual(result.name, "CUMPCTRET_1")
```