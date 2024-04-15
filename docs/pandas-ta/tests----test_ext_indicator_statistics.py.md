# `.\pandas-ta\tests\test_ext_indicator_statistics.py`

```py
# 从config模块中导入sample_data变量
from .config import sample_data
# 从context模块中导入pandas_ta模块
from .context import pandas_ta
# 从unittest模块中导入skip和TestCase类
from unittest import skip, TestCase
# 从pandas模块中导入DataFrame类
from pandas import DataFrame

# 定义测试统计扩展功能的测试类
class TestStatisticsExtension(TestCase):
    # 在所有测试方法执行前执行，设置测试数据
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data

    # 在所有测试方法执行后执行，删除测试数据
    @classmethod
    def tearDownClass(cls):
        del cls.data

    # 测试方法执行前执行的方法
    def setUp(self): pass
    # 测试方法执行后执行的方法
    def tearDown(self): pass

    # 测试计算熵扩展功能
    def test_entropy_ext(self):
        # 调用entropy方法计算熵，并将结果追加到数据中
        self.data.ta.entropy(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"ENTP_10"

    # 测试计算峰度扩展功能
    def test_kurtosis_ext(self):
        # 调用kurtosis方法计算峰度，并将结果追加到数据中
        self.data.ta.kurtosis(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"KURT_30"

    # 测试计算绝对平均偏差扩展功能
    def test_mad_ext(self):
        # 调用mad方法计算绝对平均偏差，并将结果追加到数据中
        self.data.ta.mad(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"MAD_30"

    # 测试计算中位数扩展功能
    def test_median_ext(self):
        # 调用median方法计算中位数，并将结果追加到数据中
        self.data.ta.median(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"MEDIAN_30"

    # 测试计算分位数扩展功能
    def test_quantile_ext(self):
        # 调用quantile方法计算分位数，并将结果追加到数据中
        self.data.ta.quantile(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"QTL_30_0.5"

    # 测试计算偏度扩展功能
    def test_skew_ext(self):
        # 调用skew方法计算偏度，并将结果追加到数据中
        self.data.ta.skew(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"SKEW_30"

    # 测试计算标准差扩展功能
    def test_stdev_ext(self):
        # 调用stdev方法计算标准差，并将结果追加到数据中
        self.data.ta.stdev(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"STDEV_30"

    # 测试计算所有时间序列标准差扩展功能
    def test_tos_stdevall_ext(self):
        # 调用tos_stdevall方法计算所有时间序列标准差，并将结果追加到数据中
        self.data.ta.tos_stdevall(append=True)
        # 断言数据类型为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后7列的列名为指定列表

    # 测试计算方差扩展功能
    def test_variance_ext(self):
        # 调用variance方法计算方差，并将结果追加到数据中
        self.data.ta.variance(append=True)
        # 断言数据类���为DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为"VAR_30"
```