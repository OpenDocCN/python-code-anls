# `.\pandas-ta\tests\test_utils.py`

```
# 从config模块中导入sample_data变量
from .config import sample_data
# 从context模块中导入pandas_ta模块
from .context import pandas_ta

# 从unittest模块中导入skip和TestCase类
from unittest import skip, TestCase
# 从unittest.mock模块中导入patch函数
from unittest.mock import patch

# 导入numpy模块并重命名为np
import numpy as np
# 导入numpy.testing模块并重命名为npt
import numpy.testing as npt
# 从pandas模块中导入DataFrame和Series类
from pandas import DataFrame, Series
# 从pandas.api.types模块中导入is_datetime64_ns_dtype和is_datetime64tz_dtype函数
from pandas.api.types import is_datetime64_ns_dtype, is_datetime64tz_dtype

# 定义一个数据字典
data = {
    "zero": [0, 0],
    "a": [0, 1],
    "b": [1, 0],
    "c": [1, 1],
    "crossed": [0, 1],
}

# 定义一个测试类TestUtilities，继承自TestCase类
class TestUtilities(TestCase):

    # 类方法，设置测试类的数据
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data

    # 类方法，清理测试类的数据
    @classmethod
    def tearDownClass(cls):
        del cls.data

    # 实例方法，每个测试用例执行前的初始化操作
    def setUp(self):
        self.crosseddf = DataFrame(data)
        self.utils = pandas_ta.utils

    # 实例方法，每个测试用例执行后的清理操作
    def tearDown(self):
        del self.crosseddf
        del self.utils

    # 测试用例，测试_add_prefix_suffix方法
    def test__add_prefix_suffix(self):
        # 测试添加前缀
        result = self.data.ta.hl2(append=False, prefix="pre")
        self.assertEqual(result.name, "pre_HL2")

        # 测试添加后缀
        result = self.data.ta.hl2(append=False, suffix="suf")
        self.assertEqual(result.name, "HL2_suf")

        # 测试同时添加前缀和后缀
        result = self.data.ta.hl2(append=False, prefix="pre", suffix="suf")
        self.assertEqual(result.name, "pre_HL2_suf")

        # 测试添加数字前缀和后缀
        result = self.data.ta.hl2(append=False, prefix=1, suffix=2)
        self.assertEqual(result.name, "1_HL2_2")

        # 测试添加前缀和后缀到MACD指标
        result = self.data.ta.macd(append=False, prefix="pre", suffix="suf")
        for col in result.columns:
            self.assertTrue(col.startswith("pre_") and col.endswith("_suf"))

    # 跳过测试用例
    @skip
    def test__above_below(self):
        # 测试_above_below方法，判断a是否在zero上方
        result = self.utils._above_below(self.crosseddf["a"], self.crosseddf["zero"], above=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "a_A_zero")
        npt.assert_array_equal(result, self.crosseddf["c"])

        # 测试_above_below方法，判断a是否在zero下方
        result = self.utils._above_below(self.crosseddf["a"], self.crosseddf["zero"], above=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "a_B_zero")
        npt.assert_array_equal(result, self.crosseddf["b"])

        # 测试_above_below方法，判断c是否在zero上方
        result = self.utils._above_below(self.crosseddf["c"], self.crosseddf["zero"], above=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "c_A_zero")
        npt.assert_array_equal(result, self.crosseddf["c"])

        # 测试_above_below方法，判断c是否在zero下方
        result = self.utils._above_below(self.crosseddf["c"], self.crosseddf["zero"], above=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "c_B_zero")
        npt.assert_array_equal(result, self.crosseddf["zero"])
    # 测试 utils 模块中 above 函数的功能
    def test_above(self):
        # 调用 above 函数，传入两列数据，返回结果
        result = self.utils.above(self.crosseddf["a"], self.crosseddf["zero"])
        # 断言返回结果的类型为 Series
        self.assertIsInstance(result, Series)
        # 断言返回结果的名称为 "a_A_zero"
        self.assertEqual(result.name, "a_A_zero")
        # 使用 numpy.testing.assert_array_equal 方法检查返回结果是否与预期一致
        npt.assert_array_equal(result, self.crosseddf["c"])

        # 再次调用 above 函数，参数顺序颠倒，进行测试
        result = self.utils.above(self.crosseddf["zero"], self.crosseddf["a"])
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "zero_A_a")
        npt.assert_array_equal(result, self.crosseddf["b"])

    # 测试 utils 模块中 above_value 函数的功能
    def test_above_value(self):
        # 调用 above_value 函数，传入一列数据和一个值，返回结果
        result = self.utils.above_value(self.crosseddf["a"], 0)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "a_A_0")
        npt.assert_array_equal(result, self.crosseddf["c"])

        # 再次调用 above_value 函数，传入不合法参数进行测试
        result = self.utils.above_value(self.crosseddf["a"], self.crosseddf["zero"])
        self.assertIsNone(result)

    # 测试 utils 模块中 below 函数的功能
    def test_below(self):
        # 调用 below 函数，传入两列数据，返回结果
        result = self.utils.below(self.crosseddf["zero"], self.crosseddf["a"])
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "zero_B_a")
        npt.assert_array_equal(result, self.crosseddf["c"])

        # 再次调用 below 函数，传入两列数据，返回结果
        result = self.utils.below(self.crosseddf["zero"], self.crosseddf["a"])
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "zero_B_a")
        npt.assert_array_equal(result, self.crosseddf["c"])

    # 测试 utils 模块中 below_value 函数的功能
    def test_below_value(self):
        # 调用 below_value 函数，传入一列数据和一个值，返回结果
        result = self.utils.below_value(self.crosseddf["a"], 0)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "a_B_0")
        npt.assert_array_equal(result, self.crosseddf["b"])

        # 再次调用 below_value 函数，传入不合法参数进行测试
        result = self.utils.below_value(self.crosseddf["a"], self.crosseddf["zero"])
        self.assertIsNone(result)

    # 测试 utils 模块中 combination 函数的功能
    def test_combination(self):
        # 测试 combination 函数不传入参数时的情况
        self.assertIsNotNone(self.utils.combination())

        # 测试 combination 函数传入参数 n=0 时的情况
        self.assertEqual(self.utils.combination(), 1)
        self.assertEqual(self.utils.combination(r=-1), 1)

        # 测试 combination 函数传入参数 n=10, r=4, repetition=False 时的情况
        self.assertEqual(self.utils.combination(n=10, r=4, repetition=False), 210)
        # 测试 combination 函数传入参数 n=10, r=4, repetition=True 时的情况
        self.assertEqual(self.utils.combination(n=10, r=4, repetition=True), 715)

    # 测试 utils 模块中 cross 函数的功能（above 参数为 True）
    def test_cross_above(self):
        # 调用 cross 函数，传入两列数据，返回结果
        result = self.utils.cross(self.crosseddf["a"], self.crosseddf["b"])
        self.assertIsInstance(result, Series)
        # 使用 numpy.testing.assert_array_equal 方法检查返回结果是否与预期一致
        npt.assert_array_equal(result, self.crosseddf["crossed"])

        # 再次调用 cross 函数，传入两列数据，返回结果
        result = self.utils.cross(self.crosseddf["a"], self.crosseddf["b"], above=True)
        self.assertIsInstance(result, Series)
        npt.assert_array_equal(result, self.crosseddf["crossed"])

    # 测试 utils 模块中 cross 函数的功能（above 参数为 False）
    def test_cross_below(self):
        # 调用 cross 函数，传入两列数据，返回结果
        result = self.utils.cross(self.crosseddf["b"], self.crosseddf["a"], above=False)
        self.assertIsInstance(result, Series)
        # 使用 numpy.testing.assert_array_equal 方法检查返回结果是否与预期一致
        npt.assert_array_equal(result, self.crosseddf["crossed"])
    # 测试 DataFrame 中日期相关函数的行为

    # 调用 df_dates 函数测试，传入数据 self.data
    result = self.utils.df_dates(self.data)
    # 断言结果是否为 None
    self.assertEqual(None, result)

    # 再次调用 df_dates 函数测试，传入数据 self.data 和指定日期 "1999-11-01"
    result = self.utils.df_dates(self.data, "1999-11-01")
    # 断言结果行数是否为 1
    self.assertEqual(1, result.shape[0])

    # 再次调用 df_dates 函数测试，传入数据 self.data 和多个日期
    result = self.utils.df_dates(self.data, ["1999-11-01", "2020-08-15", "2020-08-24", "2020-08-25", "2020-08-26", "2020-08-27"])
    # 断言结果行数是否为 5
    self.assertEqual(5, result.shape[0])

    # 跳过以下测试函数
    @skip
    def test_df_month_to_date(self):
        result = self.utils.df_month_to_date(self.data)

    @skip
    def test_df_quarter_to_date(self):
        result = self.utils.df_quarter_to_date(self.data)

    @skip
    def test_df_year_to_date(self):
        result = self.utils.df_year_to_date(self.data)

    # 测试 Fibonacci 函数的行为
    def test_fibonacci(self):
        # 断言返回值类型是否为 numpy 数组
        self.assertIs(type(self.utils.fibonacci(zero=True, weighted=False)), np.ndarray)

        # 断言 Fibonacci 函数返回结果是否正确
        npt.assert_array_equal(self.utils.fibonacci(zero=True), np.array([0, 1, 1]))
        npt.assert_array_equal(self.utils.fibonacci(zero=False), np.array([1, 1]))

        # 断言 Fibonacci 函数返回结果是否正确，带参数 n=0
        npt.assert_array_equal(self.utils.fibonacci(n=0, zero=True, weighted=False), np.array([0]))
        npt.assert_array_equal(self.utils.fibonacci(n=0, zero=False, weighted=False), np.array([1]))

        # 断言 Fibonacci 函数返回结果是否正确，带参数 n=5
        npt.assert_array_equal(self.utils.fibonacci(n=5, zero=True, weighted=False), np.array([0, 1, 1, 2, 3, 5]))
        npt.assert_array_equal(self.utils.fibonacci(n=5, zero=False, weighted=False), np.array([1, 1, 2, 3, 5]))

    # 测试带权重的 Fibonacci 函数的行为
    def test_fibonacci_weighted(self):
        # 断言返回值类型是否为 numpy 数组
        self.assertIs(type(self.utils.fibonacci(zero=True, weighted=True)), np.ndarray)
        # 断言 Fibonacci 函数返回结果是否正确，带参数 n=0
        npt.assert_array_equal(self.utils.fibonacci(n=0, zero=True, weighted=True), np.array([0]))
        npt.assert_array_equal(self.utils.fibonacci(n=0, zero=False, weighted=True), np.array([1]))

        # 断言 Fibonacci 函数返回结果是否正确，带参数 n=5
        npt.assert_allclose(self.utils.fibonacci(n=5, zero=True, weighted=True), np.array([0, 1 / 12, 1 / 12, 1 / 6, 1 / 4, 5 / 12]))
        npt.assert_allclose(self.utils.fibonacci(n=5, zero=False, weighted=True), np.array([1 / 12, 1 / 12, 1 / 6, 1 / 4, 5 / 12]))

    # 测试几何平均数函数的行为
    def test_geometric_mean(self):
        # 计算收益率并传入几何平均数函数进行测试
        returns = pandas_ta.percent_return(self.data.close)
        result = self.utils.geometric_mean(returns)
        # 断言结果类型是否为浮点数
        self.assertIsInstance(result, float)

        # 传入一系列数值进行测试
        result = self.utils.geometric_mean(Series([12, 14, 11, 8]))
        # 断言结果类型是否为浮点数
        self.assertIsInstance(result, float)

        # 传入一系列数值进行测试
        result = self.utils.geometric_mean(Series([100, 50, 0, 25, 0, 60]))
        # 断言结果类型是否为浮点数
        self.assertIsInstance(result, float)

        # 传入一系列数值进行测试
        series = Series([0, 1, 2, 3])
        result = self.utils.geometric_mean(series)
        # 断言结果类型是否为浮点数
        self.assertIsInstance(result, float)

        # 传入一系列数值进行测试，包括负数
        result = self.utils.geometric_mean(-series)
        # 断言结果类型是否为整数
        self.assertIsInstance(result, int)
        # 断言结果是否接近 0
        self.assertAlmostEqual(result, 0)
    # 测试获取时间函数
    def test_get_time(self):
        # 测试获取当前时间并转换为字符串
        result = self.utils.get_time(to_string=True)
        # 断言结果为字符串类型
        self.assertIsInstance(result, str)

        # 测试获取指定市场时间并转换为字符串
        result = self.utils.get_time("NZSX", to_string=True)
        # 断言结果字符串包含指定市场代码
        self.assertTrue("NZSX" in result)
        # 断言结果为字符串类型
        self.assertIsInstance(result, str)

        # 测试获取指定市场时间并转换为字符串
        result = self.utils.get_time("SSE", to_string=True)
        # 断言结果为字符串类型
        self.assertIsInstance(result, str)
        # 断言结果字符串包含指定市场代码
        self.assertTrue("SSE" in result)

    # 测试线性回归函数
    def test_linear_regression(self):
        # 创建示例数据
        x = Series([1, 2, 3, 4, 5])
        y = Series([1.8, 2.1, 2.7, 3.2, 4])

        # 进行线性回归
        result = self.utils.linear_regression(x, y)
        # 断言结果为字典类型
        self.assertIsInstance(result, dict)
        # 断言字典中'a'键对应的值为浮点型
        self.assertIsInstance(result["a"], float)
        # 断言字典中'b'键对应的值为浮点型
        self.assertIsInstance(result["b"], float)
        # 断言字典中'r'键对应的值为浮点型
        self.assertIsInstance(result["r"], float)
        # 断言字典中't'键对应的值为浮点型
        self.assertIsInstance(result["t"], float)
        # 断言字典中'line'键对应的值为Series类型
        self.assertIsInstance(result["line"], Series)

    # 测试对数几何平均函数
    def test_log_geometric_mean(self):
        # 计算收益率
        returns = pandas_ta.percent_return(self.data.close)
        # 计算对数几何平均
        result = self.utils.log_geometric_mean(returns)
        # 断言结果为浮点型
        self.assertIsInstance(result, float)

        # 测试对数几何平均函数的其他参数
        result = self.utils.log_geometric_mean(Series([12, 14, 11, 8]))
        self.assertIsInstance(result, float)

        result = self.utils.log_geometric_mean(Series([100, 50, 0, 25, 0, 60]))
        self.assertIsInstance(result, float)

        series = Series([0, 1, 2, 3])
        result = self.utils.log_geometric_mean(series)
        self.assertIsInstance(result, float)

        result = self.utils.log_geometric_mean(-series)
        # 断言结果为整型
        self.assertIsInstance(result, int)
        # 断言结果接近0
        self.assertAlmostEqual(result, 0)

    # 测试帕斯卡三角形函数
    def test_pascals_triangle(self):
        # 测试帕斯卡三角形的反向情况
        self.assertIsNone(self.utils.pascals_triangle(inverse=True), None)

        array_1 = np.array([1])
        # 测试默认参数下的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(), array_1)
        # 测试带权重的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(weighted=True), array_1)
        # 测试带权重且反向的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(weighted=True, inverse=True), np.array([0]))

        array_5 = self.utils.pascals_triangle(n=5)  # or np.array([1, 5, 10, 10, 5, 1])
        array_5w = array_5 / np.sum(array_5)
        array_5iw = 1 - array_5w
        # 测试负数行数的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(n=-5), array_5)
        # 测试负数行数的带权重的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(n=-5, weighted=True), array_5w)
        # 测试负数行数的带权重且反向的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(n=-5, weighted=True, inverse=True), array_5iw)

        # 测试正数行数的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(n=5), array_5)
        # 测试正数行数的带权重的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(n=5, weighted=True), array_5w)
        # 测试正数行数的带权重且反向的帕斯卡三角形
        npt.assert_array_equal(self.utils.pascals_triangle(n=5, weighted=True, inverse=True), array_5iw)
    # 测试函数，验证对称三角形函数的输出是否符合预期
    def test_symmetric_triangle(self):
        # 验证未加权的对称三角形函数输出是否正确
        npt.assert_array_equal(self.utils.symmetric_triangle(), np.array([1,1]))
        # 验证加权的对称三角形函数输出是否正确
        npt.assert_array_equal(self.utils.symmetric_triangle(weighted=True), np.array([0.5, 0.5])

        # 验证 n=4 时对称三角形函数输出是否正确
        array_4 = self.utils.symmetric_triangle(n=4)  # or np.array([1, 2, 2, 1])
        array_4w = array_4 / np.sum(array_4)
        npt.assert_array_equal(self.utils.symmetric_triangle(n=4), array_4)
        npt.assert_array_equal(self.utils.symmetric_triangle(n=4, weighted=True), array_4w)

        # 验证 n=5 时对称三角形函数输出是否正确
        array_5 = self.utils.symmetric_triangle(n=5)  # or np.array([1, 2, 3, 2, 1])
        array_5w = array_5 / np.sum(array_5)
        npt.assert_array_equal(self.utils.symmetric_triangle(n=5), array_5)
        npt.assert_array_equal(self.utils.symmetric_triangle(n=5, weighted=True), array_5w)

    # 测试函数，验证 tal_ma 函数的输出是否符合预期
    def test_tal_ma(self):
        # 验证不同参数输入时 tal_ma 函数的输出是否正确
        self.assertEqual(self.utils.tal_ma("sma"), 0)
        self.assertEqual(self.utils.tal_ma("Sma"), 0)
        self.assertEqual(self.utils.tal_ma("ema"), 1)
        self.assertEqual(self.utils.tal_ma("wma"), 2)
        self.assertEqual(self.utils.tal_ma("dema"), 3)
        self.assertEqual(self.utils.tal_ma("tema"), 4)
        self.assertEqual(self.utils.tal_ma("trima"), 5)
        self.assertEqual(self.utils.tal_ma("kama"), 6)
        self.assertEqual(self.utils.tal_ma("mama"), 7)
        self.assertEqual(self.utils.tal_ma("t3"), 8)

    # 测试函数，验证 zero 函数的输出是否符合预期
    def test_zero(self):
        # 验证不同参数输入时 zero 函数的输出是否正确
        self.assertEqual(self.utils.zero(-0.0000000000000001), 0)
        self.assertEqual(self.utils.zero(0), 0)
        self.assertEqual(self.utils.zero(0.0), 0)
        self.assertEqual(self.utils.zero(0.0000000000000001), 0)

        self.assertNotEqual(self.utils.zero(-0.000000000000001), 0)
        self.assertNotEqual(self.utils.zero(0.000000000000001), 0)
        self.assertNotEqual(self.utils.zero(1), 0)

    # 测试函数，验证 get_drift 函数的输出是否符合预期
    def test_get_drift(self):
        # 验证不同参数输入时 get_drift 函数的输出是否正确
        for s in [0, None, "", [], {}]:
            self.assertIsInstance(self.utils.get_drift(s), int)

        self.assertEqual(self.utils.get_drift(0), 1)
        self.assertEqual(self.utils.get_drift(1.1), 1)
        self.assertEqual(self.utils.get_drift(-1.1), 1)

    # 测试函数，验证 get_offset 函数的输出是否符合预期
    def test_get_offset(self):
        # 验证不同参数输入时 get_offset 函数的输出是否正确
        for s in [0, None, "", [], {}]:
            self.assertIsInstance(self.utils.get_offset(s), int)

        self.assertEqual(self.utils.get_offset(0), 0)
        self.assertEqual(self.utils.get_offset(-1.1), 0)
        self.assertEqual(self.utils.get_offset(1), 1)

    # 测试函数，验证 to_utc 函数的输出是否符合预期
    def test_to_utc(self):
        # 验证 to_utc 函数对数据的处理是否正确
        result = self.utils.to_utc(self.data.copy())
        self.assertTrue(is_datetime64_ns_dtype(result.index))
        self.assertTrue(is_datetime64tz_dtype(result.index))
    # 测试计算给定数据的总时间
    def test_total_time(self):
        # 计算总时间，默认单位为年
        result = self.utils.total_time(self.data)
        # 断言总时间为约30.18年
        self.assertEqual(30.182539682539684, result)

        # 计算总时间，单位为月
        result = self.utils.total_time(self.data, "months")
        # 断言总时间为约250.06个月
        self.assertEqual(250.05753361606995, result)

        # 计算总时间，单位为周
        result = self.utils.total_time(self.data, "weeks")
        # 断言总时间为约1086.57周
        self.assertEqual(1086.5714285714287, result)

        # 计算总时间，单位为天
        result = self.utils.total_time(self.data, "days")
        # 断言总时间为7606天
        self.assertEqual(7606, result)

        # 计算总时间，单位为小时
        result = self.utils.total_time(self.data, "hours")
        # 断言总时间为182544小时
        self.assertEqual(182544, result)

        # 计算总时间，单位为分钟
        result = self.utils.total_time(self.data, "minutes")
        # 断言总时间为10952640分钟
        self.assertEqual(10952640.0, result)

        # 计算总时间，单位为秒
        result = self.utils.total_time(self.data, "seconds")
        # 断言总时间为657158400秒
        self.assertEqual(657158400.0, result)

    # 测试 pandas_ta 库的版本
    def test_version(self):
        # 获取 pandas_ta 库的版本号
        result = pandas_ta.version
        # 断言版本号为字符串类型
        self.assertIsInstance(result, str)
        # 打印 pandas_ta 库的版本信息
        print(f"\nPandas TA v{result}")
```