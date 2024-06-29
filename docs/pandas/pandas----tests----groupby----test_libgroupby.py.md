# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_libgroupby.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

from pandas._libs import groupby as libgroupby  # 导入pandas库的内部模块groupby并重命名为libgroupby
from pandas._libs.groupby import (  # 从pandas库的内部模块groupby导入特定函数
    group_cumprod,
    group_cumsum,
    group_mean,
    group_sum,
    group_var,
)

from pandas.core.dtypes.common import ensure_platform_int  # 从pandas核心模块中导入ensure_platform_int函数

from pandas import isna  # 从pandas库中导入isna函数，用于检测缺失值
import pandas._testing as tm  # 导入pandas库的测试模块，并重命名为tm


class GroupVarTestMixin:
    def test_group_var_generic_1d(self):
        prng = np.random.default_rng(2)  # 创建一个基于种子2的随机数生成器

        out = (np.nan * np.ones((5, 1))).astype(self.dtype)  # 创建一个5x1的数组，并初始化为NaN，转换为指定dtype
        counts = np.zeros(5, dtype="int64")  # 创建一个长度为5的整数数组，初始化为0
        values = 10 * prng.random((15, 1)).astype(self.dtype)  # 使用随机数生成器创建一个15x1的数组，乘以10，并转换为指定dtype
        labels = np.tile(np.arange(5), (3,)).astype("intp")  # 使用np.tile创建一个重复数组，用于标记分组

        expected_out = (
            np.squeeze(values).reshape((5, 3), order="F").std(axis=1, ddof=1) ** 2
        )[:, np.newaxis]  # 计算期望输出值的方差
        expected_counts = counts + 3  # 预期计数值增加3

        self.algo(out, counts, values, labels)  # 调用被测试的算法
        assert np.allclose(out, expected_out, self.rtol)  # 断言实际输出与期望输出在指定的相对容差范围内相等
        tm.assert_numpy_array_equal(counts, expected_counts)  # 使用测试模块中的函数断言两个数组是否相等

    def test_group_var_generic_1d_flat_labels(self):
        prng = np.random.default_rng(2)  # 创建一个基于种子2的随机数生成器

        out = (np.nan * np.ones((1, 1))).astype(self.dtype)  # 创建一个1x1的数组，并初始化为NaN，转换为指定dtype
        counts = np.zeros(1, dtype="int64")  # 创建一个长度为1的整数数组，初始化为0
        values = 10 * prng.random((5, 1)).astype(self.dtype)  # 使用随机数生成器创建一个5x1的数组，乘以10，并转换为指定dtype
        labels = np.zeros(5, dtype="intp")  # 创建一个长度为5的整数数组，初始化为0，用于标记分组

        expected_out = np.array([[values.std(ddof=1) ** 2]])  # 计算期望输出值的方差
        expected_counts = counts + 5  # 预期计数值增加5

        self.algo(out, counts, values, labels)  # 调用被测试的算法

        assert np.allclose(out, expected_out, self.rtol)  # 断言实际输出与期望输出在指定的相对容差范围内相等
        tm.assert_numpy_array_equal(counts, expected_counts)  # 使用测试模块中的函数断言两个数组是否相等

    def test_group_var_generic_2d_all_finite(self):
        prng = np.random.default_rng(2)  # 创建一个基于种子2的随机数生成器

        out = (np.nan * np.ones((5, 2))).astype(self.dtype)  # 创建一个5x2的数组，并初始化为NaN，转换为指定dtype
        counts = np.zeros(5, dtype="int64")  # 创建一个长度为5的整数数组，初始化为0
        values = 10 * prng.random((10, 2)).astype(self.dtype)  # 使用随机数生成器创建一个10x2的数组，乘以10，并转换为指定dtype
        labels = np.tile(np.arange(5), (2,)).astype("intp")  # 使用np.tile创建一个重复数组，用于标记分组

        expected_out = np.std(values.reshape(2, 5, 2), ddof=1, axis=0) ** 2  # 计算期望输出值的方差
        expected_counts = counts + 2  # 预期计数值增加2

        self.algo(out, counts, values, labels)  # 调用被测试的算法
        assert np.allclose(out, expected_out, self.rtol)  # 断言实际输出与期望输出在指定的相对容差范围内相等
        tm.assert_numpy_array_equal(counts, expected_counts)  # 使用测试模块中的函数断言两个数组是否相等

    def test_group_var_generic_2d_some_nan(self):
        prng = np.random.default_rng(2)  # 创建一个基于种子2的随机数生成器

        out = (np.nan * np.ones((5, 2))).astype(self.dtype)  # 创建一个5x2的数组，并初始化为NaN，转换为指定dtype
        counts = np.zeros(5, dtype="int64")  # 创建一个长度为5的整数数组，初始化为0
        values = 10 * prng.random((10, 2)).astype(self.dtype)  # 使用随机数生成器创建一个10x2的数组，乘以10，并转换为指定dtype
        values[:, 1] = np.nan  # 将第二列设置为NaN，模拟部分缺失数据
        labels = np.tile(np.arange(5), (2,)).astype("intp")  # 使用np.tile创建一个重复数组，用于标记分组

        expected_out = np.vstack(
            [
                values[:, 0].reshape(5, 2, order="F").std(ddof=1, axis=1) ** 2,
                np.nan * np.ones(5),
            ]
        ).T.astype(self.dtype)  # 计算期望输出值的方差，并处理NaN值
        expected_counts = counts + 2  # 预期计数值增加2

        self.algo(out, counts, values, labels)  # 调用被测试的算法
        tm.assert_almost_equal(out, expected_out, rtol=0.5e-06)  # 使用测试模块中的函数断言两个数组是否几乎相等
        tm.assert_numpy_array_equal(counts, expected_counts)  # 使用测试模块中的函数断言两个数组是否相等
    # 定义一个测试方法，用于测试算法的行为是否符合 GH 10448 的回归测试要求
    def test_group_var_constant(self):
        # 创建一个包含单个 NaN 元素的二维数组，数据类型由 self.dtype 决定
        out = np.array([[np.nan]], dtype=self.dtype)
        # 创建一个包含单个整数 0 的数组，数据类型为 int64
        counts = np.array([0], dtype="int64")
        # 创建一个形状为 (3, 1) 的数组，每个元素均为 0.832845131556193，数据类型由 self.dtype 决定
        values = 0.832845131556193 * np.ones((3, 1), dtype=self.dtype)
        # 创建一个包含三个整数 0 的数组，数据类型为 intp（与平台相关的整数类型）
        labels = np.zeros(3, dtype="intp")

        # 调用 self.algo 方法，传递以上创建的数组作为参数
        self.algo(out, counts, values, labels)

        # 断言 counts 数组的第一个元素是否等于 3
        assert counts[0] == 3
        # 断言 out 数组的第一个元素是否大于等于 0
        assert out[0, 0] >= 0
        # 使用 tm.assert_almost_equal 函数断言 out 数组的第一个元素是否接近于 0.0
        tm.assert_almost_equal(out[0, 0], 0.0)
class TestGroupVarFloat64(GroupVarTestMixin):
    # 定义一个测试类，继承自GroupVarTestMixin
    __test__ = True  # 允许该测试类被pytest发现并执行

    algo = staticmethod(group_var)  # 设置算法方法为group_var静态方法
    dtype = np.float64  # 设置数据类型为64位浮点数
    rtol = 1e-5  # 设置相对误差容忍度为1e-5

    def test_group_var_large_inputs(self):
        # 测试大输入数据下的group_var函数

        # 生成一个伪随机数生成器对象
        prng = np.random.default_rng(2)

        # 初始化输出数组，包含一个NaN值的64位浮点数数组
        out = np.array([[np.nan]], dtype=self.dtype)
        
        # 初始化计数数组，包含一个整数0
        counts = np.array([0], dtype="int64")
        
        # 生成一个随机数组，加上10^12并转换为指定数据类型
        values = (prng.random(10**6) + 10**12).astype(self.dtype)
        values.shape = (10**6, 1)  # 调整形状为(1000000, 1)
        
        # 初始化标签数组，包含1000000个零
        labels = np.zeros(10**6, dtype="intp")

        # 调用算法方法处理数据
        self.algo(out, counts, values, labels)

        # 断言检查计数数组第一个元素是否等于1000000
        assert counts[0] == 10**6
        
        # 使用断言检查输出数组的第一个元素是否接近于1/12，相对误差容忍度为0.5e-3
        tm.assert_almost_equal(out[0, 0], 1.0 / 12, rtol=0.5e-3)


class TestGroupVarFloat32(GroupVarTestMixin):
    # 定义一个测试类，继承自GroupVarTestMixin
    __test__ = True  # 允许该测试类被pytest发现并执行

    algo = staticmethod(group_var)  # 设置算法方法为group_var静态方法
    dtype = np.float32  # 设置数据类型为32位浮点数
    rtol = 1e-2  # 设置相对误差容忍度为1e-2


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_group_ohlc(dtype):
    # 参数化测试函数，测试OHLC数据处理

    # 生成一个指定数据类型和种子的随机数数组
    obj = np.array(np.random.default_rng(2).standard_normal(20), dtype=dtype)

    # 初始化分组边界数组
    bins = np.array([6, 12, 20])
    
    # 初始化输出数组，全零数组
    out = np.zeros((3, 4), dtype)
    
    # 初始化计数数组，全零数组
    counts = np.zeros(len(out), dtype=np.int64)
    
    # 确保标签为整数类型，用以标记分组
    labels = ensure_platform_int(np.repeat(np.arange(3), np.diff(np.r_[0, bins])))

    # 获取libgroupby库中的group_ohlc函数
    func = libgroupby.group_ohlc
    
    # 调用group_ohlc函数处理数据
    func(out, counts, obj[:, None], labels)

    # 定义内部函数_ohlc，返回每组数据的OHLC统计量
    def _ohlc(group):
        if isna(group).all():  # 如果组内所有值为NaN
            return np.repeat(np.nan, 4)  # 返回全为NaN的数组
        return [group[0], group.max(), group.min(), group[-1]]  # 返回开盘价、最高价、最低价、收盘价

    # 生成预期结果，计算每组数据的OHLC统计量
    expected = np.array([_ohlc(obj[:6]), _ohlc(obj[6:12]), _ohlc(obj[12:])])

    # 使用断言检查输出数组是否接近于预期结果
    tm.assert_almost_equal(out, expected)
    
    # 使用断言检查计数数组是否与预期相等
    tm.assert_numpy_array_equal(counts, np.array([6, 6, 8], dtype=np.int64))

    # 将前6个元素设置为NaN
    obj[:6] = np.nan
    
    # 重新调用group_ohlc函数处理修改后的数据
    func(out, counts, obj[:, None], labels)
    
    # 更新预期结果中的第一行为NaN
    expected[0] = np.nan
    
    # 使用断言检查输出数组是否接近于更新后的预期结果
    tm.assert_almost_equal(out, expected)


def _check_cython_group_transform_cumulative(pd_op, np_op, dtype):
    """
    Check a group transform that executes a cumulative function.

    Parameters
    ----------
    pd_op : callable
        The pandas cumulative function.
    np_op : callable
        The analogous one in NumPy.
    dtype : type
        The specified dtype of the data.
    """
    is_datetimelike = False  # 初始化日期时间类似标志为False

    # 生成指定数据类型和形状的数据数组
    data = np.array([[1], [2], [3], [4]], dtype=dtype)
    
    # 初始化答案数组，与data形状相同
    answer = np.zeros_like(data)

    # 初始化标签数组，所有元素为0，表示单一分组
    labels = np.array([0, 0, 0, 0], dtype=np.intp)
    ngroups = 1  # 设置分组数为1
    
    # 调用pd_op函数处理数据，计算累积函数结果
    pd_op(answer, data, labels, ngroups, is_datetimelike)

    # 使用断言检查np_op函数计算的结果是否与答案数组第一列元素相等
    tm.assert_numpy_array_equal(np_op(data), answer[:, 0], check_dtype=False)


@pytest.mark.parametrize("np_dtype", ["int64", "uint64", "float32", "float64"])
def test_cython_group_transform_cumsum(np_dtype):
    # 参数化测试函数，测试累积求和函数

    # 设置np_dtype为指定类型的NumPy数据类型对象
    dtype = np.dtype(np_dtype).type
    
    # 获取累积求和函数和NumPy中的cumsum函数
    pd_op, np_op = group_cumsum, np.cumsum
    
    # 调用_check_cython_group_transform_cumulative函数检查累积求和函数的正确性
    _check_cython_group_transform_cumulative(pd_op, np_op, dtype)


def test_cython_group_transform_cumprod():
    # 测试累积乘积函数

    # 设置数据类型为64位浮点数
    dtype = np.float64
    
    # 获取累积乘积函数和NumPy中的cumprod函数
    pd_op, np_op = group_cumprod, np.cumprod
    
    # 调用_check_cython_group_transform_cumulative函数检查累积乘积函数的正确性
    _check_cython_group_transform_cumulative(pd_op, np_op, dtype)


def test_cython_group_transform_algos():
    # 测试各种算法函数

    is_datetimelike = False  # 初始化日期时间类似标志为False

    # 处理NaN值的情况
    # 创建包含五个元素的 numpy 数组，数据类型为 np.intp，值为 [0, 0, 0, 0, 0]
    labels = np.array([0, 0, 0, 0, 0], dtype=np.intp)
    # 设定组数为 1
    ngroups = 1

    # 创建包含五行一列的 numpy 浮点数数组，值为 [[1], [2], [3], [nan], [4]]
    data = np.array([[1], [2], [3], [np.nan], [4]], dtype="float64")
    # 创建一个与 data 维度和类型相同的全零数组 actual，并将其填充为 nan
    actual = np.zeros_like(data)
    actual.fill(np.nan)
    # 调用 group_cumprod 函数，计算累积乘积，并更新到 actual 数组中
    group_cumprod(actual, data, labels, ngroups, is_datetimelike)
    # 创建一个期望的 numpy 数组 expected，数据类型为 float64，值为 [1, 2, 6, nan, 24]
    expected = np.array([1, 2, 6, np.nan, 24], dtype="float64")
    # 使用测试工具函数 tm.assert_numpy_array_equal 检查 actual[:, 0] 是否与 expected 相等
    tm.assert_numpy_array_equal(actual[:, 0], expected)

    # 重置 actual 数组为与 data 相同维度和类型的全零数组，并填充为 nan
    actual = np.zeros_like(data)
    actual.fill(np.nan)
    # 调用 group_cumsum 函数，计算累积求和，并更新到 actual 数组中
    group_cumsum(actual, data, labels, ngroups, is_datetimelike)
    # 创建一个期望的 numpy 数组 expected，数据类型为 float64，值为 [1, 3, 6, nan, 10]
    expected = np.array([1, 3, 6, np.nan, 10], dtype="float64")
    # 使用测试工具函数 tm.assert_numpy_array_equal 检查 actual[:, 0] 是否与 expected 相等
    tm.assert_numpy_array_equal(actual[:, 0], expected)

    # 设置 is_datetimelike 为 True，表示处理日期时间类数据
    is_datetimelike = True
    # 创建包含五个元素的 numpy 时间差数组 data，数据类型为 m8[ns]，单位为纳秒
    data = np.array([np.timedelta64(1, "ns")] * 5, dtype="m8[ns]")[:, None]
    # 创建与 data 维度相同的全零数组 actual，数据类型为 int64
    actual = np.zeros_like(data, dtype="int64")
    # 调用 group_cumsum 函数，计算时间差的累积求和，并更新到 actual 数组中
    group_cumsum(actual, data.view("int64"), labels, ngroups, is_datetimelike)
    # 创建一个期望的 numpy 时间差数组 expected，单位为纳秒，值为 [1ns, 2ns, 3ns, 4ns, 5ns]
    expected = np.array(
        [
            np.timedelta64(1, "ns"),
            np.timedelta64(2, "ns"),
            np.timedelta64(3, "ns"),
            np.timedelta64(4, "ns"),
            np.timedelta64(5, "ns"),
        ]
    )
    # 使用测试工具函数 tm.assert_numpy_array_equal 检查 actual[:, 0] 是否与 expected 相等
    tm.assert_numpy_array_equal(actual[:, 0].view("m8[ns]"), expected)
# 定义测试函数，计算和测试使用 Cython 编写的组均值函数 `group_mean` 对于日期时间类型数据的处理

def test_cython_group_mean_datetimelike():
    # 创建一个形状为 (1, 1)，元素为 0 的 float64 数组
    actual = np.zeros(shape=(1, 1), dtype="float64")
    # 创建一个包含单个元素 0 的 int64 数组
    counts = np.array([0], dtype="int64")
    # 创建一个包含 timedelta64 值的 numpy 数组，转换为 float64 类型
    data = (
        np.array(
            [np.timedelta64(2, "ns"), np.timedelta64(4, "ns"), np.timedelta64("NaT")],
            dtype="m8[ns]",
        )[:, None]
        .view("int64")
        .astype("float64")
    )
    # 创建一个长度与数据数组相同的 intp 类型的全零数组
    labels = np.zeros(len(data), dtype=np.intp)

    # 调用 group_mean 函数，计算组均值
    group_mean(actual, counts, data, labels, is_datetimelike=True)

    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_numpy_array_equal(actual[:, 0], np.array([3], dtype="float64"))


# 定义测试函数，验证 `group_mean` 对于最小计数错误的处理

def test_cython_group_mean_wrong_min_count():
    # 创建一个形状为 (1, 1)，元素为 0 的 float64 数组
    actual = np.zeros(shape=(1, 1), dtype="float64")
    # 创建一个包含单个元素 0 的 int64 数组
    counts = np.zeros(1, dtype="int64")
    # 创建一个包含单个元素 0 的 float64 数组，作为数据
    data = np.zeros(1, dtype="float64")[:, None]
    # 创建一个包含单个元素 0 的 intp 类型的数组，作为标签
    labels = np.zeros(1, dtype=np.intp)

    # 使用 pytest 的上下文管理器检查是否抛出预期的 AssertionError
    with pytest.raises(AssertionError, match="min_count"):
        # 调用 group_mean 函数，并传入 is_datetimelike 和 min_count 参数
        group_mean(actual, counts, data, labels, is_datetimelike=True, min_count=0)


# 定义测试函数，验证 `group_mean` 对于非日期时间类型但包含 NaT 值的数据的处理

def test_cython_group_mean_not_datetimelike_but_has_NaT_values():
    # 创建一个形状为 (1, 1)，元素为 0 的 float64 数组
    actual = np.zeros(shape=(1, 1), dtype="float64")
    # 创建一个包含单个元素 0 的 int64 数组
    counts = np.array([0], dtype="int64")
    # 创建一个包含 timedelta64 NaT 值的 numpy 数组，转换为 float64 类型
    data = (
        np.array(
            [np.timedelta64("NaT"), np.timedelta64("NaT")],
            dtype="m8[ns]",
        )[:, None]
        .view("int64")
        .astype("float64")
    )
    # 创建一个长度与数据数组相同的 intp 类型的全零数组
    labels = np.zeros(len(data), dtype=np.intp)

    # 调用 group_mean 函数，计算组均值
    group_mean(actual, counts, data, labels, is_datetimelike=False)

    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_numpy_array_equal(
        actual[:, 0], np.array(np.divide(np.add(data[0], data[1]), 2), dtype="float64")
    )


# 定义测试函数，验证 `group_mean` 对于数据中存在 Inf 值的处理

def test_cython_group_mean_Inf_at_begining_and_end():
    # GH 50367
    # 创建一个 2x2 的 float64 数组，元素为 NaN
    actual = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype="float64")
    # 创建一个包含两个元素 0 的 int64 数组
    counts = np.array([0, 0], dtype="int64")
    # 创建一个包含 Inf 和数字值的 float64 数组
    data = np.array(
        [[np.inf, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5, np.inf]],
        dtype="float64",
    )
    # 创建一个包含标签的 intp 类型的数组
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)

    # 调用 group_mean 函数，计算组均值
    group_mean(actual, counts, data, labels, is_datetimelike=False)

    # 创建一个预期结果的 float64 数组
    expected = np.array([[np.inf, 3], [3, np.inf]], dtype="float64")

    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_numpy_array_equal(
        actual,
        expected,
    )


# 使用 pytest 的 parametrize 装饰器定义参数化测试函数，验证 `group_sum` 对于数据中存在 Inf 值的不同情况处理

@pytest.mark.parametrize(
    "values, out",
    [
        ([[np.inf], [np.inf], [np.inf]], [[np.inf], [np.inf]]),
        ([[np.inf], [np.inf], [-np.inf]], [[np.inf], [np.nan]]),
        ([[np.inf], [-np.inf], [np.inf]], [[np.inf], [np.nan]]),
        ([[np.inf], [-np.inf], [-np.inf]], [[np.inf], [-np.inf]]),
    ],
)
def test_cython_group_sum_Inf_at_begining_and_end(values, out):
    # GH #53606
    # 创建一个形状为 (2, 1)，元素为 NaN 的 float64 数组
    actual = np.array([[np.nan], [np.nan]], dtype="float64")
    # 创建一个包含两个元素 0 的 int64 数组
    counts = np.array([0, 0], dtype="int64")
    # 创建一个包含 values 中值的 float64 数组
    data = np.array(values, dtype="float64")
    # 创建一个包含标签的 intp 类型的数组
    labels = np.array([0, 1, 1], dtype=np.intp)

    # 调用 group_sum 函数，计算组和
    group_sum(actual, counts, data, labels, None, is_datetimelike=False)

    # 创建一个预期结果的 float64 数组
    expected = np.array(out, dtype="float64")

    # 使用 pytest 的 assert 函数验证结果是否符合预期
    tm.assert_numpy_array_equal(
        actual,
        expected,
    )
```