# `D:\src\scipysrc\pandas\pandas\tests\test_algos.py`

```
# 导入必要的模块和库

from datetime import datetime  # 导入datetime模块中的datetime类
import struct  # 导入struct模块

import numpy as np  # 导入NumPy库，并用np作为别名
import pytest  # 导入pytest模块

from pandas._libs import (  # 从pandas._libs中导入指定模块
    algos as libalgos,  # 导入algos模块并重命名为libalgos
    hashtable as ht,  # 导入hashtable模块并重命名为ht
)

from pandas.core.dtypes.common import (  # 从pandas.core.dtypes.common导入指定函数
    is_bool_dtype,  # 导入is_bool_dtype函数，用于检查布尔类型
    is_complex_dtype,  # 导入is_complex_dtype函数，用于检查复数类型
    is_float_dtype,  # 导入is_float_dtype函数，用于检查浮点类型
    is_integer_dtype,  # 导入is_integer_dtype函数，用于检查整数类型
    is_object_dtype,  # 导入is_object_dtype函数，用于检查对象类型
)
from pandas.core.dtypes.dtypes import (  # 从pandas.core.dtypes.dtypes导入指定类
    CategoricalDtype,  # 导入CategoricalDtype类，用于分类数据类型
    DatetimeTZDtype,  # 导入DatetimeTZDtype类，用于带时区的日期时间数据类型
)

import pandas as pd  # 导入pandas库，并用pd作为别名
from pandas import (  # 从pandas中导入多个类和函数
    Categorical,  # 导入Categorical类，用于创建分类数据
    CategoricalIndex,  # 导入CategoricalIndex类，用于分类数据的索引
    DataFrame,  # 导入DataFrame类，用于创建数据框
    DatetimeIndex,  # 导入DatetimeIndex类，用于日期时间索引
    Index,  # 导入Index类，用于通用索引
    IntervalIndex,  # 导入IntervalIndex类，用于区间索引
    MultiIndex,  # 导入MultiIndex类，用于多级索引
    NaT,  # 导入NaT，表示不可用的日期时间
    Period,  # 导入Period类，用于时期数据
    PeriodIndex,  # 导入PeriodIndex类，用于时期索引
    Series,  # 导入Series类，用于序列数据
    Timedelta,  # 导入Timedelta类，用于时间差数据
    Timestamp,  # 导入Timestamp类，用于时间戳数据
    cut,  # 导入cut函数，用于根据指定的分箱规则离散化数据
    date_range,  # 导入date_range函数，用于生成日期范围
    timedelta_range,  # 导入timedelta_range函数，用于生成时间差范围
    to_datetime,  # 导入to_datetime函数，用于将输入转换为datetime类型
    to_timedelta,  # 导入to_timedelta函数，用于将输入转换为timedelta类型
)
import pandas._testing as tm  # 导入pandas._testing模块，并用tm作为别名
import pandas.core.algorithms as algos  # 导入pandas.core.algorithms模块，并用algos作为别名
from pandas.core.arrays import (  # 从pandas.core.arrays导入指定类
    DatetimeArray,  # 导入DatetimeArray类，用于处理日期时间数组
    TimedeltaArray,  # 导入TimedeltaArray类，用于处理时间差数组
)
import pandas.core.common as com  # 导入pandas.core.common模块，并用com作为别名


class TestFactorize:
    def test_factorize_complex(self):
        # 测试复数数组的因子化
        # GH#17927
        array = np.array([1, 2, 2 + 1j], dtype=complex)  # 创建复数数组
        labels, uniques = algos.factorize(array)  # 对数组进行因子化

        expected_labels = np.array([0, 1, 2], dtype=np.intp)  # 预期的标签数组
        tm.assert_numpy_array_equal(labels, expected_labels)  # 断言标签数组与预期相等

        expected_uniques = np.array([(1 + 0j), (2 + 0j), (2 + 1j)], dtype=complex)  # 预期的唯一值数组
        tm.assert_numpy_array_equal(uniques, expected_uniques)  # 断言唯一值数组与预期相等

    def test_factorize(self, index_or_series_obj, sort):
        # 测试因子化方法
        obj = index_or_series_obj  # 设置测试对象
        result_codes, result_uniques = obj.factorize(sort=sort)  # 调用因子化方法获取结果

        constructor = Index  # 默认索引构造器为Index类
        if isinstance(obj, MultiIndex):  # 如果对象是MultiIndex类型
            constructor = MultiIndex.from_tuples  # 将构造器改为根据元组创建MultiIndex的方法
        expected_arr = obj.unique()  # 获取对象的唯一值数组
        if expected_arr.dtype == np.float16:  # 如果唯一值数组的数据类型是np.float16
            expected_arr = expected_arr.astype(np.float32)  # 将其转换为np.float32类型
        expected_uniques = constructor(expected_arr)  # 根据唯一值数组使用构造器创建预期的唯一值对象
        if (
            isinstance(obj, Index)  # 如果对象是Index类型
            and expected_uniques.dtype == bool  # 预期的唯一值对象的数据类型是布尔型
            and obj.dtype == object  # 对象的数据类型是对象型
        ):
            expected_uniques = expected_uniques.astype(object)  # 将预期的唯一值对象转换为对象型

        if sort:  # 如果需要排序
            expected_uniques = expected_uniques.sort_values()  # 对预期的唯一值对象进行排序

        # 构造一个整数ndarray，使得`expected_uniques.take(expected_codes)`等于`obj`
        expected_uniques_list = list(expected_uniques)  # 将预期的唯一值对象转换为列表
        expected_codes = [expected_uniques_list.index(val) for val in obj]  # 构造预期的代码数组
        expected_codes = np.asarray(expected_codes, dtype=np.intp)  # 将预期的代码数组转换为np.intp类型

        tm.assert_numpy_array_equal(result_codes, expected_codes)  # 断言结果的代码数组与预期相等
        tm.assert_index_equal(result_uniques, expected_uniques, exact=True)  # 断言结果的唯一值对象与预期相等

    def test_series_factorize_use_na_sentinel_false(self):
        # 测试序列因子化方法，使用NA标志为false
        # GH#35667
        values = np.array([1, 2, 1, np.nan])  # 创建包含NA的数组
        ser = Series(values)  # 创建序列对象
        codes, uniques = ser.factorize(use_na_sentinel=False)  # 调用因子化方法，不使用NA标志

        expected_codes = np.array([0, 1, 0, 2], dtype=np.intp)  # 预期的代码数组
        expected_uniques = Index([1.0, 2.0, np.nan])  # 预期的唯一值对象

        tm.assert_numpy_array_equal(codes, expected_codes)  # 断言代码数组与预期相等
        tm.assert_index_equal(uniques, expected_uniques)  # 断言唯一值对象与预期相等
    def test_basic(self):
        # 创建一个包含字符串对象的 NumPy 数组
        items = np.array(["a", "b", "b", "a", "a", "c", "c", "c"], dtype=object)
        # 调用 factorize 函数，返回编码和唯一值数组
        codes, uniques = algos.factorize(items)
        # 断言唯一值数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(uniques, np.array(["a", "b", "c"], dtype=object))

        # 再次调用 factorize 函数，使用排序参数为 True
        codes, uniques = algos.factorize(items, sort=True)
        # 预期的编码结果数组
        exp = np.array([0, 1, 1, 0, 0, 2, 2, 2], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = np.array(["a", "b", "c"], dtype=object)
        # 断言唯一值数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(uniques, exp)

        # 创建一个倒序的整数 NumPy 数组
        arr = np.arange(5, dtype=np.intp)[::-1]

        # 调用 factorize 函数，返回编码和唯一值数组
        codes, uniques = algos.factorize(arr)
        # 预期的编码结果数组
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = np.array([4, 3, 2, 1, 0], dtype=arr.dtype)
        # 断言唯一值数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(uniques, exp)

        # 再次调用 factorize 函数，使用排序参数为 True
        codes, uniques = algos.factorize(arr, sort=True)
        # 预期的编码结果数组
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = np.array([0, 1, 2, 3, 4], dtype=arr.dtype)
        # 断言唯一值数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(uniques, exp)

        # 创建一个倒序的浮点数 NumPy 数组
        arr = np.arange(5.0)[::-1]

        # 调用 factorize 函数，返回编码和唯一值数组
        codes, uniques = algos.factorize(arr)
        # 预期的编码结果数组
        exp = np.array([0, 1, 2, 3, 4], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=arr.dtype)
        # 断言唯一值数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(uniques, exp)

        # 再次调用 factorize 函数，使用排序参数为 True
        codes, uniques = algos.factorize(arr, sort=True)
        # 预期的编码结果数组
        exp = np.array([4, 3, 2, 1, 0], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=arr.dtype)
        # 断言唯一值数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(uniques, exp)

    def test_mixed(self):
        # 使用 Series 创建包含混合类型数据的对象
        # 文档示例 reshaping.rst
        x = Series(["A", "A", np.nan, "B", 3.14, np.inf])
        # 调用 factorize 函数，返回编码和唯一值数组
        codes, uniques = algos.factorize(x)
        # 预期的编码结果数组
        exp = np.array([0, 0, -1, 1, 2, 3], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = Index(["A", "B", 3.14, np.inf])
        # 断言唯一值数组与预期的 Index 相等
        tm.assert_index_equal(uniques, exp)

        # 再次调用 factorize 函数，使用排序参数为 True
        codes, uniques = algos.factorize(x, sort=True)
        # 预期的编码结果数组
        exp = np.array([2, 2, -1, 3, 0, 1], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = Index([3.14, np.inf, "A", "B"])
        # 断言唯一值数组与预期的 Index 相等
        tm.assert_index_equal(uniques, exp)

    def test_factorize_datetime64(self):
        # 创建包含 datetime64 类型数据的 Series 对象
        # M8
        v1 = Timestamp("20130101 09:00:00.00004")
        v2 = Timestamp("20130101")
        x = Series([v1, v1, v1, v2, v2, v1])
        # 调用 factorize 函数，返回编码和唯一值数组
        codes, uniques = algos.factorize(x)
        # 预期的编码结果数组
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = DatetimeIndex([v1, v2])
        # 断言唯一值数组与预期的 DatetimeIndex 相等
        tm.assert_index_equal(uniques, exp)

        # 再次调用 factorize 函数，使用排序参数为 True
        codes, uniques = algos.factorize(x, sort=True)
        # 预期的编码结果数组
        exp = np.array([1, 1, 1, 0, 0, 1], dtype=np.intp)
        # 断言编码数组与预期的 NumPy 数组相等
        tm.assert_numpy_array_equal(codes, exp)
        # 预期的唯一值数组
        exp = DatetimeIndex([v2, v1])
        # 断言唯一值数组与预期的 DatetimeIndex 相等
        tm.assert_index_equal(uniques, exp)
    def test_factorize_period(self):
        # 创建一个频率为月份的时间段对象v1和v2，并创建一个包含这些对象的Series x
        v1 = Period("201302", freq="M")
        v2 = Period("201303", freq="M")
        x = Series([v1, v1, v1, v2, v2, v1])

        # 对Series x进行因子化，返回编码codes和唯一值uniques
        codes, uniques = algos.factorize(x)
        # 期望的编码结果
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        # 断言codes与期望结果相等
        tm.assert_numpy_array_equal(codes, exp)
        # 断言uniques与PeriodIndex([v1, v2])相等
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

        # 对Series x进行因子化，同时对唯一值进行排序，返回编码codes和唯一值uniques
        codes, uniques = algos.factorize(x, sort=True)
        # 期望的编码结果
        exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
        # 断言codes与期望结果相等
        tm.assert_numpy_array_equal(codes, exp)
        # 断言uniques与PeriodIndex([v1, v2])相等
        tm.assert_index_equal(uniques, PeriodIndex([v1, v2]))

    def test_factorize_timedelta(self):
        # 创建两个时间增量对象v1和v2，并创建一个包含这些对象的Series x
        v1 = to_timedelta("1 day 1 min")
        v2 = to_timedelta("1 day")
        x = Series([v1, v2, v1, v1, v2, v2, v1])

        # 对Series x进行因子化，返回编码codes和唯一值uniques
        codes, uniques = algos.factorize(x)
        # 期望的编码结果
        exp = np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.intp)
        # 断言codes与期望结果相等
        tm.assert_numpy_array_equal(codes, exp)
        # 断言uniques与to_timedelta([v1, v2])相等
        tm.assert_index_equal(uniques, to_timedelta([v1, v2]))

        # 对Series x进行因子化，同时对唯一值进行排序，返回编码codes和唯一值uniques
        codes, uniques = algos.factorize(x, sort=True)
        # 期望的编码结果
        exp = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.intp)
        # 断言codes与期望结果相等
        tm.assert_numpy_array_equal(codes, exp)
        # 断言uniques与to_timedelta([v2, v1])相等
        tm.assert_index_equal(uniques, to_timedelta([v2, v1]))

    def test_factorize_nan(self):
        # 创建包含NaN的对象key，创建一个ObjectFactorizer对象rizer
        key = np.array([1, 2, 1, np.nan], dtype="O")
        rizer = ht.ObjectFactorizer(len(key))
        
        # 遍历不同的na_sentinel值，对key进行因子化，返回编码ids
        for na_sentinel in (-1, 20):
            ids = rizer.factorize(key, na_sentinel=na_sentinel)
            # 期望的编码结果
            expected = np.array([0, 1, 0, na_sentinel], dtype=np.intp)
            # 断言key的不同值数量与expected的不同值数量相等
            assert len(set(key)) == len(set(expected))
            # 断言key中的NaN值是否映射到na_sentinel，而不是reverse_indexer[na_sentinel]
            tm.assert_numpy_array_equal(pd.isna(key), expected == na_sentinel)
            # 断言ids与期望结果相等
            tm.assert_numpy_array_equal(ids, expected)

    def test_factorizer_with_mask(self):
        # 创建一个包含数据和掩码的数组data和mask，创建一个Int64Factorizer对象rizer
        data = np.array([1, 2, 3, 1, 1, 0], dtype="int64")
        mask = np.array([False, False, False, False, False, True])
        rizer = ht.Int64Factorizer(len(data))
        
        # 对data进行因子化，使用mask进行掩码，返回结果result
        result = rizer.factorize(data, mask=mask)
        # 期望的编码结果
        expected = np.array([0, 1, 2, 0, 0, -1], dtype=np.intp)
        # 断言result与期望结果相等
        tm.assert_numpy_array_equal(result, expected)
        # 期望的唯一值
        expected_uniques = np.array([1, 2, 3], dtype="int64")
        # 断言rizer的唯一值数组与期望的唯一值数组相等
        tm.assert_numpy_array_equal(rizer.uniques.to_array(), expected_uniques)
    def test_factorizer_object_with_nan(self):
        # GH#49549
        # 创建包含 NaN 的 NumPy 数组作为测试数据
        data = np.array([1, 2, 3, 1, np.nan])
        # 使用 ObjectFactorizer 类初始化一个对象 rizer
        rizer = ht.ObjectFactorizer(len(data))
        # 调用 factorize 方法对数据进行因子化处理
        result = rizer.factorize(data.astype(object))
        # 预期的因子化结果
        expected = np.array([0, 1, 2, 0, -1], dtype=np.intp)
        # 断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)
        # 预期的唯一值数组
        expected_uniques = np.array([1, 2, 3], dtype=object)
        # 断言 rizer 对象的唯一值属性与预期唯一值数组相等
        tm.assert_numpy_array_equal(rizer.uniques.to_array(), expected_uniques)

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), "nonsense"],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), "nonsense"],
            ),
            (
                [(1, 1), (1, 2), (0, 0), (1, 2), (1, 2, 3)],
                [0, 1, 2, 1, 3],
                [(1, 1), (1, 2), (0, 0), (1, 2, 3)],
            ),
            ([(1, 1), (1, 2), (0, 0), (1, 2)], [0, 1, 2, 1], [(1, 1), (1, 2), (0, 0)]),
        ],
    )
    def test_factorize_tuple_list(self, data, expected_codes, expected_uniques):
        # GH9454
        # 将输入数据转换为对象数组
        data = com.asarray_tuplesafe(data, dtype=object)
        # 使用 pandas 的 factorize 函数对数据进行因子化处理
        codes, uniques = pd.factorize(data)
        # 断言因子化后的 codes 数组与预期结果 expected_codes 相等
        tm.assert_numpy_array_equal(codes, np.array(expected_codes, dtype=np.intp))
        # 将预期的唯一值数组转换为对象数组
        expected_uniques_array = com.asarray_tuplesafe(expected_uniques, dtype=object)
        # 断言因子化后的 uniques 数组与预期唯一值数组相等
        tm.assert_numpy_array_equal(uniques, expected_uniques_array)

    def test_complex_sorting(self):
        # gh 12666 - check no segfault
        # 创建一个复数数组 x17，用于测试复杂排序情况
        x17 = np.array([complex(i) for i in range(17)], dtype=object)

        # 定义预期的异常消息
        msg = "'[<>]' not supported between instances of .*"
        # 使用 pytest 的 raises 断言捕获预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            # 调用 algos.factorize 进行排序，检查是否发生段错误
            algos.factorize(x17[::-1], sort=True)

    def test_numeric_dtype_factorize(self, any_real_numpy_dtype):
        # GH41132
        # 获取一个任意的 NumPy 数值类型
        dtype = any_real_numpy_dtype
        # 创建一个指定数据类型的 NumPy 数组作为测试数据
        data = np.array([1, 2, 2, 1], dtype=dtype)
        # 预期的因子化结果数组
        expected_codes = np.array([0, 1, 1, 0], dtype=np.intp)
        # 预期的唯一值数组
        expected_uniques = np.array([1, 2], dtype=dtype)

        # 调用 algos.factorize 进行因子化处理
        codes, uniques = algos.factorize(data)
        # 断言因子化后的 codes 数组与预期结果 expected_codes 相等
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 断言因子化后的 uniques 数组与预期唯一值数组相等
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_float64_factorize(self, writable):
        # 创建一个浮点数数组作为测试数据，数据类型为 np.float64
        data = np.array([1.0, 1e8, 1.0, 1e-8, 1e8, 1.0], dtype=np.float64)
        # 设置数组为可写入状态
        data.setflags(write=writable)
        # 预期的因子化结果数组
        expected_codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        # 预期的唯一值数组
        expected_uniques = np.array([1.0, 1e8, 1e-8], dtype=np.float64)

        # 调用 algos.factorize 进行因子化处理
        codes, uniques = algos.factorize(data)
        # 断言因子化后的 codes 数组与预期结果 expected_codes 相等
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 断言因子化后的 uniques 数组与预期唯一值数组相等
        tm.assert_numpy_array_equal(uniques, expected_uniques)
    # 测试对无符号 64 位整数数组进行因子分解
    def test_uint64_factorize(self, writable):
        # 创建包含 [2^64 - 1, 1, 2^64 - 1] 的 numpy 数组，数据类型为 uint64
        data = np.array([2**64 - 1, 1, 2**64 - 1], dtype=np.uint64)
        # 允许对数据进行写操作
        data.setflags(write=writable)
        # 期望的编码结果为 [0, 1, 0]，数据类型为 intp
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        # 期望的唯一值数组为 [2^64 - 1, 1]，数据类型为 uint64
        expected_uniques = np.array([2**64 - 1, 1], dtype=np.uint64)

        # 调用 algos.factorize 函数进行因子分解
        codes, uniques = algos.factorize(data)
        # 使用测试工具函数验证编码结果与期望是否相等
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 使用测试工具函数验证唯一值数组与期望是否相等
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    # 测试对有符号 64 位整数数组进行因子分解
    def test_int64_factorize(self, writable):
        # 创建包含 [2^63 - 1, -(2^63), 2^63 - 1] 的 numpy 数组，数据类型为 int64
        data = np.array([2**63 - 1, -(2**63), 2**63 - 1], dtype=np.int64)
        # 允许对数据进行写操作
        data.setflags(write=writable)
        # 期望的编码结果为 [0, 1, 0]，数据类型为 intp
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        # 期望的唯一值数组为 [2^63 - 1, -(2^63)]，数据类型为 int64
        expected_uniques = np.array([2**63 - 1, -(2**63)], dtype=np.int64)

        # 调用 algos.factorize 函数进行因子分解
        codes, uniques = algos.factorize(data)
        # 使用测试工具函数验证编码结果与期望是否相等
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 使用测试工具函数验证唯一值数组与期望是否相等
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    # 测试对字符串数组进行因子分解
    def test_string_factorize(self, writable):
        # 创建包含 ["a", "c", "a", "b", "c"] 的 numpy 数组，数据类型为 object
        data = np.array(["a", "c", "a", "b", "c"], dtype=object)
        # 允许对数据进行写操作
        data.setflags(write=writable)
        # 期望的编码结果为 [0, 1, 0, 2, 1]，数据类型为 intp
        expected_codes = np.array([0, 1, 0, 2, 1], dtype=np.intp)
        # 期望的唯一值数组为 ["a", "c", "b"]，数据类型为 object
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        # 调用 algos.factorize 函数进行因子分解
        codes, uniques = algos.factorize(data)
        # 使用测试工具函数验证编码结果与期望是否相等
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 使用测试工具函数验证唯一值数组与期望是否相等
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    # 测试对包含多种类型数据的对象数组进行因子分解
    def test_object_factorize(self, writable):
        # 创建包含 ["a", "c", None, np.nan, "a", "b", NaT, "c"] 的 numpy 数组，数据类型为 object
        data = np.array(["a", "c", None, np.nan, "a", "b", NaT, "c"], dtype=object)
        # 允许对数据进行写操作
        data.setflags(write=writable)
        # 期望的编码结果为 [0, 1, -1, -1, 0, 2, -1, 1]，数据类型为 intp
        expected_codes = np.array([0, 1, -1, -1, 0, 2, -1, 1], dtype=np.intp)
        # 期望的唯一值数组为 ["a", "c", "b"]，数据类型为 object
        expected_uniques = np.array(["a", "c", "b"], dtype=object)

        # 调用 algos.factorize 函数进行因子分解
        codes, uniques = algos.factorize(data)
        # 使用测试工具函数验证编码结果与期望是否相等
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 使用测试工具函数验证唯一值数组与期望是否相等
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    # 测试对 datetime64 数据类型数组进行因子分解
    def test_datetime64_factorize(self, writable):
        # 创建包含 [np.datetime64("2020-01-01T00:00:00.000")] 的 numpy 数组，数据类型为 datetime64[ns]
        data = np.array([np.datetime64("2020-01-01T00:00:00.000")], dtype="M8[ns]")
        # 允许对数据进行写操作
        data.setflags(write=writable)
        # 期望的编码结果为 [0]，数据类型为 intp
        expected_codes = np.array([0], dtype=np.intp)
        # 期望的唯一值数组为 ["2020-01-01T00:00:00.000000000"]，数据类型为 datetime64[ns]
        expected_uniques = np.array(["2020-01-01T00:00:00.000000000"], dtype="datetime64[ns]")

        # 调用 pd.factorize 函数进行因子分解
        codes, uniques = pd.factorize(data)
        # 使用测试工具函数验证编码结果与期望是否相等
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 使用测试工具函数验证唯一值数组与期望是否相等
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    # 测试对 RangeIndex 进行因子分解
    def test_factorize_rangeindex(self, sort):
        # 创建一个 RangeIndex 对象，其范围为 [0, 1, ..., 9]
        ri = pd.RangeIndex.from_range(range(10))
        # 期望的结果为一个包含 [0, 1, ..., 9] 的 numpy 数组和原始的 RangeIndex 对象
        expected = np.arange(10, dtype=np.intp), ri

        # 调用 algos.factorize 函数进行因子分解，sort 参数决定是否排序
        result = algos.factorize(ri, sort=sort)
        # 使用测试工具函数验证编码结果与期望是否相等
        tm.assert_numpy_array_equal(result[0], expected[0])
        # 使用测试工具函数验证 Index 对象与期望是否相等，确保完全匹配
        tm.assert_index_equal(result[1], expected[1], exact=True)

        # 调用 RangeIndex 对象的 factorize 方法进行因子分解，sort 参数决定是否排序
        result = ri.factorize(sort=sort)
        # 使用测试工具函数验证编码结果与期望是否相等
        tm.assert_numpy_array_equal(result[0], expected[0])
        # 使用测试工具函数验证 Index 对象与期望是否相等，确保完全匹配
        tm.assert_index_equal(result[1], expected[1], exact=True)
    # 定义一个测试方法，用于测试 factorize_rangeindex_decreasing 函数
    def test_factorize_rangeindex_decreasing(self, sort):
        # decreasing -> sort matters
        # 创建一个递减的 RangeIndex 对象 ri，范围从 0 到 9
        ri = pd.RangeIndex.from_range(range(10))
        # 创建期望的结果 expected，包括一个从 0 到 9 的整数数组和 ri 对象本身
        expected = np.arange(10, dtype=np.intp), ri
        
        # 将 ri 反转得到 ri2
        ri2 = ri[::-1]
        # 更新期望结果为原先整数数组的反转和 ri2 对象
        expected = expected[0], ri2
        # 如果 sort 为真，则进一步反转 expected 中的两个元素
        if sort:
            expected = expected[0][::-1], expected[1][::-1]
        
        # 使用 algos 中的 factorize 函数对 ri2 进行因子分解，并获取结果
        result = algos.factorize(ri2, sort=sort)
        # 断言 result[0] 等于期望的第一个元素，即整数数组的反转
        tm.assert_numpy_array_equal(result[0], expected[0])
        # 断言 result[1] 等于期望的第二个元素，即 ri2 对象
        tm.assert_index_equal(result[1], expected[1], exact=True)
        
        # 使用 ri2 对象自身的 factorize 方法进行因子分解，获取结果
        result = ri2.factorize(sort=sort)
        # 断言 result[0] 等于期望的第一个元素，即整数数组的反转
        tm.assert_numpy_array_equal(result[0], expected[0])
        # 断言 result[1] 等于期望的第二个元素，即 ri2 对象
        tm.assert_index_equal(result[1], expected[1], exact=True)

    # 定义一个测试方法，用于测试 deprecate_order 函数
    def test_deprecate_order(self):
        # gh 19727 - check warning is raised for deprecated keyword, order.
        # Test not valid once order keyword is removed.
        # 创建一个包含无符号 64 位整数的数组 data
        data = np.array([2**63, 1, 2**63], dtype=np.uint64)
        # 使用 pytest 检查 algos.factorize 是否会因为 order 关键字而引发 TypeError 异常
        with pytest.raises(TypeError, match="got an unexpected keyword"):
            algos.factorize(data, order=True)
        # 使用 tm.assert_produces_warning 确保 algos.factorize 没有产生警告
        with tm.assert_produces_warning(False):
            algos.factorize(data)

    # 使用 pytest 的参数化装饰器标记，定义一个测试方法，用于测试 parametrized_factorize_na_value_default 函数
    @pytest.mark.parametrize(
        "data",
        [
            np.array([0, 1, 0], dtype="u8"),
            np.array([-(2**63), 1, -(2**63)], dtype="i8"),
            np.array(["__nan__", "foo", "__nan__"], dtype="object"),
        ],
    )
    def test_parametrized_factorize_na_value_default(self, data):
        # arrays that include the NA default for that type, but isn't used.
        # 使用 algos.factorize 对 data 进行因子分解，获取结果 codes 和 uniques
        codes, uniques = algos.factorize(data)
        # 期望的唯一值为 data 中的第一个和第二个元素
        expected_uniques = data[[0, 1]]
        # 期望的因子编码数组为 [0, 1, 0]
        expected_codes = np.array([0, 1, 0], dtype=np.intp)
        # 断言 codes 等于期望的因子编码数组
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 断言 uniques 等于期望的唯一值数组
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    # 使用 pytest 的参数化装饰器标记，定义一个测试方法，用于测试 parametrized_factorize_na_value 函数
    @pytest.mark.parametrize(
        "data, na_value",
        [
            (np.array([0, 1, 0, 2], dtype="u8"), 0),
            (np.array([1, 0, 1, 2], dtype="u8"), 1),
            (np.array([-(2**63), 1, -(2**63), 0], dtype="i8"), -(2**63)),
            (np.array([1, -(2**63), 1, 0], dtype="i8"), 1),
            (np.array(["a", "", "a", "b"], dtype=object), "a"),
            (np.array([(), ("a", 1), (), ("a", 2)], dtype=object), ()),
            (np.array([("a", 1), (), ("a", 1), ("a", 2)], dtype=object), ("a", 1)),
        ],
    )
    def test_parametrized_factorize_na_value(self, data, na_value):
        # 使用 algos 中的 factorize_array 函数对 data 进行因子分解，使用给定的 na_value
        codes, uniques = algos.factorize_array(data, na_value=na_value)
        # 期望的唯一值为 data 中的第二个和第四个元素
        expected_uniques = data[[1, 3]]
        # 期望的因子编码数组为 [-1, 0, -1, 1]
        expected_codes = np.array([-1, 0, -1, 1], dtype=np.intp)
        # 断言 codes 等于期望的因子编码数组
        tm.assert_numpy_array_equal(codes, expected_codes)
        # 断言 uniques 等于期望的唯一值数组
        tm.assert_numpy_array_equal(uniques, expected_uniques)
    @pytest.mark.parametrize(
        "data, uniques",
        [  # 定义参数化测试的输入数据和期望的唯一值
            (
                np.array(["b", "a", None, "b"], dtype=object),  # 第一个参数是包含对象类型数据的 NumPy 数组
                np.array(["b", "a"], dtype=object),  # 第二个参数是期望的唯一值数组，也是对象类型
            ),
            (
                pd.array([2, 1, np.nan, 2], dtype="Int64"),  # 第一个参数是包含整数类型数据的 Pandas 数组
                pd.array([2, 1], dtype="Int64"),  # 第二个参数是期望的唯一值数组，也是整数类型
            ),
        ],
        ids=["numpy_array", "extension_array"],  # 参数化测试的标识符
    )
    def test_factorize_use_na_sentinel(self, sort, data, uniques):
        codes, uniques = algos.factorize(data, sort=sort, use_na_sentinel=True)  # 调用 algos.factorize 函数，使用 NA 哨兵值

        if sort:
            expected_codes = np.array([1, 0, -1, 1], dtype=np.intp)  # 如果 sort 为 True，期望的编码数组
            expected_uniques = algos.safe_sort(uniques)  # 如果 sort 为 True，期望的唯一值数组（经过安全排序）
        else:
            expected_codes = np.array([0, 1, -1, 0], dtype=np.intp)  # 如果 sort 为 False，期望的编码数组
            expected_uniques = uniques  # 如果 sort 为 False，期望的唯一值数组保持不变

        tm.assert_numpy_array_equal(codes, expected_codes)  # 断言编码数组是否与期望相等
        if isinstance(data, np.ndarray):
            tm.assert_numpy_array_equal(uniques, expected_uniques)  # 如果输入数据是 NumPy 数组，则断言唯一值数组是否与期望相等
        else:
            tm.assert_extension_array_equal(uniques, expected_uniques)  # 如果输入数据不是 NumPy 数组，则使用特定的扩展数组断言函数来断言唯一值数组是否与期望相等

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [  # 定义参数化测试的输入数据、期望的编码数组和期望的唯一值数组
            (
                ["a", None, "b", "a"],  # 第一个参数是包含字符串和空值的列表
                np.array([0, 1, 2, 0], dtype=np.dtype("intp")),  # 第二个参数是期望的编码数组，整数类型
                np.array(["a", np.nan, "b"], dtype=object),  # 第三个参数是期望的唯一值数组，对象类型
            ),
            (
                ["a", np.nan, "b", "a"],  # 第一个参数是包含字符串和 NaN 值的列表
                np.array([0, 1, 2, 0], dtype=np.dtype("intp")),  # 第二个参数是期望的编码数组，整数类型
                np.array(["a", np.nan, "b"], dtype=object),  # 第三个参数是期望的唯一值数组，对象类型
            ),
        ],
    )
    def test_object_factorize_use_na_sentinel_false(
        self, data, expected_codes, expected_uniques
    ):
        codes, uniques = algos.factorize(
            np.array(data, dtype=object), use_na_sentinel=False  # 调用 algos.factorize 函数，不使用 NA 哨兵值
        )

        tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)  # 断言唯一值数组是否与期望相等，严格比较 NaN 值
        tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)  # 断言编码数组是否与期望相等，严格比较 NaN 值

    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [  # 定义参数化测试的输入数据、期望的编码数组和期望的唯一值数组
            (
                np.array([1, None, 1, 2], dtype=object),  # 第一个参数是包含对象类型数据的 NumPy 数组
                np.array([0, 1, 0, 2], dtype=np.dtype("intp")),  # 第二个参数是期望的编码数组，整数类型
                np.array([1, np.nan, 2], dtype="O"),  # 第三个参数是期望的唯一值数组，对象类型
            ),
            (
                np.array([1, np.nan, 1, 2], dtype=np.float64),  # 第一个参数是包含浮点数类型数据的 NumPy 数组
                np.array([0, 1, 0, 2], dtype=np.dtype("intp")),  # 第二个参数是期望的编码数组，整数类型
                np.array([1, np.nan, 2], dtype=np.float64),  # 第三个参数是期望的唯一值数组，浮点数类型
            ),
        ],
    )
    def test_int_factorize_use_na_sentinel_false(
        self, data, expected_codes, expected_uniques
    ):
        codes, uniques = algos.factorize(data, use_na_sentinel=False)  # 调用 algos.factorize 函数，不使用 NA 哨兵值

        tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)  # 断言唯一值数组是否与期望相等，严格比较 NaN 值
        tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)  # 断言编码数组是否与期望相等，严格比较 NaN 值
    @pytest.mark.parametrize(
        "data, expected_codes, expected_uniques",
        [  # 使用 pytest 的 parametrize 装饰器，定义多组测试参数
            (
                Index(Categorical(["a", "a", "b"])),  # 创建 Index 对象，包含类别型数据 ["a", "a", "b"]
                np.array([0, 0, 1], dtype=np.intp),  # 预期的编码结果，数组类型为 intp
                CategoricalIndex(["a", "b"], categories=["a", "b"], dtype="category"),  # 预期的唯一索引对象，包含类别 ["a", "b"]
            ),
            (
                Series(Categorical(["a", "a", "b"])),  # 创建 Series 对象，包含类别型数据 ["a", "a", "b"]
                np.array([0, 0, 1], dtype=np.intp),  # 预期的编码结果，数组类型为 intp
                CategoricalIndex(["a", "b"], categories=["a", "b"], dtype="category"),  # 预期的唯一索引对象，包含类别 ["a", "b"]
            ),
            (
                Series(DatetimeIndex(["2017", "2017"], tz="US/Eastern")),  # 创建包含时区信息的 DatetimeIndex 对象
                np.array([0, 0], dtype=np.intp),  # 预期的编码结果，数组类型为 intp
                DatetimeIndex(["2017"], tz="US/Eastern"),  # 预期的 DatetimeIndex 对象，带有时区信息
            ),
        ],
    )
    def test_factorize_mixed_values(self, data, expected_codes, expected_uniques):
        # GH 19721，测试混合数值的因子化函数
        codes, uniques = algos.factorize(data)  # 调用 algos 模块的 factorize 函数进行因子化
        tm.assert_numpy_array_equal(codes, expected_codes)  # 使用测试工具 tm 来断言编码结果是否与预期一致
        tm.assert_index_equal(uniques, expected_uniques)  # 使用测试工具 tm 来断言唯一值索引是否与预期一致

    def test_factorize_interval_non_nano(self, unit):
        # GH#56099，测试区间索引的因子化处理，包括时间单位为非纳秒的情况
        left = DatetimeIndex(["2016-01-01", np.nan, "2015-10-11"]).as_unit(unit)  # 创建左边界 DatetimeIndex，并转换为指定单位
        right = DatetimeIndex(["2016-01-02", np.nan, "2015-10-15"]).as_unit(unit)  # 创建右边界 DatetimeIndex，并转换为指定单位
        idx = IntervalIndex.from_arrays(left, right)  # 使用左右边界创建 IntervalIndex
        codes, cats = idx.factorize()  # 调用 factorize 方法进行因子化
        assert cats.dtype == f"interval[datetime64[{unit}], right]"  # 断言返回的区间类型是否符合预期

        ts = Timestamp(0).as_unit(unit)  # 创建时间戳，并转换为指定单位
        idx2 = IntervalIndex.from_arrays(left - ts, right - ts)  # 使用调整后的左右边界创建新的 IntervalIndex
        codes2, cats2 = idx2.factorize()  # 调用 factorize 方法进行因子化
        assert cats2.dtype == f"interval[timedelta64[{unit}], right]"  # 断言返回的区间类型是否符合预期

        idx3 = IntervalIndex.from_arrays(
            left.tz_localize("US/Pacific"), right.tz_localize("US/Pacific")
        )  # 使用本地化的左右边界创建新的 IntervalIndex
        codes3, cats3 = idx3.factorize()  # 调用 factorize 方法进行因子化
        assert cats3.dtype == f"interval[datetime64[{unit}, US/Pacific], right]"  # 断言返回的区间类型是否符合预期
class TestUnique:
    # 定义测试类 TestUnique
    def test_ints(self):
        # 生成一个包含50个0到99之间随机整数的数组
        arr = np.random.default_rng(2).integers(0, 100, size=50)

        # 调用 algos 模块的 unique 函数，返回结果
        result = algos.unique(arr)
        
        # 断言返回的结果是一个 numpy 数组对象
        assert isinstance(result, np.ndarray)

    def test_objects(self):
        # 生成一个包含50个0到99之间随机整数，并转换为对象类型的数组
        arr = np.random.default_rng(2).integers(0, 100, size=50).astype("O")

        # 调用 algos 模块的 unique 函数，返回结果
        result = algos.unique(arr)

        # 断言返回的结果是一个 numpy 数组对象
        assert isinstance(result, np.ndarray)

    def test_object_refcount_bug(self):
        # 创建一个包含字符串对象的 numpy 数组
        lst = np.array(["A", "B", "C", "D", "E"], dtype=object)
        
        # 循环调用 algos 模块的 unique 函数，1000次，不关心返回结果
        for i in range(1000):
            len(algos.unique(lst))

    def test_index_returned(self, index):
        # GH#57043
        # 对传入的索引对象重复每个元素两次
        index = index.repeat(2)
        
        # 调用 algos 模块的 unique 函数，返回结果
        result = algos.unique(index)

        # 使用 dict.fromkeys 保持顺序，获取唯一值列表
        unique_values = list(dict.fromkeys(index.values))
        
        # 根据索引对象类型进行不同的预期结果构造
        if isinstance(index, MultiIndex):
            expected = MultiIndex.from_tuples(unique_values, names=index.names)
        else:
            expected = Index(unique_values, dtype=index.dtype)
            if isinstance(index.dtype, DatetimeTZDtype):
                expected = expected.normalize()
        
        # 使用测试框架的断言函数比较结果与预期是否相等
        tm.assert_index_equal(result, expected, exact=True)

    def test_dtype_preservation(self, any_numpy_dtype):
        # GH 15442
        # 根据传入的 numpy 数据类型选择不同的数据和唯一值列表
        if any_numpy_dtype in (tm.BYTES_DTYPES + tm.STRING_DTYPES):
            data = [1, 2, 2]
            uniques = [1, 2]
        elif is_integer_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1, 2]
        elif is_float_dtype(any_numpy_dtype):
            data = [1, 2, 2]
            uniques = [1.0, 2.0]
        elif is_complex_dtype(any_numpy_dtype):
            data = [complex(1, 0), complex(2, 0), complex(2, 0)]
            uniques = [complex(1, 0), complex(2, 0)]
        elif is_bool_dtype(any_numpy_dtype):
            data = [True, True, False]
            uniques = [True, False]
        elif is_object_dtype(any_numpy_dtype):
            data = ["A", "B", "B"]
            uniques = ["A", "B"]
        else:
            # datetime64[ns]/M8[ns]/timedelta64[ns]/m8[ns] 在其他地方测试
            data = [1, 2, 2]
            uniques = [1, 2]

        # 创建一个 pandas Series 对象，并调用其 unique 方法，返回结果
        result = Series(data, dtype=any_numpy_dtype).unique()
        
        # 构造预期结果的 numpy 数组
        expected = np.array(uniques, dtype=any_numpy_dtype)

        # 如果数据类型是字符串类型，则将预期结果转换为对象类型
        if any_numpy_dtype in tm.STRING_DTYPES:
            expected = expected.astype(object)

        # 如果预期结果的数据类型属于日期时间相关类型，进行规范化处理
        if expected.dtype.kind in ["m", "M"]:
            assert isinstance(result, (DatetimeArray, TimedeltaArray))
            result = np.array(result)
        
        # 使用测试框架的断言函数比较结果与预期是否相等
        tm.assert_numpy_array_equal(result, expected)
    def test_datetime64_dtype_array_returned(self):
        # 测试用例：检验 datetime64 数据类型数组的返回情况
        dt_arr = np.array(
            [
                "2015-01-03T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
            ],
            dtype="M8[ns]",
        )

        # 创建 datetime64 数据类型的索引
        dt_index = to_datetime(
            [
                "2015-01-03T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
                "2015-01-01T00:00:00.000000000",
            ]
        )
        # 获取唯一值并进行比较
        result = algos.unique(dt_index)
        expected = to_datetime(dt_arr)
        tm.assert_index_equal(result, expected, exact=True)

        # 创建 Series 对象并获取其唯一值
        s = Series(dt_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, dt_arr)
        assert result.dtype == dt_arr.dtype

        # 获取 Series 的值数组并获取其唯一值
        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, dt_arr)
        assert result.dtype == dt_arr.dtype

    def test_datetime_non_ns(self):
        # 测试用例：检验非纳秒级别的 datetime 数组处理
        a = np.array(["2000", "2000", "2001"], dtype="datetime64[s]")
        result = pd.unique(a)
        expected = np.array(["2000", "2001"], dtype="datetime64[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta_non_ns(self):
        # 测试用例：检验非纳秒级别的 timedelta 数组处理
        a = np.array(["2000", "2000", "2001"], dtype="timedelta64[s]")
        result = pd.unique(a)
        expected = np.array([2000, 2001], dtype="timedelta64[s]")
        tm.assert_numpy_array_equal(result, expected)

    def test_timedelta64_dtype_array_returned(self):
        # 测试用例：检验 timedelta64 数据类型数组的返回情况
        td_arr = np.array([31200, 45678, 10000], dtype="m8[ns]")

        # 创建 timedelta64 数据类型的索引
        td_index = to_timedelta([31200, 45678, 31200, 10000, 45678])
        result = algos.unique(td_index)
        expected = to_timedelta(td_arr)
        tm.assert_index_equal(result, expected)
        assert result.dtype == expected.dtype

        # 创建 Series 对象并获取其唯一值
        s = Series(td_index)
        result = algos.unique(s)
        tm.assert_numpy_array_equal(result, td_arr)
        assert result.dtype == td_arr.dtype

        # 获取 Series 的值数组并获取其唯一值
        arr = s.values
        result = algos.unique(arr)
        tm.assert_numpy_array_equal(result, td_arr)
        assert result.dtype == td_arr.dtype

    def test_uint64_overflow(self):
        # 测试用例：检验 uint64 数据类型的处理，特别是溢出情况
        s = Series([1, 2, 2**63, 2**63], dtype=np.uint64)
        exp = np.array([1, 2, 2**63], dtype=np.uint64)
        tm.assert_numpy_array_equal(algos.unique(s), exp)

    def test_nan_in_object_array(self):
        # 测试用例：检验包含 NaN 的 object 类型数组的处理
        duplicated_items = ["a", np.nan, "c", "c"]
        result = pd.unique(np.array(duplicated_items, dtype=object))
        expected = np.array(["a", np.nan, "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)
    def test_categorical(self):
        # 期望返回按照出现顺序排序的分类数据
        expected = Categorical(list("bac"))

        # 期望返回按照类别顺序排序的分类数据
        expected_o = Categorical(list("bac"), categories=list("abc"), ordered=True)

        # GH 15939
        # 创建包含重复元素的分类数据
        c = Categorical(list("baabc"))
        # 获取唯一的分类值
        result = c.unique()
        # 断言唯一化后的结果与期望相同
        tm.assert_categorical_equal(result, expected)

        # 使用算法工具包获取唯一的分类值
        result = algos.unique(c)
        # 断言唯一化后的结果与期望相同
        tm.assert_categorical_equal(result, expected)

        # 创建有序的分类数据
        c = Categorical(list("baabc"), ordered=True)
        # 获取唯一的分类值
        result = c.unique()
        # 断言唯一化后的结果与期望相同
        tm.assert_categorical_equal(result, expected_o)

        # 使用算法工具包获取唯一的分类值
        result = algos.unique(c)
        # 断言唯一化后的结果与期望相同
        tm.assert_categorical_equal(result, expected_o)

        # 创建包含分类数据的序列
        s = Series(Categorical(list("baabc")), name="foo")
        # 获取唯一的分类值
        result = s.unique()
        # 断言唯一化后的结果与期望相同
        tm.assert_categorical_equal(result, expected)

        # 使用 Pandas 工具包获取唯一的分类值
        result = pd.unique(s)
        # 断言唯一化后的结果与期望相同
        tm.assert_categorical_equal(result, expected)

        # 创建包含分类数据的索引
        ci = CategoricalIndex(Categorical(list("baabc"), categories=list("abc")))
        expected = CategoricalIndex(expected)
        # 获取唯一的索引值
        result = ci.unique()
        # 断言唯一化后的结果与期望相同
        tm.assert_index_equal(result, expected)

        # 使用 Pandas 工具包获取唯一的索引值
        result = pd.unique(ci)
        # 断言唯一化后的结果与期望相同
        tm.assert_index_equal(result, expected)

    def test_datetime64tz_aware(self, unit):
        # GH 15939
        # 创建带有时区信息的时间戳索引
        dti = Index(
            [
                Timestamp("20160101", tz="US/Eastern"),
                Timestamp("20160101", tz="US/Eastern"),
            ]
        ).as_unit(unit)
        ser = Series(dti)

        # 获取唯一的时间戳数据
        result = ser.unique()
        # 断言唯一化后的结果与期望相同
        expected = dti[:1]._data
        tm.assert_extension_array_equal(result, expected)

        # 获取唯一的时间戳索引
        result = dti.unique()
        # 断言唯一化后的结果与期望相同
        expected = dti[:1]
        tm.assert_index_equal(result, expected)

        # 使用 Pandas 工具包获取唯一的时间戳数据
        result = pd.unique(ser)
        # 断言唯一化后的结果与期望相同
        expected = dti[:1]._data
        tm.assert_extension_array_equal(result, expected)

        # 使用 Pandas 工具包获取唯一的时间戳索引
        result = pd.unique(dti)
        # 断言唯一化后的结果与期望相同
        expected = dti[:1]
        tm.assert_index_equal(result, expected)

    def test_order_of_appearance(self):
        # 9346
        # 对出现顺序的保证进行轻量级测试
        # 这些也是文档示例
        # 获取唯一的整数数组
        result = pd.unique(Series([2, 1, 3, 3]))
        # 断言唯一化后的结果与期望相同
        tm.assert_numpy_array_equal(result, np.array([2, 1, 3], dtype="int64"))

        # 获取唯一的整数数组
        result = pd.unique(Series([2] + [1] * 5))
        # 断言唯一化后的结果与期望相同
        tm.assert_numpy_array_equal(result, np.array([2, 1], dtype="int64"))

        # 创建包含对象类型数据的数组
        data = np.array(["a", "a", "b", "c"], dtype=object)
        # 获取唯一的对象类型数据
        result = pd.unique(data)
        # 断言唯一化后的结果与期望相同
        expected = np.array(["a", "b", "c"], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # 获取唯一的分类数据
        result = pd.unique(Series(Categorical(list("aabc"))))
        expected = Categorical(list("abc"))
        # 断言唯一化后的结果与期望相同
        tm.assert_categorical_equal(result, expected)
    # 测试函数，用于验证处理 datetime64 数据类型时的顺序和唯一性
    def test_order_of_appearance_dt64(self, unit):
        # 创建包含两个 Timestamp 的 Series，将其转换为指定单位的 datetime64 数据类型
        ser = Series([Timestamp("20160101"), Timestamp("20160101")]).dt.as_unit(unit)
        # 获取 Series 中唯一的值
        result = pd.unique(ser)
        # 期望结果是一个 numpy 数组，包含指定格式的日期字符串
        expected = np.array(["2016-01-01T00:00:00.000000000"], dtype=f"M8[{unit}]")
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数，用于验证带有时区的 datetime64 数据类型的顺序和唯一性
    def test_order_of_appearance_dt64tz(self, unit):
        # 创建带有时区信息的 DatetimeIndex，将其转换为指定单位的 datetime64 数据类型
        dti = DatetimeIndex(
            [
                Timestamp("20160101", tz="US/Eastern"),
                Timestamp("20160101", tz="US/Eastern"),
            ]
        ).as_unit(unit)
        # 获取 DatetimeIndex 中唯一的值
        result = pd.unique(dti)
        # 期望结果是一个 DatetimeIndex，包含指定格式和时区的日期时间字符串
        expected = DatetimeIndex(
            ["2016-01-01 00:00:00"], dtype=f"datetime64[{unit}, US/Eastern]", freq=None
        )
        # 断言两个 DatetimeIndex 是否相等
        tm.assert_index_equal(result, expected)

    # 使用参数化测试，验证处理字符串元组时的唯一性
    @pytest.mark.parametrize(
        "arg ,expected",
        [
            (("1", "1", "2"), np.array(["1", "2"], dtype=object)),
            (("foo",), np.array(["foo"], dtype=object)),
        ],
    )
    def test_tuple_with_strings(self, arg, expected):
        # GH 17108
        # 将元组转换为适用于数组的格式，确保其唯一性
        arg = com.asarray_tuplesafe(arg, dtype=object)
        # 获取数组中的唯一值
        result = pd.unique(arg)
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数，验证处理包含 None 的对象数组时的唯一性保持
    def test_obj_none_preservation(self):
        # GH 20866
        # 创建包含字符串和 None 的对象数组
        arr = np.array(["foo", None], dtype=object)
        # 获取数组中的唯一值，严格检查 NaN 值
        result = pd.unique(arr)
        # 期望结果是包含原数组中的值，严格保持 NaN 值的唯一性
        expected = np.array(["foo", None], dtype=object)
        # 断言两个 numpy 数组是否相等，严格比较 NaN 值
        tm.assert_numpy_array_equal(result, expected, strict_nan=True)

    # 测试函数，验证处理包含有符号零的数组时的唯一性
    def test_signed_zero(self):
        # GH 21866
        # 创建包含正负零的数组
        a = np.array([-0.0, 0.0])
        # 获取数组中的唯一值，确保正负零被视为等同值
        result = pd.unique(a)
        # 期望结果是只包含一个零值的数组
        expected = np.array([-0.0])  # 0.0 and -0.0 are equivalent
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数，验证处理包含不同 NaN 值的数组时的唯一性
    def test_different_nans(self):
        # GH 21866
        # 创建具有不同位模式的 NaN 值
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        # 确保创建的两个 NaN 值不相等
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        # 创建包含这两个 NaN 值的数组
        a = np.array([NAN1, NAN2])  # NAN1 and NAN2 are equivalent
        # 获取数组中的唯一值，期望结果是只包含一个 NaN 值的数组
        result = pd.unique(a)
        expected = np.array([np.nan])
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 使用参数化测试，验证处理不同类型的数组时，首个 NaN 值保持不变
    @pytest.mark.parametrize("el_type", [np.float64, object])
    def test_first_nan_kept(self, el_type):
        # GH 22295
        # 创建不同位模式的 NaN 值
        bits_for_nan1 = 0xFFF8000000000001
        bits_for_nan2 = 0x7FF8000000000001
        NAN1 = struct.unpack("d", struct.pack("=Q", bits_for_nan1))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", bits_for_nan2))[0]
        # 确保创建的两个 NaN 值不相等
        assert NAN1 != NAN1
        assert NAN2 != NAN2
        # 创建包含这两个 NaN 值的数组，指定数据类型
        a = np.array([NAN1, NAN2], dtype=el_type)
        # 获取数组中的唯一值，期望结果是只包含一个 NaN 值的数组
        result = pd.unique(a)
        assert result.size == 1
        # 使用位模式来确定保留了哪个 NaN 值
        result_nan_bits = struct.unpack("=Q", struct.pack("d", result[0]))[0]
        assert result_nan_bits == bits_for_nan1
    # 测试函数，用于验证不混淆非适用值（NA）的情况
    def test_do_not_mangle_na_values(self, unique_nulls_fixture, unique_nulls_fixture2):
        # GH 22295：GitHub 上的 issue 编号
        # 如果 unique_nulls_fixture 和 unique_nulls_fixture2 是同一个对象，则直接返回，跳过测试，因为值不是唯一的
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # 跳过，值不唯一
        # 创建包含两个对象的 numpy 数组，类型为 object
        a = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
        # 获取数组 a 中的唯一值
        result = pd.unique(a)
        # 断言结果数组的大小为 2
        assert result.size == 2
        # 断言数组 a 的第一个元素仍是 unique_nulls_fixture
        assert a[0] is unique_nulls_fixture
        # 断言数组 a 的第二个元素仍是 unique_nulls_fixture2
        assert a[1] is unique_nulls_fixture2

    # 测试函数，验证处理掩码值的唯一性
    def test_unique_masked(self, any_numeric_ea_dtype):
        # GH#48019：GitHub 上的 issue 编号
        # 创建包含带有 NA 值的 Series 对象，数据类型由 any_numeric_ea_dtype 指定
        ser = Series([1, pd.NA, 2] * 3, dtype=any_numeric_ea_dtype)
        # 获取 Series 对象中的唯一值
        result = pd.unique(ser)
        # 创建预期的扩展数组，包含值 1、NA、2，数据类型由 any_numeric_ea_dtype 指定
        expected = pd.array([1, pd.NA, 2], dtype=any_numeric_ea_dtype)
        # 使用测试工具函数验证结果数组与预期数组的内容是否一致
        tm.assert_extension_array_equal(result, expected)
# 定义一个测试函数，用于测试给定的索引、序列或数组的唯一值数量
def test_nunique_ints(index_or_series_or_array):
    # GH#36327
    # 生成一个包含随机整数的数组，范围为 [0, 20)，数量为 30
    values = index_or_series_or_array(np.random.default_rng(2).integers(0, 20, 30))
    # 调用算法模块中的 nunique_ints 函数，计算其唯一值的数量
    result = algos.nunique_ints(values)
    # 计算期望的唯一值数量，使用算法模块中的 unique 函数计算其长度
    expected = len(algos.unique(values))
    # 断言测试结果与期望值相等
    assert result == expected


class TestIsin:
    # 定义一个测试类 TestIsin，用于测试算法模块中的 isin 函数

    def test_invalid(self):
        # 测试 isin 函数对非列表类对象的输入，应该抛出 TypeError 异常
        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a `int`"
        )
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, 1)
        with pytest.raises(TypeError, match=msg):
            algos.isin(1, [1])
        with pytest.raises(TypeError, match=msg):
            algos.isin([1], 1)

    def test_basic(self):
        # 测试 isin 函数对基本输入的处理

        # 测试数组 [1, 2] 是否包含在列表 [1] 中
        result = algos.isin(np.array([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试 pandas Series [1, 2] 是否包含在列表 [1] 中
        result = algos.isin(Series([1, 2]), [1])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试 pandas Series [1, 2] 是否包含在 pandas Series [1] 中
        result = algos.isin(Series([1, 2]), Series([1]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试 pandas Series [1, 2] 是否包含在集合 {1} 中
        result = algos.isin(Series([1, 2]), {1})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试字符串数组 ["a", "b"] 是否包含在列表 ["a"] 中
        arg = np.array(["a", "b"], dtype=object)
        result = algos.isin(arg, ["a"])
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试 pandas Series ["a", "b"] 是否包含在 pandas Series ["a"] 中
        result = algos.isin(Series(arg), Series(["a"]))
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试 pandas Series ["a", "b"] 是否包含在集合 {"a"} 中
        result = algos.isin(Series(arg), {"a"})
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试字符串数组 ["a", "b"] 是否包含在列表 [1] 中
        result = algos.isin(arg, [1])
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_i8(self):
        # 测试 isin 函数对日期和时间间隔数据的处理

        # 生成日期范围为 ["20130101", "20130103"] 的 numpy 数组
        arr = date_range("20130101", periods=3).values
        # 测试数组是否包含第一个元素 arr[0] = "20130101"
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试数组是否包含 arr[0:2] = ["20130101", "20130102"]
        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试数组是否包含集合 set(arr[0:2]) = {"20130101", "20130102"}
        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 生成时间间隔范围为 ["1 day", "3 day"] 的 numpy 数组
        arr = timedelta_range("1 day", periods=3).values
        # 测试数组是否包含第一个元素 arr[0] = "1 days"
        result = algos.isin(arr, [arr[0]])
        expected = np.array([True, False, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试数组是否包含 arr[0:2] = ["1 days", "2 days"]
        result = algos.isin(arr, arr[0:2])
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 测试数组是否包含集合 set(arr[0:2]) = {"1 days", "2 days"}
        result = algos.isin(arr, set(arr[0:2]))
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
    @pytest.mark.parametrize("dtype1", ["m8[ns]", "M8[ns]", "M8[ns, UTC]", "period[D]"])
    @pytest.mark.parametrize("dtype", ["i8", "f8", "u8"])
    # 使用pytest的参数化装饰器，分别对dtype1和dtype参数进行多组参数化测试

    def test_isin_datetimelike_values_numeric_comps(self, dtype, dtype1):
        # Anything but object and we get all-False shortcut
        # 如果dtype不是object类型，算法会使用快捷方式直接返回全False的结果

        dta = date_range("2013-01-01", periods=3)._values
        # 生成一个包含3个日期的时间序列，并获取其内部的值数组

        arr = Series(dta.view("i8")).array.view(dtype1)
        # 将日期时间值转换为64位整数后，根据dtype1视图再次转换为特定的日期时间类型数组

        comps = arr.view("i8").astype(dtype)
        # 将arr视图转换为64位整数后，再按照dtype指定的类型进行类型转换

        result = algos.isin(comps, arr)
        # 使用isin函数检查comps数组中的元素是否在arr数组中，返回布尔类型的结果数组

        expected = np.zeros(comps.shape, dtype=bool)
        # 创建一个与comps形状相同的全零布尔类型数组作为期望的结果

        tm.assert_numpy_array_equal(result, expected)
        # 使用测试模块中的函数验证result数组与expected数组是否相等

    def test_large(self):
        # 生成一个包含2000000个秒级时间序列的时间范围
        s = date_range("20000101", periods=2000000, freq="s").values

        result = algos.isin(s, s[0:2])
        # 使用isin函数检查s数组中的元素是否在s数组的前两个元素中，返回布尔类型的结果数组

        expected = np.zeros(len(s), dtype=bool)
        expected[0] = True
        expected[1] = True
        # 创建一个与s长度相同的全零布尔类型数组，设置前两个元素为True作为期望的结果

        tm.assert_numpy_array_equal(result, expected)
        # 使用测试模块中的函数验证result数组与expected数组是否相等

    @pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]", "M8[ns, UTC]", "period[D]"])
    # 使用pytest的参数化装饰器，对dtype参数进行多组参数化测试

    def test_isin_datetimelike_all_nat(self, dtype):
        # GH#56427
        # 生成一个包含3个日期的时间序列，并获取其内部的值数组
        dta = date_range("2013-01-01", periods=3)._values

        # 将日期时间值转换为64位整数后，根据dtype视图再次转换为特定的日期时间类型数组
        arr = Series(dta.view("i8")).array.view(dtype)

        arr[0] = NaT
        # 将第一个元素设置为NaT（Not a Time，表示缺失时间）

        result = algos.isin(arr, [NaT])
        # 使用isin函数检查arr数组中的元素是否在[NaT]数组中，返回布尔类型的结果数组

        expected = np.array([True, False, False], dtype=bool)
        # 创建一个与arr长度相同的布尔类型数组，期望第一个元素为True，其余为False

        tm.assert_numpy_array_equal(result, expected)
        # 使用测试模块中的函数验证result数组与expected数组是否相等

    @pytest.mark.parametrize("dtype", ["m8[ns]", "M8[ns]", "M8[ns, UTC]"])
    # 使用pytest的参数化装饰器，对dtype参数进行多组参数化测试

    def test_isin_datetimelike_strings_returns_false(self, dtype):
        # GH#53111
        # 生成一个包含3个日期的时间序列，并获取其内部的值数组
        dta = date_range("2013-01-01", periods=3)._values

        # 将日期时间值转换为64位整数后，根据dtype视图再次转换为特定的日期时间类型数组
        arr = Series(dta.view("i8")).array.view(dtype)

        vals = [str(x) for x in arr]
        # 创建一个包含arr中每个元素字符串形式的列表vals

        res = algos.isin(arr, vals)
        # 使用isin函数检查arr数组中的元素是否在vals列表中，返回布尔类型的结果数组

        assert not res.any()
        # 使用断言验证res数组中的所有元素都为False

        vals2 = np.array(vals, dtype=str)
        # 将vals列表转换为NumPy数组，并指定数据类型为字符串

        res2 = algos.isin(arr, vals2)
        # 使用isin函数检查arr数组中的元素是否在vals2数组中，返回布尔类型的结果数组

        assert not res2.any()
        # 使用断言验证res2数组中的所有元素都为False

    def test_isin_dt64tz_with_nat(self):
        # the all-NaT values used to get inferred to tznaive, which was evaluated
        #  as non-matching GH#56427
        # 生成一个包含3个日期时间的时间范围，并指定时区为UTC
        dti = date_range("2016-01-01", periods=3, tz="UTC")
        ser = Series(dti)

        ser[0] = NaT
        # 将序列中的第一个元素设置为NaT（表示缺失时间）

        res = algos.isin(ser._values, [NaT])
        # 使用isin函数检查ser序列的值数组中的元素是否在[NaT]数组中，返回布尔类型的结果数组

        exp = np.array([True, False, False], dtype=bool)
        # 创建一个与ser值数组长度相同的布尔类型数组，期望第一个元素为True，其余为False

        tm.assert_numpy_array_equal(res, exp)
        # 使用测试模块中的函数验证res数组与exp数组是否相等

    def test_categorical_from_codes(self):
        # GH 16639
        vals = np.array([0, 1, 2, 0])
        # 创建一个包含整数的NumPy数组vals

        cats = ["a", "b", "c"]
        # 创建一个字符串列表，表示分类值

        Sd = Series(Categorical([1]).from_codes(vals, cats))
        # 使用from_codes方法创建一个分类序列Sd，根据vals和cats指定分类编码和分类值

        St = Series(Categorical([1]).from_codes(np.array([0, 1]), cats))
        # 使用from_codes方法创建另一个分类序列St，根据指定的编码和cats分类值

        expected = np.array([True, True, False, True])
        # 创建一个布尔类型数组，期望前两个元素为True，其余为False

        result = algos.isin(Sd, St)
        # 使用isin函数检查Sd序列中的元素是否在St序列中，返回布尔类型的结果数组

        tm.assert_numpy_array_equal(expected, result)
        # 使用测试模块中的函数验证result数组与expected数组是否相等
    def test_categorical_isin(self):
        # 创建一个包含整数的 NumPy 数组
        vals = np.array([0, 1, 2, 0])
        # 创建一个包含字符串的列表
        cats = ["a", "b", "c"]
        # 使用给定的整数和字符串创建 Categorical 对象
        cat = Categorical([1]).from_codes(vals, cats)
        # 使用另一个整数数组和相同的字符串列表创建另一个 Categorical 对象
        other = Categorical([1]).from_codes(np.array([0, 1]), cats)

        # 预期的结果，是一个布尔类型的 NumPy 数组
        expected = np.array([True, True, False, True])
        # 调用算法库中的 isin 函数来比较两个 Categorical 对象
        result = algos.isin(cat, other)
        # 使用测试框架中的方法来断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in(self):
        # GH 22160
        # nan 是一个特殊值，因为 "a is b" 不一定跟 "a == b" 相同
        # 至少，isin() 应该遵循 Python 的 "np.nan in [nan] == True" 的行为
        # 将 comps 数组设为包含一个 object 类型的 np.nan（可以被转换为 float64）
        comps = np.array([np.nan], dtype=object)  # 可能会被转换为 float64 类型
        # values 列表中包含 np.nan
        values = [np.nan]
        # 预期结果，是一个包含 True 的 NumPy 数组
        expected = np.array([True])
        # 调用算法库中的 isin 函数来比较 comps 和 values
        result = algos.isin(comps, values)
        # 使用测试框架中的方法来断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(expected, result)

    def test_same_nan_is_in_large(self):
        # https://github.com/pandas-dev/pandas/issues/22205
        # 创建一个长度为 1,000,001 的浮点数数组 s，大部分元素为 1.0，第一个元素为 np.nan
        s = np.tile(1.0, 1_000_001)
        s[0] = np.nan
        # 使用算法库中的 isin 函数来比较 s 和包含 np.nan 和 1 的数组
        result = algos.isin(s, np.array([np.nan, 1]))
        # 预期结果是一个全为 True 的布尔类型的 NumPy 数组
        expected = np.ones(len(s), dtype=bool)
        # 使用测试框架中的方法来断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)

    def test_same_nan_is_in_large_series(self):
        # https://github.com/pandas-dev/pandas/issues/22205
        # 创建一个长度为 1,000,001 的浮点数数组 s，大部分元素为 1.0，第一个元素为 np.nan
        s = np.tile(1.0, 1_000_001)
        # 使用 Pandas 的 Series 类来创建一个 Series 对象
        series = Series(s)
        s[0] = np.nan
        # 使用 Series 对象的 isin 方法来比较其与包含 np.nan 和 1 的数组
        result = series.isin(np.array([np.nan, 1]))
        # 预期结果是一个包含 True 的 Pandas Series 对象
        expected = Series(np.ones(len(s), dtype=bool))
        # 使用测试框架中的方法来断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)

    def test_same_object_is_in(self):
        # GH 22160
        # 可能会对 nan 进行特殊处理
        # 用户可能定义一个类似行为的自定义类，isin() 应该至少遵循 Python 的行为： "a in [a] == True"
        # 创建一个类 LikeNan，覆写了 __eq__ 方法返回 False，覆写 __hash__ 方法返回 0
        class LikeNan:
            def __eq__(self, other) -> bool:
                return False

            def __hash__(self):
                return 0

        # 创建一个包含 LikeNan 对象的 object 类型的 NumPy 数组
        a, b = LikeNan(), LikeNan()
        arg = np.array([a], dtype=object)

        # 对于相同对象，期望返回 True 的 NumPy 数组
        tm.assert_numpy_array_equal(algos.isin(arg, [a]), np.array([True]))
        # 对于不同对象，期望返回 False 的 NumPy 数组
        tm.assert_numpy_array_equal(algos.isin(arg, [b]), np.array([False]))
    def test_different_nans(self):
        # GH 22160
        # 所有的 NaN 被视为等价处理

        comps = [float("nan")]
        values = [float("nan")]
        assert comps[0] is not values[0]  # 不同的 NaN 对象

        # 作为 Python 对象列表：
        result = algos.isin(np.array(comps), values)
        tm.assert_numpy_array_equal(np.array([True]), result)

        # 作为对象数组：
        result = algos.isin(
            np.asarray(comps, dtype=object), np.asarray(values, dtype=object)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

        # 作为 float64 数组：
        result = algos.isin(
            np.asarray(comps, dtype=np.float64), np.asarray(values, dtype=np.float64)
        )
        tm.assert_numpy_array_equal(np.array([True]), result)

    def test_no_cast(self):
        # GH 22160
        # 确保 42 不被转换为字符串
        comps = np.array(["ss", 42], dtype=object)
        values = ["42"]
        expected = np.array([False, False])

        result = algos.isin(comps, values)
        tm.assert_numpy_array_equal(expected, result)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_empty(self, empty):
        # 参见 gh-16991
        vals = Index(["a", "b"])
        expected = np.array([False, False])

        result = algos.isin(vals, empty)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nan_objects(self):
        # GH 22119
        comps = np.array(["nan", np.nan * 1j, float("nan")], dtype=object)
        vals = np.array([float("nan")], dtype=object)
        expected = np.array([False, False, True])
        result = algos.isin(comps, vals)
        tm.assert_numpy_array_equal(expected, result)

    def test_different_nans_as_float64(self):
        # GH 21866
        # 根据位模式创建不同的 NaN，
        # 如果不特别处理，这些 NaN 将落入散列表的不同桶中
        NAN1 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000000))[0]
        NAN2 = struct.unpack("d", struct.pack("=Q", 0x7FF8000000000001))[0]
        assert NAN1 != NAN1
        assert NAN2 != NAN2

        # 检查 NAN1 和 NAN2 是否等价：
        arr = np.array([NAN1, NAN2], dtype=np.float64)
        lookup1 = np.array([NAN1], dtype=np.float64)
        result = algos.isin(arr, lookup1)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        lookup2 = np.array([NAN2], dtype=np.float64)
        result = algos.isin(arr, lookup2)
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
    def test_isin_int_df_string_search(self):
        """测试函数：test_isin_int_df_string_search
        比较整数数据框中的整数（1,2）与字符串的isin()操作("1")
        -> 由于整数1不等于字符串1，因此不应匹配值"""
        # 创建包含整数值的数据框
        df = DataFrame({"values": [1, 2]})
        # 使用isin()方法比较数据框与字符串数组的匹配情况
        result = df.isin(["1"])
        # 期望的比较结果，应该全为False
        expected_false = DataFrame({"values": [False, False]})
        # 使用测试工具比较结果与期望结果是否一致
        tm.assert_frame_equal(result, expected_false)

    def test_isin_nan_df_string_search(self):
        """测试函数：test_isin_nan_df_string_search
        比较包含NaN值的数据框（np.nan,2）与字符串的isin()操作("NaN")
        -> 由于NaN值不等于字符串NaN，因此不应匹配值"""
        # 创建包含NaN值的数据框
        df = DataFrame({"values": [np.nan, 2]})
        # 使用isin()方法比较数据框与字符串数组的匹配情况
        result = df.isin(np.array(["NaN"], dtype=object))
        # 期望的比较结果，应该全为False
        expected_false = DataFrame({"values": [False, False]})
        # 使用测试工具比较结果与期望结果是否一致
        tm.assert_frame_equal(result, expected_false)

    def test_isin_float_df_string_search(self):
        """测试函数：test_isin_float_df_string_search
        比较浮点数数据框中的浮点数（1.4245,2.32441）与字符串的isin()操作("1.4245")
        -> 由于浮点数1.4245不等于字符串1.4245，因此不应匹配值"""
        # 创建包含浮点数值的数据框
        df = DataFrame({"values": [1.4245, 2.32441]})
        # 使用isin()方法比较数据框与字符串数组的匹配情况
        result = df.isin(np.array(["1.4245"], dtype=object))
        # 期望的比较结果，应该全为False
        expected_false = DataFrame({"values": [False, False]})
        # 使用测试工具比较结果与期望结果是否一致
        tm.assert_frame_equal(result, expected_false)

    def test_isin_unsigned_dtype(self):
        """测试函数：test_isin_unsigned_dtype
        检查无符号整数数据类型的isin()操作
        -> 应该返回False，因为数值不匹配"""
        # 创建包含无符号整数的序列
        ser = Series([1378774140726870442], dtype=np.uint64)
        # 使用isin()方法比较序列与给定值的匹配情况
        result = ser.isin([1378774140726870528])
        # 期望的比较结果，应该为False
        expected = Series(False)
        # 使用测试工具比较结果与期望结果是否一致
        tm.assert_series_equal(result, expected)
class TestValueCounts:
    # 定义测试类 TestValueCounts

    def test_value_counts(self):
        # 定义测试方法 test_value_counts
        arr = np.random.default_rng(1234).standard_normal(4)
        # 生成一个包含四个标准正态分布随机数的数组 arr
        factor = cut(arr, 4)
        # 使用 cut 函数对数组 arr 进行分割，返回 factor
        
        # assert isinstance(factor, n)
        # 断言 factor 的类型为 n 类型（此处原文可能有误，应为具体类型，这里保留原注释）
        result = algos.value_counts_internal(factor)
        # 调用 algos 中的 value_counts_internal 函数，计算 factor 中每个元素的频数
        breaks = [-1.606, -1.018, -0.431, 0.155, 0.741]
        # 定义断点列表 breaks
        index = IntervalIndex.from_breaks(breaks).astype(CategoricalDtype(ordered=True))
        # 使用断点列表 breaks 创建一个区间索引，并将其转换为有序分类数据类型的索引 index
        expected = Series([1, 0, 2, 1], index=index, name="count")
        # 创建预期的 Series 对象 expected，包含指定索引 index 和名称为 "count"
        tm.assert_series_equal(result.sort_index(), expected.sort_index())
        # 使用 tm.assert_series_equal 断言 result 和 expected 的排序后索引相等

    def test_value_counts_bins(self):
        # 定义测试方法 test_value_counts_bins
        s = [1, 2, 3, 4]
        # 创建列表 s 包含元素 [1, 2, 3, 4]
        result = algos.value_counts_internal(s, bins=1)
        # 调用 algos 中的 value_counts_internal 函数，计算 s 中每个元素的频数，bins=1
        expected = Series(
            [4], index=IntervalIndex.from_tuples([(0.996, 4.0)]), name="count"
        )
        # 创建预期的 Series 对象 expected，包含指定索引和名称为 "count"
        tm.assert_series_equal(result, expected)
        # 使用 tm.assert_series_equal 断言 result 和 expected 相等

        result = algos.value_counts_internal(s, bins=2, sort=False)
        # 再次调用 value_counts_internal 函数，计算 s 中每个元素的频数，bins=2，不排序
        expected = Series(
            [2, 2],
            index=IntervalIndex.from_tuples([(0.996, 2.5), (2.5, 4.0)]),
            name="count",
        )
        # 创建预期的 Series 对象 expected，包含指定索引和名称为 "count"
        tm.assert_series_equal(result, expected)
        # 使用 tm.assert_series_equal 断言 result 和 expected 相等

    def test_value_counts_dtypes(self):
        # 定义测试方法 test_value_counts_dtypes
        result = algos.value_counts_internal(np.array([1, 1.0]))
        # 调用 algos 中的 value_counts_internal 函数，计算数组中每个元素的频数
        assert len(result) == 1
        # 断言结果 result 的长度为 1

        result = algos.value_counts_internal(np.array([1, 1.0]), bins=1)
        # 再次调用 value_counts_internal 函数，计算数组中每个元素的频数，bins=1
        assert len(result) == 1
        # 断言结果 result 的长度为 1

        result = algos.value_counts_internal(Series([1, 1.0, "1"]))  # object
        # 调用 value_counts_internal 函数，计算 Series 中每个元素的频数，元素类型为 object
        assert len(result) == 2
        # 断言结果 result 的长度为 2

        msg = "bins argument only works with numeric data"
        # 定义异常信息 msg
        with pytest.raises(TypeError, match=msg):
            # 使用 pytest.raises 检查是否会抛出 TypeError 异常，且异常信息匹配 msg
            algos.value_counts_internal(np.array(["1", 1], dtype=object), bins=1)
            # 调用 value_counts_internal 函数，尝试计算非数值数据的频数，bins=1

    def test_value_counts_nat(self):
        # 定义测试方法 test_value_counts_nat
        td = Series([np.timedelta64(10000), NaT], dtype="timedelta64[ns]")
        # 创建时间差类型的 Series 对象 td，包含两个元素
        dt = to_datetime(["NaT", "2014-01-01"])
        # 使用 to_datetime 函数转换列表为日期时间类型 dt

        for ser in [td, dt]:
            # 遍历 Series 对象列表 [td, dt]
            vc = algos.value_counts_internal(ser)
            # 调用 algos 中的 value_counts_internal 函数，计算 ser 中每个元素的频数
            vc_with_na = algos.value_counts_internal(ser, dropna=False)
            # 再次调用 value_counts_internal 函数，计算 ser 中每个元素的频数，不丢弃缺失值
            assert len(vc) == 1
            # 断言结果 vc 的长度为 1
            assert len(vc_with_na) == 2
            # 断言结果 vc_with_na 的长度为 2

        exp_dt = Series({Timestamp("2014-01-01 00:00:00"): 1}, name="count")
        # 创建预期的 Series 对象 exp_dt，包含指定时间戳和名称为 "count"
        result_dt = algos.value_counts_internal(dt)
        # 调用 algos 中的 value_counts_internal 函数，计算 dt 中每个元素的频数
        tm.assert_series_equal(result_dt, exp_dt)
        # 使用 tm.assert_series_equal 断言 result_dt 和 exp_dt 相等

        exp_td = Series({np.timedelta64(10000): 1}, name="count")
        # 创建预期的 Series 对象 exp_td，包含指定时间差和名称为 "count"
        result_td = algos.value_counts_internal(td)
        # 调用 algos 中的 value_counts_internal 函数，计算 td 中每个元素的频数
        tm.assert_series_equal(result_td, exp_td)
        # 使用 tm.assert_series_equal 断言 result_td 和 exp_td 相等

    @pytest.mark.parametrize("dtype", [object, "M8[us]"])
    # 使用 pytest.mark.parametrize 为测试方法指定参数 dtype 的多个取值
    def test_value_counts_datetime_outofbounds(self, dtype):
        # 测试处理超出范围的日期时间数据
        ser = Series(
            [
                datetime(3000, 1, 1),
                datetime(5000, 1, 1),
                datetime(5000, 1, 1),
                datetime(6000, 1, 1),
                datetime(3000, 1, 1),
                datetime(3000, 1, 1),
            ],
            dtype=dtype,
        )

        res = ser.value_counts()  # 计算序列中每个值的频数

        exp_index = Index(
            [datetime(3000, 1, 1), datetime(5000, 1, 1), datetime(6000, 1, 1)],
            dtype=dtype,
        )
        exp = Series([3, 2, 1], index=exp_index, name="count")
        tm.assert_series_equal(res, exp)  # 断言计算结果与期望结果相等

    def test_categorical(self):
        s = Series(Categorical(list("aaabbc")))  # 创建包含分类数据的序列
        result = s.value_counts()  # 计算分类数据的频数
        expected = Series(
            [3, 2, 1], index=CategoricalIndex(["a", "b", "c"]), name="count"
        )

        tm.assert_series_equal(result, expected, check_index_type=True)  # 断言计算结果与期望结果相等

        # 保持顺序？
        s = s.cat.as_ordered()  # 将分类序列转换为有序分类
        result = s.value_counts()  # 计算有序分类数据的频数
        expected.index = expected.index.as_ordered()  # 将期望结果的索引转换为有序
        tm.assert_series_equal(result, expected, check_index_type=True)  # 断言计算结果与期望结果相等

    def test_categorical_nans(self):
        s = Series(Categorical(list("aaaaabbbcc")))  # 创建包含分类数据的序列（带有 NaN 值）
        s.iloc[1] = np.nan  # 将序列中的第二个元素设置为 NaN
        result = s.value_counts()  # 计算分类数据的频数
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c"]),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)  # 断言计算结果与期望结果相等
        result = s.value_counts(dropna=False)  # 计算分类数据的频数（包括 NaN 值）
        expected = Series(
            [4, 3, 2, 1], index=CategoricalIndex(["a", "b", "c", np.nan]), name="count"
        )
        tm.assert_series_equal(result, expected, check_index_type=True)  # 断言计算结果与期望结果相等

        # 无序的情况
        s = Series(
            Categorical(list("aaaaabbbcc"), ordered=True, categories=["b", "a", "c"])
        )  # 创建有序分类数据的序列
        s.iloc[1] = np.nan  # 将序列中的第二个元素设置为 NaN
        result = s.value_counts()  # 计算分类数据的频数
        expected = Series(
            [4, 3, 2],
            index=CategoricalIndex(
                ["a", "b", "c"],
                categories=["b", "a", "c"],
                ordered=True,
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)  # 断言计算结果与期望结果相等

        result = s.value_counts(dropna=False)  # 计算分类数据的频数（包括 NaN 值）
        expected = Series(
            [4, 3, 2, 1],
            index=CategoricalIndex(
                ["a", "b", "c", np.nan], categories=["b", "a", "c"], ordered=True
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected, check_index_type=True)  # 断言计算结果与期望结果相等
    def test_categorical_zeroes(self):
        # 定义一个测试方法，测试分类数据中出现的零值情况
        # 创建一个序列 s，其元素为 Categorical 类型，包含字符串列表 "bbbaac"，指定分类为 "abcd"，并且是有序的
        s = Series(Categorical(list("bbbaac"), categories=list("abcd"), ordered=True))
        # 计算序列 s 中每个元素的频数并返回结果
        result = s.value_counts()
        # 创建一个期望的序列 expected，其元素为 [3, 2, 1, 0]，指定索引为 Categorical 类型，分类为 "abcd"，有序
        expected = Series(
            [3, 2, 1, 0],
            index=Categorical(
                ["b", "a", "c", "d"], categories=list("abcd"), ordered=True
            ),
            name="count",
        )
        # 使用测试工具 tm 检验 result 和 expected 序列是否相等
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_value_counts_dropna(self):
        # 测试 value_counts 方法在 dropna=True 和 dropna=False 时的行为
        # 检验 Series([True, True, False]) 的值出现次数，期望结果为 Series([2, 1], index=[True, False], name="count")
        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=True),
            Series([2, 1], index=[True, False], name="count"),
        )
        # 再次检验 Series([True, True, False]) 的值出现次数，期望结果为 Series([2, 1], index=[True, False], name="count")
        tm.assert_series_equal(
            Series([True, True, False]).value_counts(dropna=False),
            Series([2, 1], index=[True, False], name="count"),
        )

        # 检验 Series([True] * 3 + [False] * 2 + [None] * 5) 的值出现次数，dropna=True
        # 期望结果为 Series([3, 2], index=Index([True, False], dtype=object), name="count")
        tm.assert_series_equal(
            Series([True] * 3 + [False] * 2 + [None] * 5).value_counts(dropna=True),
            Series([3, 2], index=Index([True, False], dtype=object), name="count"),
        )
        # 检验 Series([True] * 5 + [False] * 3 + [None] * 2) 的值出现次数，dropna=False
        # 期望结果为 Series([5, 3, 2], index=[True, False, None], name="count")
        tm.assert_series_equal(
            Series([True] * 5 + [False] * 3 + [None] * 2).value_counts(dropna=False),
            Series([5, 3, 2], index=[True, False, None], name="count"),
        )
        # 检验 Series([10.3, 5.0, 5.0]) 的值出现次数，dropna=True
        # 期望结果为 Series([2, 1], index=[5.0, 10.3], name="count")
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )
        # 再次检验 Series([10.3, 5.0, 5.0]) 的值出现次数，dropna=False
        # 期望结果为 Series([2, 1], index=[5.0, 10.3], name="count")
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0]).value_counts(dropna=False),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )

        # 检验 Series([10.3, 5.0, 5.0, None]) 的值出现次数，dropna=True
        # 期望结果为 Series([2, 1], index=[5.0, 10.3], name="count")
        tm.assert_series_equal(
            Series([10.3, 5.0, 5.0, None]).value_counts(dropna=True),
            Series([2, 1], index=[5.0, 10.3], name="count"),
        )

        # 检验 Series([10.3, 10.3, 5.0, 5.0, 5.0, None]) 的值出现次数，dropna=False
        # 期望结果为 Series([3, 2, 1], index=[5.0, 10.3, None], name="count")
        result = Series([10.3, 10.3, 5.0, 5.0, 5.0, None]).value_counts(dropna=False)
        expected = Series([3, 2, 1], index=[5.0, 10.3, None], name="count")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", (np.float64, object, "M8[ns]"))
    def test_value_counts_normalized(self, dtype):
        # 测试 value_counts 方法在指定数据类型 dtype 下进行归一化处理的行为
        # 创建一个包含数字 1 出现 2 次，数字 2 出现 3 次，以及 NaN 出现 5 次的序列 s
        s = Series([1] * 2 + [2] * 3 + [np.nan] * 5)
        # 将序列 s 转换成指定数据类型 dtype 的序列 s_typed
        s_typed = s.astype(dtype)
        # 对 s_typed 序列进行归一化处理并计算值的频数，期望结果为 Series([0.5, 0.3, 0.2], index=Series([NaN, 2.0, 1.0], dtype=dtype), name="proportion")
        result = s_typed.value_counts(normalize=True, dropna=False)
        expected = Series(
            [0.5, 0.3, 0.2],
            index=Series([np.nan, 2.0, 1.0], dtype=dtype),
            name="proportion",
        )
        tm.assert_series_equal(result, expected)

        # 再次对 s_typed 序列进行归一化处理并计算值的频数，dropna=True
        # 期望结果为 Series([0.6, 0.4], index=Series([2.0, 1.0], dtype=dtype), name="proportion")
        result = s_typed.value_counts(normalize=True, dropna=True)
        expected = Series(
            [0.6, 0.4], index=Series([2.0, 1.0], dtype=dtype), name="proportion"
        )
        tm.assert_series_equal(result, expected)
    # 定义测试方法，用于测试算法在 uint64 数据类型上的值计数功能
    def test_value_counts_uint64(self):
        # 创建一个包含 2^63 的 uint64 类型的 NumPy 数组
        arr = np.array([2**63], dtype=np.uint64)
        # 创建预期的 Series 对象，包含一个索引为 2^63 的值为 1 的计数
        expected = Series([1], index=[2**63], name="count")
        # 调用待测试的算法，进行值计数操作
        result = algos.value_counts_internal(arr)

        # 使用测试框架的方法断言测试结果与预期是否相等
        tm.assert_series_equal(result, expected)

        # 创建另一个 NumPy 数组，包含一个 uint64 类型的负数和 2^63
        arr = np.array([-1, 2**63], dtype=object)
        # 创建另一个预期的 Series 对象，包含两个索引分别为 -1 和 2^63 的计数为 1
        expected = Series([1, 1], index=[-1, 2**63], name="count")
        # 再次调用待测试的算法，进行值计数操作
        result = algos.value_counts_internal(arr)

        # 再次使用测试框架的方法断言测试结果与预期是否相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，用于测试 Series 对象的值计数功能
    def test_value_counts_series(self):
        # 说明：GH#54857
        # 创建一个包含整数、浮点数和 NaN 值的 NumPy 数组
        values = np.array([3, 1, 2, 3, 4, np.nan])
        # 使用 Series 类创建一个 Series 对象，并对其进行值计数操作，分为 3 个区间
        result = Series(values).value_counts(bins=3)
        # 创建预期的 Series 对象，包含三个区间的计数结果
        expected = Series(
            [2, 2, 1],
            # 创建一个 IntervalIndex 对象，表示区间的索引
            index=IntervalIndex.from_tuples(
                [(0.996, 2.0), (2.0, 3.0), (3.0, 4.0)], dtype="interval[float64, right]"
            ),
            name="count",
        )
        # 使用测试框架的方法断言测试结果与预期是否相等
        tm.assert_series_equal(result, expected)
class TestDuplicated:
    # 定义测试类 TestDuplicated

    def test_duplicated_with_nas(self):
        # 定义测试方法 test_duplicated_with_nas

        keys = np.array([0, 1, np.nan, 0, 2, np.nan], dtype=object)
        # 创建包含 NaN 的 NumPy 数组 keys

        result = algos.duplicated(keys)
        # 调用 algos.duplicated 函数，计算重复项并存储结果
        expected = np.array([False, False, False, True, False, True])
        # 创建预期结果的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)
        # 使用 tm.assert_numpy_array_equal 检查结果是否符合预期

        result = algos.duplicated(keys, keep="first")
        # 再次调用 algos.duplicated 函数，保留第一个重复项并存储结果
        expected = np.array([False, False, False, True, False, True])
        # 更新预期结果的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)
        # 再次使用 tm.assert_numpy_array_equal 检查结果是否符合预期

        result = algos.duplicated(keys, keep="last")
        # 再次调用 algos.duplicated 函数，保留最后一个重复项并存储结果
        expected = np.array([True, False, True, False, False, False])
        # 更新预期结果的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)
        # 再次使用 tm.assert_numpy_array_equal 检查结果是否符合预期

        result = algos.duplicated(keys, keep=False)
        # 再次调用 algos.duplicated 函数，移除所有重复项并存储结果
        expected = np.array([True, False, True, True, False, True])
        # 更新预期结果的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)
        # 再次使用 tm.assert_numpy_array_equal 检查结果是否符合预期

        keys = np.empty(8, dtype=object)
        # 创建空的长度为 8 的对象数组 keys
        for i, t in enumerate(
            zip([0, 0, np.nan, np.nan] * 2, [0, np.nan, 0, np.nan] * 2)
        ):
            keys[i] = t
        # 将组合的元组添加到 keys 中

        result = algos.duplicated(keys)
        # 再次调用 algos.duplicated 函数，计算重复项并存储结果
        falses = [False] * 4
        trues = [True] * 4
        expected = np.array(falses + trues)
        # 创建更新后的预期结果的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)
        # 使用 tm.assert_numpy_array_equal 检查结果是否符合预期

        result = algos.duplicated(keys, keep="last")
        # 再次调用 algos.duplicated 函数，保留最后一个重复项并存储结果
        expected = np.array(trues + falses)
        # 更新预期结果的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)
        # 再次使用 tm.assert_numpy_array_equal 检查结果是否符合预期

        result = algos.duplicated(keys, keep=False)
        # 再次调用 algos.duplicated 函数，移除所有重复项并存储结果
        expected = np.array(trues + trues)
        # 更新预期结果的 NumPy 数组
        tm.assert_numpy_array_equal(result, expected)
        # 再次使用 tm.assert_numpy_array_equal 检查结果是否符合预期

    @pytest.mark.parametrize(
        "case",
        [
            np.array([1, 2, 1, 5, 3, 2, 4, 1, 5, 6]),
            np.array([1.1, 2.2, 1.1, np.nan, 3.3, 2.2, 4.4, 1.1, np.nan, 6.6]),
            np.array(
                [
                    1 + 1j,
                    2 + 2j,
                    1 + 1j,
                    5 + 5j,
                    3 + 3j,
                    2 + 2j,
                    4 + 4j,
                    1 + 1j,
                    5 + 5j,
                    6 + 6j,
                ]
            ),
            np.array(["a", "b", "a", "e", "c", "b", "d", "a", "e", "f"], dtype=object),
            np.array([1, 2**63, 1, 3**5, 10, 2**63, 39, 1, 3**5, 7], dtype=np.uint64),
        ],
    )
    # 使用 pytest.mark.parametrize 注解，定义多个测试用例，每个用例是一个包含不同数据类型和值的 NumPy 数组
    # 定义一个测试方法，用于测试数值对象的重复性检测函数
    def test_numeric_object_likes(self, case):
        # 期望的第一个重复项数组
        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        # 期望的最后一个重复项数组
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        # 期望的非重复项数组，使用逻辑或操作符
        exp_false = exp_first | exp_last

        # 使用算法包中的函数检测重复项，保留第一个重复项
        res_first = algos.duplicated(case, keep="first")
        # 断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(res_first, exp_first)

        # 使用算法包中的函数检测重复项，保留最后一个重复项
        res_last = algos.duplicated(case, keep="last")
        # 断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(res_last, exp_last)

        # 使用算法包中的函数检测重复项，不保留任何重复项
        res_false = algos.duplicated(case, keep=False)
        # 断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(res_false, exp_false)

        # 对索引对象进行测试
        # 针对索引对象和类别数据类型的索引对象，分别进行测试
        for idx in [Index(case), Index(case, dtype="category")]:
            # 检测索引对象中的重复项，保留第一个重复项
            res_first = idx.duplicated(keep="first")
            # 断言两个 NumPy 数组相等
            tm.assert_numpy_array_equal(res_first, exp_first)

            # 检测索引对象中的重复项，保留最后一个重复项
            res_last = idx.duplicated(keep="last")
            # 断言两个 NumPy 数组相等
            tm.assert_numpy_array_equal(res_last, exp_last)

            # 检测索引对象中的重复项，不保留任何重复项
            res_false = idx.duplicated(keep=False)
            # 断言两个 NumPy 数组相等
            tm.assert_numpy_array_equal(res_false, exp_false)

        # 对系列对象进行测试
        # 针对系列对象和类别数据类型的系列对象，分别进行测试
        for s in [Series(case), Series(case, dtype="category")]:
            # 检测系列对象中的重复项，保留第一个重复项
            res_first = s.duplicated(keep="first")
            # 断言两个系列对象相等
            tm.assert_series_equal(res_first, Series(exp_first))

            # 检测系列对象中的重复项，保留最后一个重复项
            res_last = s.duplicated(keep="last")
            # 断言两个系列对象相等
            tm.assert_series_equal(res_last, Series(exp_last))

            # 检测系列对象中的重复项，不保留任何重复项
            res_false = s.duplicated(keep=False)
            # 断言两个系列对象相等
            tm.assert_series_equal(res_false, Series(exp_false))
    # 定义测试方法，用于测试处理类似日期时间的数据的函数
    def test_datetime_likes(self):
        # 日期时间数据列表
        dt = [
            "2011-01-01",
            "2011-01-02",
            "2011-01-01",
            "NaT",
            "2011-01-03",
            "2011-01-02",
            "2011-01-04",
            "2011-01-01",
            "NaT",
            "2011-01-06",
        ]
        # 时间增量数据列表
        td = [
            "1 days",
            "2 days",
            "1 days",
            "NaT",
            "3 days",
            "2 days",
            "4 days",
            "1 days",
            "NaT",
            "6 days",
        ]

        # 不同的日期时间数据类型示例
        cases = [
            np.array([Timestamp(d) for d in dt]),  # 标准时间戳对象
            np.array([Timestamp(d, tz="US/Eastern") for d in dt]),  # 含时区信息的时间戳对象
            np.array([Period(d, freq="D") for d in dt]),  # 时间段对象
            np.array([np.datetime64(d) for d in dt]),  # NumPy datetime64 对象
            np.array([Timedelta(d) for d in td]),  # 时间增量对象
        ]

        # 预期的第一个重复项结果
        exp_first = np.array(
            [False, False, True, False, False, True, False, True, True, False]
        )
        # 预期的最后一个重复项结果
        exp_last = np.array(
            [True, True, True, True, False, False, False, False, False, False]
        )
        # 预期的非重复项结果
        exp_false = exp_first | exp_last

        # 对于每种日期时间数据类型进行测试
        for case in cases:
            # 测试保留第一个重复项的情况
            res_first = algos.duplicated(case, keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            # 测试保留最后一个重复项的情况
            res_last = algos.duplicated(case, keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            # 测试不保留重复项的情况
            res_false = algos.duplicated(case, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

            # index
            # 对于不同类型的索引进行测试
            for idx in [
                Index(case),  # 默认类型的索引
                Index(case, dtype="category"),  # 类别类型的索引
                Index(case, dtype=object),  # 对象类型的索引
            ]:
                # 测试保留第一个重复项的情况
                res_first = idx.duplicated(keep="first")
                tm.assert_numpy_array_equal(res_first, exp_first)

                # 测试保留最后一个重复项的情况
                res_last = idx.duplicated(keep="last")
                tm.assert_numpy_array_equal(res_last, exp_last)

                # 测试不保留重复项的情况
                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)

            # series
            # 对于不同类型的系列数据进行测试
            for s in [
                Series(case),  # 默认类型的系列
                Series(case, dtype="category"),  # 类别类型的系列
                Series(case, dtype=object),  # 对象类型的系列
            ]:
                # 测试保留第一个重复项的情况
                res_first = s.duplicated(keep="first")
                tm.assert_series_equal(res_first, Series(exp_first))

                # 测试保留最后一个重复项的情况
                res_last = s.duplicated(keep="last")
                tm.assert_series_equal(res_last, Series(exp_last))

                # 测试不保留重复项的情况
                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

    # 使用 pytest 的参数化装饰器，为不同的情况（索引）定义单独的测试方法
    @pytest.mark.parametrize("case", [Index([1, 2, 3]), pd.RangeIndex(0, 3)])
    def test_unique_index(self, case):
        # 断言索引中不存在重复项
        assert case.is_unique is True
        # 断言索引的重复项标记为 False
        tm.assert_numpy_array_equal(case.duplicated(), np.array([False, False, False]))
    @pytest.mark.parametrize(
        "arr, uniques",
        [  # 参数化测试数据：输入数组和预期唯一值列表
            (
                [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)],
                [(0, 0), (0, 1), (1, 0), (1, 1)],
            ),
            (
                [("b", "c"), ("a", "b"), ("a", "b"), ("b", "c")],
                [("b", "c"), ("a", "b")],
            ),
            (
                [("a", 1), ("b", 2), ("a", 3), ("a", 1)],
                [("a", 1), ("b", 2), ("a", 3)],
            ),
        ],
    )
    # 测试方法：检查处理元组的唯一性
    def test_unique_tuples(self, arr, uniques):
        # https://github.com/pandas-dev/pandas/issues/16519
        expected = np.empty(len(uniques), dtype=object)
        expected[:] = uniques

        # 准备异常消息
        msg = "unique requires a Series, Index, ExtensionArray, or np.ndarray, got list"
        # 断言抛出特定类型错误和消息
        with pytest.raises(TypeError, match=msg):
            # GH#52986
            pd.unique(arr)

        # 转换并获取唯一值结果
        res = pd.unique(com.asarray_tuplesafe(arr, dtype=object))
        # 使用测试辅助函数检查结果数组相等性
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize(
        "array,expected",
        [  # 参数化测试数据：输入复数数组和预期唯一值数组
            (
                [1 + 1j, 0, 1, 1j, 1 + 2j, 1 + 2j],
                np.array([(1 + 1j), 0j, (1 + 0j), 1j, (1 + 2j)], dtype=complex),
            )
        ],
    )
    # 测试方法：检查处理复数的唯一性
    def test_unique_complex_numbers(self, array, expected):
        # GH 17927
        # 准备异常消息
        msg = "unique requires a Series, Index, ExtensionArray, or np.ndarray, got list"
        # 断言抛出特定类型错误和消息
        with pytest.raises(TypeError, match=msg):
            # GH#52986
            pd.unique(array)

        # 获取唯一值结果
        res = pd.unique(np.array(array))
        # 使用测试辅助函数检查结果数组相等性
        tm.assert_numpy_array_equal(res, expected)
# 定义一个测试类 TestHashTable，用于测试不同类型的哈希表
class TestHashTable:
    
    # 使用 pytest 的 parametrize 装饰器为 test_hashtable_unique 方法提供多组参数
    @pytest.mark.parametrize(
        "htable, data",
        [
            (ht.PyObjectHashTable, [f"foo_{i}" for i in range(1000)]),  # 使用 PyObjectHashTable，数据为包含 1000 个元素的字符串列表
            (ht.StringHashTable, [f"foo_{i}" for i in range(1000)]),   # 使用 StringHashTable，数据同样为包含 1000 个元素的字符串列表
            (ht.Float64HashTable, np.arange(1000, dtype=np.float64)),  # 使用 Float64HashTable，数据为包含 1000 个元素的浮点数数组
            (ht.Int64HashTable, np.arange(1000, dtype=np.int64)),      # 使用 Int64HashTable，数据为包含 1000 个元素的整数数组
            (ht.UInt64HashTable, np.arange(1000, dtype=np.uint64)),    # 使用 UInt64HashTable，数据为包含 1000 个元素的无符号整数数组
        ],
    )
    
    # 定义测试方法 test_hashtable_unique，参数包括 htable 表示的哈希表类和数据 data，以及 writable 参数（未在代码中直接指定）
    def test_hashtable_unique(self, htable, data, writable):
        # 创建一个 pandas Series 对象 s，数据为参数 data
        s = Series(data)
        
        # 根据哈希表类型执行不同的操作
        if htable == ht.Float64HashTable:
            # 如果是 Float64HashTable 类型，则在第 500 个位置插入 NaN
            s.loc[500] = np.nan
        elif htable == ht.PyObjectHashTable:
            # 如果是 PyObjectHashTable 类型，则在第 500 到 502 个位置插入不同的 NaN 类型数据
            s.loc[500:502] = [np.nan, None, NaT]

        # 创建 s 的副本 s_duplicated，通过抽样方式创建重复的数据
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        # 设置 s_duplicated 的值可写入
        s_duplicated.values.setflags(write=writable)

        # 使用哈希表的 unique 方法获取 s_duplicated 的唯一值，与 drop_duplicates 方法的结果比较
        expected_unique = s_duplicated.drop_duplicates(keep="first").values
        result_unique = htable().unique(s_duplicated.values)
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # 测试 return_inverse=True 的情况，确保反向重构正确
        result_unique, result_inverse = htable().unique(
            s_duplicated.values, return_inverse=True
        )
        tm.assert_numpy_array_equal(result_unique, expected_unique)
        reconstr = result_unique[result_inverse]
        tm.assert_numpy_array_equal(reconstr, s_duplicated.values)
    
    # 使用 pytest 的 parametrize 装饰器为下面的测试方法提供相同的参数化数据
    @pytest.mark.parametrize(
        "htable, data",
        [
            (ht.PyObjectHashTable, [f"foo_{i}" for i in range(1000)]),
            (ht.StringHashTable, [f"foo_{i}" for i in range(1000)]),
            (ht.Float64HashTable, np.arange(1000, dtype=np.float64)),
            (ht.Int64HashTable, np.arange(1000, dtype=np.int64)),
            (ht.UInt64HashTable, np.arange(1000, dtype=np.uint64)),
        ],
    )
    def test_hashtable_factorize(self, htable, writable, data):
        # 创建一个 Pandas Series 对象，使用给定的数据
        s = Series(data)
        
        # 如果哈希表类型是 Float64HashTable
        if htable == ht.Float64HashTable:
            # 在浮点列中添加 NaN 值
            s.loc[500] = np.nan
        # 如果哈希表类型是 PyObjectHashTable
        elif htable == ht.PyObjectHashTable:
            # 对象列使用不同的 NaN 类型
            s.loc[500:502] = [np.nan, None, NaT]

        # 创建一个重复选择的样本
        s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
        # 设置样本的写入标志
        s_duplicated.values.setflags(write=writable)
        # 创建 NaN 掩码
        na_mask = s_duplicated.isna().values

        # 使用给定的哈希表对象对样本进行因子化
        result_unique, result_inverse = htable().factorize(s_duplicated.values)

        # drop_duplicates 有自己的 Cython 代码 (hash_table_func_helper.pxi)
        # 并且是单独测试的；保留第一次出现的元素，类似于 ht.factorize()
        # 因为 factorize 移除所有的 NaN 值，我们在这里也做同样的操作
        expected_unique = s_duplicated.dropna().drop_duplicates().values
        tm.assert_numpy_array_equal(result_unique, expected_unique)

        # 只有当逆操作正确时，重构才能成功
        # 因为 factorize 移除了 NaN 值，这些值在这里也必须被排除
        result_reconstruct = result_unique[result_inverse[~na_mask]]
        expected_reconstruct = s_duplicated.dropna().values
        tm.assert_numpy_array_equal(result_reconstruct, expected_reconstruct)
# 定义一个测试类 TestRank，用于测试排名算法的功能
class TestRank:
    
    # 使用 pytest 的参数化装饰器，定义一个参数化测试，测试 scipy 库的兼容性
    @pytest.mark.parametrize(
        "arr",
        [
            [np.nan, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 3, np.nan],
            [4.0, np.nan, 5.0, 5.0, 5.0, np.nan, 1, 2, 4.0, np.nan],
        ],
    )
    # 测试 scipy 兼容性的具体方法
    def test_scipy_compat(self, arr):
        # 导入 pytest 所需的 scipy.stats 模块，并跳过如果导入失败
        sp_stats = pytest.importorskip("scipy.stats")

        # 将传入的 arr 转换为 numpy 数组
        arr = np.array(arr)

        # 创建一个掩码，标识数组中非有限数的位置
        mask = ~np.isfinite(arr)
        # 使用自定义的排名算法处理数组
        result = libalgos.rank_1d(arr)
        # 将非有限数位置的值设为正无穷
        arr[mask] = np.inf
        # 使用 scipy 库的 rankdata 函数计算期望排名
        exp = sp_stats.rankdata(arr)
        # 将非有限数位置的期望排名设为 NaN
        exp[mask] = np.nan
        # 使用 pytest 的工具函数检查结果与期望是否几乎相等
        tm.assert_almost_equal(result, exp)

    # 测试基本的排名功能
    def test_basic(self, writable, any_int_numpy_dtype):
        # 定义期望的排名结果
        exp = np.array([1, 2], dtype=np.float64)

        # 创建一个包含整数数据的 numpy 数组
        data = np.array([1, 100], dtype=any_int_numpy_dtype)
        # 设置数组可写属性
        data.setflags(write=writable)
        # 将数组转换为 Pandas 的 Series 对象
        ser = Series(data)
        # 使用排名算法处理 Series 对象
        result = algos.rank(ser)
        # 使用 pytest 的工具函数检查结果数组与期望数组是否相等
        tm.assert_numpy_array_equal(result, exp)

    # 使用 pytest 的参数化装饰器，定义一个测试，测试 uint64 数据溢出情况
    @pytest.mark.parametrize("dtype", [np.float64, np.uint64])
    def test_uint64_overflow(self, dtype):
        # 定义期望的排名结果
        exp = np.array([1, 2], dtype=np.float64)

        # 创建一个包含数据溢出的 Series 对象
        s = Series([1, 2**63], dtype=dtype)
        # 使用排名算法处理 Series 对象
        tm.assert_numpy_array_equal(algos.rank(s), exp)

    # 测试多维数组的情况
    def test_too_many_ndims(self):
        # 创建一个多维数组，测试排名算法是否抛出预期的异常
        arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        msg = "Array with ndim > 2 are not supported"

        # 使用 pytest 的上下文管理器，检查排名算法是否抛出预期的类型错误异常
        with pytest.raises(TypeError, match=msg):
            algos.rank(arr)

    # 使用 pytest 的单核标记，定义一个测试，测试大量行数据的百分比最大排名
    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self):
        # GH 18271
        # 创建一个非常大的整数数组，测试其百分比最大排名
        values = np.arange(2**24 + 1)
        result = algos.rank(values, pct=True).max()
        # 使用断言检查结果是否等于预期值 1
        assert result == 1

        # 创建一个更大的整数数组，测试其百分比最大排名
        values = np.arange(2**25 + 2).reshape(2**24 + 1, 2)
        result = algos.rank(values, pct=True).max()
        # 使用断言检查结果是否等于预期值 1
        assert result == 1


class TestMode:
    # 定义一个测试类 TestMode，用于测试众数算法的功能

    # 测试没有众数的情况
    def test_no_mode(self):
        # 定义一个空的 Series 作为预期结果
        exp = Series([], dtype=np.float64, index=Index([], dtype=int))
        # 使用排名算法处理空数组，并使用工具函数检查结果数组是否与预期数组相等
        tm.assert_numpy_array_equal(algos.mode(np.array([])), exp.values)

    # 测试单个众数的情况
    def test_mode_single(self, any_real_numpy_dtype):
        # GH 15714
        # 定义一个单个元素的期望众数列表和数据列表
        exp_single = [1]
        data_single = [1]

        # 定义一个多个相同元素的期望众数列表和数据列表
        exp_multi = [1]
        data_multi = [1, 1]

        # 创建一个包含单个元素的 Series 对象，并使用排名算法处理其中的值
        ser = Series(data_single, dtype=any_real_numpy_dtype)
        exp = Series(exp_single, dtype=any_real_numpy_dtype)
        # 使用工具函数检查排名算法处理后的结果数组与期望数组是否相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 使用工具函数检查 Series 对象的众数方法是否返回预期的 Series 对象
        tm.assert_series_equal(ser.mode(), exp)

        # 创建一个包含多个相同元素的 Series 对象，并使用排名算法处理其中的值
        ser = Series(data_multi, dtype=any_real_numpy_dtype)
        exp = Series(exp_multi, dtype=any_real_numpy_dtype)
        # 使用工具函数检查排名算法处理后的结果数组与期望数组是否相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 使用工具函数检查 Series 对象的众数方法是否返回预期的 Series 对象
        tm.assert_series_equal(ser.mode(), exp)

    # 测试包含对象和整数的众数情况
    def test_mode_obj_int(self):
        # 定义一个整数 Series 对象作为预期结果
        exp = Series([1], dtype=int)
        # 使用排名算法处理整数 Series 对象，并使用工具函数检查处理后的结果数组是否与预期数组相等
        tm.assert_numpy_array_equal(algos.mode(exp.values), exp.values)

        # 定义一个对象类型的 Series 对象作为预期结果
        exp = Series(["a", "b", "c"], dtype=object)
        # 使用排名算法处理对象类型 Series 对象，并使用工具函数检查处理后的结果数组是否与预期数组相等
        tm.assert_numpy_array_equal(algos.mode(exp.values), exp.values)
    # 测试单一数字模式
    def test_number_mode(self, any_real_numpy_dtype):
        # 预期的单一模式值
        exp_single = [1]
        # 包含重复数字的数据
        data_single = [1] * 5 + [2] * 3

        # 预期的多模式值
        exp_multi = [1, 3]
        # 包含多个重复数字的数据
        data_multi = [1] * 5 + [2] * 3 + [3] * 5

        # 创建包含 data_single 数据的 Series 对象
        ser = Series(data_single, dtype=any_real_numpy_dtype)
        # 创建包含 exp_single 数据的 Series 对象
        exp = Series(exp_single, dtype=any_real_numpy_dtype)
        # 断言算法模式函数计算结果与预期单一模式值是否相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 断言 Series 对象的 mode 方法结果与预期单一模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

        # 创建包含 data_multi 数据的 Series 对象
        ser = Series(data_multi, dtype=any_real_numpy_dtype)
        # 创建包含 exp_multi 数据的 Series 对象
        exp = Series(exp_multi, dtype=any_real_numpy_dtype)
        # 断言算法模式函数计算结果与预期多模式值是否相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 断言 Series 对象的 mode 方法结果与预期多模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

    # 测试字符串对象模式
    def test_strobj_mode(self):
        # 预期的字符串模式值
        exp = ["b"]
        # 包含重复字符串的数据
        data = ["a"] * 2 + ["b"] * 3

        # 创建包含 data 数据的 Series 对象，数据类型为字符型
        ser = Series(data, dtype="c")
        # 创建包含 exp 数据的 Series 对象，数据类型为字符型
        exp = Series(exp, dtype="c")
        # 断言算法模式函数计算结果与预期字符串模式值是否相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 断言 Series 对象的 mode 方法结果与预期字符串模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

    # 使用参数化测试字符串对象多字符
    @pytest.mark.parametrize("dt", [str, object])
    def test_strobj_multi_char(self, dt):
        # 预期的多字符字符串模式值
        exp = ["bar"]
        # 包含多字符重复字符串的数据
        data = ["foo"] * 2 + ["bar"] * 3

        # 创建包含 data 数据的 Series 对象，数据类型为参数化传入的类型
        ser = Series(data, dtype=dt)
        # 创建包含 exp 数据的 Series 对象，数据类型为参数化传入的类型
        exp = Series(exp, dtype=dt)
        # 断言算法模式函数计算结果与预期多字符字符串模式值是否相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 断言 Series 对象的 mode 方法结果与预期多字符字符串模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

    # 测试日期类数据模式
    def test_datelike_mode(self):
        # 预期的日期模式值
        exp = Series(["1900-05-03", "2011-01-03", "2013-01-02"], dtype="M8[ns]")
        # 包含日期数据的 Series 对象
        ser = Series(["2011-01-03", "2013-01-02", "1900-05-03"], dtype="M8[ns]")
        # 断言算法模式函数计算结果与预期日期模式值是否相等
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        # 断言 Series 对象的 mode 方法结果与预期日期模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

        # 更换预期的日期模式值
        exp = Series(["2011-01-03", "2013-01-02"], dtype="M8[ns]")
        # 包含扩展日期数据的 Series 对象
        ser = Series(
            ["2011-01-03", "2013-01-02", "1900-05-03", "2011-01-03", "2013-01-02"],
            dtype="M8[ns]",
        )
        # 断言算法模式函数计算结果与新的预期日期模式值是否相等
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        # 断言 Series 对象的 mode 方法结果与新的预期日期模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

    # 测试时间间隔模式
    def test_timedelta_mode(self):
        # 预期的时间间隔模式值
        exp = Series(["-1 days", "0 days", "1 days"], dtype="timedelta64[ns]")
        # 包含时间间隔数据的 Series 对象
        ser = Series(["1 days", "-1 days", "0 days"], dtype="timedelta64[ns]")
        # 断言算法模式函数计算结果与预期时间间隔模式值是否相等
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        # 断言 Series 对象的 mode 方法结果与预期时间间隔模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

        # 更换预期的时间间隔模式值
        exp = Series(["2 min", "1 day"], dtype="timedelta64[ns]")
        # 包含扩展时间间隔数据的 Series 对象
        ser = Series(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        # 断言算法模式函数计算结果与新的预期时间间隔模式值是否相等
        tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
        # 断言 Series 对象的 mode 方法结果与新的预期时间间隔模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)

    # 测试混合数据类型模式
    def test_mixed_dtype(self):
        # 预期的混合数据类型模式值
        exp = Series(["foo"], dtype=object)
        # 包含混合数据类型的 Series 对象
        ser = Series([1, "foo", "foo"])
        # 断言算法模式函数计算结果与预期混合数据类型模式值是否相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 断言 Series 对象的 mode 方法结果与预期混合数据类型模式值是否相等
        tm.assert_series_equal(ser.mode(), exp)
    # 定义测试函数，用于测试处理无符号64位整数溢出的情况
    def test_uint64_overflow(self):
        # 创建期望的Series对象，包含一个2^63的无符号64位整数
        exp = Series([2**63], dtype=np.uint64)
        # 创建包含多个无符号64位整数的Series对象
        ser = Series([1, 2**63, 2**63], dtype=np.uint64)
        # 使用numpy.testing模块中的函数验证算法模块中的mode函数返回的结果与期望值相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 使用pandas.testing模块中的函数验证Series对象的mode方法返回的结果与期望值相等
        tm.assert_series_equal(ser.mode(), exp)

        # 创建包含两个无符号64位整数的Series对象
        exp = Series([1, 2**63], dtype=np.uint64)
        ser = Series([1, 2**63], dtype=np.uint64)
        # 使用numpy.testing模块中的函数验证算法模块中的mode函数返回的结果与期望值相等
        tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
        # 使用pandas.testing模块中的函数验证Series对象的mode方法返回的结果与期望值相等
        tm.assert_series_equal(ser.mode(), exp)

    # 定义测试函数，用于测试处理分类数据的情况
    def test_categorical(self):
        # 创建Categorical对象
        c = Categorical([1, 2])
        # 将Categorical对象转换为Series对象后，调用其mode方法获取结果，并转换为内部存储的值
        exp = c
        res = Series(c).mode()._values
        # 使用pandas.testing模块中的函数验证计算得到的结果与期望值相等
        tm.assert_categorical_equal(res, exp)

        # 创建包含混合类型数据的Categorical对象
        c = Categorical([1, "a", "a"])
        # 创建期望的Categorical对象，指定了新的分类值和类别列表
        exp = Categorical(["a"], categories=[1, "a"])
        # 将Categorical对象转换为Series对象后，调用其mode方法获取结果，并转换为内部存储的值
        res = Series(c).mode()._values
        # 使用pandas.testing模块中的函数验证计算得到的结果与期望值相等
        tm.assert_categorical_equal(res, exp)

        # 创建包含多个重复值的Categorical对象
        c = Categorical([1, 1, 2, 3, 3])
        # 创建期望的Categorical对象，指定了新的分类值和类别列表
        exp = Categorical([1, 3], categories=[1, 2, 3])
        # 将Categorical对象转换为Series对象后，调用其mode方法获取结果，并转换为内部存储的值
        res = Series(c).mode()._values
        # 使用pandas.testing模块中的函数验证计算得到的结果与期望值相等
        tm.assert_categorical_equal(res, exp)

    # 定义测试函数，用于测试处理索引对象的情况
    def test_index(self):
        # 创建Index对象
        idx = Index([1, 2, 3])
        # 创建期望的Series对象，包含与Index对象相同的数据，指定数据类型为np.int64
        exp = Series([1, 2, 3], dtype=np.int64)
        # 使用numpy.testing模块中的函数验证算法模块中的mode函数返回的结果与期望值相等
        tm.assert_numpy_array_equal(algos.mode(idx), exp.values)

        # 创建包含混合类型数据的Index对象
        idx = Index([1, "a", "a"])
        # 创建期望的Series对象，包含与Index对象相同的数据，指定数据类型为object
        exp = Series(["a"], dtype=object)
        # 使用numpy.testing模块中的函数验证算法模块中的mode函数返回的结果与期望值相等
        tm.assert_numpy_array_equal(algos.mode(idx), exp.values)

        # 创建包含多个重复值的Index对象
        idx = Index([1, 1, 2, 3, 3])
        # 创建期望的Series对象，包含与Index对象相同的数据，指定数据类型为np.int64
        exp = Series([1, 3], dtype=np.int64)
        # 使用numpy.testing模块中的函数验证算法模块中的mode函数返回的结果与期望值相等
        tm.assert_numpy_array_equal(algos.mode(idx), exp.values)

        # 创建包含时间增量数据的Index对象
        idx = Index(
            ["1 day", "1 day", "-1 day", "-1 day 2 min", "2 min", "2 min"],
            dtype="timedelta64[ns]",
        )
        # 使用pytest.raises函数验证调用算法模块中的mode函数会引发AttributeError异常，且异常消息包含"TimedeltaIndex"
        with pytest.raises(AttributeError, match="TimedeltaIndex"):
            algos.mode(idx)

    # 定义测试函数，用于测试带有名称的Series对象的mode方法
    def test_ser_mode_with_name(self):
        # 创建具有名称的Series对象
        ser = Series([1, 1, 3], name="foo")
        # 调用Series对象的mode方法获取结果
        result = ser.mode()
        # 创建期望的具有名称的Series对象
        expected = Series([1], name="foo")
        # 使用pandas.testing模块中的函数验证计算得到的结果与期望值相等
        tm.assert_series_equal(result, expected)
class TestDiff:
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_diff_datetimelike_nat(self, dtype):
        # NaT - NaT is NaT, not 0
        # 创建一个包含整数的 NumPy 数组，将其视图转换为给定的日期时间类型
        arr = np.arange(12).astype(np.int64).view(dtype).reshape(3, 4)
        # 将数组中第三列的值设置为指定日期时间类型的 NaT（Not-a-Time）
        arr[:, 2] = arr.dtype.type("NaT", "ns")
        # 使用算法库中的 diff 函数对数组进行差分计算，沿着指定轴向计算差分
        result = algos.diff(arr, 1, axis=0)

        # 创建一个期望的结果数组，填充为指定的时间增量类型的 NaT
        expected = np.ones(arr.shape, dtype="timedelta64[ns]") * 4
        expected[:, 2] = np.timedelta64("NaT", "ns")
        expected[0, :] = np.timedelta64("NaT", "ns")

        # 使用测试工具库中的 assert 函数检查 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 对数组的转置进行差分计算，沿着指定轴向计算差分
        result = algos.diff(arr.T, 1, axis=1)
        # 使用测试工具库中的 assert 函数检查 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected.T)

    def test_diff_ea_axis(self):
        # 创建一个日期范围，并获取其内部数据对象
        dta = date_range("2016-01-01", periods=3, tz="US/Pacific")._data

        msg = "cannot diff DatetimeArray on axis=1"
        # 使用 pytest 的上下文管理器检查是否引发预期的 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 使用算法库中的 diff 函数对日期时间数组进行差分计算，指定轴向不支持
            algos.diff(dta, 1, axis=1)

    @pytest.mark.parametrize("dtype", ["int8", "int16"])
    def test_diff_low_precision_int(self, dtype):
        # 创建一个低精度整数类型的 NumPy 数组
        arr = np.array([0, 1, 1, 0, 0], dtype=dtype)
        # 使用算法库中的 diff 函数对数组进行差分计算
        result = algos.diff(arr, 1)
        # 创建一个期望的结果数组，使用 float32 类型填充
        expected = np.array([np.nan, 1, 0, -1, 0], dtype="float32")
        # 使用测试工具库中的 assert 函数检查 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("op", [np.array, pd.array])
def test_union_with_duplicates(op):
    # GH#36289
    # 创建两个包含重复值的数组，使用指定的操作函数
    lvals = op([3, 1, 3, 4])
    rvals = op([2, 3, 1, 1])
    # 创建一个期望的结果数组，包含两个输入数组的并集（去除重复）
    expected = op([3, 3, 1, 1, 4, 2])
    if isinstance(expected, np.ndarray):
        # 使用算法库中的 union_with_duplicates 函数计算数组的并集，检查是否符合预期
        result = algos.union_with_duplicates(lvals, rvals)
        # 使用测试工具库中的 assert 函数检查 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)
    else:
        # 使用算法库中的 union_with_duplicates 函数计算数组的并集，检查是否符合预期
        result = algos.union_with_duplicates(lvals, rvals)
        # 使用测试工具库中的 assert 函数检查扩展数组是否相等
        tm.assert_extension_array_equal(result, expected)
```