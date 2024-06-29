# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_reductions.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入以下对象：
    NaT,  # 无效时间戳的表示
    SparseDtype,  # 稀疏数据类型
    Timestamp,  # 时间戳对象
    isna,  # 判断是否为缺失值的函数
)
from pandas.core.arrays.sparse import SparseArray  # 从 pandas 的稀疏数组模块中导入 SparseArray 类


class TestReductions:
    @pytest.mark.parametrize(  # 使用 pytest 的参数化标记定义测试参数
        "data,pos,neg",
        [
            ([True, True, True], True, False),  # 布尔数据测试
            ([1, 2, 1], 1, 0),  # 整数数据测试
            ([1.0, 2.0, 1.0], 1.0, 0.0),  # 浮点数数据测试
        ],
    )
    def test_all(self, data, pos, neg):
        # GH#17570
        out = SparseArray(data).all()  # 创建 SparseArray 对象并调用 all 方法
        assert out  # 断言结果为真

        out = SparseArray(data, fill_value=pos).all()  # 创建填充了特定值的 SparseArray 对象并调用 all 方法
        assert out  # 断言结果为真

        data[1] = neg  # 修改数据使得 all 方法返回 False
        out = SparseArray(data).all()  # 重新调用 all 方法
        assert not out  # 断言结果为假

        out = SparseArray(data, fill_value=pos).all()  # 使用填充值的对象再次调用 all 方法
        assert not out  # 断言结果为假

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([True, True, True], True, False),
            ([1, 2, 1], 1, 0),
            ([1.0, 2.0, 1.0], 1.0, 0.0),
        ],
    )
    def test_numpy_all(self, data, pos, neg):
        # GH#17570
        out = np.all(SparseArray(data))  # 使用 NumPy 的 all 函数对 SparseArray 进行操作
        assert out  # 断言结果为真

        out = np.all(SparseArray(data, fill_value=pos))  # 使用填充值的 SparseArray 对象调用 NumPy 的 all 函数
        assert out  # 断言结果为真

        data[1] = neg  # 修改数据使得 all 方法返回 False
        out = np.all(SparseArray(data))  # 使用 NumPy 的 all 函数再次操作 SparseArray
        assert not out  # 断言结果为假

        out = np.all(SparseArray(data, fill_value=pos))  # 使用填充值的 SparseArray 对象再次调用 NumPy 的 all 函数
        assert not out  # 断言结果为假

        # 在 Python 2 中会引发不同的消息。
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 来检查是否引发特定的 ValueError 异常
            np.all(SparseArray(data), out=np.array([]))  # 尝试使用 'out' 参数调用 NumPy 的 all 函数

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([False, True, False], True, False),
            ([0, 2, 0], 2, 0),
            ([0.0, 2.0, 0.0], 2.0, 0.0),
        ],
    )
    def test_any(self, data, pos, neg):
        # GH#17570
        out = SparseArray(data).any()  # 创建 SparseArray 对象并调用 any 方法
        assert out  # 断言结果为真

        out = SparseArray(data, fill_value=pos).any()  # 创建填充了特定值的 SparseArray 对象并调用 any 方法
        assert out  # 断言结果为真

        data[1] = neg  # 修改数据使得 any 方法返回 False
        out = SparseArray(data).any()  # 重新调用 any 方法
        assert not out  # 断言结果为假

        out = SparseArray(data, fill_value=pos).any()  # 使用填充值的对象再次调用 any 方法
        assert not out  # 断言结果为假

    @pytest.mark.parametrize(
        "data,pos,neg",
        [
            ([False, True, False], True, False),
            ([0, 2, 0], 2, 0),
            ([0.0, 2.0, 0.0], 2.0, 0.0),
        ],
    )
    def test_numpy_any(self, data, pos, neg):
        # GH#17570
        out = np.any(SparseArray(data))  # 使用 NumPy 的 any 函数对 SparseArray 进行操作
        assert out  # 断言结果为真

        out = np.any(SparseArray(data, fill_value=pos))  # 使用填充值的 SparseArray 对象调用 NumPy 的 any 函数
        assert out  # 断言结果为真

        data[1] = neg  # 修改数据使得 any 方法返回 False
        out = np.any(SparseArray(data))  # 使用 NumPy 的 any 函数再次操作 SparseArray
        assert not out  # 断言结果为假

        out = np.any(SparseArray(data, fill_value=pos))  # 使用填充值的 SparseArray 对象再次调用 NumPy 的 any 函数
        assert not out  # 断言结果为假

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 来检查是否引发特定的 ValueError 异常
            np.any(SparseArray(data), out=out)  # 尝试使用 'out' 参数调用 NumPy 的 any 函数
    # 定义测试方法，验证稀疏数组的求和功能
    def test_sum(self):
        # 创建一个包含10个浮点数的NumPy数组
        data = np.arange(10).astype(float)
        # 使用SparseArray类对数组进行封装，并计算其总和
        out = SparseArray(data).sum()
        # 断言总和结果为45.0
        assert out == 45.0

        # 将数组中索引为5的位置设置为NaN
        data[5] = np.nan
        # 使用填充值2重新创建SparseArray，并计算其总和
        out = SparseArray(data, fill_value=2).sum()
        # 断言总和结果为40.0
        assert out == 40.0

        # 使用填充值NaN重新创建SparseArray，并计算其总和
        out = SparseArray(data, fill_value=np.nan).sum()
        # 断言总和结果为40.0
        assert out == 40.0

    # 使用参数化测试装饰器定义测试方法，测试稀疏数组的求和功能与最小计数
    @pytest.mark.parametrize(
        "arr",
        [[0, 1, np.nan, 1], [0, 1, 1]],
    )
    @pytest.mark.parametrize("fill_value", [0, 1, np.nan])
    @pytest.mark.parametrize("min_count, expected", [(3, 2), (4, np.nan)])
    def test_sum_min_count(self, arr, fill_value, min_count, expected):
        # GH#25777
        # 使用SparseArray类创建稀疏数组，设置填充值和输入数组arr
        sparray = SparseArray(np.array(arr), fill_value=fill_value)
        # 对稀疏数组进行求和，指定最小计数为min_count
        result = sparray.sum(min_count=min_count)
        # 如果期望结果为NaN，则断言结果也为NaN
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            # 否则断言结果与期望值相等
            assert result == expected

    # 定义测试方法，验证布尔类型稀疏数组的求和功能与最小计数
    def test_bool_sum_min_count(self):
        # 使用SparseArray类创建布尔类型稀疏数组，填充值为True
        spar_bool = SparseArray([False, True] * 5, dtype=np.bool_, fill_value=True)
        # 对布尔类型稀疏数组进行求和，设置最小计数为1
        res = spar_bool.sum(min_count=1)
        # 断言求和结果为5
        assert res == 5
        # 再次对布尔类型稀疏数组进行求和，设置最小计数为11
        res = spar_bool.sum(min_count=11)
        # 断言结果为NaN
        assert isna(res)

    # 定义测试方法，验证使用NumPy的sum函数对稀疏数组的求和功能
    def test_numpy_sum(self):
        # 创建一个包含10个浮点数的NumPy数组
        data = np.arange(10).astype(float)
        # 使用NumPy的sum函数计算SparseArray的总和
        out = np.sum(SparseArray(data))
        # 断言总和结果为45.0
        assert out == 45.0

        # 将数组中索引为5的位置设置为NaN
        data[5] = np.nan
        # 使用NumPy的sum函数计算填充值为2的SparseArray的总和
        out = np.sum(SparseArray(data, fill_value=2))
        # 断言总和结果为40.0
        assert out == 40.0

        # 使用NumPy的sum函数计算填充值为NaN的SparseArray的总和
        out = np.sum(SparseArray(data, fill_value=np.nan))
        # 断言总和结果为40.0
        assert out == 40.0

        # 使用pytest的raises断言捕获值错误，并匹配错误消息
        msg = "the 'dtype' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.sum(SparseArray(data), dtype=np.int64)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.sum(SparseArray(data), out=out)

    # 定义测试方法，验证稀疏数组的均值功能
    def test_mean(self):
        # 创建一个包含10个浮点数的NumPy数组
        data = np.arange(10).astype(float)
        # 使用SparseArray类对数组进行封装，并计算其均值
        out = SparseArray(data).mean()
        # 断言均值结果为4.5
        assert out == 4.5

        # 将数组中索引为5的位置设置为NaN
        data[5] = np.nan
        # 使用SparseArray类对数组进行封装，并重新计算其均值
        out = SparseArray(data).mean()
        # 断言均值结果为40.0 / 9
        assert out == 40.0 / 9

    # 定义测试方法，验证使用NumPy的mean函数对稀疏数组的均值功能
    def test_numpy_mean(self):
        # 创建一个包含10个浮点数的NumPy数组
        data = np.arange(10).astype(float)
        # 使用NumPy的mean函数计算SparseArray的均值
        out = np.mean(SparseArray(data))
        # 断言均值结果为4.5
        assert out == 4.5

        # 将数组中索引为5的位置设置为NaN
        data[5] = np.nan
        # 使用NumPy的mean函数计算填充值为2的SparseArray的均值
        out = np.mean(SparseArray(data, fill_value=2))
        # 断言均值结果为40.0 / 9
        assert out == 40.0 / 9

        # 使用pytest的raises断言捕获值错误，并匹配错误消息
        msg = "the 'dtype' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.mean(SparseArray(data), dtype=np.int64)

        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.mean(SparseArray(data), out=out)
    # 定义一个测试类 TestMinMax，用于测试 SparseArray 类中的最大最小值相关功能
    class TestMinMax:
        
        # 使用 pytest 的参数化标记，定义了多组测试数据，包括原始数据、预期最大值和最小值
        @pytest.mark.parametrize(
            "raw_data,max_expected,min_expected",
            [
                (np.arange(5.0), [4], [0]),  # 测试从0到4的连续浮点数数组
                (-np.arange(5.0), [0], [-4]),  # 测试从0到-4的负数浮点数数组
                (np.array([0, 1, 2, np.nan, 4]), [4], [0]),  # 测试包含NaN的整数数组
                (np.array([np.nan] * 5), [np.nan], [np.nan]),  # 测试全为NaN的数组
                (np.array([]), [np.nan], [np.nan]),  # 测试空数组
            ],
        )
        
        # 测试处理 NaN 值的情况下，计算最大值和最小值
        def test_nan_fill_value(self, raw_data, max_expected, min_expected):
            # 创建 SparseArray 对象，用给定的原始数据初始化
            arr = SparseArray(raw_data)
            # 计算 SparseArray 中的最大值和最小值
            max_result = arr.max()
            min_result = arr.min()
            # 断言计算出的最大值和最小值是否在预期范围内
            assert max_result in max_expected
            assert min_result in min_expected
    
            # 使用 skipna=False 参数重新计算最大值和最小值
            max_result = arr.max(skipna=False)
            min_result = arr.min(skipna=False)
            # 如果原始数据中包含 NaN 值，则断言计算出的最大值和最小值也是 NaN
            if np.isnan(raw_data).any():
                assert np.isnan(max_result)
                assert np.isnan(min_result)
            else:
                # 否则，断言计算出的最大值和最小值是否在预期范围内
                assert max_result in max_expected
                assert min_result in min_expected
    
        # 使用 pytest 的参数化标记，定义了多组测试数据，包括填充值、预期最大值和最小值
        @pytest.mark.parametrize(
            "fill_value,max_expected,min_expected",
            [
                (100, 100, 0),  # 测试填充值为100的整数数组
                (-100, 1, -100),  # 测试填充值为-100的整数数组
            ],
        )
        
        # 测试填充值情况下，计算最大值和最小值
        def test_fill_value(self, fill_value, max_expected, min_expected):
            # 创建 SparseArray 对象，用给定的填充值数组初始化
            arr = SparseArray(
                np.array([fill_value, 0, 1]), dtype=SparseDtype("int", fill_value)
            )
            # 计算 SparseArray 中的最大值
            max_result = arr.max()
            # 断言计算出的最大值是否等于预期最大值
            assert max_result == max_expected
    
            # 计算 SparseArray 中的最小值
            min_result = arr.min()
            # 断言计算出的最小值是否等于预期最小值
            assert min_result == min_expected
    
        # 测试仅包含填充值的情况下，计算最大值和最小值
        def test_only_fill_value(self):
            # 设置填充值
            fv = 100
            # 创建 SparseArray 对象，用给定的填充值数组初始化
            arr = SparseArray(np.array([fv, fv, fv]), dtype=SparseDtype("int", fv))
            # 断言 SparseArray 对象中有效稀疏值的长度为0
            assert len(arr._valid_sp_values) == 0
    
            # 断言计算 SparseArray 中的最大值和最小值是否等于填充值
            assert arr.max() == fv
            assert arr.min() == fv
            assert arr.max(skipna=False) == fv
            assert arr.min(skipna=False) == fv
    
        # 使用 pytest 的参数化标记，定义了多组测试数据，包括函数名和数据数组
        # 另外还定义了数据类型和预期结果
        @pytest.mark.parametrize("func", ["min", "max"])
        @pytest.mark.parametrize("data", [np.array([]), np.array([np.nan, np.nan])])
        @pytest.mark.parametrize(
            "dtype,expected",
            [
                (SparseDtype(np.float64, np.nan), np.nan),  # 测试浮点类型数据和预期NaN结果
                (SparseDtype(np.float64, 5.0), np.nan),  # 测试浮点类型数据和预期NaN结果
                (SparseDtype("datetime64[ns]", NaT), NaT),  # 测试日期时间类型数据和NaT结果
                (SparseDtype("datetime64[ns]", Timestamp("2018-05-05")), NaT),  # 测试日期时间类型数据和NaT结果
            ],
        )
        
        # 测试没有有效值的情况下，计算最大值和最小值
        def test_na_value_if_no_valid_values(self, func, data, dtype, expected):
            # 创建 SparseArray 对象，用给定的数据和数据类型初始化
            arr = SparseArray(data, dtype=dtype)
            # 使用 getattr 函数调用指定的计算函数（min 或 max）
            result = getattr(arr, func)()
            # 如果预期结果是 NaT，则断言计算结果是 NaT 或者是有效的 NaT 值
            if expected is NaT:
                assert result is NaT or np.isnat(result)
            else:
                # 否则，断言计算结果是 NaN
                assert np.isnan(result)
    # 使用 pytest 的 parametrize 装饰器为下面的测试方法参数化多组参数
    @pytest.mark.parametrize(
        "arr,argmax_expected,argmin_expected",
        [
            # 使用 SparseArray 对象及其期望的 argmax 和 argmin 结果进行参数化
            (SparseArray([1, 2, 0, 1, 2]), 1, 2),
            (SparseArray([-1, -2, 0, -1, -2]), 2, 1),
            (SparseArray([np.nan, 1, 0, 0, np.nan, -1]), 1, 5),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2]), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=-1), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=0), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=1), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=2), 5, 2),
            (SparseArray([np.nan, 1, 0, 0, np.nan, 2], fill_value=3), 5, 2),
            (SparseArray([0] * 10 + [-1], fill_value=0), 0, 10),
            (SparseArray([0] * 10 + [-1], fill_value=-1), 0, 10),
            (SparseArray([0] * 10 + [-1], fill_value=1), 0, 10),
            (SparseArray([-1] + [0] * 10, fill_value=0), 1, 0),
            (SparseArray([1] + [0] * 10, fill_value=0), 0, 1),
            (SparseArray([-1] + [0] * 10, fill_value=-1), 1, 0),
            (SparseArray([1] + [0] * 10, fill_value=1), 0, 1),
        ],
    )
    # 测试方法，用于测试 SparseArray 类的 argmax 和 argmin 方法的正确性
    def test_argmax_argmin(self, arr, argmax_expected, argmin_expected):
        # 调用 SparseArray 的 argmax 和 argmin 方法得到实际结果
        argmax_result = arr.argmax()
        argmin_result = arr.argmin()
        # 断言实际结果与期望结果相等
        assert argmax_result == argmax_expected
        assert argmin_result == argmin_expected

    # 使用 pytest 的 parametrize 装饰器为下面的测试方法参数化多组参数
    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    # 测试空的 SparseArray 对象调用 argmax 和 argmin 方法时的异常情况
    def test_empty_array(self, method):
        # 准备错误消息，说明试图对空序列调用 argmax 或 argmin 方法
        msg = f"attempt to get {method} of an empty sequence"
        # 创建空的 SparseArray 对象
        arr = SparseArray([])
        # 使用 pytest 的 raises 方法检查是否抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # 使用 getattr 方法动态调用 arr 对象的 method 方法（即 argmax 或 argmin）
            getattr(arr, method)()
```