# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_astype.py`

```
# 导入所需的库和模块
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from pandas._libs.sparse import IntIndex  # 从pandas._libs.sparse模块中导入IntIndex类

from pandas import (  # 从pandas库中导入以下类和函数
    SparseDtype,
    Timestamp,
)
import pandas._testing as tm  # 导入pandas._testing模块并重命名为tm
from pandas.core.arrays.sparse import SparseArray  # 从pandas.core.arrays.sparse模块中导入SparseArray类


class TestAstype:
    def test_astype(self):
        # float -> float
        arr = SparseArray([None, None, 0, 2])  # 创建SparseArray对象arr，包含指定元素
        result = arr.astype("Sparse[float32]")  # 将arr转换为Sparse[float32]类型
        expected = SparseArray([None, None, 0, 2], dtype=np.dtype("float32"))  # 创建期望的SparseArray对象expected，指定dtype为float32
        tm.assert_sp_array_equal(result, expected)  # 断言result与expected相等

        dtype = SparseDtype("float64", fill_value=0)  # 创建SparseDtype类型dtype，指定dtype为float64，填充值为0
        result = arr.astype(dtype)  # 将arr转换为指定dtype类型
        expected = SparseArray._simple_new(
            np.array([0.0, 2.0], dtype=dtype.subtype), IntIndex(4, [2, 3]), dtype
        )  # 创建期望的SparseArray对象expected，使用_simple_new方法创建
        tm.assert_sp_array_equal(result, expected)  # 断言result与expected相等

        dtype = SparseDtype("int64", 0)  # 创建SparseDtype类型dtype，指定dtype为int64，填充值为0
        result = arr.astype(dtype)  # 将arr转换为指定dtype类型
        expected = SparseArray._simple_new(
            np.array([0, 2], dtype=np.int64), IntIndex(4, [2, 3]), dtype
        )  # 创建期望的SparseArray对象expected，使用_simple_new方法创建
        tm.assert_sp_array_equal(result, expected)  # 断言result与expected相等

        arr = SparseArray([0, np.nan, 0, 1], fill_value=0)  # 创建SparseArray对象arr，包含指定元素和填充值
        with pytest.raises(ValueError, match="NA"):  # 使用pytest检测异常，期望抛出值错误，并匹配"NA"
            arr.astype("Sparse[i8]")  # 将arr转换为Sparse[i8]类型

    def test_astype_bool(self):
        a = SparseArray([1, 0, 0, 1], dtype=SparseDtype(int, 0))  # 创建SparseArray对象a，包含指定元素和指定dtype
        result = a.astype(bool)  # 将a转换为bool类型
        expected = np.array([1, 0, 0, 1], dtype=bool)  # 创建期望的NumPy数组对象expected，指定dtype为bool
        tm.assert_numpy_array_equal(result, expected)  # 断言result与expected相等

        # update fill value
        result = a.astype(SparseDtype(bool, False))  # 将a转换为指定dtype类型，更新填充值
        expected = SparseArray(
            [True, False, False, True], dtype=SparseDtype(bool, False)
        )  # 创建期望的SparseArray对象expected，指定dtype和填充值
        tm.assert_sp_array_equal(result, expected)  # 断言result与expected相等

    def test_astype_all(self, any_real_numpy_dtype):
        vals = np.array([1, 2, 3])  # 创建NumPy数组vals，包含指定元素
        arr = SparseArray(vals, fill_value=1)  # 创建SparseArray对象arr，指定元素和填充值
        typ = np.dtype(any_real_numpy_dtype)  # 创建NumPy dtype类型typ，指定为any_real_numpy_dtype
        res = arr.astype(typ)  # 将arr转换为指定dtype类型
        tm.assert_numpy_array_equal(res, vals.astype(any_real_numpy_dtype))  # 断言res与vals的指定dtype相等
    @pytest.mark.parametrize(
        "arr, dtype, expected",
        [   # 参数化测试：定义输入数组、目标数据类型和预期输出
            (
                SparseArray([0, 1]),
                "float",
                SparseArray([0.0, 1.0], dtype=SparseDtype(float, 0.0)),
            ),
            (SparseArray([0, 1]), bool, SparseArray([False, True])),
            (
                SparseArray([0, 1], fill_value=1),
                bool,
                SparseArray([False, True], dtype=SparseDtype(bool, True)),
            ),
            pytest.param(
                SparseArray([0, 1]),
                "datetime64[ns]",
                SparseArray(
                    np.array([0, 1], dtype="datetime64[ns]"),
                    dtype=SparseDtype("datetime64[ns]", Timestamp("1970")),
                ),
            ),
            (
                SparseArray([0, 1, 10]),
                str,
                SparseArray(["0", "1", "10"], dtype=SparseDtype(str, "0")),
            ),
            (SparseArray(["10", "20"]), float, SparseArray([10.0, 20.0])),
            (
                SparseArray([0, 1, 0]),
                object,
                SparseArray([0, 1, 0], dtype=SparseDtype(object, 0)),
            ),
        ],
    )
    # 定义测试方法：测试不同的数据类型转换，验证转换后的稀疏数组是否符合预期
    def test_astype_more(self, arr, dtype, expected):
        result = arr.astype(arr.dtype.update_dtype(dtype))
        tm.assert_sp_array_equal(result, expected)

    def test_astype_nan_raises(self):
        arr = SparseArray([1.0, np.nan])
        # 测试方法：测试转换包含 NaN 值的稀疏数组是否引发 ValueError 异常
        with pytest.raises(ValueError, match="Cannot convert non-finite"):
            arr.astype(int)

    def test_astype_copy_false(self):
        # 测试方法：测试修复 GH#34456 Bug 的情况，确保在 astype_nansafe 中使用 .astype 而不是 .view
        arr = SparseArray([1, 2, 3])

        dtype = SparseDtype(float, 0)

        result = arr.astype(dtype, copy=False)
        expected = SparseArray([1.0, 2.0, 3.0], fill_value=0.0)
        tm.assert_sp_array_equal(result, expected)

    def test_astype_dt64_to_int64(self):
        # 测试方法：测试将 datetime64 转换为 int64 的情况，验证转换后的稀疏数组是否与预期一致
        # 修复 GH#49631 Bug，匹配非稀疏情况下的行为
        values = np.array(["NaT", "2016-01-02", "2016-01-03"], dtype="M8[ns]")

        arr = SparseArray(values)
        result = arr.astype("int64")
        expected = values.astype("int64")
        tm.assert_numpy_array_equal(result, expected)

        # 验证也能够转换为等效的 Sparse[int64]
        dtype_int64 = SparseDtype("int64", np.iinfo(np.int64).min)
        result2 = arr.astype(dtype_int64)
        tm.assert_numpy_array_equal(result2.to_numpy(), expected)

        # 修复 GH#50087 Bug，确保无论填充值是否为 NaT，稀疏数组转换行为与非稀疏数组一致
        dtype = SparseDtype("datetime64[ns]", values[1])
        arr3 = SparseArray(values, dtype=dtype)
        result3 = arr3.astype("int64")
        tm.assert_numpy_array_equal(result3, expected)
```