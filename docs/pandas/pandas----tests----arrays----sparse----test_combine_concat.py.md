# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_combine_concat.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

import pandas as pd  # 导入Pandas库，用于数据操作和分析
import pandas._testing as tm  # 导入Pandas测试模块
from pandas.core.arrays.sparse import SparseArray  # 导入Pandas稀疏数组模块


class TestSparseArrayConcat:
    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_basic(self, kind):
        a = SparseArray([1, 0, 0, 2], kind=kind)  # 创建稀疏数组a，指定类型为kind
        b = SparseArray([1, 0, 2, 2], kind=kind)  # 创建稀疏数组b，指定类型为kind

        result = SparseArray._concat_same_type([a, b])
        # 无法对稀疏索引本身做任何断言，因为我们没有在to_concat中合并稀疏块跨数组
        # 在结果中断言稀疏值数组是否与期望结果相等
        expected = np.array([1, 2, 1, 2, 2], dtype="int64")
        tm.assert_numpy_array_equal(result.sp_values, expected)
        assert result.kind == kind  # 断言结果的稀疏数组类型为kind

    @pytest.mark.parametrize("kind", ["integer", "block"])
    def test_uses_first_kind(self, kind):
        other = "integer" if kind == "block" else "block"  # 根据kind选择另一种类型
        a = SparseArray([1, 0, 0, 2], kind=kind)  # 创建稀疏数组a，指定类型为kind
        b = SparseArray([1, 0, 2, 2], kind=other)  # 创建稀疏数组b，指定类型为other

        result = SparseArray._concat_same_type([a, b])
        expected = np.array([1, 2, 1, 2, 2], dtype="int64")
        tm.assert_numpy_array_equal(result.sp_values, expected)
        assert result.kind == kind  # 断言结果的稀疏数组类型为kind


@pytest.mark.parametrize(
    "other, expected_dtype",
    [
        # 兼容的dtype -> 保留稀疏
        (pd.Series([3, 4, 5], dtype="int64"), pd.SparseDtype("int64", 0)),
        # (pd.Series([3, 4, 5], dtype="Int64"), pd.SparseDtype("int64", 0)),
        # 不兼容的dtype -> Sparse[common dtype]
        (pd.Series([1.5, 2.5, 3.5], dtype="float64"), pd.SparseDtype("float64", 0)),
        # 不兼容的dtype -> Sparse[object] dtype
        (pd.Series(["a", "b", "c"], dtype=object), pd.SparseDtype(object, 0)),
        # 具有兼容类别的分类变量 -> 类别的dtype
        (pd.Series([3, 4, 5], dtype="category"), np.dtype("int64")),
        (pd.Series([1.5, 2.5, 3.5], dtype="category"), np.dtype("float64")),
        # 具有不兼容类别的分类变量 -> object dtype
        (pd.Series(["a", "b", "c"], dtype="category"), np.dtype(object)),
    ],
)
def test_concat_with_non_sparse(other, expected_dtype):
    # https://github.com/pandas-dev/pandas/issues/34336
    s_sparse = pd.Series([1, 0, 2], dtype=pd.SparseDtype("int64", 0))

    result = pd.concat([s_sparse, other], ignore_index=True)
    expected = pd.Series(list(s_sparse) + list(other)).astype(expected_dtype)
    tm.assert_series_equal(result, expected)

    result = pd.concat([other, s_sparse], ignore_index=True)
    expected = pd.Series(list(other) + list(s_sparse)).astype(expected_dtype)
    tm.assert_series_equal(result, expected)
```