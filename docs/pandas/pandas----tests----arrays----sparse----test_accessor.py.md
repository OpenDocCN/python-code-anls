# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_accessor.py`

```
import string  # 导入 string 模块，用于处理字符串操作

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from pandas import SparseDtype  # 从 Pandas 中导入 SparseDtype 类型
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.core.arrays.sparse import SparseArray  # 从 Pandas 中导入 SparseArray 类

class TestSeriesAccessor:
    def test_to_dense(self):
        ser = pd.Series([0, 1, 0, 10], dtype="Sparse[int64]")  # 创建一个稀疏整数类型的 Pandas Series
        result = ser.sparse.to_dense()  # 将稀疏 Series 转换为密集 Series
        expected = pd.Series([0, 1, 0, 10])  # 期望的密集 Series
        tm.assert_series_equal(result, expected)  # 断言结果与期望相等

    @pytest.mark.parametrize("attr", ["npoints", "density", "fill_value", "sp_values"])
    def test_get_attributes(self, attr):
        arr = SparseArray([0, 1])  # 创建一个 SparseArray 对象
        ser = pd.Series(arr)  # 创建一个包含 SparseArray 的 Pandas Series

        result = getattr(ser.sparse, attr)  # 获取稀疏 Series 的指定属性值
        expected = getattr(arr, attr)  # 获取 SparseArray 对象的相同属性值
        assert result == expected  # 断言结果与期望相等

    def test_from_coo(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")  # 导入并检查 SciPy 的稀疏矩阵模块是否可用

        row = [0, 3, 1, 0]
        col = [0, 3, 1, 2]
        data = [4, 5, 7, 9]

        sp_array = scipy_sparse.coo_matrix((data, (row, col)))  # 创建一个 SciPy 的 COO 稀疏矩阵
        result = pd.Series.sparse.from_coo(sp_array)  # 从 COO 矩阵创建 Pandas 的稀疏 Series

        index = pd.MultiIndex.from_arrays(
            [
                np.array([0, 0, 1, 3], dtype=np.int32),
                np.array([0, 2, 1, 3], dtype=np.int32),
            ],
        )
        expected = pd.Series([4, 9, 7, 5], index=index, dtype="Sparse[int]")  # 期望的稀疏 Series
        tm.assert_series_equal(result, expected)  # 断言结果与期望相等

    @pytest.mark.parametrize(
        "sort_labels, expected_rows, expected_cols, expected_values_pos",
        [
            (
                False,
                [("b", 2), ("a", 2), ("b", 1), ("a", 1)],
                [("z", 1), ("z", 2), ("x", 2), ("z", 0)],
                {1: (1, 0), 3: (3, 3)},
            ),
            (
                True,
                [("a", 1), ("a", 2), ("b", 1), ("b", 2)],
                [("x", 2), ("z", 0), ("z", 1), ("z", 2)],
                {1: (1, 2), 3: (0, 1)},
            ),
        ],
    )
    def test_to_coo(
        self, sort_labels, expected_rows, expected_cols, expected_values_pos
    ):
        sp_sparse = pytest.importorskip("scipy.sparse")  # 导入并检查 SciPy 的稀疏矩阵模块是否可用

        values = SparseArray([0, np.nan, 1, 0, None, 3], fill_value=0)  # 创建一个 SparseArray 对象
        index = pd.MultiIndex.from_tuples(
            [
                ("b", 2, "z", 1),
                ("a", 2, "z", 2),
                ("a", 2, "z", 1),
                ("a", 2, "x", 2),
                ("b", 1, "z", 1),
                ("a", 1, "z", 0),
            ]
        )
        ss = pd.Series(values, index=index)  # 创建一个包含 SparseArray 的 Pandas Series

        expected_A = np.zeros((4, 4))  # 创建一个预期的稀疏矩阵
        for value, (row, col) in expected_values_pos.items():
            expected_A[row, col] = value  # 填充预期的稀疏矩阵

        A, rows, cols = ss.sparse.to_coo(  # 将稀疏 Series 转换为 COO 格式
            row_levels=(0, 1), column_levels=(2, 3), sort_labels=sort_labels
        )
        assert isinstance(A, sp_sparse.coo_matrix)  # 断言 A 是 SciPy 的 COO 矩阵类型
        tm.assert_numpy_array_equal(A.toarray(), expected_A)  # 断言稀疏矩阵 A 的数组形式与预期相等
        assert rows == expected_rows  # 断言行标签与预期相等
        assert cols == expected_cols  # 断言列标签与预期相等
    # 定义一个测试方法，用于测试非稀疏系列（Series）是否会触发异常
    def test_non_sparse_raises(self):
        # 创建一个 Pandas 系列（Series），其中包含整数 1, 2, 3
        ser = pd.Series([1, 2, 3])
        # 使用 pytest 的上下文管理器来捕获期望的异常，即 AttributeError，匹配错误消息中包含 ".sparse"
        with pytest.raises(AttributeError, match=".sparse"):
            # 在稀疏属性上访问密度（density），预期会触发 AttributeError 异常
            ser.sparse.density
    # 定义 TestFrameAccessor 类，用于测试 DataFrame 的访问器功能
    class TestFrameAccessor:
        
        # 定义测试方法，验证访问器引发 AttributeError 异常，匹配 "sparse" 字符串
        def test_accessor_raises(self):
            # 创建一个包含列"A"的DataFrame，每列分别包含0和1
            df = pd.DataFrame({"A": [0, 1]})
            # 使用 pytest 断言检查访问名为 "sparse" 的属性时是否引发了 AttributeError 异常
            with pytest.raises(AttributeError, match="sparse"):
                df.sparse

        # 使用 pytest 的参数化装饰器标记，测试从稀疏矩阵创建 DataFrame 的功能
        @pytest.mark.parametrize("format", ["csc", "csr", "coo"])
        @pytest.mark.parametrize("labels", [None, list(string.ascii_letters[:10])])
        @pytest.mark.parametrize("dtype", [np.complex128, np.float64, np.int64, bool])
        def test_from_spmatrix(self, format, labels, dtype):
            # 导入 scipy.sparse 模块，如果不存在则跳过测试
            sp_sparse = pytest.importorskip("scipy.sparse")

            # 创建稀疏数据类型对象 SparseDtype
            sp_dtype = SparseDtype(dtype)

            # 生成一个 10x10 的稀疏单位矩阵
            sp_mat = sp_sparse.eye(10, format=format, dtype=dtype)
            # 使用稀疏矩阵创建 DataFrame，指定索引和列名
            result = pd.DataFrame.sparse.from_spmatrix(sp_mat, index=labels, columns=labels)
            # 生成一个与稀疏矩阵对应的密集矩阵
            mat = np.eye(10, dtype=dtype)
            # 使用 MaskedArray 填充密集矩阵，创建期望的 DataFrame，转换为指定的稀疏数据类型
            expected = pd.DataFrame(
                np.ma.array(mat, mask=(mat == 0)).filled(sp_dtype.fill_value),
                index=labels,
                columns=labels,
            ).astype(sp_dtype)
            # 使用测试工具函数检查两个 DataFrame 是否相等
            tm.assert_frame_equal(result, expected)

        # 使用 pytest 的参数化装饰器标记，测试包含显式零值的稀疏矩阵创建 DataFrame 的功能
        @pytest.mark.parametrize("format", ["csc", "csr", "coo"])
        @pytest.mark.parametrize("dtype", [np.int64, bool])
        def test_from_spmatrix_including_explicit_zero(self, format, dtype):
            # 导入 scipy.sparse 模块，如果不存在则跳过测试
            sp_sparse = pytest.importorskip("scipy.sparse")

            # 创建稀疏数据类型对象 SparseDtype
            sp_dtype = SparseDtype(dtype)

            # 生成一个具有 50% 密度的随机稀疏矩阵，格式为指定的格式和数据类型
            sp_mat = sp_sparse.random(10, 2, density=0.5, format=format, dtype=dtype)
            # 将稀疏矩阵的第一个数据元素设为零
            sp_mat.data[0] = 0
            # 使用稀疏矩阵创建 DataFrame
            result = pd.DataFrame.sparse.from_spmatrix(sp_mat)
            # 将稀疏矩阵转换为密集矩阵
            mat = sp_mat.toarray()
            # 使用 MaskedArray 填充密集矩阵，创建期望的 DataFrame，转换为指定的稀疏数据类型
            expected = pd.DataFrame(
                np.ma.array(mat, mask=(mat == 0)).filled(sp_dtype.fill_value)
            ).astype(sp_dtype)
            # 使用测试工具函数检查两个 DataFrame 是否相等
            tm.assert_frame_equal(result, expected)

        # 使用 pytest 的参数化装饰器标记，测试从稀疏矩阵创建 DataFrame 并指定列名的功能
        @pytest.mark.parametrize(
            "columns",
            [["a", "b"], pd.MultiIndex.from_product([["A"], ["a", "b"]]), ["a", "a"]],
        )
        def test_from_spmatrix_columns(self, columns):
            # 导入 scipy.sparse 模块，如果不存在则跳过测试
            sp_sparse = pytest.importorskip("scipy.sparse")

            # 创建稀疏数据类型对象 SparseDtype，指定数据类型为 np.float64
            sp_dtype = SparseDtype(np.float64)

            # 生成一个具有 50% 密度的随机稀疏矩阵
            sp_mat = sp_sparse.random(10, 2, density=0.5)
            # 使用稀疏矩阵创建 DataFrame，指定列名
            result = pd.DataFrame.sparse.from_spmatrix(sp_mat, columns=columns)
            # 将稀疏矩阵转换为密集矩阵
            mat = sp_mat.toarray()
            # 使用 MaskedArray 填充密集矩阵，创建期望的 DataFrame，转换为指定的稀疏数据类型
            expected = pd.DataFrame(
                np.ma.array(mat, mask=(mat == 0)).filled(sp_dtype.fill_value),
                columns=columns,
            ).astype(sp_dtype)
            # 使用测试工具函数检查两个 DataFrame 是否相等
            tm.assert_frame_equal(result, expected)

        # 使用 pytest 的参数化装饰器标记，测试不同列名和数据类型的组合
        @pytest.mark.parametrize(
            "columns", [("A", "B"), (1, 2), (1, pd.NA), (0.1, 0.2), ("x", "x"), (0, 0)]
        )
        @pytest.mark.parametrize("dtype", [np.complex128, np.float64, np.int64, bool])
    def test_to_coo(self, columns, dtype):
        # 导入 pytest 模块并检查是否可用 scipy.sparse 模块
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 创建稀疏数据类型对象
        sp_dtype = SparseDtype(dtype)

        # 生成一个稀疏随机矩阵作为预期输出，格式为 COO
        expected = sp_sparse.random(10, 2, density=0.5, format="coo", dtype=dtype)
        
        # 将预期矩阵转换为密集矩阵
        mat = expected.toarray()
        
        # 创建一个 Pandas 数据帧，将密集矩阵转换为稀疏类型，并指定数据类型
        result = pd.DataFrame(
            np.ma.array(mat, mask=(mat == 0)).filled(sp_dtype.fill_value),
            columns=columns,
            dtype=sp_dtype,
        ).sparse.to_coo()
        
        # 断言结果矩阵与预期矩阵的非零元素数量相等
        assert (result != expected).nnz == 0

    def test_to_coo_midx_categorical(self):
        # GH#50996
        # 导入 pytest 模块并检查是否可用 scipy.sparse 模块
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 创建一个多级索引
        midx = pd.MultiIndex.from_arrays(
            [
                pd.CategoricalIndex(list("ab"), name="x"),
                pd.CategoricalIndex([0, 1], name="y"),
            ]
        )

        # 创建一个稀疏系列对象
        ser = pd.Series(1, index=midx, dtype="Sparse[int]")
        
        # 将稀疏系列转换为 COO 格式，指定行和列的级别
        result = ser.sparse.to_coo(row_levels=["x"], column_levels=["y"])[0]
        
        # 创建预期的 COO 矩阵
        expected = sp_sparse.coo_matrix(
            (np.array([1, 1]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2)
        )
        
        # 断言结果矩阵与预期矩阵的非零元素数量相等
        assert (result != expected).nnz == 0

    def test_to_dense(self):
        # 创建一个包含稀疏数组的数据帧
        df = pd.DataFrame(
            {
                "A": SparseArray([1, 0], dtype=SparseDtype("int64", 0)),
                "B": SparseArray([1, 0], dtype=SparseDtype("int64", 1)),
                "C": SparseArray([1.0, 0.0], dtype=SparseDtype("float64", 0.0)),
            },
            index=["b", "a"],
        )
        
        # 将稀疏数据帧转换为密集形式
        result = df.sparse.to_dense()
        
        # 创建预期的密集数据帧
        expected = pd.DataFrame(
            {"A": [1, 0], "B": [1, 0], "C": [1.0, 0.0]}, index=["b", "a"]
        )
        
        # 使用断言比较结果和预期数据帧
        tm.assert_frame_equal(result, expected)

    def test_density(self):
        # 创建一个包含稀疏数组的数据帧
        df = pd.DataFrame(
            {
                "A": SparseArray([1, 0, 2, 1], fill_value=0),
                "B": SparseArray([0, 1, 1, 1], fill_value=0),
            }
        )
        
        # 计算稀疏数据帧的密度
        res = df.sparse.density
        
        # 预期的密度值
        expected = 0.75
        
        # 断言计算结果与预期值相等
        assert res == expected

    @pytest.mark.parametrize("dtype", ["int64", "float64"])
    @pytest.mark.parametrize("dense_index", [True, False])
    def test_series_from_coo(self, dtype, dense_index):
        # 导入 pytest 模块并检查是否可用 scipy.sparse 模块
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 创建一个 COO 格式的稀疏矩阵
        A = sp_sparse.eye(3, format="coo", dtype=dtype)
        
        # 从 COO 矩阵创建稀疏系列对象
        result = pd.Series.sparse.from_coo(A, dense_index=dense_index)

        # 创建预期的稀疏系列对象的索引
        index = pd.MultiIndex.from_tuples(
            [
                np.array([0, 0], dtype=np.int32),
                np.array([1, 1], dtype=np.int32),
                np.array([2, 2], dtype=np.int32),
            ],
        )
        
        # 创建预期的稀疏系列对象
        expected = pd.Series(SparseArray(np.array([1, 1, 1], dtype=dtype)), index=index)
        
        # 如果设置了 dense_index，重新索引预期结果
        if dense_index:
            expected = expected.reindex(pd.MultiIndex.from_product(index.levels))
        
        # 使用断言比较结果和预期稀疏系列对象
        tm.assert_series_equal(result, expected)
    # 定义测试函数：当传入 COO 格式不正确时，应该引发异常
    def test_series_from_coo_incorrect_format_raises(self):
        # 引入 pytest 库，并检查是否可用，如果不可用则跳过此测试
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 创建一个 CSR 矩阵作为测试数据
        m = sp_sparse.csr_matrix(np.array([[0, 1], [0, 0]]))
        
        # 使用 pytest 的断言检查是否引发指定类型的异常，并验证异常信息匹配预期
        with pytest.raises(
            TypeError, match="Expected coo_matrix. Got csr_matrix instead."
        ):
            # 调用 Pandas 的 sparse.from_coo 方法，预期引发异常
            pd.Series.sparse.from_coo(m)

    # 定义测试函数：验证在 DataFrame 中存在名为 'sparse' 的列时的行为
    def test_with_column_named_sparse(self):
        # 创建一个包含名为 'sparse' 的列的 DataFrame
        df = pd.DataFrame({"sparse": pd.arrays.SparseArray([1, 2])})
        
        # 使用断言验证 'sparse' 列是否是 SparseFrameAccessor 类型的对象
        assert isinstance(df.sparse, pd.core.arrays.sparse.accessor.SparseFrameAccessor)
```