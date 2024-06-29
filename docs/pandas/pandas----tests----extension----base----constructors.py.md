# `D:\src\scipysrc\pandas\pandas\tests\extension\base\constructors.py`

```
    # 导入 NumPy 库，并将其命名为 np
    import numpy as np
    # 导入 pytest 库，用于编写和运行测试
    import pytest

    # 导入 Pandas 库，并将其命名为 pd
    import pandas as pd
    # 导入 Pandas 内部测试模块，用于测试辅助功能
    import pandas._testing as tm
    # 导入 Pandas 扩展数组接口
    from pandas.api.extensions import ExtensionArray
    # 导入 Pandas 内部模块，用于块的管理
    from pandas.core.internals.blocks import EABackedBlock

    # 定义基础构造器测试类
    class BaseConstructorsTests:
        # 测试从序列创建扩展数组
        def test_from_sequence_from_cls(self, data):
            # 调用类方法 _from_sequence 创建结果对象 result
            result = type(data)._from_sequence(data, dtype=data.dtype)
            # 使用测试模块验证结果对象和输入数据相等
            tm.assert_extension_array_equal(result, data)

            # 将数据切片为空并再次创建结果对象 result
            data = data[:0]
            result = type(data)._from_sequence(data, dtype=data.dtype)
            # 使用测试模块验证结果对象和输入数据相等
            tm.assert_extension_array_equal(result, data)

        # 测试从标量创建扩展数组
        def test_array_from_scalars(self, data):
            # 创建包含若干数据元素的标量列表 scalars
            scalars = [data[0], data[1], data[2]]
            # 调用对象方法 _from_sequence 创建结果对象 result
            result = data._from_sequence(scalars, dtype=data.dtype)
            # 验证结果对象的类型与输入对象类型相同
            assert isinstance(result, type(data))

        # 测试 Series 构造器
        def test_series_constructor(self, data):
            # 使用输入数据创建 Series 对象 result，关闭数据复制选项
            result = pd.Series(data, copy=False)
            # 验证结果 Series 对象的数据类型与输入数据类型相同
            assert result.dtype == data.dtype
            # 验证结果 Series 对象的长度与输入数据长度相同
            assert len(result) == len(data)
            # 如果结果对象的管理器存在 blocks 属性，则验证第一个块为 EABackedBlock 类型
            if hasattr(result._mgr, "blocks"):
                assert isinstance(result._mgr.blocks[0], EABackedBlock)
            # 验证结果 Series 对象的底层数组为输入数据对象

            # 使用结果对象 result 创建另一个 Series 对象 result2
            result2 = pd.Series(result)
            # 验证结果 Series 对象 result2 的数据类型与输入数据类型相同
            assert result2.dtype == data.dtype
            # 如果结果对象的管理器存在 blocks 属性，则验证第一个块为 EABackedBlock 类型
            if hasattr(result._mgr, "blocks"):
                assert isinstance(result2._mgr.blocks[0], EABackedBlock)

        # 测试无数据情况下的 Series 构造器
        def test_series_constructor_no_data_with_index(self, dtype, na_value):
            # 创建指定索引的 Series 对象 result
            result = pd.Series(index=[1, 2, 3], dtype=dtype)
            # 创建预期的 Series 对象 expected，其中每个元素为指定的缺失值
            expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
            # 使用测试模块验证结果 Series 对象与预期 Series 对象相等
            tm.assert_series_equal(result, expected)

            # GH 33559 - 空索引的情况
            # 创建空索引的 Series 对象 result
            result = pd.Series(index=[], dtype=dtype)
            # 创建预期的空 Series 对象 expected
            expected = pd.Series([], index=pd.Index([], dtype="object"), dtype=dtype)
            # 使用测试模块验证结果 Series 对象与预期 Series 对象相等
            tm.assert_series_equal(result, expected)

        # 测试带索引的标量缺失值情况下的 Series 构造器
        def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
            # 创建带索引和标量缺失值的 Series 对象 result
            result = pd.Series(na_value, index=[1, 2, 3], dtype=dtype)
            # 创建预期的 Series 对象 expected，其中每个元素为指定的缺失值
            expected = pd.Series([na_value] * 3, index=[1, 2, 3], dtype=dtype)
            # 使用测试模块验证结果 Series 对象与预期 Series 对象相等
            tm.assert_series_equal(result, expected)

        # 测试带索引的标量情况下的 Series 构造器
        def test_series_constructor_scalar_with_index(self, data, dtype):
            # 获取数据的第一个标量值
            scalar = data[0]
            # 创建带索引的标量 Series 对象 result
            result = pd.Series(scalar, index=[1, 2, 3], dtype=dtype)
            # 创建预期的 Series 对象 expected，其中每个元素为指定的标量值
            expected = pd.Series([scalar] * 3, index=[1, 2, 3], dtype=dtype)
            # 使用测试模块验证结果 Series 对象与预期 Series 对象相等
            tm.assert_series_equal(result, expected)

            # 创建带索引的标量 Series 对象 result
            result = pd.Series(scalar, index=["foo"], dtype=dtype)
            # 创建预期的 Series 对象 expected，其中每个元素为指定的标量值
            expected = pd.Series([scalar], index=["foo"], dtype=dtype)
            # 使用测试模块验证结果 Series 对象与预期 Series 对象相等
            tm.assert_series_equal(result, expected)

        # 使用 pytest 的参数化功能，为 from_series 参数分别传入 True 和 False 的情况
        @pytest.mark.parametrize("from_series", [True, False])
    # 从字典数据构造 DataFrame 的测试方法，支持从 Series 构造
    def test_dataframe_constructor_from_dict(self, data, from_series):
        # 如果 from_series 为真，则将 data 转换为 Series
        if from_series:
            data = pd.Series(data)
        # 用给定的数据构造一个 DataFrame，列名为 "A"
        result = pd.DataFrame({"A": data})
        # 断言结果 DataFrame 的列 "A" 的数据类型与原始数据的数据类型相同
        assert result.dtypes["A"] == data.dtype
        # 断言结果 DataFrame 的形状与原始数据的长度一致，列数为 1
        assert result.shape == (len(data), 1)
        # 如果结果 DataFrame 有 "_mgr" 属性的 "blocks" 存在，则断言第一个 block 是 EABackedBlock 类型
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        # 断言结果 DataFrame 的第一个 block 的值是 ExtensionArray 类型
        assert isinstance(result._mgr.blocks[0].values, ExtensionArray)

    # 从 Series 构造 DataFrame 的测试方法
    def test_dataframe_from_series(self, data):
        # 用给定的 Series 数据构造一个 DataFrame
        result = pd.DataFrame(pd.Series(data))
        # 断言结果 DataFrame 的第一列的数据类型与原始数据的数据类型相同
        assert result.dtypes[0] == data.dtype
        # 断言结果 DataFrame 的形状与原始数据的长度一致，列数为 1
        assert result.shape == (len(data), 1)
        # 如果结果 DataFrame 有 "_mgr" 属性的 "blocks" 存在，则断言第一个 block 是 EABackedBlock 类型
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)
        # 断言结果 DataFrame 的第一个 block 的值是 ExtensionArray 类型
        assert isinstance(result._mgr.blocks[0].values, ExtensionArray)

    # 测试当 Series 的长度与给定索引的长度不匹配时是否抛出 ValueError
    def test_series_given_mismatched_index_raises(self, data):
        # 期望的错误消息
        msg = r"Length of values \(3\) does not match length of index \(5\)"
        # 使用 pytest 的 raises 方法来断言是否抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            pd.Series(data[:3], index=[0, 1, 2, 3, 4])

    # 从数据类型构造 Series 和 DataFrame 的测试方法
    def test_from_dtype(self, data):
        # 用原始数据的数据类型构造期望的 Series
        expected = pd.Series(data)
        # 使用指定的数据类型构造 Series，并断言与期望的 Series 相等
        result = pd.Series(list(data), dtype=data.dtype)
        tm.assert_series_equal(result, expected)
        
        # 使用字符串形式的数据类型构造 Series，并断言与期望的 Series 相等
        result = pd.Series(list(data), dtype=str(data.dtype))
        tm.assert_series_equal(result, expected)
        
        # 用原始数据的数据类型构造期望的 DataFrame，并将结果 DataFrame 与期望的 DataFrame 进行比较
        expected = pd.DataFrame(data).astype(data.dtype)
        result = pd.DataFrame(list(data), dtype=data.dtype)
        tm.assert_frame_equal(result, expected)
        
        # 使用字符串形式的数据类型构造 DataFrame，并将结果 DataFrame 与期望的 DataFrame 进行比较
        result = pd.DataFrame(list(data), dtype=str(data.dtype))
        tm.assert_frame_equal(result, expected)

    # 测试从数据构造 Pandas 数组的方法
    def test_pandas_array(self, data):
        # 使用原始数据构造 Pandas 数组，断言与原始数据相等
        result = pd.array(data)
        tm.assert_extension_array_equal(result, data)

    # 测试从数据构造 Pandas 数组并指定数据类型的方法
    def test_pandas_array_dtype(self, data):
        # 使用原始数据构造 Pandas 数组，并指定为 np.dtype(object)，断言与期望的 NumpyExtensionArray 相等
        result = pd.array(data, dtype=np.dtype(object))
        expected = pd.arrays.NumpyExtensionArray(np.asarray(data, dtype=object))
        tm.assert_equal(result, expected)

    # 测试构造空 DataFrame 的方法
    def test_construct_empty_dataframe(self, dtype):
        # 构造一个指定列名和数据类型的空 DataFrame
        result = pd.DataFrame(columns=["a"], dtype=dtype)
        # 构造一个期望的空 DataFrame，列名为 "a"，数据类型为给定的 dtype，索引为 pd.RangeIndex(0)
        expected = pd.DataFrame({"a": pd.array([], dtype=dtype)}, index=pd.RangeIndex(0))
        # 断言结果 DataFrame 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试空数组的创建
    def test_empty(self, dtype):
        # 根据给定的数据类型创建一个数组类型的类
        cls = dtype.construct_array_type()
        # 使用该类的方法创建一个空数组，形状为 (4,)，数据类型为 dtype
        result = cls._empty((4,), dtype=dtype)
        # 断言结果是 cls 类型的实例
        assert isinstance(result, cls)
        # 断言结果的数据类型是预期的 dtype
        assert result.dtype == dtype
        # 断言结果的形状为 (4,)
        assert result.shape == (4,)

        # 使用 ExtensionDtype 类的方法创建一个空数组，形状为 (4,)
        # GH#19600 表示该操作关联到 GitHub 上的 Issue #19600
        result2 = dtype.empty((4,))
        # 断言结果是 cls 类型的实例
        assert isinstance(result2, cls)
        # 断言结果的数据类型是预期的 dtype
        assert result2.dtype == dtype
        # 断言结果的形状为 (4,)
        assert result2.shape == (4,)

        # 使用 dtype 类的方法创建一个空数组，形状为 (4,)
        result2 = dtype.empty(4)
        # 断言结果是 cls 类型的实例
        assert isinstance(result2, cls)
        # 断言结果的数据类型是预期的 dtype
        assert result2.dtype == dtype
        # 断言结果的形状为 (4,)
        assert result2.shape == (4,)
```