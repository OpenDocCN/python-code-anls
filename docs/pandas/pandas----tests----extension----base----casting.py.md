# `D:\src\scipysrc\pandas\pandas\tests\extension\base\casting.py`

```
    # 导入 numpy 库并用 np 别名引用
    import numpy as np
    # 导入 pytest 库
    import pytest

    # 导入 pandas 库中的 _test_decorators 模块
    import pandas.util._test_decorators as td

    # 导入 pandas 库并用 pd 别名引用
    import pandas as pd
    # 导入 pandas 库中的 _testing 模块并用 tm 别名引用
    import pandas._testing as tm
    # 从 pandas.core.internals.blocks 模块中导入 NumpyBlock 类
    from pandas.core.internals.blocks import NumpyBlock


    class BaseCastingTests:
        """Casting to and from ExtensionDtypes"""

        # 测试将 Series 转换为 object 类型的方法
        def test_astype_object_series(self, all_data):
            # 创建一个名为 "A" 的 Series 对象
            ser = pd.Series(all_data, name="A")
            # 执行类型转换为 object
            result = ser.astype(object)
            # 断言结果的数据类型为 Python 内置的 object 类型
            assert result.dtype == np.dtype(object)
            # 如果结果对象具有 _mgr 属性下的 blocks 属性
            if hasattr(result._mgr, "blocks"):
                # 获取第一个块对象
                blk = result._mgr.blocks[0]
                # 断言第一个块对象为 NumpyBlock 类型
                assert isinstance(blk, NumpyBlock)
                # 断言第一个块对象的数据类型为 object
                assert blk.is_object
            # 断言结果的数据存储在 numpy 数组中
            assert isinstance(result._mgr.array, np.ndarray)
            # 断言 numpy 数组的数据类型为 object
            assert result._mgr.array.dtype == np.dtype(object)

        # 测试将 DataFrame 转换为 object 类型的方法
        def test_astype_object_frame(self, all_data):
            # 创建一个包含所有数据的 DataFrame 对象，列名为 "A"
            df = pd.DataFrame({"A": all_data})

            # 执行类型转换为 object
            result = df.astype(object)
            # 如果结果对象具有 _mgr 属性下的 blocks 属性
            if hasattr(result._mgr, "blocks"):
                # 获取第一个块对象
                blk = result._mgr.blocks[0]
                # 断言第一个块对象为 NumpyBlock 类型
                assert isinstance(blk, NumpyBlock), type(blk)
                # 断言第一个块对象的数据类型为 object
                assert blk.is_object
            # 获取结果 DataFrame 的第一个块的值数组
            arr = result._mgr.blocks[0].values
            # 断言该值数组为 numpy 数组
            assert isinstance(arr, np.ndarray)
            # 断言值数组的数据类型为 object
            assert arr.dtype == np.dtype(object)

            # 检查结果的数据类型是否与原始 DataFrame 的数据类型相同
            comp = result.dtypes == df.dtypes
            assert not comp.any()

        # 测试将 Series 转换为列表的方法
        def test_tolist(self, data):
            # 将 Series 对象转换为列表
            result = pd.Series(data).tolist()
            # 创建期望的列表结果
            expected = list(data)
            # 断言转换后的结果与期望结果相等
            assert result == expected

        # 测试将 Series 转换为 str 类型的方法
        def test_astype_str(self, data):
            # 将 Series 的前五个元素转换为 str 类型
            result = pd.Series(data[:5]).astype(str)
            # 创建期望的 Series 结果，所有元素转换为 str 类型
            expected = pd.Series([str(x) for x in data[:5]], dtype=str)
            # 断言转换后的结果与期望结果相等
            tm.assert_series_equal(result, expected)

        # 参数化测试，测试将 Series 转换为指定可为空的字符串类型的方法
        @pytest.mark.parametrize(
            "nullable_string_dtype",
            [
                "string[python]",
                pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
            ],
        )
        def test_astype_string(self, data, nullable_string_dtype):
            # 将 Series 的前五个元素转换为指定的可为空的字符串类型
            result = pd.Series(data[:5]).astype(nullable_string_dtype)
            # 创建期望的 Series 结果，如果元素不是 bytes 类型则转换为 str 类型，否则解码为 str 类型
            expected = pd.Series(
                [str(x) if not isinstance(x, bytes) else x.decode() for x in data[:5]],
                dtype=nullable_string_dtype,
            )
            # 断言转换后的结果与期望结果相等
            tm.assert_series_equal(result, expected)

        # 测试将 Series 或 DataFrame 转换为 numpy 数组的方法
        def test_to_numpy(self, data):
            # 创建期望的 numpy 数组
            expected = np.asarray(data)

            # 将 Series 对象转换为 numpy 数组
            result = data.to_numpy()
            # 断言转换后的结果与期望结果相等
            tm.assert_equal(result, expected)

            # 将 Series 对象转换为 numpy 数组
            result = pd.Series(data).to_numpy()
            # 断言转换后的结果与期望结果相等
            tm.assert_equal(result, expected)

        # 测试将空 DataFrame 转换为指定类型的方法
        def test_astype_empty_dataframe(self, dtype):
            # 创建一个空 DataFrame
            df = pd.DataFrame()
            # 执行类型转换为指定类型
            result = df.astype(dtype)
            # 断言转换后的结果与原始 DataFrame 相等
            tm.assert_frame_equal(result, df)

        # 参数化测试的参数，测试拷贝操作的影响
        @pytest.mark.parametrize("copy", [True, False])
    # 定义一个测试方法 test_astype_own_type，用于验证在 dtype 相等且 copy=False 的情况下，astype 返回原始对象。
    def test_astype_own_type(self, data, copy):
        # 确保当 dtype 相等且 copy=False 时，astype 返回原始对象
        # 参考：https://github.com/pandas-dev/pandas/issues/28488
        # 调用 data 的 astype 方法，将其转换为相同的 dtype，并根据 copy 参数决定是否复制数据
        result = data.astype(data.dtype, copy=copy)
        # 使用断言检查返回的结果是否为原始 data 对象，且根据 copy 参数判断是否复制了对象
        assert (result is data) is (not copy)
        # 使用测试框架的方法验证 result 和 data 的内容是否相等
        tm.assert_extension_array_equal(result, data)
```