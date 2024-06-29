# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_set_axis.py`

```
    # 导入 numpy 库，用于数值计算
    import numpy as np
    # 导入 pytest 库，用于编写和运行测试用例
    import pytest

    # 从 pandas 库中导入 DataFrame 和 Series 类
    from pandas import (
        DataFrame,
        Series,
    )
    # 导入 pandas._testing 模块，用于测试辅助函数
    import pandas._testing as tm

    # 定义一个共享的测试类 SharedSetAxisTests
    class SharedSetAxisTests:
        # pytest 的 fixture 方法，返回一个待实现的对象
        @pytest.fixture
        def obj(self):
            raise NotImplementedError("Implemented by subclasses")

        # 测试方法：测试 set_axis 方法设置索引在 Series 和 DataFrame 中的应用
        def test_set_axis(self, obj):
            # GH14636; 这个测试用例测试同时设置 Series 和 DataFrame 的索引
            new_index = list("abcd")[: len(obj)]
            expected = obj.copy()
            expected.index = new_index
            result = obj.set_axis(new_index, axis=0)
            tm.assert_equal(expected, result)

        # 测试方法：测试 set_axis 方法的 copy 关键字参数 GH#47932
        def test_set_axis_copy(self, obj):
            new_index = list("abcd")[: len(obj)]

            # 复制原始对象
            orig = obj.iloc[:]
            expected = obj.copy()
            expected.index = new_index

            # 使用 set_axis 方法设置新的索引
            result = obj.set_axis(new_index, axis=0)
            tm.assert_equal(expected, result)
            assert result is not obj
            # 检查我们没有进行复制操作
            if obj.ndim == 1:
                assert tm.shares_memory(result, obj)
            else:
                assert all(
                    tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                    for i in range(obj.shape[1])
                )

            result = obj.set_axis(new_index, axis=0)
            tm.assert_equal(expected, result)
            assert result is not obj
            # 再次检查我们没有进行复制操作
            if obj.ndim == 1:
                assert tm.shares_memory(result, obj)
            else:
                assert any(
                    tm.shares_memory(result.iloc[:, i], obj.iloc[:, i])
                    for i in range(obj.shape[1])
                )

            res = obj.set_axis(new_index)
            tm.assert_equal(expected, res)
            # 再次检查我们没有进行复制操作
            if res.ndim == 1:
                assert tm.shares_memory(res, orig)
            else:
                assert all(
                    tm.shares_memory(res.iloc[:, i], orig.iloc[:, i])
                    for i in range(res.shape[1])
                )

        # 测试方法：测试 set_axis 方法的未命名关键字参数警告
        def test_set_axis_unnamed_kwarg_warns(self, obj):
            # 省略 "axis" 参数的情况
            new_index = list("abcd")[: len(obj)]

            expected = obj.copy()
            expected.index = new_index

            result = obj.set_axis(new_index)
            tm.assert_equal(result, expected)

        # 使用 pytest 的参数化装饰器，测试 set_axis 方法的无效 "axis" 参数值
        @pytest.mark.parametrize("axis", [3, "foo"])
        def test_set_axis_invalid_axis_name(self, axis, obj):
            # 错误的 "axis" 参数值情况
            with pytest.raises(ValueError, match="No axis named"):
                obj.set_axis(list("abc"), axis=axis)

        # 测试方法：测试 set_axis 方法设置索引时的非集合类型错误情况
        def test_set_axis_setattr_index_not_collection(self, obj):
            # 错误的类型
            msg = (
                r"Index\(\.\.\.\) must be called with a collection of some "
                r"kind, None was passed"
            )
            with pytest.raises(TypeError, match=msg):
                obj.index = None
    # 定义一个测试方法，用于测试在给定对象上设置轴属性时长度不匹配的情况
    def test_set_axis_setattr_index_wrong_length(self, obj):
        # 构造错误长度的错误消息
        msg = (
            f"Length mismatch: Expected axis has {len(obj)} elements, "
            f"new values have {len(obj)-1} elements"
        )
        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            obj.index = np.arange(len(obj) - 1)

        # 如果对象是二维的
        if obj.ndim == 2:
            # 再次使用 pytest 检查是否会抛出 ValueError 异常，并匹配长度不匹配的错误消息
            with pytest.raises(ValueError, match="Length mismatch"):
                # 设置对象的列属性为当前列属性的每隔一个元素的切片
                obj.columns = obj.columns[::2]
# TestDataFrameSetAxis 类，继承自 SharedSetAxisTests 类
class TestDataFrameSetAxis(SharedSetAxisTests):
    
    # pytest 的 fixture 方法，用于创建测试对象
    @pytest.fixture
    def obj(self):
        # 创建一个 DataFrame 对象 df，包含三列数据 A, B, C，以及指定的索引
        df = DataFrame(
            {"A": [1.1, 2.2, 3.3], "B": [5.0, 6.1, 7.2], "C": [4.4, 5.5, 6.6]},
            index=[2010, 2011, 2012],
        )
        return df  # 返回创建的 DataFrame 对象作为测试对象


# TestSeriesSetAxis 类，继承自 SharedSetAxisTests 类
class TestSeriesSetAxis(SharedSetAxisTests):
    
    # pytest 的 fixture 方法，用于创建测试对象
    @pytest.fixture
    def obj(self):
        # 创建一个 Series 对象 ser，包含从 0 到 3 的整数值，并指定索引和数据类型
        ser = Series(np.arange(4), index=[1, 3, 5, 7], dtype="int64")
        return ser  # 返回创建的 Series 对象作为测试对象
```