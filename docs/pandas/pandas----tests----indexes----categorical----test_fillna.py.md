# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_fillna.py`

```
    # 导入 numpy 库，并将其重命名为 np
    import numpy as np
    # 导入 pytest 库
    import pytest

    # 从 pandas 库中导入 CategoricalIndex 类
    from pandas import CategoricalIndex
    # 导入 pandas._testing 模块并将其重命名为 tm
    import pandas._testing as tm


    class TestFillNA:
        # 定义测试类 TestFillNA
        def test_fillna_categorical(self):
            # GH#11343: 创建一个 CategoricalIndex，包含浮点数和 NaN 值，指定名称为 "x"
            idx = CategoricalIndex([1.0, np.nan, 3.0, 1.0], name="x")
            # 用类别中的值填充 NaN 值，创建期望的 CategoricalIndex
            exp = CategoricalIndex([1.0, 1.0, 3.0, 1.0], name="x")
            # 使用 assert_index_equal 函数比较填充后的结果和期望结果
            tm.assert_index_equal(idx.fillna(1.0), exp)

            # 获取 CategoricalIndex 的数据对象
            cat = idx._data

            # 尝试用不在类别中的值填充，预期会引发 TypeError 错误，匹配错误消息 "Cannot setitem on a Categorical with a new category"
            with pytest.raises(TypeError, match="Cannot setitem on a Categorical with a new category"):
                cat.fillna(2.0)

            # 对整个索引进行填充，将 NaN 值替换为 2.0
            result = idx.fillna(2.0)
            # 将索引转换为对象类型后再进行填充，以获取预期的结果
            expected = idx.astype(object).fillna(2.0)
            # 使用 assert_index_equal 函数比较填充后的结果和预期结果
            tm.assert_index_equal(result, expected)

        def test_fillna_copies_with_no_nas(self):
            # 当没有 NaN 值需要填充时，应当仍然获得一个副本，但对于 CategoricalIndex 方法来说，获得视图也是可以的
            ci = CategoricalIndex([0, 1, 1])
            # 对 CategoricalIndex 进行填充操作
            result = ci.fillna(0)
            # 检查返回的结果对象是否不是原始对象的引用
            assert result is not ci
            # 检查返回的结果对象和原始对象是否共享内存
            assert tm.shares_memory(result, ci)

            # 但在 EA 级别上，我们始终会获得一个副本
            cat = ci._data
            # 对 Categorical 数据对象进行填充操作
            result = cat.fillna(0)
            # 检查返回的结果对象的 ndarray 是否不是原始对象的 ndarray
            assert result._ndarray is not cat._ndarray
            # 检查返回的结果对象的 ndarray 是否没有基类
            assert result._ndarray.base is None
            # 检查返回的结果对象和原始对象是否不共享内存
            assert not tm.shares_memory(result, cat)

        def test_fillna_validates_with_no_nas(self):
            # 即使 fillna 操作没有实际的 NaN 值需要填充，我们仍然会验证填充值的有效性
            ci = CategoricalIndex([2, 3, 3])
            cat = ci._data

            # 对于没有需要填充的情况，我们不会进行类型转换
            res = ci.fillna(False)
            # 使用 assert_index_equal 函数比较填充后的结果和原始结果
            tm.assert_index_equal(res, ci)

            # 直接在 Categorical 对象上进行相同的检查
            with pytest.raises(TypeError, match="Cannot setitem on a Categorical with a new category"):
                cat.fillna(False)
```