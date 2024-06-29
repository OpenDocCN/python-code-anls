# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_take.py`

```
import pytest  # 导入 pytest 模块

import pandas._testing as tm  # 导入 pandas 测试工具模块

class TestDataFrameTake:
    def test_take_slices_not_supported(self, float_frame):
        # GH#51539: 标识 GitHub 上的 issue 编号
        df = float_frame  # 将 float_frame 赋给变量 df

        slc = slice(0, 4, 1)  # 创建一个切片对象 slc
        # 断言调用 take 方法时会引发 TypeError 异常，异常信息中包含 "slice"
        with pytest.raises(TypeError, match="slice"):
            df.take(slc, axis=0)
        with pytest.raises(TypeError, match="slice"):
            df.take(slc, axis=1)

    def test_take(self, float_frame):
        # homogeneous: 同类的
        order = [3, 1, 2, 0]  # 定义一个索引顺序列表 order
        for df in [float_frame]:  # 对于 float_frame 执行以下循环
            # 使用 take 方法按指定顺序重新排列行，并断言结果与预期相等
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            # 使用 take 方法按指定顺序重新排列列，并断言结果与预期相等，不检查列名
            result = df.take(order, axis=1)
            expected = df.loc[:, ["D", "B", "C", "A"]]
            tm.assert_frame_equal(result, expected, check_names=False)

        # negative indices: 负索引
        order = [2, 1, -1]
        for df in [float_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            result = df.take(order, axis=0)  # 重复调用 take 方法
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["C", "B", "D"]]
            tm.assert_frame_equal(result, expected, check_names=False)

        # illegal indices: 非法索引
        msg = "indices are out-of-bounds"  # 定义异常信息字符串
        # 断言调用 take 方法时会引发 IndexError 异常，异常信息包含 "indices are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, 30], axis=0)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, -31], axis=0)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, 5], axis=1)
        with pytest.raises(IndexError, match=msg):
            df.take([3, 1, 2, -5], axis=1)

    def test_take_mixed_type(self, float_string_frame):
        # mixed-dtype: 混合数据类型
        order = [4, 1, 2, 0, 3]  # 定义一个索引顺序列表 order
        for df in [float_string_frame]:  # 对于 float_string_frame 执行以下循环
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["foo", "B", "C", "A", "D"]]
            tm.assert_frame_equal(result, expected)

        # negative indices
        order = [4, 1, -2]
        for df in [float_string_frame]:
            result = df.take(order, axis=0)
            expected = df.reindex(df.index.take(order))
            tm.assert_frame_equal(result, expected)

            # axis = 1
            result = df.take(order, axis=1)
            expected = df.loc[:, ["foo", "B", "D"]]
            tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试混合类型数据框按指定顺序取值的情况
    def test_take_mixed_numeric(self, mixed_float_frame, mixed_int_frame):
        # 指定取值顺序
        order = [1, 2, 0, 3]
        # 对于每个数据框（包括浮点型和整型混合数据框）
        for df in [mixed_float_frame, mixed_int_frame]:
            # 按指定顺序在指定轴上取值，生成结果数据框
            result = df.take(order, axis=0)
            # 生成预期结果数据框，通过重新索引原数据框的索引来实现
            expected = df.reindex(df.index.take(order))
            # 使用测试框架验证结果数据框与预期数据框是否相等
            tm.assert_frame_equal(result, expected)

            # 在轴上进行另一种方式的取值操作（axis = 1）
            result = df.take(order, axis=1)
            # 生成预期结果数据框，仅包含指定的列顺序
            expected = df.loc[:, ["B", "C", "A", "D"]]
            # 使用测试框架验证结果数据框与预期数据框是否相等
            tm.assert_frame_equal(result, expected)
```