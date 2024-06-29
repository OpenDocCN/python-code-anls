# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_set_value.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas.core.dtypes.common import is_float_dtype  # 从 Pandas 核心模块中导入检查浮点类型的函数

from pandas import (  # 从 Pandas 库中导入以下对象
    DataFrame,  # 数据框对象，用于处理二维数据
    isna,  # 检查缺失值的函数
)
import pandas._testing as tm  # 导入 Pandas 测试模块的别名 tm


class TestSetValue:  # 定义一个测试类 TestSetValue
    def test_set_value(self, float_frame):  # 定义测试方法 test_set_value，接收一个 float_frame 参数
        for idx in float_frame.index:  # 遍历 float_frame 的索引
            for col in float_frame.columns:  # 遍历 float_frame 的列
                float_frame._set_value(idx, col, 1)  # 在 float_frame 中设置值为 1
                assert float_frame[col][idx] == 1  # 断言检查设置是否成功

    def test_set_value_resize(self, float_frame, using_infer_string):  # 定义测试方法 test_set_value_resize，接收 float_frame 和 using_infer_string 参数
        res = float_frame._set_value("foobar", "B", 0)  # 在 float_frame 中设置特定值，返回结果保存在 res 中
        assert res is None  # 断言 res 应为 None
        assert float_frame.index[-1] == "foobar"  # 断言 float_frame 的最后一个索引为 "foobar"
        assert float_frame._get_value("foobar", "B") == 0  # 断言从 float_frame 中获取特定值为 0

        float_frame.loc["foobar", "qux"] = 0  # 在 float_frame 中指定位置设置值为 0
        assert float_frame._get_value("foobar", "qux") == 0  # 断言从 float_frame 中获取特定值为 0

        res = float_frame.copy()  # 复制 float_frame 到 res
        res._set_value("foobar", "baz", "sam")  # 在 res 中设置特定值
        if using_infer_string:  # 如果 using_infer_string 为真
            assert res["baz"].dtype == "string"  # 断言 res 中 "baz" 列的数据类型为字符串
        else:
            assert res["baz"].dtype == np.object_  # 断言 res 中 "baz" 列的数据类型为 NumPy 对象类型
        res = float_frame.copy()  # 再次复制 float_frame 到 res
        res._set_value("foobar", "baz", True)  # 在 res 中设置特定值为 True
        assert res["baz"].dtype == np.object_  # 断言 res 中 "baz" 列的数据类型为 NumPy 对象类型

        res = float_frame.copy()  # 再次复制 float_frame 到 res
        res._set_value("foobar", "baz", 5)  # 在 res 中设置特定值为 5
        assert is_float_dtype(res["baz"])  # 断言 res 中 "baz" 列的数据类型为浮点类型
        assert isna(res["baz"].drop(["foobar"])).all()  # 断言 res 中除了 "foobar" 行外，其余值均为缺失值

        with tm.assert_produces_warning(  # 使用 tm 模块的断言检查警告
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            res._set_value("foobar", "baz", "sam")  # 在 res 中设置特定值为 "sam"
        assert res.loc["foobar", "baz"] == "sam"  # 断言 res 中 "foobar" 行 "baz" 列的值为 "sam"

    def test_set_value_with_index_dtype_change(self):  # 定义测试方法 test_set_value_with_index_dtype_change
        df_orig = DataFrame(  # 创建一个 DataFrame 对象 df_orig
            np.random.default_rng(2).standard_normal((3, 3)),  # 使用随机数生成数据填充
            index=range(3),  # 设置索引为范围 0 到 2
            columns=list("ABC"),  # 设置列名为 A, B, C
        )

        # this is actually ambiguous as the 2 is interpreted as a positional
        # so column is not created
        df = df_orig.copy()  # 复制 df_orig 到 df
        df._set_value("C", 2, 1.0)  # 在 df 中设置特定值为 1.0
        assert list(df.index) == list(df_orig.index) + ["C"]  # 断言 df 的索引应为 df_orig 的索引加上 "C"
        # assert list(df.columns) == list(df_orig.columns) + [2]

        df = df_orig.copy()  # 再次复制 df_orig 到 df
        df.loc["C", 2] = 1.0  # 在 df 中指定位置设置值为 1.0
        assert list(df.index) == list(df_orig.index) + ["C"]  # 断言 df 的索引应为 df_orig 的索引加上 "C"
        # assert list(df.columns) == list(df_orig.columns) + [2]

        # create both new
        df = df_orig.copy()  # 再次复制 df_orig 到 df
        df._set_value("C", "D", 1.0)  # 在 df 中设置特定值为 1.0
        assert list(df.index) == list(df_orig.index) + ["C"]  # 断言 df 的索引应为 df_orig 的索引加上 "C"
        assert list(df.columns) == list(df_orig.columns) + ["D"]  # 断言 df 的列名应为 df_orig 的列名加上 "D"

        df = df_orig.copy()  # 再次复制 df_orig 到 df
        df.loc["C", "D"] = 1.0  # 在 df 中指定位置设置值为 1.0
        assert list(df.index) == list(df_orig.index) + ["C"]  # 断言 df 的索引应为 df_orig 的索引加上 "C"
        assert list(df.columns) == list(df_orig.columns) + ["D"]  # 断言 df 的列名应为 df_orig 的列名加上 "D"
```