# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_copy.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类

class TestCopy:
    @pytest.mark.parametrize("attr", ["index", "columns"])
    def test_copy_index_name_checking(self, float_frame, attr):
        # 不希望在复制后能够修改存储在其他位置的索引
        ind = getattr(float_frame, attr)  # 获取 float_frame 对象的索引或列属性
        ind.name = None  # 将索引或列属性的名称设为 None
        cp = float_frame.copy()  # 复制 float_frame 对象
        getattr(cp, attr).name = "foo"  # 设置复制对象的索引或列属性名称为 "foo"
        assert getattr(float_frame, attr).name is None  # 断言原始对象的索引或列属性名称仍为 None

    def test_copy(self, float_frame, float_string_frame):
        cop = float_frame.copy()  # 复制 float_frame 对象
        cop["E"] = cop["A"]  # 在复制对象中添加新列 "E"，值与 "A" 列相同
        assert "E" not in float_frame  # 断言 "E" 列不在原始 float_frame 对象中

        # 复制对象
        copy = float_string_frame.copy()  # 复制 float_string_frame 对象
        assert copy._mgr is not float_string_frame._mgr  # 断言复制对象的内部管理器不同于原始对象的内部管理器

    def test_copy_consolidates(self):
        # GH#42477
        # 创建一个 DataFrame 对象 df，包含两列 'a' 和 'b'，并添加 10 列随机数列
        df = DataFrame(
            {
                "a": np.random.default_rng(2).integers(0, 100, size=55),
                "b": np.random.default_rng(2).integers(0, 100, size=55),
            }
        )

        for i in range(10):
            # 添加命名为 'n_i' 的随机数列
            df.loc[:, f"n_{i}"] = np.random.default_rng(2).integers(0, 100, size=55)

        assert len(df._mgr.blocks) == 11  # 断言 df 对象的内部管理器包含 11 个块
        result = df.copy()  # 复制 df 对象
        assert len(result._mgr.blocks) == 1  # 断言复制对象的内部管理器只包含 1 个块
```