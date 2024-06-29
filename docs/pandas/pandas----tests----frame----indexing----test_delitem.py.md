# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_delitem.py`

```
# 导入所需的库
import re
import numpy as np
import pytest
from pandas import (
    DataFrame,
    MultiIndex,
)

# 定义一个测试类 TestDataFrameDelItem
class TestDataFrameDelItem:
    # 测试删除 DataFrame 中的列
    def test_delitem(self, float_frame):
        # 删除列"A"
        del float_frame["A"]
        # 断言列"A"不在 DataFrame 中
        assert "A" not in float_frame

    # 测试删除 MultiIndex 中的列
    def test_delitem_multiindex(self):
        # 创建一个 MultiIndex
        midx = MultiIndex.from_product([["A", "B"], [1, 2]])
        # 创建一个 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), columns=midx)
        # 断言 DataFrame 的列数为4
        assert len(df.columns) == 4
        # 断言("A",)在 DataFrame 的列中
        assert ("A",) in df.columns
        # 断言"A"在 DataFrame 的列中
        assert "A" in df.columns

        # 获取列"A"的数据
        result = df["A"]
        # 断言结果是一个 DataFrame
        assert isinstance(result, DataFrame)
        # 删除列"A"
        del df["A"]

        # 断言 DataFrame 的列数为2
        assert len(df.columns) == 2

        # 断言("A",)不在 DataFrame 的列中，尝试删除会引发 KeyError
        assert ("A",) not in df.columns
        with pytest.raises(KeyError, match=re.escape("('A',)")):
            del df[("A",)]

        # 从 GH 2770 到 GH 19027，删除/删除的 MultiIndex 级别的行为发生了变化
        # MultiIndex 不再包含被删除的级别
        assert "A" not in df.columns
        with pytest.raises(KeyError, match=re.escape("('A',)")):
            del df["A"]

    # 测试删除 DataFrame 中的列
    def test_delitem_corner(self, float_frame):
        # 复制 DataFrame
        f = float_frame.copy()
        # 删除列"D"
        del f["D"]
        # 断言 DataFrame 的列数为3
        assert len(f.columns) == 3
        # 尝试再次删除列"D"会引发 KeyError
        with pytest.raises(KeyError, match=r"^'D'$"):
            del f["D"]
        # 删除列"B"
        del f["B"]
        # 断言 DataFrame 的列数为2
        assert len(f.columns) == 2

    # 测试删除 MultiIndex 中的列
    def test_delitem_col_still_multiindex(self):
        # 创建一个 MultiIndex
        arrays = [["a", "b", "c", "top"], ["", "", "", "OD"], ["", "", "", "wx"]]
        tuples = sorted(zip(*arrays))
        index = MultiIndex.from_tuples(tuples)

        # 创建一个 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), columns=index)
        # 删除列("a", "", "")
        del df[("a", "", "")]
        # 断言 DataFrame 的列仍然是一个 MultiIndex
        assert isinstance(df.columns, MultiIndex)
```