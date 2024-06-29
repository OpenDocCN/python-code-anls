# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_to_dict_of_blocks.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 pandas 库中导入 DataFrame、MultiIndex 类
from pandas import (
    DataFrame,
    MultiIndex,
)

# 从 pandas 库中导入 _testing 模块作为 tm
import pandas._testing as tm

# 从 pandas.core.arrays 中导入 NumpyExtensionArray 类
from pandas.core.arrays import NumpyExtensionArray


class TestToDictOfBlocks:
    def test_no_copy_blocks(self, float_frame):
        # GH#9607
        # 使用 float_frame 创建 DataFrame，并进行复制
        df = DataFrame(float_frame, copy=True)
        
        # 获取 DataFrame 的第一个列名
        column = df.columns[0]

        _last_df = None
        
        # 使用 _to_dict_of_blocks 方法获取 DataFrame 的数据块字典
        blocks = df._to_dict_of_blocks()
        
        # 遍历数据块字典中的每个数据块
        for _df in blocks.values():
            _last_df = _df
            
            # 如果当前数据块中包含指定的列名
            if column in _df:
                # 修改该列的值（加 1）
                _df.loc[:, column] = _df[column] + 1
        
        # 断言最后一个数据块不为空且修改后的列与原始 DataFrame 列不相等
        assert _last_df is not None and not _last_df[column].equals(df[column])


def test_to_dict_of_blocks_item_cache():
    # 调用 to_dict_of_blocks 方法不应影响 item_cache
    df = DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    
    # 添加一个新的列，其中包含 NumpyExtensionArray 对象
    df["c"] = NumpyExtensionArray(np.array([1, 2, None, 3], dtype=object))
    
    # 获取 DataFrame 的内部管理器
    mgr = df._mgr
    
    # 断言数据块的数量为 3，即未合并
    assert len(mgr.blocks) == 3  # i.e. not consolidated
    
    # 获取 DataFrame 中的一列，从而填充 item_cache["b"]
    ser = df["b"]  # populates item_cache["b"]
    
    # 调用 _to_dict_of_blocks 方法
    df._to_dict_of_blocks()
    
    # 使用 pytest 断言，捕获 ValueError 异常，匹配字符串 "read-only"
    with pytest.raises(ValueError, match="read-only"):
        # 尝试修改 ser 中的第一个值
        ser.values[0] = "foo"


def test_set_change_dtype_slice():
    # GH#8850
    # 创建包含 MultiIndex 的列索引
    cols = MultiIndex.from_tuples([("1st", "a"), ("2nd", "b"), ("3rd", "c")])
    
    # 创建 DataFrame，并指定列索引
    df = DataFrame([[1.0, 2, 3], [4.0, 5, 6]], columns=cols)
    
    # 修改名为 "2nd" 的列，使其每个值乘以 2.0
    df["2nd"] = df["2nd"] * 2.0
    
    # 使用 _to_dict_of_blocks 方法获取 DataFrame 的数据块字典
    blocks = df._to_dict_of_blocks()
    
    # 断言数据块字典的键按字母顺序排列
    assert sorted(blocks.keys()) == ["float64", "int64"]
    
    # 使用 pandas._testing 模块的 assert_frame_equal 方法断言两个 DataFrame 相等
    tm.assert_frame_equal(
        blocks["float64"], DataFrame([[1.0, 4.0], [4.0, 10.0]], columns=cols[:2])
    )
    tm.assert_frame_equal(blocks["int64"], DataFrame([[3], [6]], columns=cols[2:]))
```