# `D:\src\scipysrc\pandas\pandas\tests\frame\common.py`

```
# 从未来导入注解支持，允许使用类型注解中的 `annotations`
from __future__ import annotations

# 引入类型检查
from typing import TYPE_CHECKING

# 从 pandas 中导入 DataFrame、concat 函数
from pandas import (
    DataFrame,
    concat,
)

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从 pandas._typing 中导入 AxisInt 类型
    from pandas._typing import AxisInt


# 检查混合浮点数类型的函数
def _check_mixed_float(df, dtype=None):
    # 预设的数据类型字典
    dtypes = {"A": "float32", "B": "float32", "C": "float16", "D": "float64"}
    # 如果传入参数 dtype 是字符串，则将所有键的类型设为该字符串
    if isinstance(dtype, str):
        dtypes = {k: dtype for k, v in dtypes.items()}
    # 如果传入参数 dtype 是字典，则更新数据类型字典
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    # 如果预设的数据类型字典中有键 "A"，则断言 DataFrame 的 "A" 列的数据类型与预设的一致
    if dtypes.get("A"):
        assert df.dtypes["A"] == dtypes["A"]
    # 如果预设的数据类型字典中有键 "B"，则断言 DataFrame 的 "B" 列的数据类型与预设的一致
    if dtypes.get("B"):
        assert df.dtypes["B"] == dtypes["B"]
    # 如果预设的数据类型字典中有键 "C"，则断言 DataFrame 的 "C" 列的数据类型与预设的一致
    if dtypes.get("C"):
        assert df.dtypes["C"] == dtypes["C"]
    # 如果预设的数据类型字典中有键 "D"，则断言 DataFrame 的 "D" 列的数据类型与预设的一致
    if dtypes.get("D"):
        assert df.dtypes["D"] == dtypes["D"]


# 检查混合整数类型的函数
def _check_mixed_int(df, dtype=None):
    # 预设的数据类型字典
    dtypes = {"A": "int32", "B": "uint64", "C": "uint8", "D": "int64"}
    # 如果传入参数 dtype 是字符串，则将所有键的类型设为该字符串
    if isinstance(dtype, str):
        dtypes = {k: dtype for k, v in dtypes.items()}
    # 如果传入参数 dtype 是字典，则更新数据类型字典
    elif isinstance(dtype, dict):
        dtypes.update(dtype)
    # 如果预设的数据类型字典中有键 "A"，则断言 DataFrame 的 "A" 列的数据类型与预设的一致
    if dtypes.get("A"):
        assert df.dtypes["A"] == dtypes["A"]
    # 如果预设的数据类型字典中有键 "B"，则断言 DataFrame 的 "B" 列的数据类型与预设的一致
    if dtypes.get("B"):
        assert df.dtypes["B"] == dtypes["B"]
    # 如果预设的数据类型字典中有键 "C"，则断言 DataFrame 的 "C" 列的数据类型与预设的一致
    if dtypes.get("C"):
        assert df.dtypes["C"] == dtypes["C"]
    # 如果预设的数据类型字典中有键 "D"，则断言 DataFrame 的 "D" 列的数据类型与预设的一致
    if dtypes.get("D"):
        assert df.dtypes["D"] == dtypes["D"]


# 将多个 DataFrame 沿指定轴合并为一个新的 DataFrame
def zip_frames(frames: list[DataFrame], axis: AxisInt = 1) -> DataFrame:
    """
    将一个 DataFrame 列表进行合并，假设这些 DataFrame 都有第一个 DataFrame 的索引/列名。

    Parameters
    ----------
    frames : list[DataFrame]
        要合并的 DataFrame 列表
    axis : AxisInt, optional
        合并的轴向，默认为 1 (列合并)

    Returns
    -------
    new_frame : DataFrame
        合并后的新 DataFrame
    """
    # 如果合并轴是 1
    if axis == 1:
        # 取第一个 DataFrame 的列名
        columns = frames[0].columns
        # 对于每个列名，在每个 DataFrame 中取出对应列的数据，并合并成一个列表
        zipped = [f.loc[:, c] for c in columns for f in frames]
        # 使用 concat 函数按列合并所有数据，并返回结果
        return concat(zipped, axis=1)
    else:
        # 取第一个 DataFrame 的索引
        index = frames[0].index
        # 对于每个索引，在每个 DataFrame 中取出对应行的数据，并合并成一个列表
        zipped = [f.loc[i, :] for i in index for f in frames]
        # 使用 DataFrame 构造函数根据合并后的数据列表创建新的 DataFrame，并返回结果
        return DataFrame(zipped)
```