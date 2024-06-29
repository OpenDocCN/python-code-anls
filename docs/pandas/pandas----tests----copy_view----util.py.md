# `D:\src\scipysrc\pandas\pandas\tests\copy_view\util.py`

```
# 从 pandas 库中导入特定的类和函数：Categorical（分类数据）、Index（索引）、Series（序列）
from pandas import (
    Categorical,
    Index,
    Series,
)
# 从 pandas 的核心数组模块导入 BaseMaskedArray 类
from pandas.core.arrays import BaseMaskedArray


# 定义一个函数 get_array，用于获取 DataFrame 列或 Series 的数组表示
def get_array(obj, col=None):
    """
    Helper method to get array for a DataFrame column or a Series.

    Equivalent of df[col].values, but without going through normal getitem,
    which triggers tracking references / CoW (and we might be testing that
    this is done by some other operation).
    """
    # 如果 obj 是 Index 类型
    if isinstance(obj, Index):
        # 直接获取 Index 对象的值数组
        arr = obj._values
    # 如果 obj 是 Series 类型，并且未指定列名 col 或者 obj 的列名与 col 相符
    elif isinstance(obj, Series) and (col is None or obj.name == col):
        # 获取 Series 对象的值数组
        arr = obj._values
    else:
        # 断言确保 col 不为空
        assert col is not None
        # 获取列 col 在 DataFrame 中的位置索引
        icol = obj.columns.get_loc(col)
        # 断言确保 icol 是整数类型
        assert isinstance(icol, int)
        # 通过列索引 icol 获取 DataFrame 中指定列的数组表示
        arr = obj._get_column_array(icol)
    
    # 如果 arr 是 BaseMaskedArray 类型的对象，则返回其数据
    if isinstance(arr, BaseMaskedArray):
        return arr._data
    # 如果 arr 是 Categorical 类型的对象，则直接返回 arr
    elif isinstance(arr, Categorical):
        return arr
    # 否则，返回 arr 对象的 _ndarray 属性，如果不存在则返回 arr 本身
    return getattr(arr, "_ndarray", arr)
```