# `D:\src\scipysrc\sympy\sympy\tensor\array\tests\test_ndim_array_conversions.py`

```
# 导入 SymPy 库中的不同类型的多维数组类
from sympy.tensor.array import (ImmutableDenseNDimArray,
        ImmutableSparseNDimArray, MutableDenseNDimArray, MutableSparseNDimArray)
# 导入 SymPy 库中的符号变量 x, y, z
from sympy.abc import x, y, z

# 定义测试函数，用于测试多维数组的转换操作
def test_NDim_array_conv():
    # 创建可变密集多维数组 MD，其元素为符号变量 x, y, z
    MD = MutableDenseNDimArray([x, y, z])
    # 创建可变稀疏多维数组 MS，其元素为符号变量 x, y, z
    MS = MutableSparseNDimArray([x, y, z])
    # 创建不可变密集多维数组 ID，其元素为符号变量 x, y, z
    ID = ImmutableDenseNDimArray([x, y, z])
    # 创建不可变稀疏多维数组 IS，其元素为符号变量 x, y, z
    IS = ImmutableSparseNDimArray([x, y, z])

    # 断言：将可变密集多维数组 MD 转换为不可变密集多维数组后，应当与 ID 相等
    assert MD.as_immutable() == ID
    # 断言：将可变密集多维数组 MD 转换为可变形式后，应当与 MD 自身相等
    assert MD.as_mutable() == MD

    # 断言：将可变稀疏多维数组 MS 转换为不可变稀疏多维数组后，应当与 IS 相等
    assert MS.as_immutable() == IS
    # 断言：将可变稀疏多维数组 MS 转换为可变形式后，应当与 MS 自身相等
    assert MS.as_mutable() == MS

    # 断言：不可变密集多维数组 ID 转换为不可变形式后，应当与 ID 自身相等
    assert ID.as_immutable() == ID
    # 断言：不可变密集多维数组 ID 转换为可变形式后，应当与 MD 相等
    assert ID.as_mutable() == MD

    # 断言：不可变稀疏多维数组 IS 转换为不可变形式后，应当与 IS 自身相等
    assert IS.as_immutable() == IS
    # 断言：不可变稀疏多维数组 IS 转换为可变形式后，应当与 MS 相等
    assert IS.as_mutable() == MS
```