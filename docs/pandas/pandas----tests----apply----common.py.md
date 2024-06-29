# `D:\src\scipysrc\pandas\pandas\tests\apply\common.py`

```
# 从 pandas 库中导入基础包中的 transformation_kernels 函数
from pandas.core.groupby.base import transformation_kernels

# 生成一个新的列表 series_transform_kernels，包含 transformation_kernels 返回的排序后的所有元素，除去 "cumcount"
series_transform_kernels = [
    x for x in sorted(transformation_kernels) if x != "cumcount"
]

# 生成一个新的列表 frame_transform_kernels，包含 transformation_kernels 返回的排序后的所有元素，除去 "cumcount"
frame_transform_kernels = [x for x in sorted(transformation_kernels) if x != "cumcount"]
```