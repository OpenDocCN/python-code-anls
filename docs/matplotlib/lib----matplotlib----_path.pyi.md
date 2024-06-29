# `D:\src\scipysrc\matplotlib\lib\matplotlib\_path.pyi`

```py
# 从 collections.abc 模块导入 Sequence 类
from collections.abc import Sequence

# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 从当前包中的 transforms 模块导入 BboxBase 类
from .transforms import BboxBase

# 定义一个名为 affine_transform 的函数，接受两个参数：points 和 trans，都是 NumPy 数组，返回一个 NumPy 数组
def affine_transform(points: np.ndarray, trans: np.ndarray) -> np.ndarray:
    ...

# 定义一个名为 count_bboxes_overlapping_bbox 的函数，接受两个参数：bbox 和 bboxes，bbox 是 BboxBase 类型，bboxes 是 Sequence[BboxBase] 类型，返回一个整数
def count_bboxes_overlapping_bbox(bbox: BboxBase, bboxes: Sequence[BboxBase]) -> int:
    ...

# 定义一个名为 update_path_extents 的函数，接受五个参数：path、trans、rect、minpos、ignore
# 函数本身不指定返回类型，通常用于更新路径的范围信息
def update_path_extents(path, trans, rect, minpos, ignore):
    ...
```