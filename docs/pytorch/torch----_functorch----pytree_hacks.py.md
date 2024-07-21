# `.\pytorch\torch\_functorch\pytree_hacks.py`

```py
# 导入警告模块，用于处理警告信息
import warnings

# 导入 torch.utils._pytree 中的 tree_map_ 和 treespec_pprint 函数
# TODO: 迁移完 pytree 实用程序后删除此文件
from torch.utils._pytree import tree_map_, treespec_pprint

# 定义模块的公开接口列表
__all__ = ["tree_map_", "treespec_pprint"]

# 使用警告模块捕获警告信息
with warnings.catch_warnings():
    # 设置警告过滤器，确保捕获所有警告
    warnings.simplefilter("always")
    # 发出警告，提醒用户 `torch._functorch.pytree_hacks` 已过时，并将在将来的版本中移除
    warnings.warn(
        "`torch._functorch.pytree_hacks` is deprecated and will be removed in a future release. "
        "Please `use torch.utils._pytree` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```