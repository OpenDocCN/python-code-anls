# `.\pytorch\test\typing\fail\creation_ops.py`

```py
# flake8: noqa
# 导入 torch 库
import torch

# 创建包含单个元素 3 的整数张量，指定数据类型为 int32
torch.tensor(
    [3],
    dtype="int32",  # E: Argument "dtype" to "tensor" has incompatible type "str"; expected "dtype | None"  [arg-type]
)

# 创建包含 3 个元素的全为 1 的张量，指定数据类型为 int32
torch.ones(  # E: No overload variant of "ones" matches argument types "int", "str"
    3, dtype="int32"
)

# 创建包含 3 个元素的全为 0 的张量，指定数据类型为 int32
torch.zeros(  # E: No overload variant of "zeros" matches argument types "int", "str"
    3, dtype="int32"
)
```