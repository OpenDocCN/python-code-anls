# `.\pytorch\test\typing\reveal\tensor_copy.py`

```
# 导入 PyTorch 库
import torch

# 创建一个形状为 (2, 3) 的随机张量 t
t = torch.randn(2, 3)
# 使用类型推断显示 t 的类型为 Tensor
reveal_type(t)  # E: {Tensor}

# 创建另一个形状为 (2, 3) 的随机张量 u
u = torch.randn(2, 3)
# 使用类型推断显示 u 的类型为 Tensor
reveal_type(u)  # E: {Tensor}

# 将张量 u 的值复制到张量 t 中（in-place 操作）
t.copy_(u)
# 使用类型推断显示 t 的类型仍为 Tensor，因为 copy_ 操作不改变张量类型
reveal_type(t)  # E: {Tensor}

# 检查张量 t 和 u 的所有元素是否相等，并返回一个布尔张量 r
r = (t == u).all()
# 使用类型推断显示 r 的类型为 Tensor，因为 all() 返回一个布尔张量
reveal_type(r)  # E: {Tensor}
```