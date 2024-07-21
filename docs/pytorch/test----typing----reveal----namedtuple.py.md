# `.\pytorch\test\typing\reveal\namedtuple.py`

```
# 导入 PyTorch 库
import torch

# 创建一个 2x2 的张量
t = torch.tensor([[3.0, 1.5], [2.0, 1.5]])

# 对张量 t 进行排序，返回排序后的结果，包括排序后的值和索引
t_sort = t.sort()

# 检查排序后的张量的第一个元素的值是否为 1.5
t_sort[0][0, 0] == 1.5  # noqa: B015

# 检查排序后的张量的索引的第一个元素的值是否为 1
t_sort.indices[0, 0] == 1  # noqa: B015

# 检查排序后的张量的值的第一个元素的值是否为 1.5
t_sort.values[0, 0] == 1.5  # noqa: B015

# 显示 t_sort 的类型提示信息，应为 torch.return_types.sort
reveal_type(t_sort)  # E: torch.return_types.sort

# 对张量 t 进行 QR 分解，返回分解后的结果，包括 Q 矩阵和 R 矩阵
t_qr = torch.linalg.qr(t)

# 检查 QR 分解结果中 R 矩阵的形状是否为 [2, 2]
t_qr[0].shape == [2, 2]  # noqa: B015

# 检查 QR 分解结果中 Q 矩阵的形状是否为 [2, 2]
t_qr.Q.shape == [2, 2]  # noqa: B015

# TODO: 修复此处的类型提示，应为 Tuple[{Tensor}, {Tensor}, fallback=torch.return_types.qr]
# 显示 t_qr 的类型提示信息，当前显示为 Any 类型
reveal_type(t_qr)  # E: Any
```