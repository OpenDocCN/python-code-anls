# `.\pytorch\test\typing\reveal\tensor_constructors.py`

```py
# 引入 torch 库，用于张量操作
# mypy: disable-error-code="possibly-undefined"
# flake8: noqa
import torch
# 从 torch.testing._internal.common_utils 中导入 TEST_NUMPY 变量
from torch.testing._internal.common_utils import TEST_NUMPY

# 如果 TEST_NUMPY 变量为真，则导入 numpy 库
if TEST_NUMPY:
    import numpy as np

# 从 PyTorch 文档可知，有多种方法可以创建张量：
# https://pytorch.org/docs/stable/tensors.html

# torch.tensor() 方法用于从数据创建张量
reveal_type(torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]))  # E: {Tensor}
reveal_type(torch.tensor([0, 1]))  # E: {Tensor}
reveal_type(
    torch.tensor(
        [[0.11111, 0.222222, 0.3333333]],
        dtype=torch.float64,
        device=torch.device("cuda:0"),
    )
)  # E: {Tensor}
reveal_type(torch.tensor(3.14159))  # E: {Tensor}

# torch.sparse_coo_tensor 方法用于创建稀疏的 COO 格式张量
i = torch.tensor([[0, 1, 1], [2, 0, 2]])  # E: {Tensor}
v = torch.tensor([3, 4, 5], dtype=torch.float32)  # E: {Tensor}
reveal_type(torch.sparse_coo_tensor(i, v, [2, 4]))  # E: {Tensor}
reveal_type(torch.sparse_coo_tensor(i, v))  # E: {Tensor}
reveal_type(
    torch.sparse_coo_tensor(
        i, v, [2, 4], dtype=torch.float64, device=torch.device("cuda:0")
    )
)  # E: {Tensor}
reveal_type(torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1]))  # E: {Tensor}
reveal_type(
    torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])
)  # E: {Tensor}

# torch.as_tensor 方法用于从数组创建张量
if TEST_NUMPY:
    a = np.array([1, 2, 3])
    reveal_type(torch.as_tensor(a))  # E: {Tensor}
    reveal_type(torch.as_tensor(a, device=torch.device("cuda")))  # E: {Tensor}

# torch.as_strided 方法用于创建具有指定大小和步幅的张量视图
x = torch.randn(3, 3)
reveal_type(torch.as_strided(x, (2, 2), (1, 2)))  # E: {Tensor}
reveal_type(torch.as_strided(x, (2, 2), (1, 2), 1))  # E: {Tensor}

# torch.from_numpy 方法用于从 numpy 数组创建张量
if TEST_NUMPY:
    a = np.array([1, 2, 3])
    reveal_type(torch.from_numpy(a))  # E: {Tensor}

# torch.zeros/zeros_like 方法用于创建全零张量
reveal_type(torch.zeros(2, 3))  # E: {Tensor}
reveal_type(torch.zeros(5))  # E: {Tensor}
reveal_type(torch.zeros_like(torch.empty(2, 3)))  # E: {Tensor}

# torch.ones/ones_like 方法用于创建全一张量
reveal_type(torch.ones(2, 3))  # E: {Tensor}
reveal_type(torch.ones(5))  # E: {Tensor}
reveal_type(torch.ones_like(torch.empty(2, 3)))  # E: {Tensor}

# torch.arange 方法用于创建等差数列张量
reveal_type(torch.arange(5))  # E: {Tensor}
reveal_type(torch.arange(1, 4))  # E: {Tensor}
reveal_type(torch.arange(1, 2.5, 0.5))  # E: {Tensor}

# torch.range 方法（已弃用）替代为 torch.arange

# torch.linspace 方法用于创建均匀间隔的数值张量
reveal_type(torch.linspace(3, 10, steps=5))  # E: {Tensor}
reveal_type(torch.linspace(-10, 10, steps=5))  # E: {Tensor}
reveal_type(torch.linspace(start=-10, end=10, steps=5))  # E: {Tensor}
reveal_type(torch.linspace(start=-10, end=10, steps=1))  # E: {Tensor}

# torch.logspace 方法用于创建以对数刻度均匀分布的张量
reveal_type(torch.logspace(start=-10, end=10, steps=5))  # E: {Tensor}
reveal_type(torch.logspace(start=0.1, end=1.0, steps=5))  # E: {Tensor}
reveal_type(torch.logspace(start=0.1, end=1.0, steps=1))  # E: {Tensor}
reveal_type(torch.logspace(start=2, end=2, steps=1, base=2))  # E: {Tensor}

# torch.eye 方法用于创建单位矩阵张量
reveal_type(torch.eye(3))  # E: {Tensor}
# 使用 torch.empty 创建一个形状为 (2, 3) 的空张量
reveal_type(torch.empty(2, 3))  # E: {Tensor}
# 使用 torch.empty_like 创建一个与给定张量相同形状的空张量，并指定数据类型为 torch.int64
reveal_type(torch.empty_like(torch.empty(2, 3), dtype=torch.int64))  # E: {Tensor}
# 使用 torch.empty_strided 创建一个指定形状和步幅的空张量
reveal_type(torch.empty_strided((2, 3), (1, 2)))  # E: {Tensor}

# 使用 torch.full 创建一个形状为 (2, 3) 的张量，填充值为 3.141592
reveal_type(torch.full((2, 3), 3.141592))  # E: {Tensor}
# 使用 torch.full_like 创建一个与给定张量相同形状的张量，并填充值为 2.71828
reveal_type(torch.full_like(torch.full((2, 3), 3.141592), 2.71828))  # E: {Tensor}

# 使用 torch.quantize_per_tensor 对给定张量进行量化，指定量化参数和数据类型
reveal_type(
    torch.quantize_per_tensor(
        torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8
    )
)  # E: {Tensor}

# 使用 torch.quantize_per_channel 对给定张量进行通道间量化，指定量化参数和数据类型
x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
quant = torch.quantize_per_channel(
    x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8
)
# 显示张量 x 的类型
reveal_type(x)  # E: {Tensor}

# 使用 torch.dequantize 对给定量化张量进行反量化
reveal_type(torch.dequantize(x))  # E: {Tensor}

# 使用 torch.complex 创建一个复数张量，实部和虚部分别为 real 和 imag
real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
reveal_type(torch.complex(real, imag))  # E: {Tensor}

# 使用 torch.polar 根据绝对值和角度创建极坐标形式的复数张量
abs = torch.tensor([1, 2], dtype=torch.float64)
pi = torch.acos(torch.zeros(1)).item() * 2
angle = torch.tensor([pi / 2, 5 * pi / 4], dtype=torch.float64)
reveal_type(torch.polar(abs, angle))  # E: {Tensor}

# 使用 torch.heaviside 对输入张量进行海维赛德阶跃函数计算
inp = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
reveal_type(torch.heaviside(inp, values))  # E: {Tensor}
```