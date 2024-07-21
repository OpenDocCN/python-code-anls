# `.\pytorch\test\typing\pass\creation_ops.py`

```
# mypy: disable-error-code="possibly-undefined"
# flake8: noqa
# 引入类型断言模块，用于类型检查
from typing_extensions import assert_type

# 引入 PyTorch 库
import torch
# 引入测试用的 NumPy 模块
from torch.testing._internal.common_utils import TEST_NUMPY

# 如果 TEST_NUMPY 为真，则引入 NumPy
if TEST_NUMPY:
    import numpy as np

# 从文档中得知，有多种方式可以创建张量：
# https://pytorch.org/docs/stable/tensors.html

# torch.tensor()
# 创建浮点型张量
torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
# 创建整型张量
torch.tensor([0, 1])
# 在指定设备上创建双精度浮点型张量
torch.tensor(
    [[0.11111, 0.222222, 0.3333333]], dtype=torch.float64, device=torch.device("cuda:0")
)
# 创建标量张量
torch.tensor(3.14159)

# torch.sparse_coo_tensor
# 创建稀疏 COO 格式张量
i = torch.tensor([[0, 1, 1], [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
torch.sparse_coo_tensor(i, v, [2, 4])
torch.sparse_coo_tensor(i, v)
torch.sparse_coo_tensor(
    i, v, [2, 4], dtype=torch.float64, device=torch.device("cuda:0")
)
torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])

# torch.as_tensor
# 将 Python 列表转换为张量
a = [1, 2, 3]
torch.as_tensor(a)
# 将 Python 列表转换为指定设备上的张量
torch.as_tensor(a, device=torch.device("cuda"))

# torch.as_strided
# 创建步长不同的张量视图
x = torch.randn(3, 3)
torch.as_strided(x, (2, 2), (1, 2))
torch.as_strided(x, (2, 2), (1, 2), 1)

# torch.from_numpy
# 如果 TEST_NUMPY 为真，则从 NumPy 数组创建张量
if TEST_NUMPY:
    torch.from_numpy(np.array([1, 2, 3]))

# torch.zeros/zeros_like
# 创建全零张量
torch.zeros(2, 3)
torch.zeros((2, 3))
torch.zeros([2, 3])
torch.zeros(5)
# 创建与给定张量形状和数据类型相同的全零张量
torch.zeros_like(torch.empty(2, 3))

# torch.ones/ones_like
# 创建全一张量
torch.ones(2, 3)
torch.ones((2, 3))
torch.ones([2, 3])
torch.ones(5)
# 创建与给定张量形状和数据类型相同的全一张量
torch.ones_like(torch.empty(2, 3))

# torch.arange
# 创建等差数列张量
torch.arange(5)
torch.arange(1, 4)
torch.arange(1, 2.5, 0.5)

# torch.range
# 不推荐使用，已废弃

# torch.linspace
# 创建均匀间隔的数列张量
torch.linspace(3, 10, steps=5)
torch.linspace(-10, 10, steps=5)
torch.linspace(start=-10, end=10, steps=5)
torch.linspace(start=-10, end=10, steps=1)

# torch.logspace
# 创建对数间隔的数列张量
torch.logspace(start=-10, end=10, steps=5)
torch.logspace(start=0.1, end=1.0, steps=5)
torch.logspace(start=0.1, end=1.0, steps=1)
torch.logspace(start=2, end=2, steps=1, base=2)

# torch.eye
# 创建单位矩阵张量
torch.eye(3)

# torch.empty/empty_like/empty_strided
# 创建未初始化的张量
torch.empty(2, 3)
torch.empty((2, 3))
torch.empty([2, 3])
# 创建与给定张量形状和数据类型相同的未初始化张量
torch.empty_like(torch.empty(2, 3), dtype=torch.int64)
torch.empty_strided((2, 3), (1, 2))

# torch.full/full_like
# 创建填充指定值的张量
torch.full((2, 3), 3.141592)
# 创建与给定张量形状和填充值相同的张量
torch.full_like(torch.full((2, 3), 3.141592), 2.71828)

# torch.quantize_per_tensor
# 对张量进行按张量量化
torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8)

# torch.quantize_per_channel
# 对张量进行按通道量化
x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
quant = torch.quantize_per_channel(
    x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8
)

# torch.dequantize
# 反量化量化后的张量
torch.dequantize(x)

# torch.complex
# 创建复数张量
real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
torch.complex(real, imag)

# torch.polar
# 根据极坐标创建张量
abs = torch.tensor([1, 2], dtype=torch.float64)
pi = torch.acos(torch.zeros(1)).item() * 2
angle = torch.tensor([pi / 2, 5 * pi / 4], dtype=torch.float64)
# 使用 torch 模块的 polar 函数计算极坐标中的复数表示，返回复数的绝对值和角度
torch.polar(abs, angle)

# 使用 torch 模块的 heaviside 函数，根据输入张量 inp 中的值和阈值 values 的比较结果，
# 返回每个元素的阶跃函数结果（0 或 1）
inp = torch.tensor([-1.5, 0, 2.0])
values = torch.tensor([0.5])
torch.heaviside(inp, values)

# 创建一个 torch 的可训练参数，其值为空（未初始化），通常用于神经网络的权重或偏置参数
p = torch.nn.Parameter(torch.empty(1))
# 断言 p 的类型为 torch.nn.Parameter 类型，用于验证参数类型是否正确
assert_type(p, torch.nn.Parameter)
```