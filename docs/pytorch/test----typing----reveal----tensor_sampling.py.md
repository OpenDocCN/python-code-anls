# `.\pytorch\test\typing\reveal\tensor_sampling.py`

```py
# flake8: noqa
# 导入 torch 库
import torch

# 获取全局随机种子值
reveal_type(torch.seed())  # E: int

# 设置手动随机种子
reveal_type(torch.manual_seed(3))  # E: torch._C.Generator

# 获取初始化时的随机种子值
reveal_type(torch.initial_seed())  # E: int

# 获取随机数生成器状态
reveal_type(torch.get_rng_state())  # E: {Tensor}

# 生成服从伯努利分布的随机数
reveal_type(torch.bernoulli(torch.empty(3, 3).uniform_(0, 1)))  # E: {Tensor}

# 从多项分布中抽取样本
weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
reveal_type(torch.multinomial(weights, 2))  # E: {Tensor}

# 生成服从正态分布的随机数
reveal_type(torch.normal(2, 3, size=(1, 4)))  # E: {Tensor}

# 生成服从泊松分布的随机数
reveal_type(torch.poisson(torch.rand(4, 4) * 5))  # E: {Tensor}

# 生成均匀分布的随机数
reveal_type(torch.rand(4))  # E: {Tensor}
reveal_type(torch.rand(2, 3))  # E: {Tensor}

# 生成与给定张量形状相同的均匀分布的随机数
a = torch.rand(4)
reveal_type(torch.rand_like(a))  # E: {Tensor}

# 生成指定范围内的随机整数
reveal_type(torch.randint(3, 5, (3,)))  # E: {Tensor}
reveal_type(torch.randint(10, (2, 2)))  # E: {Tensor}
reveal_type(torch.randint(3, 10, (2, 2)))  # E: {Tensor}

# 生成与给定张量形状相同的指定范围内的随机整数
b = torch.randint(3, 50, (3, 4))
reveal_type(torch.randint_like(b, 3, 10))  # E: {Tensor}

# 生成服从标准正态分布的随机数
reveal_type(torch.randn(4))  # E: {Tensor}
reveal_type(torch.randn(2, 3))  # E: {Tensor}

# 生成与给定张量形状相同的服从标准正态分布的随机数
c = torch.randn(2, 3)
reveal_type(torch.randn_like(c))  # E: {Tensor}

# 生成随机排列
reveal_type(torch.randperm(4))  # E: {Tensor}

# 创建 Sobol 序列引擎实例
d = torch.quasirandom.SobolEngine(dimension=5)
reveal_type(d)  # E: torch.quasirandom.SobolEngine
# 从 Sobol 序列中抽取样本
reveal_type(d.draw())  # E: {Tensor}
```