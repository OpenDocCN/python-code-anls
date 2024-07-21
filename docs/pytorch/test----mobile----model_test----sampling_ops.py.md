# `.\pytorch\test\mobile\model_test\sampling_ops.py`

```
import torch  # 导入 PyTorch 库


# 定义一个自定义的 PyTorch 模块，用于示例演示
class SamplingOpsModule(torch.nn.Module):
    def forward(self):
        # 创建一个 3x3 的空张量，并在 [0, 1) 的均匀分布中填充随机数
        a = torch.empty(3, 3).uniform_(0.0, 1.0)
        size = (1, 4)  # 定义一个元组 size = (1, 4)
        weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)  # 创建一个张量 weights
        return len(  # 返回以下操作的数量
            # torch.bernoulli 生成服从伯努利分布的随机数张量
            torch.bernoulli(a),
            # torch.multinomial 根据给定的权重进行多项式抽样
            torch.multinomial(weights, 2),
            # torch.normal 生成服从正态分布的随机数张量
            torch.normal(2.0, 3.0, size),
            # torch.poisson 生成服从泊松分布的随机数张量
            torch.poisson(a),
            # torch.rand 生成服从 [0, 1) 均匀分布的随机数张量
            torch.rand(2, 3),
            # torch.rand_like 生成与输入张量 a 具有相同形状的随机数张量
            torch.rand_like(a),
            # torch.randint 生成指定范围内的整数张量
            torch.randint(10, size),
            # torch.randint_like 生成与输入张量 a 形状相同的整数张量
            torch.randint_like(a, 4),
            # torch.rand 生成指定形状的随机数张量
            torch.rand(4),
            # torch.randn_like 生成与输入张量 a 形状相同的标准正态分布的随机数张量
            torch.randn_like(a),
            # torch.randperm 生成给定范围内的随机排列张量
            torch.randperm(4),
            # a.bernoulli_ 以张量 a 的元素为概率进行伯努利抽样，替换原张量的值
            a.bernoulli_(),
            # a.cauchy_ 以张量 a 的元素为参数进行柯西分布抽样，替换原张量的值
            a.cauchy_(),
            # a.exponential_ 以张量 a 的元素为参数进行指数分布抽样，替换原张量的值
            a.exponential_(),
            # a.geometric_ 以张量 a 的元素为概率进行几何分布抽样，替换原张量的值
            a.geometric_(0.5),
            # a.log_normal_ 以张量 a 的元素为参数进行对数正态分布抽样，替换原张量的值
            a.log_normal_(),
            # a.normal_ 以张量 a 的元素为参数进行正态分布抽样，替换原张量的值
            a.normal_(),
            # a.random_ 以张量 a 的元素为参数进行均匀分布抽样，替换原张量的值
            a.random_(),
            # a.uniform_ 以张量 a 的元素为参数进行均匀分布抽样，替换原张量的值
            a.uniform_(),
        )
```