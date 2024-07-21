# `.\pytorch\test\ao\sparsity\test_parametrization.py`

```py
# Owner(s): ["module: unknown"]

# 导入所需的模块和库
import logging  # 导入日志模块
import torch  # 导入PyTorch库

from torch import nn  # 导入神经网络模块
from torch.ao.pruning.sparsifier import utils  # 导入稀疏化工具
from torch.nn.utils import parametrize  # 导入参数化工具
from torch.testing._internal.common_utils import TestCase  # 导入测试用例类

# 配置日志格式和级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class ModelUnderTest(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        
        # 定义线性层和序列容器
        self.linear = nn.Linear(16, 16, bias=bias)
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=bias),  # 第一个线性层
            nn.Linear(16, 16, bias=bias)   # 第二个线性层
        )

        # 确保权重不是随机的
        self.linear.weight = nn.Parameter(torch.zeros_like(self.linear.weight) + 1.0)
        self.seq[0].weight = nn.Parameter(torch.zeros_like(self.seq[0].weight) + 2.0)
        self.seq[1].weight = nn.Parameter(torch.zeros_like(self.seq[1].weight) + 3.0)
        
        # 如果有偏置，设置偏置参数
        if bias:
            self.linear.bias = nn.Parameter(torch.zeros_like(self.linear.bias) + 10.0)
            self.seq[0].bias = nn.Parameter(torch.zeros_like(self.seq[0].bias) + 20.0)
            self.seq[1].bias = nn.Parameter(torch.zeros_like(self.seq[1].bias) + 30.0)

    def forward(self, x):
        # 前向传播函数
        x = self.linear(x)
        x = self.seq(x)
        return x


class TestFakeSparsity(TestCase):
    def test_masking_logic(self):
        # 测试掩码逻辑
        model = nn.Linear(16, 16, bias=False)
        model.weight = nn.Parameter(torch.eye(16))
        x = torch.randn(3, 16)
        self.assertEqual(torch.mm(x, torch.eye(16)), model(x))

        # 创建全零掩码和FakeSparsity对象
        mask = torch.zeros(16, 16)
        sparsity = utils.FakeSparsity(mask)
        
        # 将参数化策略注册到模型的权重上
        parametrize.register_parametrization(model, "weight", sparsity)

        x = torch.randn(3, 16)
        self.assertEqual(torch.zeros(3, 16), model(x))

    def test_weights_parametrized(self):
        # 测试权重参数化
        model = ModelUnderTest(bias=False)

        # 断言模型的线性层和序列层没有被参数化
        assert not hasattr(model.linear, "parametrizations")
        assert not hasattr(model.seq[0], "parametrizations")
        assert not hasattr(model.seq[1], "parametrizations")
        
        # 创建全零掩码和FakeSparsity对象，并注册到各个层的权重上
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[1], "weight", utils.FakeSparsity(mask)
        )

        # 断言模型的线性层和序列层已经被正确参数化
        assert hasattr(model.linear, "parametrizations")
        assert parametrize.is_parametrized(model.linear, "weight")
        assert hasattr(model.seq[0], "parametrizations")
        assert parametrize.is_parametrized(model.linear, "weight")
        assert hasattr(model.seq[1], "parametrizations")
        assert parametrize.is_parametrized(model.linear, "weight")
    # 定义一个测试方法 `test_jit_trace`，用于测试 JIT 追踪功能

    # 创建一个 `ModelUnderTest` 的实例 `model`，并设置 `bias=False`
    model = ModelUnderTest(bias=False)

    # 创建一个 16x16 的单位矩阵 `mask`
    mask = torch.eye(16)
    # 使用 `parametrize` 函数注册 `model.linear` 的权重参数，并使用 `FakeSparsity` 对象
    parametrize.register_parametrization(
        model.linear, "weight", utils.FakeSparsity(mask)
    )

    # 重新创建一个 16x16 的单位矩阵 `mask`
    mask = torch.eye(16)
    # 使用 `parametrize` 函数注册 `model.seq[0]` 的权重参数，并使用 `FakeSparsity` 对象
    parametrize.register_parametrization(
        model.seq[0], "weight", utils.FakeSparsity(mask)
    )

    # 再次重新创建一个 16x16 的单位矩阵 `mask`
    mask = torch.eye(16)
    # 使用 `parametrize` 函数注册 `model.seq[1]` 的权重参数，并使用 `FakeSparsity` 对象
    parametrize.register_parametrization(
        model.seq[1], "weight", utils.FakeSparsity(mask)
    )

    # JIT 追踪过程开始

    # 创建一个大小为 (3, 16) 的全为1的张量 `example_x`
    example_x = torch.ones(3, 16)
    # 使用 `torch.jit.trace_module` 方法对 `model` 进行追踪，输入 `{"forward": example_x}`
    model_trace = torch.jit.trace_module(model, {"forward": example_x})

    # 创建一个大小为 (3, 16) 的随机张量 `x`
    x = torch.randn(3, 16)
    # 使用 `model` 对 `x` 进行前向传播，得到预测结果 `y`
    y = model(x)
    # 使用 JIT 追踪后的 `model_trace` 对 `x` 进行前向传播，得到预测结果 `y_hat`
    y_hat = model_trace(x)

    # 断言 `y_hat` 和 `y` 相等
    self.assertEqual(y_hat, y)
```