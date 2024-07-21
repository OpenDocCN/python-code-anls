# `.\pytorch\torch\ao\pruning\_experimental\pruner\parametrization.py`

```py
# my`
# 允许未类型定义的函数返回值
# mypy: allow-untyped-defs
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 的神经网络模块导入 nn
from torch.nn.utils.parametrize import is_parametrized  # 从 torch.nn.utils.parametrize 导入 is_parametrized 函数

# 定义函数，检查模块是否包含指定的参数化
def module_contains_param(module, parametrization):
    # 如果模块是参数化的
    if is_parametrized(module):
        # 检查模块的参数列表中是否有任何参数的类型与传入的参数化类型匹配
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for key, param_list in module.parametrizations.items()
        )
    return False  # 如果模块不是参数化的，返回 False

# 定义一个用于结构化剪枝参数化的类
class FakeStructuredSparsity(nn.Module):
    r"""
    用于结构化剪枝的参数化。类似于 FakeSparsity，这个参数化应该附加到
    'weight' 或其他需要掩码的参数上。

    这个参数化使用的是按行的 bool 掩码，而不是逐元素的 bool 掩码。
    """

    # 初始化方法，接收一个掩码作为参数
    def __init__(self, mask):
        super().__init__()  # 调用父类的初始化方法
        self.register_buffer("mask", mask)  # 注册一个缓冲区，保存掩码

    # 前向传播方法，执行掩码与输入的乘法
    def forward(self, x):
        assert isinstance(self.mask, torch.Tensor)  # 确保掩码是一个张量
        assert self.mask.shape[0] == x.shape[0]  # 确保掩码的行数与输入的行数相同
        shape = [1] * len(x.shape)  # 创建一个与输入张量维度相同的形状列表，初始为全 1
        shape[0] = -1  # 将第一个维度设置为 -1
        return self.mask.reshape(shape) * x  # 返回掩码与输入张量相乘的结果

    # 定义状态字典方法，避免保存掩码
    def state_dict(self, *args, **kwargs):
        return {}  # 返回空字典，避免保存掩码

# 定义 BiasHook 类，用于处理偏置剪枝
class BiasHook:
    # 初始化方法，接收参数化和是否剪枝偏置的标志
    def __init__(self, parametrization, prune_bias):
        self.param = parametrization  # 保存参数化对象
        self.prune_bias = prune_bias  # 保存是否剪枝偏置的标志

    # 调用方法，在前向传播过程中应用偏置剪枝
    def __call__(self, module, input, output):
        # 如果模块具有偏置属性
        if getattr(module, "_bias", None) is not None:
            bias = module._bias.data  # 获取偏置数据
            if self.prune_bias:
                bias[~self.param.mask] = 0  # 如果需要剪枝，设置未被掩码的偏置值为 0

            # 将偏置重新形状，以便在输出维度上广播
            idx = [1] * len(output.shape)  # 创建一个与输出张量维度相同的形状列表，初始为全 1
            idx[1] = -1  # 将第二个维度设置为 -1
            bias = bias.reshape(idx)  # 重新调整偏置的形状

            output += bias  # 将偏置加到输出上
        return output  # 返回处理后的输出
```