# `.\pytorch\torch\ao\pruning\_experimental\pruner\base_structured_sparsifier.py`

```py
# mypy: allow-untyped-defs
# 引入 itertools 库中的 chain 函数，用于迭代集合中的元素
from itertools import chain
# 从 operator 库中引入 getitem 函数，用于获取集合中的元素
from operator import getitem
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import nn  # 导入 PyTorch 中的神经网络模块
from torch.fx import symbolic_trace  # 导入 PyTorch 中的函数用于符号跟踪
from torch.nn.utils import parametrize  # 导入 PyTorch 中的神经网络参数化工具
from typing import Type, Set, Dict, Callable, Tuple, Optional, Union  # 导入类型注解

from torch.ao.pruning import BaseSparsifier  # 从 PyTorch 中的剪枝模块导入 BaseSparsifier 类
from .parametrization import FakeStructuredSparsity, BiasHook, module_contains_param  # 导入本地的参数化相关模块
from .match_utils import apply_match, MatchAllNode  # 导入本地的匹配工具和匹配节点类
from .prune_functions import (
    prune_linear,  # 从本地的剪枝函数模块导入线性层剪枝函数
    prune_linear_linear,  # 导入线性层到线性层剪枝函数
    prune_linear_activation_linear,  # 导入线性层-激活函数-线性层剪枝函数
    prune_conv2d,  # 导入卷积层剪枝函数
    prune_conv2d_conv2d,  # 导入卷积层到卷积层剪枝函数
    prune_conv2d_activation_conv2d,  # 导入卷积层-激活函数-卷积层剪枝函数
    prune_conv2d_activation_pool_conv2d,  # 导入卷积层-激活函数-池化层-卷积层剪枝函数
    prune_conv2d_pool_activation_conv2d,  # 导入卷积层-池化层-激活函数-卷积层剪枝函数
    prune_conv2d_pool_flatten_linear,  # 导入卷积层-池化层-展开-线性层剪枝函数
    prune_lstm_output_linear,  # 导入 LSTM 输出-线性层剪枝函数
    prune_lstm_output_layernorm_linear,  # 导入 LSTM 输出-层归一化-线性层剪枝函数
)


def _get_supported_structured_pruning_modules():
    """
    返回支持的结构化剪枝模块集合，包括 nn.Linear, nn.Conv2d, nn.LSTM。
    """
    SUPPORTED_STRUCTURED_PRUNING_MODULES = {  # 如果未指定，则添加到配置中
        nn.Linear,
        nn.Conv2d,
        nn.LSTM,
    }
    return SUPPORTED_STRUCTURED_PRUNING_MODULES


def _get_supported_activation_functions():
    """
    返回支持的激活函数集合，包括 F.relu, F.rrelu, 等等。
    """
    SUPPORTED_ACTIVATION_FUNCTIONS = {
        F.relu,
        F.rrelu,
        F.hardtanh,
        F.relu6,
        F.sigmoid,
        F.hardsigmoid,
        F.tanh,
        F.silu,
        F.mish,
        F.hardswish,
        F.elu,
        F.celu,
        F.selu,
        F.hardshrink,
        F.leaky_relu,
        F.logsigmoid,
        F.softplus,
        F.prelu,
        F.softsign,
        F.tanhshrink,
        F.gelu,
    }
    return SUPPORTED_ACTIVATION_FUNCTIONS


def _get_supported_activation_modules():
    """
    返回支持的激活函数模块集合，包括 nn.ReLU, nn.RReLU, 等等。
    """
    SUPPORTED_ACTIVATION_MODULES = {
        nn.ReLU,
        nn.RReLU,
        nn.Hardtanh,
        nn.ReLU6,
        nn.Sigmoid,
        nn.Hardsigmoid,
        nn.Tanh,
        nn.SiLU,
        nn.Mish,
        nn.Hardswish,
        nn.ELU,
        nn.CELU,
        nn.SELU,
        nn.Hardshrink,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.Softplus,
        nn.PReLU,
        nn.Softsign,
        nn.Tanhshrink,
        nn.GELU,
    }
    return SUPPORTED_ACTIVATION_MODULES


def _get_default_structured_pruning_patterns() -> Dict[
    Tuple[Union[Type[nn.Module], Callable, MatchAllNode, str], ...],
    Callable[..., None],
]:
    """
    返回默认的结构化剪枝模式字典，包含每个激活函数/模块的 conv2d/linear 转换模式。
    """
    patterns: Dict[
        Tuple[Union[Type[nn.Module], Callable, MatchAllNode, str], ...],
        Callable[..., None],
    ]
    ```
    # 定义一个字典，用于存储不同类型神经网络模块和操作序列到剪枝函数的映射关系
    {
        # 对于 nn.Linear 输出，使用 prune_linear 函数进行剪枝
        (nn.Linear, "output"): prune_linear,
        # 对于两个 nn.Linear 模块相连的情况，使用 prune_linear_linear 函数进行剪枝
        (nn.Linear, nn.Linear): prune_linear_linear,
        # 对于 nn.Conv2d 输出，使用 prune_conv2d 函数进行剪枝
        (nn.Conv2d, "output"): prune_conv2d,
        # 对于两个 nn.Conv2d 模块相连的情况，使用 prune_conv2d_conv2d 函数进行剪枝
        (nn.Conv2d, nn.Conv2d): prune_conv2d_conv2d,
        # TODO LSTM 结构化剪枝当前不支持返回状态，需要找到一种显式匹配 getitem(0) 而不是 getitem 的方法
        # 这还需要改变剪枝函数本身的实现
        # 对于 nn.LSTM -> getitem(0) -> nn.Linear 的序列，使用 prune_lstm_output_linear 函数进行剪枝
        (nn.LSTM, getitem, nn.Linear): prune_lstm_output_linear,
        # 对于 nn.LSTM -> getitem(0) -> nn.LayerNorm -> nn.Linear 的序列，使用 prune_lstm_output_layernorm_linear 函数进行剪枝
        (nn.LSTM, getitem, nn.LayerNorm, nn.Linear): prune_lstm_output_layernorm_linear,
    }

    # 遍历所有支持的激活函数和模块，添加到待剪枝操作序列中
    for activation in chain(
        _get_supported_activation_functions(), _get_supported_activation_modules()
    ):
        patterns.update(
            {
                # 当前模式: 线性层 -> 激活函数 -> 线性层
                (nn.Linear, activation, nn.Linear): prune_linear_activation_linear,
                # 当前模式: 二维卷积层 -> 激活函数 -> 二维卷积层
                (nn.Conv2d, activation, nn.Conv2d): prune_conv2d_activation_conv2d,
                # 当前模式: 二维卷积层 -> 激活函数 -> 平均池化 -> 二维卷积层
                (
                    nn.Conv2d,
                    activation,
                    nn.AvgPool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                (
                    nn.Conv2d,
                    activation,
                    F.avg_pool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                (
                    nn.Conv2d,
                    activation,
                    nn.MaxPool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                (
                    nn.Conv2d,
                    activation,
                    F.max_pool2d,
                    nn.Conv2d,
                ): prune_conv2d_activation_pool_conv2d,
                # 当前模式: 二维卷积层 -> 池化层 -> 激活函数 -> 二维卷积层
                (
                    nn.Conv2d,
                    nn.AvgPool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                (
                    nn.Conv2d,
                    F.avg_pool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                (
                    nn.Conv2d,
                    nn.MaxPool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                (
                    nn.Conv2d,
                    F.max_pool2d,
                    activation,
                    nn.Conv2d,
                ): prune_conv2d_pool_activation_conv2d,
                # 当前模式: 二维卷积层 -> 自适应池化 -> 展平 -> 线性层
                (
                    nn.Conv2d,
                    nn.AdaptiveAvgPool2d,
                    nn.Flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
                (
                    nn.Conv2d,
                    nn.AdaptiveAvgPool2d,
                    torch.flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
                (
                    nn.Conv2d,
                    nn.AdaptiveMaxPool2d,
                    nn.Flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
                (
                    nn.Conv2d,
                    nn.AdaptiveMaxPool2d,
                    torch.flatten,
                    nn.Linear,
                ): prune_conv2d_pool_flatten_linear,
            }
        )
    return patterns
class BaseStructuredSparsifier(BaseSparsifier):
    r"""Base class for structured pruning.

    Abstract methods that need to be implemented:
        - update_mask: Function to compute a new mask for all keys in the
            `groups` attribute.

    Args:
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    """

    def __init__(self, defaults, patterns=None):
        super().__init__(defaults)  # 调用父类的初始化方法
        if patterns is None:
            patterns = _get_default_structured_pruning_patterns()  # 如果没有提供patterns，则使用默认的结构化剪枝模式
        self.patterns = patterns  # 初始化patterns属性为传入的patterns参数

    def make_config_from_model(
        self,
        model: nn.Module,
        SUPPORTED_MODULES: Optional[Set[Type]] = None,
    ) -> None:
        if SUPPORTED_MODULES is None:
            SUPPORTED_MODULES = _get_supported_structured_pruning_modules()  # 如果没有指定支持的模块类型，则获取默认支持的结构化剪枝模块类型
        super().make_config_from_model(model, SUPPORTED_MODULES=SUPPORTED_MODULES)  # 调用父类的make_config_from_model方法

    def _prepare(self, *args, **kwargs) -> None:
        r"""This function will attach the FakeStructuredSparsity parameterizations
        and BiasHooks at the appropriate points in the model.
        """
        for config in self.groups:  # 遍历self.groups中的配置项
            module = config["module"]  # 获取模块对象
            tensor_name = config["tensor_name"]  # 获取张量名称
            parametrization = config.get("parametrization", FakeStructuredSparsity)  # 获取参数化方法，如果未指定则使用FakeStructuredSparsity
            tensor = getattr(module, tensor_name)  # 获取模块中的张量对象

            mask = config.get(
                "mask",
                torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device),
            )  # 获取掩码，如果未指定则创建一个全为True的张量作为默认掩码
            self.state[config["tensor_fqn"]]["mask"] = mask  # 将掩码保存在状态中

            parametrize.register_parametrization(
                module, tensor_name, parametrization(mask)
            )  # 注册参数化方法到指定的模块和张量名称上

            # if linear / conv, we add in bias hooks
            if isinstance(module, (nn.Linear, nn.Conv2d)):  # 如果模块是线性层或卷积层
                prune_bias = config.get("prune_bias", True)  # 获取是否剪枝偏置参数，默认为True
                if module.bias is not None:
                    module.register_parameter(
                        "_bias", nn.Parameter(module.bias.detach())
                    )  # 注册一个名为"_bias"的参数，并将当前的偏置值分离出来
                    module.bias = None  # 将模块的偏置置为None
                    module.prune_bias = prune_bias  # 设置是否剪枝偏置的属性为传入的prune_bias值

                module.register_forward_hook(
                    BiasHook(module.parametrizations.weight[0], prune_bias)
                )  # 注册前向钩子函数BiasHook，针对权重参数和是否剪枝偏置
    def prune(self) -> None:
        r"""
        This function will FX symbolically trace the model and then find instances of the patterns
        defined in self.patterns (by default SUPPORTED_STRUCTURED_PRUNING_PATTERNS ).

        For each pattern, it will apply to corresponding conversion function, which will modify the output
        and input size expected by the modules within the pattern
        """

        # 对模型进行符号化跟踪，将其存储在 self.traced 中
        self.traced = symbolic_trace(self.model)
        # 获取模型中所有模块的字典表示
        modules = dict(self.traced.named_modules())

        # 目前的实现是通过迭代所有模式来查找匹配，如果速度慢，可以考虑使用 Trie 结构进行存储以实现更快的查找
        for node in self.traced.graph.nodes:
            for pattern, convert_fn in self.patterns.items():
                # 应用匹配函数尝试匹配当前节点到定义的模式
                matched = apply_match(modules, pattern, node, [])
                if matched is None:
                    continue

                # 获取第一个模块，并检查其参数化情况以及是否包含 FakeStructuredSparsity 参数
                first_module = modules.get(node.target)
                if (
                    first_module is not None
                    and parametrize.is_parametrized(first_module)
                    and module_contains_param(first_module, FakeStructuredSparsity)
                ):
                    # 构建需要转换的模块列表
                    convert_block = []
                    for node in matched:
                        if node.op == "call_module":
                            convert_block.append(modules.get(node.target))
                        elif node.op == "call_function":
                            convert_block.append(node.target)
                    # 应用转换函数到匹配的模块列表
                    convert_fn(*convert_block)

        # 检查所有模块，如果任何模块仍包含 FakeStructuredSparsity 参数，则抛出异常
        for module in self.traced.modules():
            if module_contains_param(module, FakeStructuredSparsity):
                raise Exception(
                    f"Error: {module} still contains FakeStructuredSparsity parametrizations!"
                )

        # 对跟踪到的图进行 lint 操作，确保其结构正确
        self.traced.graph.lint()
        # 重新编译跟踪到的模型
        self.traced.recompile()
        # 返回更新后的符号化跟踪模型
        return self.traced
```