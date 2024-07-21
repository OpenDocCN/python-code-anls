# `.\pytorch\test\fx\quantization.py`

```
r"""
**This file is EXPERIMENTAL and is mostly used for testing purposes! Do not
rely on it for anything!**
"""

# 导入必要的模块和类
import operator  # 导入 operator 模块，用于操作符相关的函数和类
import sys  # 导入 sys 模块，用于系统相关的功能
from typing import Optional  # 从 typing 模块导入 Optional 类型，用于声明可选类型

import torch  # 导入 PyTorch 模块
from torch.fx import Graph, GraphModule, Node  # 从 torch.fx 导入 Graph, GraphModule, Node 类
from torch.fx.graph import map_arg  # 从 torch.fx.graph 导入 map_arg 函数
from torch.fx.proxy import Proxy  # 从 torch.fx.proxy 导入 Proxy 类
from torch.nn.utils import fuse_conv_bn_weights  # 从 torch.nn.utils 导入 fuse_conv_bn_weights 函数

# can be a
#  module type, a builtin function, or a string to match target


def _minmax_scale_zeropoint(
    min_val, max_val, qmin=-127, qmax=128, eps=torch.finfo(torch.float32).eps
):
    """
    计算量化的 scale 和 zero_point。

    Args:
        min_val: 最小值
        max_val: 最大值
        qmin: 量化的最小值，默认为 -127
        qmax: 量化的最大值，默认为 128
        eps: 浮点数精度，默认为 torch.finfo(torch.float32).eps

    Returns:
        scale: 缩放因子
        zero_point: 零点
    """
    min_val = min(0.0, min_val)  # 计算最小值
    max_val = max(0.0, max_val)  # 计算最大值
    if max_val == min_val:
        return 1.0, 0  # 特殊情况，返回默认值
    else:
        scale = (max_val - min_val) / float(qmax - qmin)  # 计算缩放因子
        scale = max(scale, eps)  # 确保缩放因子不小于 eps
        zero_point = qmin - round(min_val / scale)  # 计算零点
        zero_point = max(qmin, zero_point)  # 确保零点不小于 qmin
        zero_point = min(qmax, zero_point)  # 确保零点不大于 qmax
        zero_point = int(zero_point)  # 转换为整数
        return scale, zero_point  # 返回 scale 和 zero_point


class MinMaxObserver:
    """
    最小最大观察器类，用于观察节点值的最小和最大值。
    """

    def __init__(self, quantizer, node):
        self.min, self.max = float("inf"), float("-inf")  # 初始化最小值和最大值为无穷大和负无穷
        self.all_tensors = True  # 初始化为 True，表示所有值都是张量

    def observe(self, node, env):
        """
        观察节点值，并更新最小和最大值。

        Args:
            node: 节点对象
            env: 环境变量
        """
        v = env[node.name]  # 获取节点的值
        if not isinstance(v, torch.Tensor):
            self.all_tensors = False  # 如果值不是张量，则更新标记为 False
            return
        self.max = max(self.max, float(v.max()))  # 更新最大值
        self.min = min(self.min, float(v.min()))  # 更新最小值

    def scale_zeropoint(self):
        """
        计算量化的 scale 和 zero_point。

        Returns:
            scale: 缩放因子
            zero_point: 零点
        """
        return _minmax_scale_zeropoint(self.min, self.max, qmin=0, qmax=255)  # 调用计算函数


class NoObserver:
    """
    无观察器类，不执行任何操作。
    """

    def __init__(self, quantizer, node):
        pass

    def observe(self, node, env):
        pass


_DEFAULT_QUANTIZATION_PATTERNS = {}  # 默认的量化模式字典


def register_pattern(pattern):
    """
    注册量化模式函数的装饰器。

    Args:
        pattern: 量化模式

    Returns:
        insert 函数
    """

    def insert(fn):
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn  # 将函数注册到默认量化模式字典中
        return fn

    return insert  # 返回插入函数


@register_pattern(operator.add)
class Add(MinMaxObserver):
    """
    加法操作的量化模式类，继承自最小最大观察器。
    """

    def quantize(self, quantizer, node, load_arg):
        """
        对加法操作进行量化。

        Args:
            quantizer: 量化器对象
            node: 节点对象
            load_arg: 载入参数函数

        Returns:
            量化后的节点对象
        """
        if not self.all_tensors:
            return NotImplemented  # 如果不是所有值都是张量，则返回未实现
        scale, zeropoint = self.scale_zeropoint()  # 计算 scale 和 zero_point
        return quantizer.quantized_graph.create_node(
            "call_function",
            torch.ops.quantized.add,
            load_arg(node.args),
            {"scale": scale, "zero_point": zeropoint},
        )


class Relu(NoObserver):
    """
    ReLU 操作的量化模式类，继承自无观察器。
    """

    def quantize(self, quantizer, node, load_arg):
        """
        对 ReLU 操作进行量化。

        Args:
            quantizer: 量化器对象
            node: 节点对象
            load_arg: 载入参数函数

        Returns:
            量化后的操作结果
        """
        return torch.relu(
            load_arg(node.args[0])
        )  # torch.relu 直接在量化张量上执行工作？


# these ops have quantized equivalents that do not need any extra information
@register_pattern(torch.nn.ReLU)
@register_pattern(torch.nn.AvgPool2d)
@register_pattern(torch.nn.MaxPool2d)
@register_pattern(torch.nn.AdaptiveAvgPool2d)
class CopyNode(NoObserver):
    """
    拷贝节点的量化模式类，继承自无观察器。
    """

    def quantize(self, quantizer, node, load_arg):
        """
        对拷贝节点进行量化。

        Args:
            quantizer: 量化器对象
            node: 节点对象
            load_arg: 载入参数函数

        Returns:
            量化后的节点对象
        """
        return quantizer.quantized_graph.node_copy(node, load_arg)


class IdentityModule(torch.nn.Module):
    """
    身份模块类，继承自 torch.nn.Module。
    """

    def forward(self, x):
        """
        前向传播函数，返回输入值 x 本身。

        Args:
            x: 输入张量

        Returns:
            x: 输入张量
        """
        return x


# handle conv, maybe followed by bn, maybe followed by relu
@register_pattern(torch.nn.modules.conv.Conv2d)
@register_pattern((torch.nn.ReLU, torch.nn.modules.conv.Conv2d))
@register_pattern(
    (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.conv.Conv2d)
)
@register_pattern(
    (
        torch.nn.ReLU,
        (torch.nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.conv.Conv2d),
    )
)
class ConvNormRelu(MinMaxObserver):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node, self.bn_node = None, None
        # 检查节点是否是ReLU，如果是则将其赋给self.relu_node，并继续处理其参数
        if isinstance(quantizer.modules[node.target], torch.nn.ReLU):
            self.relu_node = node
            node = node.args[0]
        # 检查节点是否是BatchNorm2d，如果是则将其赋给self.bn_node，并获取其相关的BatchNorm模块
        if isinstance(quantizer.modules[node.target], torch.nn.BatchNorm2d):
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            node = node.args[0]
        # 确保节点是Conv2d类型，将其赋给self.conv_node，并获取对应的Conv2d模块
        assert isinstance(quantizer.modules[node.target], torch.nn.modules.Conv2d)
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

    def quantize(self, quantizer, node, load_arg):
        # 获取当前节点对应的Conv2d模块的权重和偏置
        mod = self.conv
        weight, bias = mod.weight, mod.bias

        # 如果存在BatchNorm2d节点，则融合Conv2d和BatchNorm2d的权重和偏置
        if self.bn_node is not None:
            weight, bias = fuse_conv_bn_weights(
                weight,
                bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )

        # 计算权重的最小值和最大值
        min_val, max_val = float(weight.min()), float(weight.max())

        # 获取激活函数的缩放因子和零点偏移量
        act_scale, act_zp = self.scale_zeropoint()

        # 计算权重的缩放因子和零点偏移量，并将权重量化为torch.qint8类型
        weight_scale, weight_zp = _minmax_scale_zeropoint(min_val, max_val)
        qweight = torch.quantize_per_tensor(
            weight, weight_scale, weight_zp, torch.qint8
        )

        # 根据是否存在ReLU节点选择合适的量化卷积操作类
        ctor = (
            torch.ao.nn.intrinsic.quantized.ConvReLU2d
            if self.relu_node is not None
            else torch.ao.nn.quantized.Conv2d
        )

        # 创建量化卷积操作对象qconv，并设置其权重和偏置
        qconv = ctor(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, bias)
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)

        # 获取父节点和节点名，并将量化卷积操作对象设置为父节点的属性
        parent_name, name = _parent_name(self.conv_node.target)
        setattr(quantizer.modules[parent_name], name, qconv)

        # 如果存在BatchNorm2d节点，将其替换为一个什么都不做的IdentityModule对象
        if self.bn_node is not None:
            parent_bn, bn_name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], bn_name, IdentityModule())

        # 返回量化后的节点
        return quantizer.quantized_graph.create_node(
            "call_module",
            self.conv_node.target,
            (load_arg(self.conv_node.args[0]),),
            {},
        )
# 将目标字符串按最后一个点号分割，返回分割结果的元组
def _parent_name(target):
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]  # 如果没有分割点，则返回空字符串和整个字符串作为元组
    else:
        return r[0], r[1]  # 返回分割后的结果作为元组的两个元素


class DefaultQuant(MinMaxObserver):
    def quantize(self, input):
        assert self.all_tensors  # 断言所有张量都被包含在内
        scale, zeropoint = self.scale_zeropoint()  # 获取量化的缩放因子和零点
        return torch.quantize_per_tensor(
            Proxy(input), scale, zeropoint, torch.quint8
        ).node  # 返回输入张量的量化结果的节点


def matches(modules, node, pattern, max_uses=sys.maxsize):
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern  # 将模式拆分为自身匹配和参数匹配
    else:
        self_match = pattern
        arg_matches = None

    if len(node.users) > max_uses:
        return False  # 如果节点的使用次数超过最大限制，则返回False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != "call_module":
            return False  # 如果节点不是调用模块操作，则返回False
        if not isinstance(modules[node.target], self_match):
            return False  # 如果模块类型不是预期的self_match类型，则返回False
    elif callable(self_match):
        if node.op != "call_function" or node.target is not self_match:
            return False  # 如果节点不是调用函数操作或目标函数不是self_match，则返回False
    elif node.target != self_match:
        return False  # 如果节点的目标不等于self_match，则返回False

    if not arg_matches:
        return True  # 如果没有参数匹配要求，则返回True

    if len(arg_matches) != len(node.args):
        return False  # 如果参数匹配的数量与节点参数数量不一致，则返回False

    return all(
        matches(modules, node, arg_match, max_uses=1)
        for node, arg_match in zip(node.args, arg_matches)
    )  # 对节点的每个参数递归调用matches函数进行匹配


class Quantizer:
    def __init__(
        self, mod, patterns=_DEFAULT_QUANTIZATION_PATTERNS, quant_ctor=DefaultQuant
    ):
        self.root = mod  # 根模块
        self.graph = mod.graph  # 模块的计算图
        self.quant_ctor = quant_ctor  # 量化器构造函数

        # 缓存用于观察的信息
        self.state_dict = self.root.state_dict()  # 获取根模块的状态字典
        self.modules = dict(self.root.named_modules())  # 获取根模块下的所有模块字典

        # 匹配将被量化的模式
        self.matches = self._find_matches(patterns)  # 找到符合模式的匹配项
        # 找到未量化的匹配节点的输入，这些节点必须被量化，需要测量统计信息
        # 为每个节点初始化一个quant_ctor对象
        self.quants = self._find_quants(quant_ctor)  # 找到需要量化的节点的输入
    def observe(self, args):
        # 定义一个函数 observe，用于执行对计算图的解释器操作
        # 这里的代码主要是对计算图的解释器
        # 将其抽象化可能是可行的，但直接看到具体操作有时也很有用
        # 可以直接进行修改和调试。
        # 或许我们应该提供一个示例解释器，供用户复制粘贴后进行编辑。
        args_iter = iter(args)  # 创建参数迭代器
        env = {}  # 初始化环境字典

        def load_arg(a):
            # 将参数 a 映射为环境字典中对应节点名的值
            return map_arg(a, lambda node: env[node.name])

        output_node: Optional[Node] = None  # 初始化输出节点为 None
        for node in self.graph.nodes:  # 遍历计算图中的每个节点
            if node.op == "placeholder":
                result = next(args_iter)  # 如果节点为占位符，从参数迭代器中获取下一个参数作为结果
            elif node.op == "get_attr":
                result = self.state_dict[node.target]  # 如果节点是获取属性操作，从状态字典中获取对应的值
            elif node.op == "call_function":
                # 如果节点是函数调用操作，调用节点指定的函数，并传入对应参数和关键字参数
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == "call_method":
                # 如果节点是方法调用操作，调用对象的方法，并传入对应参数和关键字参数
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == "call_module":
                # 如果节点是模块调用操作，调用模块的方法，并传入对应参数和关键字参数
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == "output":
                # 如果节点是输出节点，返回其参数作为最终结果
                return load_arg(node.args[0])

            env[node.name] = result  # 将节点名和结果存入环境字典
            root_node, obj = self.matches.get(node.name, (None, None))
            if root_node is node:
                obj.observe(node, env)  # 如果节点是匹配的根节点，则调用对象的 observe 方法
            if node.name in self.quants:
                self.quants[node.name].observe(node, env)  # 如果节点名在量化字典中，调用量化对象的 observe 方法

        raise RuntimeError("Graph had no output node!")  # 如果计算图没有输出节点，则抛出运行时错误
    # 定义一个方法 quantize，用于量化当前对象的图形表示
    def quantize(self):
        # 创建一个新的 Graph 对象来存储量化后的图形
        self.quantized_graph = Graph()

        # 创建两个空的环境字典
        env = {}
        quant_env = {}

        # 定义一个内部方法 load_arg，根据是否量化来加载参数
        def load_arg(n, quantized):
            if not quantized:
                # 如果未量化，检查环境中是否有该节点的非量化版本
                if n.name not in env and n.name in quant_env:
                    # 如果量化环境中存在该节点，则从量化环境中反量化并返回其节点
                    env[n.name] = Proxy(quant_env[n.name]).dequantize().node
                return env[n.name]
            else:
                # 如果已量化，检查量化环境中是否有该节点的量化版本
                if n.name not in quant_env and n.name in env:
                    # 如果非量化环境中存在该节点，则对其进行量化并存储到量化环境中
                    quant_env[n.name] = self.quants[n.name].quantize(env[n.name])
                return quant_env[n.name]

        # 定义一个内部方法 copy_recursive，递归复制节点及其子节点
        def copy_recursive(node):
            # 定义内部方法 load_or_emit，根据环境加载节点或递归处理
            def load_or_emit(n):
                if n.name in env or n.name in quant_env:
                    # 如果节点在环境中存在，则加载其非量化版本
                    return load_arg(n, quantized=False)
                else:
                    # 否则递归复制该节点
                    return copy_recursive(n)

            # 将复制后的节点存储在环境中，并返回复制后的节点
            r = env[node.name] = self.quantized_graph.node_copy(
                node, lambda n: load_arg(n, quantized=False)
            )
            return r

        # 遍历当前对象的图形节点
        for node in self.graph.nodes:
            # 获取与当前节点相关的根节点和对象
            root_node, obj = self.matches.get(node.name, (None, None))
            if root_node is None:
                # 如果不存在根节点，则直接复制节点到量化图形中
                env[node.name] = self.quantized_graph.node_copy(
                    node, lambda n: load_arg(n, quantized=False)
                )
            elif root_node is node:
                # 如果根节点为当前节点，则进行量化操作
                r = obj.quantize(
                    self,
                    node,
                    lambda a: map_arg(a, lambda n: load_arg(n, quantized=True)),
                )
                if r is NotImplemented:
                    # 如果量化器选择不量化节点，则递归复制节点及其子节点
                    env[node.name] = copy_recursive(node)
                else:
                    # 否则将量化后的结果存储到量化环境中
                    quant_env[node.name] = r

        # 返回由原始根节点和量化后的图形组成的 GraphModule 对象
        return GraphModule(self.root, self.quantized_graph)

    # 定义一个方法 _find_matches，用于查找与给定模式匹配的节点
    def _find_matches(self, patterns):
        # 获取根模块中的命名模块并创建一个模块字典
        modules = dict(self.root.named_modules())
        # 创建一个空的匹配映射字典
        match_map = {}  # node name -> (root_node, match_value?)

        # 定义一个内部方法 apply_match，将匹配应用到节点上
        def apply_match(pattern, node, match):
            if isinstance(pattern, tuple):
                s, *args = pattern
                apply_match(s, node, match)
                # 递归应用子模式到节点的参数上
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match)
            else:
                # 将匹配的节点与其匹配值存储到匹配映射中
                match_map[node.name] = match

        # 逆序遍历当前对象的图形节点
        for node in reversed(self.graph.nodes):
            # 如果节点不在匹配映射中，则尝试将其与给定模式进行匹配
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if matches(modules, node, pattern):
                        # 如果节点与模式匹配，则应用匹配到该节点上
                        apply_match(pattern, node, (node, value(self, node)))

        # 返回完整的匹配映射字典
        return match_map
    # 定义一个方法 `_find_quants`，接受一个 `quant_ctor` 参数来构造量化器
    def _find_quants(self, quant_ctor):
        # 创建一个空字典来存储量化信息
        quants = {}

        # 定义一个内部函数 `visit_arg`，用于处理节点的参数
        def visit_arg(n):
            # 注意：即使对于已经量化的节点，我们也需要测量其量化信息。
            # 这是因为每个匹配都可以选择返回 NotImplemented（例如，对于不适当的数据类型的 __add__）。
            # 如果节点的名称不在 quants 字典中，将使用 quant_ctor 方法构造该节点的量化信息并存储在 quants 中。
            if n.name not in quants:
                quants[n.name] = quant_ctor(self, n)

        # 遍历图中的每个节点
        for node in self.graph.nodes:
            # 如果节点的名称在 self.matches 中
            if node.name in self.matches:
                # 对节点的位置参数应用 visit_arg 函数
                map_arg(node.args, visit_arg)
                # 对节点的关键字参数应用 visit_arg 函数
                map_arg(node.kwargs, visit_arg)

        # 返回存储了所有量化信息的 quants 字典
        return quants
```