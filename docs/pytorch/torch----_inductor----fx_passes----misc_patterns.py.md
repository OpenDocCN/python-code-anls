# `.\pytorch\torch\_inductor\fx_passes\misc_patterns.py`

```
# 指定静态类型检查中允许未标记类型的定义
# 导入 functools 模块，用于提供缓存功能
import functools

# 导入类型相关模块
from typing import Dict, Set, Tuple

# 导入 PyTorch 库
import torch

# 导入私有模块，用于计数器
from torch._dynamo.utils import counters

# 导入 Torch 操作重载相关模块
from torch._ops import OpOverload, OpOverloadPacket

# 导入模式匹配相关函数和装饰器
from ..pattern_matcher import fwd_only, register_replacement

# 设置 aten 别名，用于方便调用 Torch 操作
aten = torch.ops.aten

# 使用 functools 模块提供的 lru_cache 装饰器，用于缓存函数返回值
@functools.lru_cache(None)
def _misc_patterns_init():
    # 导入相关模块并设置本地变量
    from .joint_graph import patterns as joint_graph_patterns
    from .post_grad import pass_patterns as post_grad_patterns_all

    # 从 pass_patterns 中获取中等优先级的模式
    post_grad_patterns = post_grad_patterns_all[1]

    # 根据当前设备是否可用选择设备类型
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 定义函数：生成随机排列索引并执行加法操作
    def randperm_index_add_pattern(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return torch.index_add(x, dim=0, source=y, index=index), index

    # 定义函数：生成随机排列索引并执行不安全的索引放置操作
    def randperm_index_add_replacement(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return (
            torch.ops.aten._unsafe_index_put(
                x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False
            ),
            index,
        )

    # 注册上述两个函数作为替换模式，限定于指定的设备和模式列表
    register_replacement(
        randperm_index_add_pattern,
        randperm_index_add_replacement,
        [torch.empty(4, 8, device=device), torch.empty(2, 8, device=device)],
        fwd_only,
        [post_grad_patterns, joint_graph_patterns],
    )

    # 定义函数：生成随机排列索引并执行索引操作
    def randperm_index_pattern(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten.index(x, (index,)), index

    # 定义函数：生成随机排列索引并执行不安全的索引操作
    def randperm_index_replacement(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten._unsafe_index(x, (index,)), index

    # 注册上述两个函数作为替换模式，限定于指定的设备和模式列表，并传递标量修正信息
    register_replacement(
        randperm_index_pattern,
        randperm_index_replacement,
        [torch.empty(4, 8, device=device)],
        fwd_only,
        [post_grad_patterns, joint_graph_patterns],
        scalar_workaround={"slice_shape": 42},
    )


# 定义类 NumpyCompatNormalization，用于定义 numpy 兼容性的归一化处理
class NumpyCompatNormalization:
    # 定义字典属性 numpy_compat，映射 torch 的维度相关参数到 numpy 的对应参数
    numpy_compat: Dict[str, Tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    # 定义反向映射字典属性 inverse_mapping，暂未指定具体内容
    inverse_mapping: Dict[str, str]
    # 定义缓存字典属性 cache，映射 torch.fx.graph.Target 到已处理的集合
    cache: Dict["torch.fx.graph.Target", Set[str]]
    def __init__(self):
        # 初始化一个空的缓存字典，用于存储可调用对象与其可替换参数的映射关系
        self.cache = {}  # callable -> tuple of replaceable args e.g. ["axis"]

        # 初始化一个空的反向映射字典
        self.inverse_mapping = {}

        # 遍历self.numpy_compat中的映射关系，建立反向映射字典
        for actual_kwarg, numpy_kwargs in self.numpy_compat.items():
            for numpy_kwarg in numpy_kwargs:
                # 确保numpy_kwarg没有重复出现在inverse_mapping中
                assert numpy_kwarg not in self.inverse_mapping
                # 建立numpy_kwarg到actual_kwarg的映射关系
                self.inverse_mapping[numpy_kwarg] = actual_kwarg

    def __call__(self, graph: torch.fx.Graph):
        # 遍历图中的每个节点
        for node in graph.nodes:
            # 如果节点的操作不是函数调用，则跳过
            if node.op != "call_function":
                continue
            # 如果node.target是OpOverload或OpOverloadPacket类型的实例，则跳过
            if isinstance(node.target, (OpOverload, OpOverloadPacket)):
                # 只适用于torch操作；例如torch.stack(axis=1)适用，torch.ops.aten.stack(axis=1)不适用。
                continue

            # 获取节点的关键字参数
            kwargs = node.kwargs

            # 如果node.target已经在缓存中，则从缓存中获取可替换参数集合
            if node.target in self.cache:
                replaceable_kwargs = self.cache[node.target]
            else:
                # 否则，从torch.fx.operator_schemas中获取node.target的签名
                signatures = torch.fx.operator_schemas.get_signature_for_torch_op(
                    node.target
                )
                signatures = () if signatures is None else signatures
                replaceable_kwargs = set()
                # 遍历每个签名的参数，将其对应的numpy兼容参数加入replaceable_kwargs中
                for sig in signatures:
                    for param_name in sig.parameters.keys():
                        if param_name in self.numpy_compat:
                            replaceable_kwargs.update(self.numpy_compat[param_name])

                # 将replaceable_kwargs存入缓存中
                self.cache[node.target] = replaceable_kwargs

            # 如果replaceable_kwargs为空，则跳过当前节点
            if not replaceable_kwargs:
                continue

            # 初始化一个空的新关键字参数字典
            new_kwargs = {}
            # 标记关键字参数是否发生变化
            kwargs_changed = False

            # 遍历原始的关键字参数
            for k, v in kwargs.items():
                # 如果当前参数k在可替换参数中
                if k in replaceable_kwargs:
                    # 标记参数发生了变化
                    kwargs_changed = True
                    # 将参数替换为其对应的actual_kwarg
                    new_kwargs[self.inverse_mapping[k]] = v
                else:
                    # 否则保持原样
                    new_kwargs[k] = v

            # 如果参数发生了变化
            if kwargs_changed:
                # 使用不可变的字典更新节点的关键字参数
                node.kwargs = torch.fx.immutable_collections.immutable_dict(new_kwargs)
                # 更新计数器
                counters["inductor"]["numpy_compat_normalization"] += 1
# 创建一个 NumpyCompatNormalization 的实例对象并赋值给 numpy_compat_normalization 变量
numpy_compat_normalization = NumpyCompatNormalization()
```