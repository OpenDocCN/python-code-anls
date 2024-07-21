# `.\pytorch\test\dynamo\test_repros.py`

```
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""

# Owner(s): ["module: dynamo"]
# 导入需要使用的模块和库
import collections  # Python 标准库中的集合模块
import contextlib  # 提供一些上下文管理工具的模块
import copy  # 提供复制对象的功能
import functools  # 提供函数修饰器和高阶函数的模块
import gc  # Python 的垃圾回收模块
import inspect  # 用于检查 Python 对象的模块
import itertools  # 提供用于构建迭代器的函数的模块
import random  # 生成随机数的模块
import unittest  # Python 的单元测试框架
import warnings  # 用于处理警告的模块
import weakref  # Python 弱引用的模块
from abc import ABC  # Python 中用于定义抽象基类的模块
from collections import namedtuple  # 命名元组的模块
from copy import deepcopy  # 用于深拷贝对象的函数
from enum import Enum  # 枚举类型的模块
from functools import wraps  # 提供用于创建装饰器的函数
from typing import Any, Dict, Iterator, List, Tuple  # Python 的类型提示模块

import numpy as np  # 数值计算库

import torch  # PyTorch 深度学习库

# 导入 PyTorch 的私有测试模块和工具
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils

# 导入 PyTorch Functorch 的配置和库
import torch._functorch.config
import torch.library
import torch.utils._pytree as pytree

from torch import nn  # PyTorch 中的神经网络模块
from torch._dynamo.debug_utils import same_two_models  # 模型比较工具
from torch._dynamo.testing import CompileCounter, rand_strided, same  # 测试相关的工具
from torch._inductor.utils import fresh_inductor_cache  # Inductor 缓存相关工具
from torch.nn import functional as F  # PyTorch 中的函数模块

# 导入 PyTorch 内部测试和工具函数
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import (
    disable_translation_validation_if_dynamic_shapes,
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.two_tensor import TwoTensor  # 用于测试的双张量类


_orig_module_call = torch.nn.Module.__call__  # 保存原始的 Module 类的 __call__ 方法

# 自定义运算符，仅支持 CPU 和 Meta
lib = torch.library.Library("test_sample", "DEF")  # 定义一个名为 "test_sample" 的库对象
lib.define("foo(Tensor self) -> Tensor")  # 定义 foo 函数，接受一个 Tensor 并返回一个 Tensor
lib.impl("foo", torch.sin, "CPU")  # 实现 foo 函数，使用 torch.sin 函数在 CPU 上执行


requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")  # 跳过测试，除非 CUDA 可用


_GLOBAL_CPU_TENSOR = torch.randn(3)  # 创建一个包含三个随机数的全局 CPU 张量


def exists(val):
    return val is not None  # 判断 val 是否为 None 的辅助函数


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x  # 如果 x 是 None，则直接返回 x
        return fn(x, *args, **kwargs)  # 否则调用传入的函数 fn，并返回其结果

    return inner  # 返回内部函数 inner 的引用


def is_fx_tracing_test() -> bool:
    """
    Copied from the hpc trainer codebase
    """
    return torch.nn.Module.__call__ is not _orig_module_call  # 判断是否为 FX 跟踪测试的辅助函数


def has_detectron2():
    try:
        from detectron2.layers.mask_ops import _paste_masks_tensor_shape

        return _paste_masks_tensor_shape is not None  # 检查是否存在 Detectron2 的辅助函数
    except ImportError:
        return False  # 如果导入失败，则返回 False


def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
    # from detectron2 mask_ops.py

    device = masks.device  # 获取 masks 张量的设备信息

    if skip_empty and not torch.jit.is_scripting():
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )  # 计算 boxes 张量的最小值，并进行下界截断，转换为整数类型
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(
            dtype=torch.int32
        )  # 计算 boxes 张量第二列的最大值，并进行上界截断，转换为整数类型
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(
            dtype=torch.int32
        )  # 计算 boxes 张量第三列的最大值，并进行上界截断，转换为整数类型
    else:
        x0_int, y0_int = 0, 0  # 否则使用默认的整数坐标值
        x1_int, y1_int = img_w, img_h  # 使用图像的宽高作为整数坐标值
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # 按列切分 boxes 张量，每个部分是 Nx1 维度

    N = masks.shape[0]  # 获取 masks 张量的批量大小
    # 创建一个包含从 y0_int 到 y1_int 的浮点数张量 img_y，每个元素加上 0.5
    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    # 创建一个包含从 x0_int 到 x1_int 的浮点数张量 img_x，每个元素加上 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    # 根据 y0、y1 的范围将 img_y 标准化到 [-1, 1] 的区间
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    # 根据 x0、x1 的范围将 img_x 标准化到 [-1, 1] 的区间
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y 的形状为 (N, w), (N, h)

    # 创建 gx 和 gy 张量，用于存放 grid 坐标信息，形状为 (N, h, w)
    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    # 将 gx 和 gy 合并为 grid 张量，形状为 (N, h, w, 2)，其中最后一个维度分别表示 x 和 y 的坐标
    grid = torch.stack([gx, gy], dim=3)

    # 如果不是在 JIT 脚本模式下，并且 masks 的数据类型不是浮点型，则将 masks 转换为 float 类型
    if not torch.jit.is_scripting():
        if not masks.dtype.is_floating_point:
            masks = masks.float()
    # 使用 F.grid_sample 函数对 masks 进行空间变换，采样 grid 中指定位置的像素值，不使用对齐角点选项
    img_masks = F.grid_sample(masks, grid.to(masks.dtype), align_corners=False)

    # 如果 skip_empty 为真，并且不是在 JIT 脚本模式下，返回处理后的 img_masks 和 ROI 区域的切片信息
    if skip_empty and not torch.jit.is_scripting():
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    # 否则，返回处理后的 img_masks 和空元组
    else:
        return img_masks[:, 0], ()
def global_fn(x):
    return torch.sin(x)


# 定义一个全局函数 global_fn，返回输入张量 x 的正弦值
def global_fn(x):
    return torch.sin(x)



def cat(tensors, dim=0):
    # from detectron2 wrappers.py
    # 断言输入 tensors 是列表或元组类型
    assert isinstance(tensors, (list, tuple))
    # 如果 tensors 只有一个元素，直接返回该元素
    if len(tensors) == 1:
        return tensors[0]
    # 使用 PyTorch 的 torch.cat 函数对 tensors 中的张量沿指定维度 dim 进行拼接
    return torch.cat(tensors, dim)



def shapes_to_tensor(x, device=None):
    # from detectron2 wrappers.py
    # 如果当前处于脚本化状态，直接将 x 转换为张量并返回
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    # 如果当前处于追踪状态，断言 x 中的所有元素都是 torch.Tensor 类型
    if torch.jit.is_tracing():
        assert all(
            isinstance(t, torch.Tensor) for t in x
        ), "Shape should be tensor during tracing!"
        # 使用 torch.stack 将所有张量堆叠成一个新的张量 ret
        ret = torch.stack(x)
        # 如果 ret 的设备与指定的 device 不同，则将 ret 转移到指定的 device 上
        if ret.device != device:
            ret = ret.to(device=device)
        return ret
    # 默认情况下，将 x 转换为张量并返回
    return torch.as_tensor(x, device=device)



fw_graph = [None]
bw_graph = [None]

def aot_graph_capture_backend(gm, args):
    from functorch.compile import min_cut_rematerialization_partition
    from torch._functorch.aot_autograd import aot_module_simplified

    def fw_compiler(gm, _):
        # 将前向图 gm 存储到 fw_graph[0] 中
        fw_graph[0] = gm
        return gm

    def bw_compiler(gm, _):
        # 将反向图 gm 存储到 bw_graph[0] 中
        bw_graph[0] = gm
        return gm

    # 使用 functorch 库中的 aot_module_simplified 函数对模型 gm 进行简化编译
    return aot_module_simplified(
        gm,
        args,
        fw_compiler,
        bw_compiler,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )



class Boxes:
    # from detectron2 poolers.py
    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        # 确定 tensor 的设备，如果 tensor 不是 torch.Tensor 类型，则使用 CPU 设备
        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        # 将 tensor 转换为 torch.float32 类型的张量
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        # 如果 tensor 元素数为 0，使用 reshape 重塑成 (-1, 4) 形状的张量
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        # 断言 tensor 是二维张量且最后一维长度为 4
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        # 将处理后的 tensor 分配给类属性 self.tensor
        self.tensor = tensor

    def __len__(self) -> int:
        # 返回 self.tensor 的第一维长度，即行数
        return self.tensor.shape[0]

    @property
    def device(self):
        # 返回 self.tensor 的设备信息
        return self.tensor.device



def convert_boxes_to_pooler_format(box_lists):
    # from detectron2 structures.py
    # 将输入 box_lists 中所有 Boxes 实例的 tensor 属性拼接成一个张量 boxes
    boxes = torch.cat([x.tensor for x in box_lists], dim=0)
    # 使用 shapes_to_tensor 函数将 box_lists 中每个 Boxes 实例的长度转换为张量 sizes
    sizes = shapes_to_tensor([x.__len__() for x in box_lists], device=boxes.device)
    # 创建索引张量 indices，用于表示每个 box 在拼接后的张量 boxes 中的索引
    indices = torch.repeat_interleave(
        torch.arange(len(box_lists), dtype=boxes.dtype, device=boxes.device), sizes
    )
    # 沿着第二维度将 indices 和 boxes 拼接成最终的池化器格式张量
    return cat([indices[:, None], boxes], dim=1)



ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput",
    ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"],
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    # 定义一个包含字符串元素的列表，表示不同类型的数据或状态
    ["hidden_states", "all_hidden_states", "all_attentions", "past_buckets_states"],
    def __init__(self):
        # 调用父类构造函数初始化对象
        super().__init__()
        # 定义丢弃率为0.5的Dropout层
        self.dropout = 0.5
        # 初始化512维度的LayerNorm层
        self.layer_norm = torch.nn.LayerNorm(512, eps=1.0e-12)
        # 创建包含单个线性层的列表
        self.layers = [torch.nn.Linear(256, 256)]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=[None] * 6,
        num_hashes=None,
        use_cache=False,
        orig_sequence_length=64,
        output_hidden_states=False,
        output_attentions=False,
    ):
        # 对输入的hidden_states进行处理
        # 如果输出隐藏状态设置为True，则将隐藏状态添加到all_hidden_states列表中
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # 对每一层进行循环处理
        for layer_id, (layer, layer_head_mask) in enumerate(zip(self.layers, head_mask)):
            # 对注意力输出进行当前层的处理
            attn_output = layer(attn_output)
            # 将当前层的注意力输出添加到all_buckets元组中
            all_buckets = all_buckets + (attn_output,)

            # 如果输出隐藏状态设置为True，则将隐藏状态添加到all_hidden_states列表中
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

        # 将注意力输出和隐藏状态连接起来并返回
        return torch.cat([attn_output, hidden_states], dim=-1)
        # 函数定义的结尾括号
        ):
        # 用于存储所有隐藏状态和注意力的列表，可以根据需要填充
        all_hidden_states = []
        all_attentions = []

        # 初始化过去桶状态列表，与层数相同
        past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # 对隐藏状态进行拼接，用于可逆ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        
        # 使用自定义的可逆函数应用层次操作
        hidden_states = _ReversibleFunction.apply(
            hidden_states,            # 输入的隐藏状态张量
            self.layers,              # 模型的层列表
            attention_mask,           # 注意力掩码
            head_mask,                # 头部掩码
            num_hashes,               # 哈希数
            all_hidden_states,        # 存储所有隐藏状态的列表
            all_attentions,           # 存储所有注意力的列表
            past_buckets_states,      # 存储过去桶状态的列表
            use_cache,                # 是否使用缓存
            orig_sequence_length,     # 原始序列长度
            output_hidden_states,     # 是否输出隐藏状态
            output_attentions,        # 是否输出注意力
        )

        # 应用层归一化到拼接的隐藏状态
        hidden_states = self.layer_norm(hidden_states)

        # 应用dropout
        hidden_states = torch.nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # 返回ReformerEncoderOutput对象，包含处理后的隐藏状态和其他信息
        return ReformerEncoderOutput(
            hidden_states=hidden_states,          # 处理后的隐藏状态
            all_hidden_states=all_hidden_states,  # 所有隐藏状态的列表
            all_attentions=all_attentions,        # 所有注意力的列表
            past_buckets_states=past_buckets_states,  # 过去桶状态的列表
        )
class ListConfig:
    # 列表配置类，支持迭代和处理列表内容
    class ValueNode:
        # 值节点类，用于存储单个值
        def __init__(self, value):
            self.value = value

        def _dereference_node(self):
            # 返回节点本身，用于解除引用
            return self

        def _is_missing(self):
            # 始终返回 False，表示节点不缺失
            return False

        def _value(self):
            # 返回节点存储的值
            return self.value

    # 基于 omegaconfig.listconfig 的示例
    class ListIterator(Iterator[Any]):
        # 列表迭代器类，用于迭代 ListConfig 类的内容
        def __init__(self, lst: Any, resolve: bool) -> None:
            self.resolve = resolve
            # 获取列表内容并创建迭代器
            self.iterator = iter(lst.__dict__["_content"])
            self.index = 0

        def __next__(self) -> Any:
            # 获取下一个元素
            x = next(self.iterator)
            if self.resolve:
                # 如果需要解析，则尝试解析节点
                x = x._dereference_node()
                if x._is_missing():
                    raise AssertionError

            self.index = self.index + 1
            if isinstance(x, ListConfig.ValueNode):
                # 如果是 ValueNode 类型，则返回其值
                return x._value()
            # 如果不是 ValueNode 类型，抛出断言错误
            raise AssertionError

    def __iter__(self):
        # 返回默认的迭代器，启用解析
        return self._iter_ex(True)

    def _iter_ex(self, resolve: bool) -> Iterator[Any]:
        try:
            # 尝试创建 ListIterator 实例并返回
            return ListConfig.ListIterator(self, resolve)
        except Exception:
            # 如果出现异常，则抛出断言错误
            raise AssertionError from None

    def __init__(self):
        # 初始化列表配置对象，包含三个 ValueNode 实例
        self._content = [
            ListConfig.ValueNode(1),
            ListConfig.ValueNode(3),
            ListConfig.ValueNode(torch.tensor([7.0])),
        ]


def longformer_chunk(hidden_states, window_overlap=256):
    """将隐藏状态转换为重叠的块。块大小为2w，重叠大小为w"""

    # 非重叠的块，大小为2w
    hidden_states = hidden_states.view(
        hidden_states.size(0),
        hidden_states.size(1) // (window_overlap * 2),
        window_overlap * 2,
        hidden_states.size(2),
    )

    # 使用 `as_strided` 创建重叠块，重叠大小为 window_overlap
    chunk_size = list(hidden_states.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(hidden_states.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)


class PartialT5(torch.nn.Module):
    # 高度简化的 T5Attention 前缀
    def __init__(self):
        super().__init__()
        # 初始化 T5 模型中的 Linear 层
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        query_length=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        def shape(states):
            """对隐藏状态进行投影"""
            return states.view(batch_size, -1, 8, 64).transpose(1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """将隐藏状态正确投影到键/查询状态"""
            if key_value_states is None:
                # 自注意力
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # 交叉注意力
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # 自注意力
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # 交叉注意力
                    hidden_states = past_key_value
            return hidden_states

        # 获取查询状态
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # 获取键/值状态
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # 计算注意力分数
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # (此处省略了部分代码)

        return scores, value_states
class ChunkReformerFeedForward(torch.nn.Module):
    # 从 HF modeling_reformer.py 简化而来的类定义
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(256, eps=1e-12)
        self.dense = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, 256)

    def forward(self, attention_output):
        # 将 forward_chunk 方法应用到输入的 attention_output 上
        return apply_chunking_to_forward(
            self.forward_chunk,
            attention_output + 1,
        )

    def forward_chunk(self, hidden_states):
        # 对隐藏状态应用层归一化
        hidden_states = self.layer_norm(hidden_states)
        # 对归一化后的隐藏状态应用线性变换 dense
        hidden_states = self.dense(hidden_states)
        # 应用输出层线性变换 output
        return self.output(hidden_states)


def apply_chunking_to_forward(forward_fn, *input_tensors):
    # 从 HF model_utils.py 简化而来的函数，用于将输入分块应用到 forward_fn 中
    assert len(input_tensors) > 0
    # 获取输入张量的形状
    tensor_shape = input_tensors[0].shape[1]
    # 确保所有输入张量在第一个维度（通常是批次维度）上具有相同的形状
    assert all(input_tensor.shape[1] == tensor_shape for input_tensor in input_tensors)
    # 获取 forward_fn 函数的参数数量
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    # 如果 forward_fn 函数的参数数量与输入张量数量不一致，则引发 ValueError
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError

    # 将 forward_fn 应用到输入张量上，并返回结果
    return forward_fn(*input_tensors)


def _validate_model_kwargs(fn, model_kwargs):
    # 从 transformers.generation.utils._validate_model_kwargs 简化而来的函数
    unused_model_args = []
    # 获取函数 fn 的参数集合
    model_args = set(inspect.signature(fn).parameters)
    # 遍历 model_kwargs 中的每个键值对
    for key, value in model_kwargs.items():
        # 如果值不为 None 且键不在函数 fn 的参数集合中，则将键添加到 unused_model_args 中
        if value is not None and key not in model_args:
            unused_model_args.append(key)
    # 如果存在未使用的 model_kwargs 键，则引发 ValueError 异常
    if unused_model_args:
        raise ValueError(
            f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
            " generate arguments will also show up in this list)"
        )


class FakeMamlInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个线性层，输入大小为 784，输出大小为 5
        self.linear = torch.nn.Linear(784, 5)

    def forward(self, x, ignored=None, bn_training=False):
        # 将输入 x 展平并应用线性层
        return self.linear(x.view(x.shape[0], -1))


class PartialMaml(torch.nn.Module):
    # 简化版的 maml.meta.Meta.finetuning
    def __init__(self):
        super().__init__()
        # 创建 FakeMamlInner 的实例作为网络的一部分
        self.net = FakeMamlInner()
        # 设置测试阶段的更新步数为 10
        self.update_step_test = 10
        # 设置更新的学习率为 0.4
        self.update_lr = 0.4
    # 定义前向传播方法，用于元学习中的测试阶段
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        # 获取查询集大小
        querysz = x_qry.size(0)

        # 初始化一个列表，用于存储不同更新步骤后的准确率
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # 为了不破坏 running_mean/variance 和 bn_weight/bias 的状态，
        # 我们在副本模型上进行微调，而不是在 self.net 上进行操作
        net = deepcopy(self.net)

        # 1. 运行第 i 个任务并计算 k=0 时的损失
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        # 计算损失对网络参数的梯度
        grad = torch.autograd.grad(loss, net.parameters())
        # 使用梯度更新快速权重，生成新的参数列表
        fast_weights = [
            p[1] - self.update_lr * p[0] for p in zip(grad, net.parameters())
        ]

        # 在第一次更新之前的损失和准确率
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # 计算正确预测的数量并转换为标量
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # 在第一次更新后的损失和准确率
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # 计算正确预测的数量并转换为标量
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        # 释放 net 变量占用的内存
        del net

        # 计算准确率并返回作为张量
        accs = torch.tensor(corrects) / querysz

        return accs
def softmax_backward_data(parent, grad_output, output, dim, self):
    from torch import _softmax_backward_data

    # 调用 PyTorch 库函数 _softmax_backward_data，用于计算 softmax 的反向传播结果
    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


class XSoftmax(torch.autograd.Function):
    # transformers.models.deberta.modeling_deberta.XSoftmax

    @staticmethod
    def forward(self, input, mask, dim):
        # 设置对象属性 dim，用于在 backward 方法中使用
        self.dim = dim
        # 计算掩码的逆，将不需要的部分替换为极小值
        rmask = ~(mask.to(torch.bool))
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        # 对替换后的张量进行 softmax 操作，dim 参数由 forward 方法传入
        output = torch.softmax(output, self.dim)
        # 将填充部分重新置为 0
        output.masked_fill_(rmask, 0)
        # 保存需要用于反向传播的张量
        self.save_for_backward(output, rmask)
        return output

    @staticmethod
    def backward(self, grad_output):
        # 从保存的张量中恢复 output 和 rmask
        (output, rmask) = self.saved_tensors
        # 调用 softmax 反向传播的自定义函数 softmax_backward_data
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None


class ModelOutput(collections.OrderedDict):
    """based on file_utils.py in HuggingFace"""

    def __getitem__(self, k):
        if isinstance(k, str):
            # 如果键是字符串，返回内部字典对应键的值
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            # 否则按索引返回元组形式的值
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # 如果属性名在当前字典的键中，并且值不为空，则设置属性值
            # 避免递归错误，不调用 self.__setitem__
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # 设置字典键值对，并确保键名有效
        super().__setitem__(key, value)
        # 避免递归错误，不调用 self.__setattr__
        super().__setattr__(key, value)

    def to_tuple(self):
        # 返回有序字典的值的元组形式
        return tuple(self[k] for k in self.keys())


def create_rand_mask_from_inputs(
    from_blocked_mask,
    to_blocked_mask,
    rand_attn,
    num_attention_heads,
    num_rand_blocks,
    batch_size,
    from_seq_length,
    from_block_size,
):
    """taken from HF modeling_big_bird.py"""
    # 计算 num_windows，即每个序列长度块数减去两个边界块的数量
    num_windows = from_seq_length // from_block_size - 2
    # 构建随机掩码，通过索引对 to_blocked_mask 和 rand_attn 的元素进行堆叠
    rand_mask = torch.stack(
        [p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)]
    )
    # 调整随机掩码的形状以匹配注意力头和块的数量
    rand_mask = rand_mask.view(
        batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size
    )
    # 使用 einsum 函数计算从 from_blocked_mask 到 rand_mask 的乘积
    rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
    return rand_mask


class SequentialAppendList(torch.nn.Sequential):
    """from timm/models/vovnet.py"""

    def forward(self, x: torch.Tensor, concat_list: List[torch.Tensor]) -> torch.Tensor:
        # 遍历序列中的模块，并将它们的输出附加到 concat_list 中
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        # 在指定维度上连接 concat_list 中的所有张量
        x = torch.cat(concat_list, dim=1)
        return x, concat_list


class BatchNormAct2d(torch.nn.BatchNorm2d):
    """Taken from timm"""
    # 初始化 BatchNormAct 模块的实例
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        act_layer=torch.nn.ReLU,  # 激活函数，默认为 ReLU
        inplace=True,
    ):
        # 调用父类 BatchNorm2d 的初始化方法，传入参数设置
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        # 根据 act_layer 参数初始化激活函数层，设置是否原地操作
        self.act = act_layer(inplace=inplace)
    
    @torch.jit.ignore
    # 通过 Python 方法进行前向传播，调用父类的 forward 方法
    def _forward_python(self, x):
        return super().forward(x)
    
    # 实现模块的前向传播
    def forward(self, x):
        # 如果正在进行 Torch 脚本化（scripting），则调用 _forward_jit 方法
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            # 否则调用 Python 方法进行前向传播
            x = self._forward_python(x)
        # 对前向传播结果应用激活函数
        x = self.act(x)
        return x
# 从 huggingface model_utils.py 中获取参数的数据类型
def get_parameter_dtype(parameter):
    try:
        # 返回参数的数据类型
        return next(parameter.parameters()).dtype
    except StopIteration:
        # 对于 PyTorch 1.5 中 nn.DataParallel 的兼容性

        # 在模块中查找张量属性的辅助函数
        def find_tensor_attributes(module):
            # 创建包含模块中所有张量属性的元组列表
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        # 生成器，用于获取模块中的命名成员
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        # 获取第一个元组
        first_tuple = next(gen)
        # 返回第一个元组中张量的数据类型
        return first_tuple[1].dtype


class DummyConfig:
    # 定义注意力层类型的列表
    attn_layers = ["local", "lsh", "local", "lsh", "local", "lsh"]
    # 定义 LSH 注意力层的分块长度
    lsh_attn_chunk_length = 64
    # 定义本地注意力层的分块长度
    local_attn_chunk_length = 64


def _get_min_chunk_len(config):
    # 从 hf_Reformer 中获取最小的分块长度
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        # 如果只有一种注意力层类型且为 "lsh"，返回 LSH 注意力层的分块长度
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == "local":
        # 如果只有一种注意力层类型且为 "local"，返回本地注意力层的分块长度
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(
        ["lsh", "local"]
    ):
        # 如果有两种注意力层类型且包含 "lsh" 和 "local"，返回两者中较小的分块长度
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    else:
        # 抛出未实现的错误，只能选择 'lsh' 和 'local' 作为注意力层类型
        raise NotImplementedError(
            f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
            "attn layer types from ['lsh', 'local'] only."
        )


def _stable_argsort(vector, dim):
    # 这个函数通过缩放向量使得 torch.argsort 变得稳定
    # torch.argsort 本身不是稳定的排序算法
    scale_offset = torch.arange(vector.shape[dim], device=vector.device).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    # 返回在指定维度上稳定排序后的索引
    return torch.argsort(scaled_vector, dim=dim)


def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(buckets):
    # 不需要梯度计算
    with torch.no_grad():
        # 基于哈希进行排序
        sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

        # 创建简单的索引来散布，以便进行撤消排序
        indices = (
            torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
            .view(1, 1, -1)
            .expand(sorted_bucket_idx.shape)
        )

        # 获取撤消排序
        undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
        undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

    # 返回排序后的索引和撤消排序的索引
    return sorted_bucket_idx, undo_sorted_bucket_idx


class CustomList1(list):
    # 自定义列表类，继承自 list
    def __call__(self, x):
        # 对列表中的每个处理器调用处理函数
        for processor in self:
            x = processor(x)
        return x

    def clear(self):
        # 这个方法防止 RestrictedListSubclassVariable 触发
        pass  # 不执行任何操作


class CustomList2(list):
    # 自定义列表类，继承自 list
    def __call__(self, x):
        # 对列表中的每个处理器调用处理函数
        for processor in self:
            x = processor(x)
        return x
    # 定义一个方法，计算实例对象自身的长度乘以 10，并返回结果
    def length_times_10(self):
        return len(self) * 10

    # 定义一个方法，向实例对象自身的列表中追加两次给定的元素 x
    def append_twice(self, x):
        self.extend([x, x])
# 合并处理条件列表的函数，将自定义列表与默认列表合并并检查类型冲突
def _merge_criteria_processor_list(default_list, custom_list):
    # 如果自定义列表为空，则直接返回默认列表
    if len(custom_list) == 0:
        return default_list
    # 遍历默认列表中的每个元素
    for default in default_list:
        # 遍历自定义列表中的每个元素
        for custom in custom_list:
            # 如果自定义元素的类型与默认元素相同，抛出数值错误异常
            if type(custom) is type(default):
                raise ValueError
    # 将自定义列表中的元素扩展到默认列表中
    default_list.extend(custom_list)
    # 返回合并后的列表
    return default_list


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, dropout) -> None:
        super().__init__()
        # 第一个线性层，输入维度为 d_model，输出维度为 dim_feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation  # 激活函数
        self.dropout1 = nn.Dropout(dropout)  # 第一个 Dropout 层
        # 第二个线性层，输入维度为 dim_feedforward，输出维度为 d_model
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)  # 第二个 Dropout 层

    def forward(self, x):
        # 前向传播函数，依次经过激活、第一个线性、第一个 Dropout、第二线性、第二 Dropout 层
        return self.dropout2(
            self.linear2(self.dropout1(self.activation(self.linear1(x))))
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        # 多头注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 第一个 LayerNorm 层
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # 第二个 LayerNorm 层
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)  # Dropout 层
        # FeedForwardLayer 类
        self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        # 使用自注意力机制和 LayerNorm 层进行前向传播
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        # 使用 FeedForwardLayer 和 LayerNorm 层进行前向传播
        x = self.norm2(x + self._ff_block(x))
        return x

    # 自注意力机制模块
    def _sa_block(self, x, attn_mask, key_padding_mask):
        # 执行自注意力机制
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        # 使用 Dropout 层进行正则化
        return self.dropout(x)

    # FeedForwardLayer 模块
    def _ff_block(self, x):
        return self.ff_block(x)


class MockModule(torch.nn.Module):
    def inner_fn(self, left, right):
        # 检查左右两个元组是否相等
        return tuple(left) == tuple(right)

    def fn(self, tensor):
        # 如果输入的张量类型为整数，则返回 False
        if type(tensor) is int:
            return False

        # 执行张量的加法操作（但未保存结果）
        torch.add(tensor, tensor)
        # 调用 inner_fn 方法，比较张量形状是否与 (1, 2, 3) 相同
        return self.inner_fn(tensor.shape, (1, 2, 3))


class IncByOne:
    def __init__(self, x):
        # 将输入值增加 1 并保存
        self.x = x + 1


class IncByTwo:
    def __init__(self, x):
        # 将输入值增加 2 并保存
        self.x = x + 2


class ReproTests(torch._dynamo.test_case.TestCase):
    pass  # 用于测试的空测试案例类，未添加任何额外的行为
    # 定义测试函数 test_do_paste_mask，用于测试 _do_paste_mask 函数的多种输入情况
    def test_do_paste_mask(self):
        # 清除 Torch 的动态计数器
        torch._dynamo.utils.counters.clear()
        # 创建编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用编译器优化 _do_paste_mask 函数
        opt__do_paste_mask = torch.compile(_do_paste_mask, backend=cnt)
        
        # 调用优化后的 _do_paste_mask 函数，传入不同的参数组合进行测试
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),     # 随机生成的张量
            torch.tensor([[0.0, 1, 2, 4]]) * 1,   # 经过张量运算得到的张量
            427,    # 整数参数
            640,    # 整数参数
            True,   # 布尔值参数
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 3,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 4,
            612,
            612,
            True,
        )
        opt__do_paste_mask(
            torch.randn(1, 1, 28, 28),
            torch.tensor([[0.0, 1, 2, 4]]) * 2,
            427,
            640,
            False,
        )
        
        # 断言验证计数器的帧数和操作数在预期范围内
        self.assertIn(cnt.frame_count, (5, 7))
        self.assertIn(cnt.op_count, (104, 106, 127))

    # 定义测试函数 test_convert_boxes_to_pooler_format，用于测试 convert_boxes_to_pooler_format 函数
    def test_convert_boxes_to_pooler_format(self):
        # 创建两组 Boxes 对象
        boxes1 = [
            Boxes(torch.arange(0, 8).reshape((2, 4))),
            Boxes(torch.arange(8, 16).reshape((2, 4))),
        ]
        boxes2 = [
            Boxes(torch.arange(16, 20).reshape((1, 4))),
            Boxes(torch.arange(20, 24).reshape((1, 4))),
        ]
        
        # 调用 convert_boxes_to_pooler_format 函数，将 Boxes 转换为池化器格式
        correct1 = convert_boxes_to_pooler_format(boxes1)
        correct2 = convert_boxes_to_pooler_format(boxes2)
        
        # 获取函数引用并使用优化器进行优化
        fn = convert_boxes_to_pooler_format
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        
        # 断言优化后的函数返回值与预期结果相同
        self.assertTrue(same(opt_fn(boxes1), correct1))
        self.assertTrue(same(opt_fn(boxes2), correct2))
        
        # 根据配置条件验证计数器的帧数和操作数是否符合预期
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """10""")
        else:
            self.assertExpectedInline(cnt.frame_count, """4""")
            self.assertExpectedInline(cnt.op_count, """14""")
    # 定义一个测试方法，用于验证箱子列表的长度计算函数
    def test_boxes_len(self):
        # 定义一个内部函数，计算箱子列表的长度
        def fn(boxes):
            return len(boxes) + boxes.__len__() + boxes.tensor

        # 创建一个包含数字序列的箱子对象
        boxes1 = Boxes(torch.arange(0, 8).reshape((2, 4)))
        # 创建一个用于计数编译次数的对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 对计数函数进行优化和断言验证
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        # 断言优化后的函数与预期结果相同
        self.assertTrue(same(opt_fn(boxes1), boxes1.tensor + 4.0))

        # 根据配置判断是否假设静态，默认情况下
        if torch._dynamo.config.assume_static_by_default:
            # 断言编译帧计数与预期结果相同
            self.assertExpectedInline(cnt.frame_count, """1""")
            # 断言操作计数与预期结果相同
            self.assertExpectedInline(cnt.op_count, """1""")
        else:
            # 断言编译帧计数与预期结果相同
            self.assertExpectedInline(cnt.frame_count, """1""")
            # 断言操作计数与预期结果相同
            self.assertExpectedInline(cnt.op_count, """6""")

    # 定义一个内部方法，用于测试Reformer模型
    def _reformer(self, nopython):
        # 创建一个随机输入张量
        input = torch.randn([1, 64, 256])
        # 创建一个Reformer编码器模型
        model = ReformerEncoder()
        # 设定随机种子
        torch.manual_seed(1337)
        # 使用深拷贝复制模型以确保正确性
        correct = copy.deepcopy(model)(input)
        # 创建一个用于计数编译次数的对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 再次设置相同的随机种子
        torch.manual_seed(1337)
        # 对模型进行优化和断言验证
        opt_model = torch._dynamo.optimize(cnt, nopython=nopython)(model)
        # 断言优化后的模型在相同输入上与正确结果相同
        self.assertTrue(same(opt_model(input), correct))
        # 返回计数对象
        return cnt

    # 标记需要CUDA支持的测试方法
    @requires_cuda
    # 定义一个测试方法，验证子Alpha标量复现问题
    def test_sub_alpha_scalar_repro(self):
        # 使用AOT eager模式编译函数
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sub(1, alpha=2)

        # 在CUDA设备上测试函数
        f(torch.ones(2, device="cuda", dtype=torch.float64))

    # https://github.com/pytorch/pytorch/issues/113010
    # 定义一个测试方法，验证非连续输出重载问题
    def test_out_overload_non_contiguous(self):
        # 定义一个函数，计算绝对值并将结果写入指定张量
        def f(x, y):
            return torch.abs(x, out=y.T)

        # 使用AOT eager模式编译函数
        f_compiled = torch.compile(f, backend="aot_eager")

        # 创建输入和参考输出张量
        x_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_ref = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        # 创建测试输入和测试输出张量
        x_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        y_test = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        # 计算参考输出
        out_ref = f(x_ref, y_ref)
        # 使用编译函数计算测试输出
        out_test = f_compiled(x_test, y_test)
        # 断言计算结果一致
        self.assertEqual(out_ref, out_test)
        # 断言输出张量一致
        self.assertEqual(y_ref, y_test)

    # https://github.com/pytorch/pytorch/issues/109053
    # 定义一个测试方法，验证视图数据类型重载问题
    def test_view_dtype_overload(self):
        # 定义一个函数，将输入张量视图转换为torch.int32类型
        def f(x):
            return x.view(torch.int32)

        # 使用AOT eager模式编译函数
        f_compiled = torch.compile(f, backend="aot_eager")

        # 创建一个requires_grad=True的输入张量
        x1 = torch.ones(4, requires_grad=True)
        # 计算参考输出
        out_ref = f(x1)
        # 使用编译函数计算测试输出
        out_test = f_compiled(x1)
        # 断言计算结果一致
        self.assertEqual(out_ref, out_test)

        # 创建一个requires_grad=False的输入张量
        x2 = torch.ones(4, requires_grad=False)
        # 计算参考输出
        out_ref = f(x2)
        # 使用编译函数计算测试输出
        out_test = f_compiled(x2)
        # 断言计算结果一致
        self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/90552
    # 定义一个测试方法，验证未命名参数重载问题
    def test_unnamed_parameter_overload(self):
        # 此处为待补充的代码
    # 定义测试函数，验证 leaf 是否需要梯度的问题
    def test_intermediate_leaf_requires_grad(self):
        # 定义一个函数 f，返回一个需要梯度的 leaf 和 leaf 的两倍
        def f(x):
            # 创建一个值全为1的张量 leaf，并指定需要计算梯度
            leaf = torch.ones(2, requires_grad=True)
            # 返回 leaf 和 leaf 的两倍
            return leaf, leaf * 2

        # 编译函数 f 到 AOT（Ahead-Of-Time）模式
        f_compiled = torch.compile(f, backend="aot_eager")
        # 创建一个形状为 (2, 2) 的浮点型张量 x
        x = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        # 调用函数 f，获取 leaf 和 out
        leaf, out = f(x)
        # 调用编译后的函数 f_compiled，获取 leaf_test 和 out_test
        leaf_test, out_test = f_compiled(x)
        # 对 out 中的所有元素求和，并进行反向传播
        out.sum().backward()
        # 对 out_test 中的所有元素求和，并进行反向传播
        out_test.sum().backward()
        # 断言 leaf 和 leaf_test 的梯度是否相等
        self.assertEqual(leaf.grad, leaf_test.grad)

    # https://github.com/pytorch/pytorch/issues/113263
    # 测试在跟踪过程中 unpack hooks 不会运行的情况
    def test_unpack_hooks_dont_run_during_tracing(self):
        # 定义一个简单的函数 f，返回 x 和 y 的乘积
        def f(x, y):
            return x * y

        # 编译函数 f 到 AOT（Ahead-Of-Time）模式
        f_compiled = torch.compile(f, backend="aot_eager")

        # 初始化 pack_count 和 unpack_count 计数器
        pack_count = 0
        unpack_count = 0

        # 定义 pack_hook，用于对输入 x 进行处理并增加 pack_count
        def pack_hook(x):
            nonlocal pack_count
            pack_count += 1
            return x

        # 定义 unpack_hook，用于对输入 x 进行处理并增加 unpack_count
        # 在编译过程中，unpack hook 不应该运行
        def unpack_hook(x):
            nonlocal unpack_count
            unpack_count += 1
            return x

        # 创建张量 x 和 y，其中 x 需要计算梯度，y 不需要
        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)

        # 使用 torch.autograd.graph.saved_tensors_hooks 包装 f_compiled 的调用
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            # 调用编译后的函数 f_compiled，获取 out_test
            out_test = f_compiled(x, y)
            # 断言 pack_count 为 1，unpack_count 为 0
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 0)
            # 对 out_test 中的所有元素求和，并进行反向传播
            out_test.sum().backward()
            # 断言 pack_count 为 1，unpack_count 为 1
            self.assertEqual(pack_count, 1)
            self.assertEqual(unpack_count, 1)

    # https://github.com/pytorch/pytorch/issues/113263
    # 测试可以禁用 unpack hooks 的情况
    def test_unpack_hooks_can_be_disabled(self):
        # 定义一个简单的函数 f，返回 x 和 y 的乘积
        def f(x, y):
            return x * y

        # 编译函数 f 到 AOT（Ahead-Of-Time）模式
        f_compiled = torch.compile(f, backend="aot_eager")

        # 创建张量 x 和 y，其中 x 需要计算梯度，y 不需要
        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)

        # 在编译过程中禁用 saved tensors hooks
        with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
            # 调用编译后的函数 f_compiled，获取 out_test
            out_test = f_compiled(x, y)
            # 对 out_test 中的所有元素求和，并进行反向传播
            out_test.sum().backward()

    # https://github.com/pytorch/pytorch/issues/113263
    # 测试在编译区域内禁用 unpack hooks 的情况
    def test_disabling_unpack_hooks_within_compiled_region(self):
        # 定义函数 g，对输入 z 进行加法操作并返回结果
        def g(z):
            # 在自动求导图中禁用 saved tensors hooks
            with torch.autograd.graph.disable_saved_tensors_hooks("hooks are disabled"):
                return z + 5

        # 定义一个函数 f，返回 x 和 y 的乘积，以及调用 g 处理后的结果
        def f(x, y):
            z = x * y
            return g(z)

        # 编译函数 f 到 AOT（Ahead-Of-Time）模式
        f_compiled = torch.compile(f, backend="aot_eager")

        # 创建张量 x 和 y，其中 x 需要计算梯度，y 不需要
        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=False)

        # 调用编译后的函数 f_compiled，获取 out_test
        out_test = f_compiled(x, y)
        # 对 out_test 中的所有元素求和，并进行反向传播
        out_test.sum().backward()

    # See https://github.com/pytorch/pytorch/issues/97745
    # 定义一个测试函数，用于测试 GAN 模型的反向传播是否能够成功进行第二次
    def test_gan_repro_trying_to_backward_through_the_graph_a_second_time(self):
        # 定义一个内部函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 创建一个大小为 (2, 2) 的张量 c，值全为 1
            c = torch.ones(2, 2)
            # 创建一个大小为 (2, 2) 的张量 d，值全为 1
            d = torch.ones(2, 2)
            # 计算矩阵 a 和 c 的乘积，得到张量 e
            e = torch.matmul(a, c)
            # 计算损失 g_loss，为 e 与 d 之间绝对值的平均值
            g_loss = torch.abs(e - d).mean()
            # 对 g_loss 进行反向传播
            g_loss.backward()
            # 计算虚假预测 fake_d_pred，为 b 与 e 的乘积，e 使用 detach() 方法使其不参与后续计算图
            fake_d_pred = torch.matmul(b, e.detach())
            # 计算损失 d_loss，为 fake_d_pred 的平均值
            d_loss = fake_d_pred.mean()
            # 对 d_loss 进行反向传播
            d_loss.backward()

        # 创建一个大小为 (2, 2) 的随机张量 a_ref，并设置 requires_grad=True
        a_ref = torch.randn(2, 2, requires_grad=True)
        # 创建一个大小为 (2, 2) 的随机张量 b_ref，并设置 requires_grad=True
        b_ref = torch.randn(2, 2, requires_grad=True)
        # 调用函数 f，传入 a_ref 和 b_ref，返回结果 out_ref
        out_ref = f(a_ref, b_ref)

        # 克隆并分离 a_ref，创建新的张量 a_test，并设置 requires_grad=True
        a_test = a_ref.clone().detach().requires_grad_(True)
        # 克隆并分离 b_ref，创建新的张量 b_test，并设置 requires_grad=True
        b_test = b_ref.clone().detach().requires_grad_(True)
        # 使用编译后的 torch 函数执行函数 f，传入 a_test 和 b_test，返回结果 out_test
        out_test = torch.compile(f, backend="aot_eager")(a_test, b_test)

        # 断言 out_ref 和 out_test 相等
        self.assertEqual(out_ref, out_test)
        # 断言 a_ref 和 a_test 的梯度相等
        self.assertEqual(a_ref.grad, a_test.grad)
        # 断言 b_ref 和 b_test 的梯度相等
        self.assertEqual(b_ref.grad, b_test.grad)

    # 定义一个测试函数，用于测试元组枚举作为字典键的情况
    # https://github.com/pytorch/pytorch/issues/111603
    def test_tuple_enum_as_key_dict(self):
        # 定义一个枚举类 MyEnum，包含一个成员 A 值为 "a"
        class MyEnum(Enum):
            A = "a"

        # 定义一个简单的神经网络模型类 SomeModel
        class SomeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 定义一个线性层，输入维度为 1，输出维度为 1
                self.linear = torch.nn.Linear(1, 1)

            # 前向传播函数，接受参数 x，返回线性层的计算结果
            def forward(self, x) -> torch.Tensor:
                return self.linear(x[MyEnum.A])

        # 创建一个字典 x，以 MyEnum.A 作为键，值为大小为 (8, 1) 的随机张量
        x = {MyEnum.A: torch.rand(8, 1)}
        # 实例化 SomeModel 类，创建模型 model_pytorch
        model_pytorch = SomeModel()
        # 编译模型 model_pytorch
        model = torch.compile(model_pytorch)
        # 执行两次模型计算
        model(x)
        y = model(x)
        # 断言 y 与 model_pytorch(x) 的结果相等
        self.assertEqual(y, model_pytorch(x))

    # 定义一个测试函数，用于测试嵌入层反向传播的广播分解
    def test_embedding_backward_broadcasting_decomp(self):
        # 定义一个函数 f，接受梯度输出 grad_output 和索引 indices 作为参数
        def f(grad_output, indices):
            # 定义权重数目 num_weights 为 10
            num_weights = 10
            # 定义填充索引 padding_idx 为 1
            padding_idx = 1
            # 设置是否按频率缩放梯度 scale_grad_by_freq 为 True
            scale_grad_by_freq = True
            # 调用 torch.ops.aten.embedding_dense_backward 函数，传入相关参数，返回结果
            return torch.ops.aten.embedding_dense_backward(
                grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
            )

        # 编译函数 f，并指定后端为 "aot_eager"
        f_compiled = torch.compile(f, backend="aot_eager")

        # 创建大小为 (2, 4, 3) 的全一张量 grad_output，数据类型为 torch.float16
        grad_output = torch.ones(2, 4, 3, dtype=torch.float16)
        # 创建大小为 (2, 4) 的全一张量 indices，数据类型为 torch.int64
        indices = torch.ones(2, 4, dtype=torch.int64)

        # 调用原始函数 f，传入 grad_output 和 indices，返回结果 out_ref
        out_ref = f(grad_output, indices)
        # 调用编译后的函数 f_compiled，传入 grad_output 和 indices，返回结果 out_test
        out_test = f_compiled(grad_output, indices)

        # 断言 out_ref 和 out_test 相等
        self.assertEqual(out_ref, out_test)

    # 定义一个测试函数，用于测试 Reformer 模型的评估
    def test_reformer_eval(self):
        # 使用 torch.no_grad 上下文管理器，执行函数 self._reformer，并返回结果 cnt
        with torch.no_grad():
            cnt = self._reformer(nopython=True)
        # 断言 cnt 的 frame_count 为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 cnt 的 op_count 为 11

    # 定义一个测试函数，用于测试 Reformer 模型的训练
    def test_reformer_train(self):
        # 使用 torch.enable_grad 上下文管理器，执行函数 self._reformer，并返回结果 cnt
        with torch.enable_grad():
            cnt = self._reformer(nopython=False)
        # 根据 torch._dynamo.config.inline_inbuilt_nn_modules 的值选择期望的 op_count
        expected_op_count = (
            """11""" if torch._dynamo.config.inline_inbuilt_nn_modules else """5"""
        )

        # 断言 cnt 的 frame_count 为 1
        self.assertExpectedInline(cnt.frame_count, """1""")
        # 断言 cnt 的 op_count 为 expected_op_count

    # 禁用动态形状验证的翻译验证
    @disable_translation_validation_if_dynamic_shapes
    # 定义一个测试函数，用于测试长形模型的分块处理
    def test_longformer_chunk(self):
        # 创建随机张量 input1，形状为 [1, 4096, 1]
        input1 = torch.randn([1, 4096, 1])
        # 创建随机张量 input2，形状为 [12, 4096, 64]
        input2 = torch.randn([12, 4096, 64])
        # 对 input1 和 input2 分别调用 longformer_chunk 函数，得到正确的处理结果
        correct1 = longformer_chunk(input1)
        correct2 = longformer_chunk(input2)
        # 将 longformer_chunk 函数赋值给变量 fn
        fn = longformer_chunk
        # 创建一个计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 optimize_assert 装饰器优化 fn 函数，并赋值给 opt_fn
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        # 断言经过优化后的结果与正确结果相同
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))
        self.assertTrue(same(opt_fn(input1), correct1))
        self.assertTrue(same(opt_fn(input2), correct2))

        # 根据配置条件断言计数器的帧数和操作数
        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """14""")
            else:
                self.assertExpectedInline(cnt.frame_count, """2""")
                self.assertExpectedInline(cnt.op_count, """4""")
        else:
            self.assertExpectedInline(cnt.frame_count, """2""")
            self.assertExpectedInline(cnt.op_count, """35""")

    # 定义一个测试函数，用于测试 Hugging Face T5 模型的前向传播
    def test_hf_t5_forward(self):
        # 创建随机张量 input，形状为 [1, 2048, 512]
        input = torch.randn([1, 2048, 512])
        # 创建 PartialT5 模型对象
        model = PartialT5()
        # 对 input 调用 model 的前向传播，得到正确的输出
        correct = model(input)
        # 创建一个计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 optimize_assert 装饰器优化 model 的前向传播方法，并赋值给 opt_model
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        # 断言经过优化后的模型对 input 的输出与正确结果相同
        self.assertTrue(same(opt_model(input), correct))

        # 根据配置条件断言计数器的帧数和操作数
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")

    # 定义一个测试函数，测试模型是否在跳过文件中
    def test_module_in_skipfiles(self):
        # 创建一个简单的线性模型 nn.Linear(10, 10)
        model = nn.Linear(10, 10)
        # 创建一个计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译模型，指定 backend 为 cnt，fullgraph 为 True，传入大小为 [5, 10] 的随机张量
        torch.compile(model, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        # 断言计数器的帧数和操作数
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    # 定义一个测试函数，测试函数是否在跳过文件中
    def test_function_in_skipfiles(self):
        # 创建一个计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 编译 torch.sin 函数，指定 backend 为 cnt，fullgraph 为 True，传入大小为 [5, 10] 的随机张量
        torch.compile(torch.sin, backend=cnt, fullgraph=True)(torch.randn([5, 10]))
        # 断言计数器的帧数和操作数
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    # 定义一个测试函数，测试动态形状的切片操作
    def test_slicing_dynamic_shape(self):
        # 定义一个函数 fn，接受一个参数 y
        def fn(y):
            # 创建一个长度为 8 的全为 1 的张量 x
            x = torch.ones(8)
            # 获取 y 的第一个元素作为索引 idx
            idx = y[0]
            # 对 x 进行切片操作，从索引 idx 到末尾，赋值给 out
            out = x[idx:]
            # 返回经过加 3 和乘 5 处理后的 out
            return (out + 3) * 5

        # 创建一个编译计数器对象 counter
        counter = torch._dynamo.testing.CompileCounter()
        # 使用 optimize 装饰器优化函数 fn，并赋值给 opt_fn
        opt_fn = torch._dynamo.optimize(counter)(fn)
        # 对 opt_fn 使用 torch.tensor([4]) 进行调用，并赋值给 out
        out = opt_fn(torch.tensor([4]))
        # 断言 out 的形状为 [4]
        self.assertEqual(list(out.shape), [4])

        # 断言计数器的操作数和帧数
        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 1)

        # 对 opt_fn 使用 torch.ones(10, dtype=torch.long) 进行调用，并赋值给 out
        out = opt_fn(torch.ones(10, dtype=torch.long))
        # 断言 out 的形状为 [7]
        self.assertEqual(list(out.shape), [7])
    # 定义测试函数 test_slicing_dynamic_shape_setitem，用于测试动态切片和赋值操作
    def test_slicing_dynamic_shape_setitem(self):
        # 定义嵌套函数 fn，接受输入长度和新的张量作为参数
        def fn(input_lengths: torch.Tensor, new_ones_1):
            # 获取 input_lengths 张量的第三个元素
            getitem_13 = input_lengths[3]
            # 在 new_ones_1 张量的第三个元素之后的所有元素上设置值为 0
            new_ones_1[(3, slice(getitem_13, None, None))] = 0
            # 将修改后的 new_ones_1 张量赋值给 setitem_13
            setitem_13 = new_ones_1
            # 返回包含 setitem_13 的元组
            return (setitem_13,)

        # 生成一个形状为 [10] 的随机张量 x，数据类型为 torch.int64
        x = torch.randn(10).to(dtype=torch.int64)
        # 生成一个形状为 [10, 204] 的随机张量 y
        y = torch.randn(10, 204)
        # 调用 fn 函数，记录其返回值为 ref
        ref = fn(x, y)
        # 使用 torch._dynamo.optimize("aot_eager") 对 fn 进行优化
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 使用优化后的函数 opt_fn 对 x, y 进行计算，记录其结果为 res
        res = opt_fn(x, y)
        # 断言 ref 和 res 结果相同
        self.assertTrue(same(ref, res))

    # 定义测试函数 test_chunk_reformer_ff，用于测试分块 Reformer 前馈传播模型
    def test_chunk_reformer_ff(self):
        # 生成一个形状为 [1, 4096, 256] 的随机张量 input
        input = torch.randn([1, 4096, 256])
        # 创建 ChunkReformerFeedForward 模型
        model = ChunkReformerFeedForward()
        # 对 input 应用模型，记录结果为 correct
        correct = model(input)
        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize_assert(cnt) 对 model 进行优化
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        # 断言优化后的模型 opt_model 对 input 的输出与 correct 相同
        self.assertTrue(same(opt_model(input), correct))
        
        # 断言帧计数 cnt.frame_count 等于 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作计数 cnt.op_count 小于等于 10
        self.assertLessEqual(cnt.op_count, 10)

    # 见: https://github.com/pytorch/pytorch/issues/80067
    # NB: 移除 expectedFailure 后，别忘了取消注释或者调整下面的 assertEqual
    @unittest.expectedFailure
    @torch._dynamo.config.patch(
        fake_tensor_propagation=True, capture_scalar_outputs=True
    )
    # 定义测试函数 test_maml_item_capture，用于测试 MAML 捕获项目
    def test_maml_item_capture(self):
        # 生成形状为 [5, 1, 28, 28] 的随机张量 a
        a = torch.randn(5, 1, 28, 28)
        # 生成形状为 [5]，数据类型为 torch.int64 的全零张量 b
        b = torch.zeros(5, dtype=torch.int64)
        # 生成形状为 [75, 1, 28, 28] 的随机张量 c
        c = torch.randn(75, 1, 28, 28)
        # 生成形状为 [75]，数据类型为 torch.int64 的全零张量 d
        d = torch.zeros(75, dtype=torch.int64)
        # 创建 PartialMaml 模型
        model = PartialMaml()
        # 对输入 a, b, c, d 应用模型，记录结果为 correct
        correct = model(a, b, c, d)
        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize(cnt) 对 model 进行优化
        opt_model = torch._dynamo.optimize(cnt)(model)
        # 循环 10 次，断言优化后的模型 opt_model 对 a, b, c, d 的输出与 correct 相同
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        # 如果 torch._dynamo.config.assume_static_by_default 为真
        # 则断言帧计数 cnt.frame_count 等于 "2"
        # 否则断言帧计数 cnt.frame_count 等于 "3"
        # TODO(jansel): 弄清楚为什么操作计数依赖于导入
        self.assertIn(cnt.op_count, (36, 35, 34, 29, 28, 27))

    # 见: https://github.com/pytorch/pytorch/issues/80067
    @torch._dynamo.config.patch(capture_scalar_outputs=False)
    # 定义测试函数 test_maml_no_item_capture，用于测试 MAML 无项目捕获
    def test_maml_no_item_capture(self):
        # 生成形状为 [5, 1, 28, 28] 的随机张量 a
        a = torch.randn(5, 1, 28, 28)
        # 生成形状为 [5]，数据类型为 torch.int64 的全零张量 b
        b = torch.zeros(5, dtype=torch.int64)
        # 生成形状为 [75, 1, 28, 28] 的随机张量 c
        c = torch.randn(75, 1, 28, 28)
        # 生成形状为 [75]，数据类型为 torch.int64 的全零张量 d
        d = torch.zeros(75, dtype=torch.int64)
        # 创建 PartialMaml 模型
        model = PartialMaml()
        # 对输入 a, b, c, d 应用模型，记录结果为 correct
        correct = model(a, b, c, d)
        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize(cnt) 对 model 进行优化
        opt_model = torch._dynamo.optimize(cnt)(model)
        # 循环 10 次，断言优化后的模型 opt_model 对 a, b, c, d 的输出与 correct 相同
        for _ in range(10):
            self.assertTrue(same(opt_model(a, b, c, d), correct))

        # 如果 torch._dynamo.config.assume_static_by_default 为真
        # 则断言帧计数 cnt.frame_count 等于 "4"
        # 否则断言帧计数 cnt.frame_count 等于 "5"
    def test_hf_model_output(self):
        # 创建模拟的模型输出对象，包含三个张量 a, b, c，每个都是大小为 10 的随机张量
        ex = ModelOutput(a=torch.randn(10), b=torch.randn(10), c=torch.randn(10))

        # 定义一个函数 fn1，接受一个字典 x 并返回 x["a"] + 1
        def fn1(x):
            return x["a"] + 1

        # 定义一个函数 fn2，接受一个对象 x 并返回 x.a + 1
        def fn2(x):
            return x.a + 1

        # 定义一个函数 fn3，接受一个对象 x 并返回 x 转换为元组后的第一个元素 + 1
        def fn3(x):
            return x.to_tuple()[0] + 1

        # 定义一个函数 fn4，接受一个对象 x 并返回 x 的第一个元素 + 1
        def fn4(x):
            return x[0] + 1

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        
        # 遍历上述四个函数，并对每个函数进行优化和断言测试
        for fn in (fn1, fn2, fn3, fn4):
            cnt.clear()
            # 使用优化和断言函数装饰当前的函数 fn
            opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
            # 断言优化后的函数对于模型输出 ex 的处理结果与 ex.a + 1 相同
            self.assertTrue(same(opt_fn(ex), ex.a + 1))
            # 断言编译计数器的帧计数为 1
            self.assertEqual(cnt.frame_count, 1)
            # 断言编译计数器的操作计数为 1
            self.assertEqual(cnt.op_count, 1)

    @disable_translation_validation_if_dynamic_shapes
    def test_create_rand_mask_from_inputs(self):
        # 定义测试函数的参数 args，包含不同形状和类型的张量及整数
        args = [
            torch.randn([1, 64, 64]),
            torch.randn([1, 64, 64]),
            torch.zeros([1, 12, 62, 3], dtype=torch.int64),
            12,
            3,
            1,
            4096,
            64,
        ]
        # 调用函数 create_rand_mask_from_inputs 生成正确的输出 correct
        correct = create_rand_mask_from_inputs(*args)
        # 获取函数 create_rand_mask_from_inputs 的引用
        fn = create_rand_mask_from_inputs

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用优化和断言函数装饰 create_rand_mask_from_inputs 函数
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        # 断言优化后的函数对于参数 args 的处理结果与 correct 相同
        self.assertTrue(same(opt_fn(*args), correct))
        # 根据动态形状的默认设置断言编译计数器的帧计数和操作计数
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """8""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """11""")

    def test_rng_state(self):
        # 定义一个函数 fn，用于测试随机数生成器状态的保存和恢复
        def fn():
            # 获取当前随机数生成器的状态
            state = torch.get_rng_state()
            # 生成一个包含 1000 个随机数的张量
            before = torch.rand(1000)
            # 恢复之前保存的随机数生成器状态
            torch.set_rng_state(state)
            # 再次生成 1000 个随机数的张量
            after = torch.rand(1000)
            # 返回两个生成的张量，用于后续的断言比较
            return before, after

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用优化函数装饰 fn 函数
        opt_fn = torch._dynamo.optimize(cnt)(fn)

        # 调用优化后的函数，获取返回的两个张量 before 和 after
        before, after = opt_fn()
        # 断言在优化后的函数中，生成的两个张量相等
        self.assertTrue(same(before, after))
        # 断言编译计数器的帧计数为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言编译计数器的操作计数为 2，代表两次 rand 操作
        self.assertEqual(cnt.op_count, 2)

        # 尝试导出函数 fn，期望导出失败并捕获特定的异常
        try:
            graph, _ = torch._dynamo.export(fn)()
            # 在这个情况下，预期的导出操作会抛出异常，如果没有抛出异常则测试失败
            self.fail("unexpected export success")
        except torch._dynamo.exc.Unsupported:
            # 捕获到预期的异常，测试通过
            pass

    def test_threading_local(self):
        # 导入 threading 模块，用于创建线程局部变量
        import threading

        # 创建一个线程局部变量对象 foo
        foo = threading.local()
        # 在 foo 对象中存储一个随机数张量
        foo.x = torch.rand(1)

        # 定义一个函数 f，接受一个输入 x，并在返回时将 x 与 foo.x 连接起来
        def f(x):
            return torch.cat([x, foo.x])

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用优化函数装饰 f 函数，并标记为不使用 JIT 编译
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)

        # 创建一个输入张量 inp，包含一个元素为 1 的张量
        inp = torch.ones(1)
        # 调用原始函数 f，并记录输出结果
        out = f(inp)
        # 调用优化后的函数 opt_f，并记录输出结果
        opt_out = opt_f(inp)
        # 断言优化后的输出与原始输出一致
        self.assertEqual(opt_out, out)
        # 断言编译计数器的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
    # 定义一个测试方法，测试 SequentialAppendList 类的行为
    def test_seq_append_list(self):
        # 创建一个形状为 (4, 10) 的随机张量 x
        x = torch.randn(4, 10)
        # 创建一个 SequentialAppendList 模型，包含两个线性层和两个 ReLU 激活层
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )
        # l1 是一个包含张量 x 的列表
        l1 = [x]
        # l2 也是一个包含张量 x 的列表，用于测试模型
        l2 = [x]
        # 使用模型处理输入 x 和列表 l1，获取正确的输出和处理后的列表
        correct, _ = model(x, l1)
        # 使用 CompileCounter 进行优化模型的计数器
        cnt = torch._dynamo.testing.CompileCounter()
        # 对模型进行优化，生成优化后的模型 opt_model
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        # 使用优化后的模型处理输入 x 和列表 l2，获取处理后的结果和列表
        result, l3 = opt_model(x, l2)
        # 断言处理后的结果与正确的输出相同
        self.assertTrue(same(result, correct))
        # 断言列表 l1 和 l2 是相同的对象
        self.assertTrue(same(l1, l2))
        # 断言列表 l2 和 l3 是相同的对象
        self.assertIs(l2, l3)
        # 断言计数器的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言计数器的操作数为 5
        self.assertEqual(cnt.op_count, 5)

    # 定义一个测试方法，测试 BatchNormAct2d 类的行为
    def test_batch_norm_act(self):
        # 创建一个形状为 (5, 1, 28, 28) 的随机张量 a
        a = torch.randn(5, 1, 28, 28)
        # 创建一个 BatchNormAct2d 模型，并设置为评估模式
        model = BatchNormAct2d(1).eval()
        # 获取模型处理输入 a 的正确输出
        correct = model(a)
        # 使用 CompileCounter 进行优化模型的计数器
        cnt = torch._dynamo.testing.CompileCounter()
        # 如果未启用特殊整数优化，则使用普通优化
        if not torch._dynamo.config.specialize_int:
            # 对模型进行优化，生成优化后的模型 opt_model
            opt_model = torch._dynamo.optimize(cnt)(model)
            # 断言优化后的模型处理输入 a 后的结果与正确输出相同
            self.assertTrue(same(opt_model(a), correct))
            return

        # 对模型进行优化，生成优化后的模型 opt_model
        opt_model = torch._dynamo.optimize_assert(cnt)(model)
        # 断言优化后的模型处理输入 a 后的结果与正确输出相同
        self.assertTrue(same(opt_model(a), correct))
        # 断言计数器的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言计数器的操作数为 2
        self.assertEqual(cnt.op_count, 2)

    # 定义一个测试方法，测试获取模型参数数据类型的函数
    def test_get_parameter_dtype(self):
        # 创建一个 SequentialAppendList 模型，包含一个线性层和一个 ReLU 激活层
        model = SequentialAppendList(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

        # 定义一个函数 fn，接收模型和输入张量 x，返回 x 加上随机噪声的结果
        def fn(model, x):
            return x + torch.randn(10, dtype=get_parameter_dtype(model))

        # 使用 CompileCounter 进行优化函数的计数器
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 进行优化，生成优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        # 断言优化后的函数处理模型和随机张量后的数据类型为 torch.float32
        self.assertEqual(opt_fn(model, torch.randn(10)).dtype, torch.float32)
        # 断言计数器的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言计数器的操作数为 2
        self.assertEqual(cnt.op_count, 2)

    # 定义一个测试方法，测试创建 torch.nn.Parameter 的行为
    def test_nn_parameter(self):
        # 定义一个函数 test_fn，创建一个形状为 (5, 5) 的参数张量 a
        def test_fn():
            a = torch.nn.Parameter(torch.randn(5, 5))
            # 断言张量 a 是 torch.nn.Parameter 类的实例
            self.assertTrue(isinstance(a, torch.nn.Parameter))
            return a

        # 使用 CompileCounter 进行优化函数的计数器
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 test_fn 进行优化，生成优化后的函数 opt_test_fn
        opt_test_fn = torch._dynamo.optimize(cnt)(test_fn)
        # 调用优化后的函数，获取输出结果 out
        out = opt_test_fn()
        # 断言输出结果 out 是 torch.nn.Parameter 类的实例
        self.assertTrue(isinstance(out, torch.nn.Parameter))

    # 定义一个测试方法，测试 torch.Size 对象的行为
    def test_Size(self):
        # 定义一个函数 test_fn，创建一个形状为 (4,) 的随机张量 a 和 torch.Size 对象 x
        def test_fn():
            a = torch.randn(4)
            x = torch.Size([1, 2, 3])
            # 断言对象 x 是 torch.Size 类的实例
            assert isinstance(x, torch.Size)
            # 引发图形中断，并检查 SizeVariable 对象的重建
            self.assertIsInstance(x, torch.Size)
            return a

        # 使用 CompileCounter 进行优化函数的计数器
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 test_fn 进行优化，生成优化后的函数 opt_test_fn
        opt_test_fn = torch._dynamo.optimize(cnt)(test_fn)
        # 调用优化后的函数 opt_test_fn
        opt_test_fn()

    # 详见 https://github.com/pytorch/pytorch/issues/100067
    def test_copy_weird_strides(self):
        # This test requires inductor's copy() decomp to preserve strides properly.
        # 定义测试函数，参数为一个张量 a
        def test_fn(a):
            # 创建一个全零张量 b，形状为 (48, 4, 256, 513)
            b = torch.zeros(48, 4, 256, 513)
            # 将张量 a 的部分数据复制到 b 的指定位置
            b[:, 0, 1:256, 1:256] = a
            # 将 b 变形为形状 (4, 12, 1024, 513) 的张量 c
            c = b.view(4, 12, 1024, 513)
            # 对 c 进行维度交换，变成形状 (4, 1024, 12, 513) 的张量 d
            d = c.transpose(2, 1)
            # 在 d 上每个元素加 1
            d.add_(1)
            # 返回修改后的张量 d
            return d

        # 定义张量的形状、步幅、数据类型、设备和是否需要梯度
        sh, st, dt, dev, rg = (
            (48, 255, 255),  # 形状
            (787968, 513, 1),  # 步幅
            torch.float16,  # 数据类型
            "cpu",  # 设备
            True,  # 是否需要梯度
        )
        # 生成一个具有指定形状和步幅的随机张量 a，并设置需要梯度
        a = rand_strided(sh, st, dt, dev).requires_grad_(rg)
        # 编译 test_fn 函数，使用特定的后端（aot_eager_decomp_partition）
        compiled_f = torch.compile(test_fn, backend="aot_eager_decomp_partition")
        # 分别用 a 调用原始函数和编译后的函数
        out1 = test_fn(a)
        out2 = compiled_f(a)
        # 断言两个输出张量相等
        self.assertEqual(out1, out2)

    def test_indexing_with_list(self):
        # 定义内部测试函数
        def test_fn():
            # 定义运行测试的嵌套函数，接受一个张量和索引列表
            def run_test(tensor, *idx):
                # 将张量转换为 NumPy 数组并进行形状比较
                npt = tensor.numpy()
                assert npt[idx].shape == tensor[idx].shape

            # 创建一个从 0 到 9 的张量 x
            x = torch.arange(0, 10)
            # 定义测试用例列表
            cases = [
                [None, None],  # 第一个测试用例
                [1, None],  # 第二个测试用例
            ]

            # 遍历测试用例列表
            for case in cases:
                # 使用 run_test 函数运行测试
                run_test(x, *case)

            # 返回形状为 (4,) 的随机张量
            return torch.randn(4)

        # 创建编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 test_fn 函数进行优化编译
        opt_test_fn = torch._dynamo.optimize(cnt)(test_fn)
        # 执行优化后的函数
        opt_test_fn()

    def test_reformer_min_chunk_len(self):
        # 定义函数 fn，接受一个配置参数 cfg
        def fn(cfg):
            # 创建一个形状为 10 的空张量 t
            t = torch.empty(10)
            # 用 _get_min_chunk_len 函数填充张量 t
            t.fill_(_get_min_chunk_len(cfg))
            # 返回张量 t 的第一个元素
            return t[0]

        # 创建一个虚拟的配置对象 cfg
        cfg = DummyConfig()
        # 创建编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化编译和断言优化
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        # 断言优化后的 fn 函数返回值等于 64
        self.assertEqual(opt_fn(cfg), 64)
        # 根据默认配置假设静态情况下，断言编译帧数为 1 和操作数为 3
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """3""")
        else:
            # 否则断言编译帧数为 1 和操作数为 4
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """4""")

    def test_reformer_sorting(self):
        # 创建一个形状为 [1, 12, 4096] 的全零张量 x
        x = torch.zeros([1, 12, 4096], dtype=torch.int64)
        # 使用 _get_sorted_bucket_idx_and_undo_sorted_bucket_idx 函数获取正确结果
        correct = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(x)
        # 将函数 fn 赋值为 _get_sorted_bucket_idx_and_undo_sorted_bucket_idx 函数
        fn = _get_sorted_bucket_idx_and_undo_sorted_bucket_idx

        # 创建编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化编译和断言优化
        opt_fn = torch._dynamo.optimize_assert(cnt)(fn)
        # 断言优化后的 fn 函数返回值与正确结果相同
        self.assertTrue(same(opt_fn(x), correct))
        # 根据默认配置假设静态情况下，断言编译帧数为 1 和操作数为 14
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """14""")
        else:
            # 否则断言编译帧数为 1 和操作数为 27
            self.assertExpectedInline(cnt.frame_count, """1""")
            self.assertExpectedInline(cnt.op_count, """27""")
    def test_recursive_map(self):
        # 定义测试用例，用于验证递归映射函数 _recursive_map 的功能
        # https://github.com/pytorch/torchdynamo/issues/132
        def _recursive_map(struct, batch_dim=0):
            # 遍历结构体中的每一个键值对
            for k, v in struct.items():
                # 如果值不为 None
                if v is not None:
                    # 如果值是字典类型，则递归调用 _recursive_map 函数处理
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        # 否则，保持原样
                        struct[k] = v

        # 定义一个简单的示例函数 toy_example，用于展示如何使用 _recursive_map
        def toy_example(a, b, v):
            # 对输入张量 a 进行变换
            x = a / (torch.abs(a) + 1)
            # 如果 v 不为 None，则进行递归映射操作
            if v is not None:
                _recursive_map(v)
            # 返回计算结果
            return x * b

        # 初始化一个计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 toy_example 函数进行优化，使用 cnt 进行性能计数
        opt_toy_example = torch._dynamo.optimize(cnt)(toy_example)
        # 调用优化后的 toy_example 函数，传入随机张量和一个嵌套字典作为参数
        opt_toy_example(
            torch.randn(10),
            torch.randn(10),
            {"layer0": {"memory_keys": torch.randn(10)}},
        )
        # 断言优化过程中的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言优化过程中的操作计数为 4
        self.assertEqual(cnt.op_count, 4)

    def test_issue114171(self):
        # 定义一个 CPU 设备对象
        device = torch.device("cpu")

        # 定义一个全连接神经网络函数 fcnn
        def fcnn(in_dim, out_dim, hidden_dim, activation=torch.nn.GELU):
            # 构建神经网络层列表
            layers = [
                torch.nn.Linear(in_dim, hidden_dim, device=device),
                activation(),
                torch.nn.Linear(hidden_dim, out_dim, device=device),
            ]
            # 返回一个序列化的神经网络模型
            return torch.nn.Sequential(*layers)

        # 定义一个测试模型类 testmodel
        class testmodel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 ModuleList 定义交互网络列表
                self.interaction_networks = torch.nn.ModuleList(
                    [fcnn(262, 1174, 400) for _ in range(4)]
                )

            # 定义一个交互函数 interact
            def interact(self, x, cycle):
                # 返回指定序号的交互网络的处理结果
                return self.interaction_networks[cycle](x)

        # 创建一个 testmodel 实例
        model = testmodel()
        # 使用 torch.compile 对模型的 interact 方法进行编译，开启完整图模式和动态模式
        forward_aot = torch.compile(
            model.interact, fullgraph=True, dynamic=True, backend="eager"
        )

        # 创建一个随机张量作为输入 x
        x = torch.rand([111, 262], device=device)
        # 调用编译后的 forward_aot 方法，传入 x 和一个指定的循环次数
        y2 = forward_aot(x, 2)  # 之前失败的情况

    def test_issue175(self):
        # 定义注意力头数和模型维度
        n_heads = 2
        d_model = 64
        # 创建一个 TransformerEncoderLayer 模型
        model = TransformerEncoderLayer(d_model, n_heads)
        # 创建一个随机输入张量
        inp = torch.randn(1, d_model)
        # 初始化一个计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对模型进行优化，使用 cnt 进行性能计数，启用无 Python 编译
        opt_model = torch._dynamo.optimize(cnt, nopython=True)(model)
        # 调用优化后的模型，传入随机输入张量 inp
        opt_model(inp)
        opt_model(inp)
        # 断言优化过程中的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)

        # 根据配置断言优化过程中的操作计数为 15 或 12
        self.assertEqual(
            15 if torch._dynamo.config.inline_inbuilt_nn_modules else 12, cnt.op_count
        )

    def test_exec_import(self):
        # 定义一个函数 fn1，使用 exec 导入 math 模块
        def fn1():
            exec("import math")

        # 定义一个函数 fn2，尝试使用 math 模块中的函数 sqrt(4)，捕获 NameError 异常
        def fn2():
            try:
                math.sqrt(4)
                return False
            except NameError:
                return True

        # 定义一个函数 fn3，调用 fn1 导入 math 模块，然后调用 fn2 进行检测
        def fn3():
            fn1()
            return fn2()

        # 断言 fn3 执行返回 True
        self.assertTrue(fn3())
        # 对 fn3 函数进行即时优化
        opt_fn3 = torch._dynamo.optimize("eager")(fn3)
        # 断言优化后的 fn3 函数执行返回 True
        self.assertTrue(opt_fn3())
    # 测试执行带通配符导入的情况
    def test_exec_wildcard_import(self):
        # 测试全局变量在不同帧之间不会传递
        def fn1():
            # 在当前帧中执行 "from torch import *" 导入操作
            exec("from torch import *")

        def fn2():
            # 使用 torch.zeros 创建一个张量 x
            x = torch.zeros(4)
            # 对张量 x 进行累加操作
            for i in range(5):
                x = x + i
            return x

        def fn3():
            # 调用 fn1 函数，执行导入操作
            fn1()
            # 调用 fn2 函数，执行张量操作
            return fn2()

        # 调用 fn3 函数，获取其返回值作为参考值
        ref = fn3()
        # 对 fn3 函数应用 torch._dynamo.optimize("eager") 优化
        opt_fn3 = torch._dynamo.optimize("eager")(fn3)
        # 执行优化后的函数，获取其返回值作为结果
        res = opt_fn3()
        # 使用 assertTrue 进行断言比较 ref 和 res 是否相同
        self.assertTrue(same(ref, res))

    # 测试在图中断开实例时的 with 语句使用情况
    def test_with_on_graph_break_inst(self):
        def reversible(x):
            # 打印 "Hello world"，导致图中断开，从而内联失败
            print("Hello world")
            return torch.sin(torch.cos(x))

        def fn(x):
            with torch.enable_grad():
                a = torch.sin(x)
                b = reversible(a)
                c = torch.sigmoid(b)
                c.sum().backward()
                return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch._dynamo.optimize("eager")(fn)
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    # 测试嵌套图中断时的 with 语句使用情况
    def test_with_on_graph_break_nested(self):
        def reversible(x):
            # 调用 torch._dynamo.graph_break()，导致图中断，内联失败
            torch._dynamo.graph_break()
            return torch.sin(torch.cos(x))

        def fn(x):
            # 嵌套上下文管理器之前的情况
            with torch.no_grad():
                with torch.enable_grad():
                    a = torch.sin(x)
                    b = reversible(a)
                    c = torch.sigmoid(b)
                    c.sum().backward()
                    return x.grad

        x = torch.randn(3, requires_grad=True)
        x.grad = None
        with torch.no_grad():
            ref = fn(x)

        x.grad = None
        opt_fn = torch._dynamo.optimize("eager")(fn)
        with torch.no_grad():
            res = opt_fn(x)
        self.assertTrue(same(ref, res))

    # 测试在图中断后，梯度模式保持正确状态
    # 参考：https://github.com/pytorch/torchdynamo/issues/1446
    def test_grad_mode_carrying_correct_state_after_graph_break(self):
        def fn(x):
            with torch.no_grad():
                y = x * 3
                # 打印 "Break"，引发图中断
                print("Break")
                z = x + 2
            return y, z

        x = torch.randn(3, requires_grad=True)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        y, z = opt_fn(x)
        # 断言 y 和 z 是否不需要梯度计算
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)
    def test_abc_setattr(self):
        # 测试确保在 __setattr__ 调用时正确地中止执行

        # TODO: 未确保正确推断 ABC 类作为 ClassVariables
        # （不测试修复 'super()' 的问题）

        class BaseModule(torch.nn.Module, ABC):
            def blah(self, x):
                return x + 1

        class Derived(BaseModule):
            def __setattr__(self, name, value) -> None:
                super().__setattr__(name, value)

            def forward(self, x):
                # 预期在 __setattr__ 处发生图中断
                self.foo = 0
                return self.blah(x)

            def blah(self, x):
                return super().blah(x)

        x = torch.randn(3, requires_grad=True)
        mod = Derived()
        opt_mod = torch._dynamo.optimize("eager")(mod)
        opt_mod(x)

        # 不确定这个测试在测试什么。先前在 __dict__ 上的图中断，所以计数器 >= 2。
        # 使用 __dict__ 支持后，就不会有图中断。
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)
        self.assertGreaterEqual(torch._dynamo.utils.counters["frames"]["total"], 1)

    @torch._dynamo.config.patch("suppress_errors", True)
    def test_guard_fail_tensor_bool(self):
        @torch._dynamo.disable(recursive=False)
        def fn():
            condition_shape = (5, 5)
            dtypes = (torch.bool,)
            shapes = (
                (),
                (5,),
                (1, 5),
            )

            tensors = [
                torch.empty(shape, dtype=dtype).fill_(17)
                for shape, dtype in itertools.product(shapes, dtypes)
            ]

            x_vals = (5.0, *tensors)
            y_vals = (6.0, *tensors)

            @torch._dynamo.disable
            def get_expected(condition, x, y):
                x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                return torch.from_numpy(
                    np.where(condition.cpu().numpy(), x_np, y_np)
                ).to(common_dtype)

            for x, y in zip(x_vals, y_vals):
                condition = torch.empty(*condition_shape, dtype=torch.bool).bernoulli_()
                common_dtype = torch.result_type(x, y)

                def check_equal(condition, x, y):
                    # NumPy 会主动提升为 double，因此将输出强制转换为正确的 dtype
                    expected = get_expected(condition, x, y)
                    result = torch.where(condition, x, y)
                    assert torch.allclose(expected, result)

                check_equal(condition, x, y)
                check_equal(condition, y, x)

        fn()
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_fn()
    def test_guard_fail_nested_tuple(self):
        def fn(args):
            # 返回一个空的张量和 args[0] 的两倍
            return torch.ones(()), args[0] * 2

        # 对 args[1][0] 和 args[1][1] 进行张量检查
        args1 = (torch.ones(1), (torch.ones(1), torch.ones(1)))
        args2 = (torch.ones(1), torch.ones(1))
        # 使用 torch._dynamo.optimize("eager") 优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 对 args1 和 args2 应用优化后的函数
        ref = opt_fn(args1)
        res = opt_fn(args2)

        self.assertTrue(same(ref, res))

    def test_nullcontext1(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            # 对 x 应用正弦函数
            x = x.sin()
            with ctx:
                # 在上下文 ctx 中，对 x 应用余弦函数
                x = x.cos()
            # 再次对 x 应用正弦函数
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext()), y.sin().cos().sin()))

    def test_nullcontext2(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, ctx):
            # 对 x 应用正弦函数
            x = x.sin()
            with ctx():
                # 在上下文 ctx 中，对 x 应用余弦函数
                x = x.cos()
            # 再次对 x 应用正弦函数
            x = x.sin()
            return x

        y = torch.randn(10)
        self.assertTrue(same(fn(y, contextlib.nullcontext), y.sin().cos().sin()))

    def test_no_grad_inline(self):
        @torch.no_grad()
        def a(x):
            # 返回 x 的正弦函数值
            return x.sin()

        @torch.compile(backend="eager", fullgraph=True)
        def b(x):
            # 返回 a(x) 的余弦函数值
            return a(x).cos()

        y = torch.randn(10)
        self.assertTrue(same(b(y), y.sin().cos()))

    def test_longtensor_list(self):
        for partition in [0, 5, 10]:

            @torch._dynamo.disable
            def rand_gen():
                rand_vals = [random.randint(5, 10) for _ in range(10)]
                # 混合了张量和 np.array 的列表
                return list(np.array(rand_vals[:partition])) + [
                    torch.tensor(val) for val in rand_vals[partition:]
                ]

            def fn(x):
                random_list = rand_gen()
                # 创建一个 LongTensor
                z = torch.LongTensor(random_list)
                return x * z

            x = torch.ones(10) * 2

            random.seed(0)
            ref0 = fn(x)
            ref1 = fn(x)

            random.seed(0)
            # 使用 torch._dynamo.optimize("eager") 优化 fn 函数
            opt_fn = torch._dynamo.optimize("eager")(fn)
            res0 = opt_fn(x)
            res1 = opt_fn(x)

            self.assertTrue(same(ref0, res0))
            self.assertTrue(same(ref1, res1))

    def test_primtorch(self):
        @torch._dynamo.optimize("eager")
        def fn(x):
            # 调用 torch._refs.abs 函数
            torch._refs.abs(x)

        fn(torch.randn(3))

    @unittest.expectedFailure
    # 在 inline_call 中出现的断言失败信息
    # [('inline in skipfiles: bind ...python3.10/inspect.py', 1)]
    def test_primtorch_no_graph_break(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            # 调用 torch._refs.abs 函数
            torch._refs.abs(x)

        fn(torch.randn(3))

    def test_torch_tensor_ops_no_graph_break(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            # 调用 torch.Tensor.abs_ 方法
            torch.Tensor.abs_(x)

        fn(torch.randn(3))
    @unittest.skipIf(
        not isinstance(torch.ops.aten.abs, torch._ops.OpOverloadPacket),
        "old pt doesn't work",
    )
    # 定义一个测试函数，如果 torch.ops.aten.abs 不是 OpOverloadPacket 类型，则跳过测试
    def test_torch_ops_aten(self):
        # 定义一个使用 torch.ops.aten.absolute 运算的函数优化器装饰器
        @torch._dynamo.optimize("eager", nopython=True)
        # 定义一个函数 fn，对输入 x 计算绝对值
        def fn(x):
            return torch.ops.aten.absolute(x)

        # 对 torch.randn(3) 执行函数 fn
        fn(torch.randn(3))

    # 测试 GELU 激活函数的内联操作
    def test_hf_gelu_inline(self):
        # 定义一个 GELUActivation 类，继承自 nn.Module
        class GELUActivation(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.functional.gelu

            def forward(self, input):
                return self.act(input)

        # 定义一个使用函数优化器装饰器的函数 fn，应用 GELUActivation
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            return GELUActivation()(x)

        # 生成一个形状为 (10,) 的随机张量 y
        y = torch.randn(10)
        # 断言 fn(y) 和 nn.functional.gelu(y) 的结果相同
        self.assertTrue(same(fn(y), nn.functional.gelu(y)))

        # 定义一个使用函数优化器装饰器的函数 fn_returns
        @torch._dynamo.optimize("eager", nopython=True)
        def fn_returns(x):
            return GELUActivation(), x + 1

        # 执行 fn_returns(y)，获取返回值 act 和 _
        act, _ = fn_returns(y)
        # 断言 act 是 GELUActivation 类型的实例
        self.assertIsInstance(act, GELUActivation)
        # 断言 act.act 是 nn.functional.gelu 函数
        self.assertIs(act.act, nn.functional.gelu)
        # 断言 act 具有 "_buffers" 属性，以验证 __init__ 方法被调用
        self.assertTrue(hasattr(act, "_buffers"))

    # 测试 Dropout 操作的内联实现
    def test_dropout_inline(self):
        # 定义一个使用函数优化器装饰器的函数 fn，应用 Dropout 操作
        @torch._dynamo.optimize("eager")
        def fn(x):
            return torch.nn.Dropout(0.1)(x)

        # 生成一个形状为 (10,) 的随机张量 y
        y = torch.randn(10)
        # 设置随机种子
        torch.manual_seed(1337)
        # 计算使用 nn.functional.dropout(y, 0.1) 的结果作为参考
        ref = nn.functional.dropout(y, 0.1)
        # 再次设置相同的随机种子
        torch.manual_seed(1337)
        # 执行 fn(y) 得到结果 res
        res = fn(y)
        # 断言 res 和 ref 的结果相同
        self.assertTrue(same(ref, res))

    # 测试使用布尔掩码进行 setitem 操作的差异
    def test_setitem_boolean_mask_diff(self):
        # 定义一个函数 fn，接受输入 x, b, y，对 x 执行 setitem 操作
        def fn(x, b, y):
            x = x.clone()
            x[b] = y
            return x

        # 对 fn 应用 "aot_eager" 函数优化器装饰器
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 生成一个形状为 (4,) 的随机张量 x，要求梯度
        x = torch.randn(4, requires_grad=True)
        # 生成一个布尔张量 b
        b = torch.tensor([True, False, True, False])
        # 生成一个形状为 (2,) 的随机张量 y，要求梯度
        y = torch.randn(2, requires_grad=True)
        # 执行优化后的函数 opt_fn(x, b, y)
        opt_fn(x, b, y)

    # 测试使用元组布尔掩码进行 setitem 操作的差异
    def test_setitem_tuple_boolean_mask_diff(self):
        # 定义一个函数 fn，接受输入 x, b, y，对 x 执行 setitem 操作
        def fn(x, b, y):
            x = x.clone()
            x[:, b] = y
            return x

        # 对 fn 应用 "aot_eager" 函数优化器装饰器
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 生成一个形状为 (8, 4) 的随机张量 x，要求梯度
        x = torch.randn(8, 4, requires_grad=True)
        # 生成一个布尔张量 b
        b = torch.tensor([True, False, True, False])
        # 生成一个形状为 (2,) 的随机张量 y，要求梯度
        y = torch.randn(2, requires_grad=True)
        # 执行优化后的函数 opt_fn(x, b, y)
        opt_fn(x, b, y)

    # 测试 torch.Tensor.abs_ 函数的优化
    def test_torch_tensor_ops(self):
        # 定义一个函数 fn，对输入 x 执行 torch.Tensor.abs_ 操作
        def fn(x):
            return torch.Tensor.abs_(x)

        # 生成一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 对 fn 应用 "eager" 函数优化器装饰器
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 执行 fn(x) 得到结果 y
        y = fn(x)
        # 执行优化后的函数 opt_fn(x) 得到结果 y_
        y_ = opt_fn(x)
        # 断言 y 和 y_ 的结果相同
        self.assertTrue(same(y, y_))
    def test_guard_ordering_shape_fail(self):
        # 创建一个 MockModule 对象
        m = MockModule()
        # 对 m 应用 eager 优化
        opt_m = torch._dynamo.optimize("eager")(m)
        # 调用 opt_m 的 fn 方法，传入一个形状为 (5, 5) 的全一张量
        opt_m.fn(torch.ones((5, 5)))
        # 再次调用 opt_m 的 fn 方法，传入一个整数 -3
        opt_m.fn(-3)

    def test_tensor_isinstance_tuple(self):
        @torch._dynamo.optimize("eager")
        def fn():
            # 创建一个形状为 (5, 5) 的全一张量 t
            t = torch.ones(5, 5)
            # 如果 t 不是 int 或者 torch.Tensor 的实例，则抛出 ValueError 异常
            if not isinstance(t, (int, torch.Tensor)):
                # 构造错误消息字符串，格式化类型信息和期望的类型元组
                msg = str.format(
                    "{0} is not an instance of {1}",
                    type(t),
                    (int, torch.Tensor),
                )
                # 抛出 ValueError 异常，带有错误消息
                raise ValueError(msg)
            return True

        # 调用 fn 函数
        fn()

    def test_isinstance_dtype(self):
        @torch._dynamo.optimize("eager", nopython=True)
        def fn(x):
            # 检查 torch.bfloat16 是否是 torch.dtype 的实例，但没有使用结果
            isinstance(torch.bfloat16, torch.dtype)
            return x

        # 调用 fn 函数，传入一个形状为 (3,) 的随机张量
        fn(torch.randn(3))

    def test_isinstance_storage(self):
        @torch._dynamo.optimize("eager")
        def fn(x):
            # 创建一个字节序列 f
            f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
            # 从字节序列 f 创建一个 torch.BoolStorage 对象
            bools = torch.BoolStorage.from_buffer(f, "big")
            # 断言 bools 是 torch.BoolStorage 的实例，用于验证结果
            assert isinstance(bools, torch.BoolStorage)
            return x

        # 调用 fn 函数，传入一个形状为 (3,) 的随机张量
        fn(torch.randn(3))

    def test_issue111522(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x, y):
            # 返回 x 和 y.a 的和，y 为类 A 的实例
            return x + y.a

        class A:
            a = 2

        # 使用 torch.zeros(2) 和 A() 调用 f 函数，期望返回全为 2.0 的张量
        self.assertEqual(f(torch.zeros(2), A()), torch.full([2], 2.0))

        # 删除类 A 的属性 a
        del A.a

        # 当缺少属性时，图形会中断，期望引发 torch._dynamo.exc.Unsupported 异常
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            # 使用 torch.zeros(2) 和 A() 调用 f 函数
            f(torch.zeros(2), A())

    def test_dict_list_values(self):
        def inner_fn(args):
            # 返回 args 中每个元素的第二个元素的形状
            return [x[1].shape for x in args]

        @torch._dynamo.optimize("eager")
        def fn(tensors):
            # 使用 itertools.count() 和 tensors["args"] 的组合调用 inner_fn 函数
            return inner_fn(zip(itertools.count(), tensors["args"]))

        # 第一次调用 fn 函数，传入包含三个张量的字典
        fn({"args": [torch.ones(5, 5), torch.ones(5, 6), torch.ones(5, 7)]})
        # 第二次调用 fn 函数，传入包含一个张量的字典
        fn({"args": [torch.ones(5, 5)]})

    def test_dict_iter(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                # 创建一个字典 z
                z = {"my": 1, "const": 2, "dict": 3, "variable": 4}
                tot = 0
                # 遍历字典 z 中的键，并将值累加到 tot 中
                for key in z:
                    tot += z[key]

                return tot

        # 创建一个形状为 [1] 的张量 x
        x = torch.tensor([0])
        # 创建 MyMod 的实例 model
        model = MyMod()
        # 对 model 应用 eager 和 nopython 优化
        opt_model = torch._dynamo.optimize("eager", nopython=True)(model)
        # 使用 x 调用 opt_model 的 forward 方法，得到结果 y
        y = opt_model(x)

        # 验证 y 是否等于 10
        self.assertEqual(y, 10)
    # 定义一个测试函数，用于测试 torch.sort 函数的输出
    def test_sort_out(self):
        # 定义数据类型和设备
        dtype = torch.float32
        device = "cpu"

        # 定义内部函数 fn
        def fn():
            # 创建一个随机张量，并选择其中的第一列
            tensor = torch.randn((3, 5), dtype=dtype, device=device)[:, 0]
            # 创建两个标量张量，用于接收排序后的值和索引
            values1 = torch.tensor(0, dtype=dtype, device=device)
            indices1 = torch.tensor(0, dtype=torch.long, device=device)
            # 对 tensor 进行排序，并将结果写入 values1 和 indices1
            torch.sort(tensor, out=(values1, indices1))
            # 断言 values1 和 indices1 的步长为 (1,)
            self.assertEqual(values1.stride(), (1,))
            self.assertEqual(indices1.stride(), (1,))

        # 调用内部函数 fn
        fn()
        # 对 fn 进行优化，使用 torch._dynamo.optimize("eager") 进行装饰
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 执行优化后的函数
        opt_fn()

    # 定义另一个测试函数，测试 torch.sort 函数在 Module 中的应用
    def test_sort_out2(self):
        # 定义一个继承自 torch.nn.Module 的子类
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册两个缓冲区，分别用于存储排序后的张量和其索引
                self.register_buffer("sorted", torch.ones(4, 4))
                self.register_buffer("indices", torch.ones(4, 4, dtype=torch.long))

            # 定义 forward 方法，接收输入 x
            def forward(self, x):
                # 对输入张量 x 进行排序，并将结果写入 self.sorted 和 self.indices
                torch.sort(x, out=(self.sorted, self.indices))
                # 返回元组，包含原始输入 x、排序后的张量和其索引
                return (x + 1, self.sorted, self.indices)

        # 创建一个随机张量作为输入
        x = torch.randn(4, 4)
        # 实例化 MyModule 类
        m = MyModule()
        # 调用 MyModule 实例，计算其输出作为参考结果
        ref = m(x)
        # 对 MyModule 实例进行优化，使用 torch._dynamo.optimize("eager") 进行装饰
        opt_m = torch._dynamo.optimize("eager")(m)
        # 计算优化后 MyModule 的输出
        res = opt_m(x)
        # 断言优化前后的输出是否一致
        self.assertTrue(same(ref, res))

    # 定义测试函数，测试 torch.sigmoid 函数的输出
    def test_sigmoid_out(self):
        # 定义数据类型和设备
        dtype = torch.float32
        device = "cpu"

        # 定义内部函数 fn
        def fn():
            # 创建一个随机张量作为输入
            inp = torch.randn((3, 5), dtype=dtype, device=device)
            # 创建一个标量张量，用于接收 sigmoid 函数的输出
            out1 = torch.tensor(0, dtype=dtype, device=device)
            # 对输入张量进行 sigmoid 操作，并将结果写入 out1
            torch.sigmoid(inp, out=out1)
            # 断言 out1 的元素数目为 15
            self.assertEqual(out1.numel(), 15)

        # 调用内部函数 fn
        fn()
        # 对 fn 进行优化，使用 torch._dynamo.optimize("eager") 进行装饰
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 执行优化后的函数
        opt_fn()

    # 定义另一个测试函数，测试 torch.sigmoid 函数在 Module 中的应用
    def test_sigmoid_out2(self):
        # 定义一个继承自 torch.nn.Module 的子类
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个缓冲区，用于存储 sigmoid 函数的输出
                self.register_buffer("base", torch.ones(4, 4))

            # 定义 forward 方法，接收输入 x
            def forward(self, x):
                # 对输入张量 x 进行 sigmoid 操作，并将结果写入 self.base
                torch.sigmoid(x, out=self.base)
                # 返回 x 与 self.base 相加后的结果
                return x + self.base

        # 创建一个随机张量作为输入
        x = torch.randn(4, 4)
        # 实例化 MyModule 类
        m = MyModule()
        # 调用 MyModule 实例，计算其输出作为参考结果
        ref = m(x)
        # 对 MyModule 实例进行优化，使用 torch._dynamo.optimize("eager") 进行装饰
        opt_m = torch._dynamo.optimize("eager")(m)
        # 计算优化后 MyModule 的输出
        res = opt_m(x)
        # 断言优化前后的输出是否一致
        self.assertTrue(same(ref, res))

    # 定义测试函数，测试切片操作在列表可变对象上的影响
    def test_slice_into_list_mutable(self):
        # 定义一个继承自 torch.nn.Module 的子类
        class Mod(torch.nn.Module):
            # 定义 forward 方法，接收列表参数 listy
            def forward(self, listy):
                # 对列表 listy 进行切片操作，获取索引为 3 到 5 的子列表 x
                x = listy[3:5]
                # 循环 10 次
                for i in range(10):
                    # 生成绝对值大于 1 的随机张量 z
                    z = torch.abs(torch.randn(10)) + 1
                    # 将 z 赋值给 x 的第一个元素
                    x[0] = z
                # 返回列表 x
                return x

        # 实例化 Mod 类
        m = Mod()
        # 创建一个包含 10 个相同张量的列表 listy
        listy = [torch.randn(10)] * 10

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 Mod 类进行优化，使用 torch._dynamo.optimize 进行装饰，并开启 nopython 模式
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        # 调用优化后的 Mod 实例的 forward 方法
        opt_m.forward(listy)

        # 断言编译计数器的 frame_count 属性为 1
        self.assertEqual(cnt.frame_count, 1)

    # torch._dynamo.config.patch 的装饰器，用于配置捕获标量输出
    def test_issue111918(self):
        # 创建编译计数器对象
        cnt = CompileCounter()

        # 使用 torch.compile 装饰器编译函数 fn，使用指定的后端并启用动态模式
        @torch.compile(backend=cnt, dynamic=True)
        def fn(x):
            # 对输入张量 x 进行加法操作
            x = x + 1
            # 将张量 x 转换为标量，并赋值给变量 y
            y = x.item()
            # 如果 y 大于 2，则返回 x 的两倍
            if y > 2:
                return x * 2
            else:
                return x * 3

        # 创建输入张量 x
        x = torch.tensor([3.0])
        # 调用 fn 函数
        fn(x)
        # 断言编译计数器的帧数为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言编译计数器的操作数为 4
        self.assertEqual(cnt.op_count, 4)

        # 重置 Torch 动态图
        torch._dynamo.reset()
        # 使用 torch.compile 函数优化 fn 函数，使用完整图并指定后端为 "eager"
        fn = torch.compile(fn, fullgraph=True, backend="eager")
        # 断言调用 fn 函数时会抛出 torch._dynamo.exc.UserError 异常
        with self.assertRaises(torch._dynamo.exc.UserError):
            fn(x)

    def test_vdd_duplicate_error(self):
        # 定义函数 fn，接受两个参数 a 和 dt
        def fn(a, dt):
            # 获取 dt 的键列表
            keys = list(dt._jt_dict.keys())
            # 计算 torch.cos 对 dt 的第一个键值的余弦值
            p = torch.cos(dt._jt_dict[keys[0]]._value)
            # 计算 torch.sin 对 a 的正弦值
            q = torch.sin(a)
            # 计算 torch.sigmoid 对 dt 的第一个键值的值的 sigmoid 值
            r = torch.sigmoid(dt._jt_dict[keys[0]]._value)
            # 返回 p + q + r 的结果
            return p + q + r

        # 定义类 Value，初始化时生成一个包含随机数的张量 _value
        class Value:
            def __init__(self):
                self._value = torch.randn(4)

        # 定义类 Sample，初始化时生成一个空字典 _jt_dict，并添加一个名为 "POSITION_ID" 的键值对
        class Sample:
            def __init__(self):
                self._jt_dict = {}
                self._jt_dict["POSITION_ID"] = Value()

        # 创建张量 a，包含随机数
        a = torch.randn(4)
        # 创建 Sample 类的实例 sample
        sample = Sample()

        # 调用 fn 函数，传入 a 和 sample 作为参数，并将结果保存到 ref
        ref = fn(a, sample)

        # 使用 torch._dynamo.optimize 函数优化 fn 函数，使用 "eager" 模式并启用 nopython
        optimized_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        # 调用优化后的函数 optimized_fn，传入 a 和 sample 作为参数，并将结果保存到 res
        res = optimized_fn(a, sample)

        # 断言 ref 和 res 的结果相同
        self.assertTrue(same(ref, res))

    def test_specialized_stride(self):
        # 定义函数 f
        def f():
            # 创建一个大小为 4 的空张量 e
            e = torch.empty(4)
            # 对张量 e 进行切片操作，步长为 2，赋值给变量 x
            x = e[::2]
            # 返回张量 x 的步长
            return x.stride()

        # 断言函数 f 的结果与优化后的 f 函数的结果相同
        self.assertEqual(f(), torch._dynamo.optimize("eager")(f)())

    def test_out_none(self):
        # https://github.com/pytorch/pytorch/issues/92814
        # 定义函数 fn，接受一个输入参数 input
        def fn(input):
            # 使用 torch.nn.functional.normalize 函数对 input 进行归一化，dim=0，out=None
            return torch.nn.functional.normalize(input, dim=0, out=None)

        # 创建一个大小为 1 的随机张量 x
        x = torch.rand([1])
        # 断言函数 fn 的结果与优化后的 fn 函数的结果相同
        self.assertEqual(fn(x), torch._dynamo.optimize("eager")(fn)(x))

    def test_multi_import(self):
        # 如果没有安装 detectron2，跳过该测试
        if not has_detectron2():
            raise unittest.SkipTest("requires detectron2")

        # 使用 torch._dynamo.optimize 装饰器优化函数 to_bitmasks，使用 "eager" 模式并禁用 nopython
        @torch._dynamo.optimize("eager", nopython=True)
        def to_bitmasks(boxes):
            # 从 detectron2.layers.mask_ops 模块导入 _paste_masks_tensor_shape 和 paste_masks_in_image 函数
            from detectron2.layers.mask_ops import (
                _paste_masks_tensor_shape,
                paste_masks_in_image,
            )

            # 如果 paste_masks_in_image 和 _paste_masks_tensor_shape 都不为空，则返回 boxes + 1
            if (
                paste_masks_in_image is not None
                and _paste_masks_tensor_shape is not None
            ):
                return boxes + 1

        # 断言调用 to_bitmasks 函数时所有元素是否均为 1
        self.assertTrue((to_bitmasks(torch.zeros(10)) == torch.ones(10)).all())

    def test_multi_dot_import(self):
        # 定义函数 fn1，接受一个参数 x
        def fn1(x):
            # 返回张量 x 的正弦值
            return torch.sin(x)

        # 定义函数 fn
        def fn(x):
            # 导入 torch.fx 模块
            import torch.fx

            # 对函数 fn1 进行符号化跟踪，结果赋值给 _
            _ = torch.fx.symbolic_trace(fn1)
            # 返回 x 的两倍
            return x * 2

        # 创建一个大小为 10 的随机张量 x
        x = torch.randn(10)
        # 调用 fn 函数，传入 x 作为参数
        fn(x)
        # 创建一个编译计数器对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 torch._dynamo.optimize 函数优化 fn 函数，传入 cnt 作为参数
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 调用优化后的函数 opt_fn，传入 x 作为参数
        opt_fn(x)
        # 断言编译计数器的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
    # 测试相对导入功能的方法
    def test_relative_import(self):
        try:
            # 导入当前包中的 utils 模块，忽略 F401 警告
            from . import utils as _  # noqa: F401

            # 定义一个函数 fn，使用相对导入的方式导入 tensor_for_import_testing 函数
            def fn(x):
                from .utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        except ImportError:
            # 如果导入失败，则使用全局导入方式导入 utils 模块
            def fn(x):
                from utils import tensor_for_import_testing

                return x * 2 * tensor_for_import_testing

        # 生成一个形状为 (10,) 的随机张量 x
        x = torch.randn(10)
        # 调用 fn 函数
        fn(x)
        # 创建一个 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理张量 x
        opt_fn(x)
        # 断言编译帧数量为 1
        self.assertEqual(cnt.frame_count, 1)

    # 测试没有模块名称的相对导入功能
    def test_relative_import_no_modulename(self):
        try:
            # 导入当前包中的 utils 模块，忽略 F401 警告
            from . import utils as _  # noqa: F401

            # 定义一个函数 fn，使用相对导入的方式导入 tensor_for_import_testing 函数
            def fn(x):
                from . import utils

                return x * 2 * utils.tensor_for_import_testing

        except ImportError:
            # 如果导入失败，则使用全局导入方式导入 utils 模块
            def fn(x):
                import utils

                return x * 2 * utils.tensor_for_import_testing

        # 生成一个形状为 (10,) 的随机张量 x
        x = torch.randn(10)
        # 调用 fn 函数
        fn(x)
        # 创建一个 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        # 使用优化后的函数 opt_fn 处理张量 x
        opt_fn(x)
        # 断言编译帧数量为 1
        self.assertEqual(cnt.frame_count, 1)

    # 测试 bigbird_unsqueeze_inplace 函数
    def test_bigbird_unsqueeze_inplace(self):
        def fn(reshape_2):
            # 复制 reshape_2 张量到 view_2
            view_2 = reshape_2.clone()
            # 在 view_2 张量上进行就地 unsqueeze 操作，在维度 2 上扩展
            view_2.unsqueeze_(2)
            # 将 view_2 张量按维度 2 连接起来，得到 cat_11 张量
            cat_11 = torch.cat([view_2], dim=2)
            # 将 cat_11 张量 reshape 成 (2, 12, 64, -1) 的形状，得到 view_13 张量
            view_13 = cat_11.view((2, 12, 64, -1))
            return (view_13,)

        # 生成一个形状为 (2, 12, 64, 64) 的随机张量 x，同时需要梯度计算
        x = torch.randn(2, 12, 64, 64, requires_grad=True)
        # 调用 fn 函数得到 ref
        ref = fn(x)
        # 使用 "aot_eager" 优化方式对 fn 函数进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 使用优化后的函数 opt_fn 处理张量 x，得到 res
        res = opt_fn(x)
        # 断言 ref 和 res 是否相同
        self.assertTrue(same(ref, res))

    # 测试 issue1466_size_aot_autograd 函数
    def test_issue1466_size_aot_autograd(self):
        def fn(x):
            # 执行张量乘法操作并计算尺寸
            y = x * 2
            x_size = x.size()
            # 触发图断裂
            print("arf")
            # 使用张量操作和尺寸计算结果
            z = y.view(x_size) + 1
            return z

        # 生成一个形状为 (2, 3) 的随机张量 x，同时需要梯度计算
        x = torch.randn(2, 3, requires_grad=True)
        # 调用 fn 函数得到 ref
        ref = fn(x)
        # 使用 "aot_eager" 优化方式对 fn 函数进行优化，并将结果赋给 opt_fn
        opt_fn = torch._dynamo.optimize("aot_eager")(fn)
        # 使用优化后的函数 opt_fn 处理张量 x，得到 res
        res = opt_fn(x)
        # 断言 ref 和 res 是否相同
        self.assertTrue(same(ref, res))
    def test_ellipsis(self):
        # 定义一个名为 Repro 的内部类，继承自 torch.nn.Module
        class Repro(torch.nn.Module):
            # 构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 初始化 LayerNorm 层，设置维度为 (256,)，epsilon=1e-06，启用元素级仿射变换
                self.lnorm = torch.nn.LayerNorm((256,), eps=1e-06, elementwise_affine=True)
                # 初始化 Linear 层，输入维度为 256，输出维度为 256，启用偏置
                self.linear = torch.nn.Linear(in_features=256, out_features=256, bias=True)

            # 前向传播函数
            def forward(self, cat_10):
                # 对输入进行 LayerNorm 处理
                lnorm = self.lnorm(cat_10)
                # 使用切片操作获取 lnorm 的部分数据
                getitem_64 = lnorm[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
                # 将获取的数据输入到 Linear 层中
                linear = self.linear(getitem_64)
                return (linear,)

        # 定义输入参数列表
        args = [torch.randn(2, 197, 256)]

        # 创建 Repro 类的实例
        mod = Repro()
        # 对模型进行优化处理，使用 torch._dynamo.optimize 进行 eager 模式的优化，开启无 Python 解释器优化
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)

        # 断言优化前后模型输出一致
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_reinplacing(self):
        # 定义一个名为 MockModule 的内部类，继承自 torch.nn.Module
        class MockModule(torch.nn.Module):
            # 构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 初始化两个 Embedding 层
                self.self_layoutlm_embeddings_x_position_embeddings = torch.nn.Embedding(1024, 768)
                self.self_layoutlm_embeddings_y_position_embeddings = torch.nn.Embedding(1024, 768)

            # 前向传播函数
            def forward(self, getitem_1, getitem_2, add):
                # 使用 self_layoutlm_embeddings_x_position_embeddings 处理 getitem_1
                self_layoutlm_embeddings_x_position_embeddings = self.self_layoutlm_embeddings_x_position_embeddings(getitem_1)
                # 使用 self_layoutlm_embeddings_y_position_embeddings 处理 getitem_2
                self_layoutlm_embeddings_y_position_embeddings = self.self_layoutlm_embeddings_y_position_embeddings(getitem_2)
                # 将 add 与处理后的 Embedding 层输出进行加法运算
                add_1 = add + self_layoutlm_embeddings_x_position_embeddings
                add_2 = add_1 + self_layoutlm_embeddings_y_position_embeddings
                return (add_2,)

        # 创建 MockModule 类的实例
        mod = MockModule()
        # 对模型进行优化处理，使用 torch._dynamo.optimize 进行 aot_eager_decomp_partition 模式的优化
        opt_mod = torch._dynamo.optimize("aot_eager_decomp_partition")(mod)

        # 定义输入参数列表
        args = [
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512), (2048, 4), torch.int64, "cpu", False),
            ((2, 512, 768), (393216, 768, 1), torch.float32, "cpu", True),
        ]
        # 根据 args 创建张量，并设置 requires_grad 属性
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]
        # 断言优化前后模型输出一致
        self.assertTrue(same_two_models(mod, opt_mod, args))

    def test_optimized_deepcopy(self):
        # 创建名为 Foo 的类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 构造函数，初始化模块
            def __init__(self):
                super().__init__()
                # 初始化 Linear 层，输入维度为 2，输出维度为 3，启用偏置
                self.fc = torch.nn.Linear(in_features=2, out_features=3, bias=True)

            # 前向传播函数
            def forward(self, x):
                return self.fc(x)

        # 创建 Foo 类的实例
        mod = Foo()
        # 对模型进行优化处理，使用 torch._dynamo.optimize 进行 eager 模式的优化
        opt_mod = torch._dynamo.optimize("eager")(mod)
        # 定义输入参数列表
        args = [torch.randn(1, 2)]
        # 断言优化前后模型输出一致
        self.assertTrue(same_two_models(mod, opt_mod, args))
    def test_class_member(self):
        # 定义一个内部类 Foo 继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 类成员变量 a 初始化为 4
            a = 4
            # 类成员变量 b 初始化为一个 3x4 的张量，元素全为 1
            b = torch.ones(3, 4)

            # 构造函数，调用父类构造函数并初始化实例变量 c 为 4
            def __init__(self):
                super().__init__()
                self.c = 4

            # 前向传播函数，计算输入张量 x 的余弦值并加上类成员变量 a、b、c 的和
            def forward(self, x):
                return x.cos() + self.a + self.b + self.c

        # 创建 Foo 类的实例 mod
        mod = Foo()
        # 对 mod 进行优化编译，使用 eager 模式并开启无 Python 模式
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)
        # 创建输入参数，一个大小为 3x4 的随机张量
        args = (torch.randn(3, 4),)
        # 断言优化编译后的 mod 和原始 mod 在相同输入下的输出相等
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_named_buffers(self):
        # 定义一个内部类 Foo 继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 构造函数，调用父类构造函数并注册两个缓冲区 x 和 y
            def __init__(self):
                super().__init__()
                self.register_buffer("x", torch.ones(3))
                self.register_buffer("y", torch.ones(3))

            # 前向传播函数，遍历所有已命名的缓冲区并计算它们的和，然后将结果加到输入张量的余弦值上
            def forward(self, inp):
                res = 0
                for name, buffer in self.named_buffers():
                    res += buffer.sum()

                return inp.cos() + res

        # 创建 Foo 类的实例 mod
        mod = Foo()
        # 对 mod 进行优化编译，使用 eager 模式并开启无 Python 模式
        opt_mod = torch._dynamo.optimize("eager", nopython=True)(mod)
        # 创建输入参数，一个大小为 3x4 的随机张量
        args = (torch.randn(3, 4),)
        # 断言优化编译后的 mod 和原始 mod 在相同输入下的输出相等
        self.assertTrue(same(mod(*args), opt_mod(*args)))

    def test_requires_grad_guards_with_grad_mode1(self):
        # 定义函数 f，根据输入张量 x 的 requires_grad 属性返回不同的计算结果
        def f(x):
            if x.requires_grad:
                return x + 1
            else:
                return x + 2

        # 创建一个 requires_grad 为 True 的大小为 2 的张量 x
        x = torch.ones(2, requires_grad=True)

        # 编译函数 f
        f_compiled = torch.compile(f)
        with torch.no_grad():
            # 编译推断图
            f_compiled(x)

        # 测试：比较未编译和编译后在同一输入下的输出结果
        out_ref = f(x.detach())
        out = f_compiled(x.detach())

        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_requires_grad_guards_with_grad_mode2(self):
        # 创建一个 requires_grad 为 True 的大小为 2 的张量 x
        x = torch.ones(2, requires_grad=True)
        # 创建 x 的克隆，并分离计算图，同时要求保留 requires_grad 属性
        x_ref = x.clone().detach().requires_grad_(True)

        # 创建一个线性模型 m，输入维度为 2，输出维度为 2
        m = torch.nn.Linear(2, 2)
        # 编译模型 m
        m_compiled = torch.compile(m)

        with torch.no_grad():
            # 编译推断图
            m_compiled(x)

        # 测试：比较未编译和编译后在同一输入下的输出结果
        out_ref = m(x_ref)
        out = m_compiled(x)
        self.assertEqual(out_ref, out)
        self.assertEqual(out_ref.requires_grad, out.requires_grad)

    def test_is_symbolic_tracing(self):
        # 确保此处不会破坏图形
        def fn(x):
            if is_fx_tracing_test():
                return x * 2
            return x * 4

        # 创建一个大小为 4 的随机张量 a
        a = torch.randn(4)
        ref = fn(a)
        # 优化编译函数 fn
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(a)
        self.assertTrue(same(ref, res))
    def test_tokenization(self):
        from collections import UserDict  # 导入 UserDict 类

        class BatchEncoding(UserDict):  # 定义 BatchEncoding 类，继承自 UserDict
            """
            Copied from tokenization
            """

            def __init__(
                self,
                data,
            ):
                super().__init__(data)  # 调用父类 UserDict 的初始化方法

            def __getattr__(self, item: str):
                try:
                    return self.data[item]  # 尝试从 data 字典中获取 item 对应的值
                except KeyError as e:
                    raise AttributeError from e  # 捕获 KeyError 并抛出 AttributeError

        def tokenization(x):
            encoding = BatchEncoding({"key": x})  # 创建 BatchEncoding 实例 encoding，包含一个键为 "key" 的项
            return encoding["key"]  # 返回 encoding 中键为 "key" 的值

        opt_fn = torch._dynamo.optimize("eager")(tokenization)  # 对 tokenization 函数进行优化
        x = torch.rand((1, 4))  # 创建一个形状为 (1, 4) 的随机张量 x
        ref = tokenization(x)  # 调用 tokenization 函数，传入 x，获取结果作为 ref
        res = opt_fn(x)  # 调用经过优化后的函数 opt_fn，传入 x，获取结果作为 res
        self.assertTrue(same(ref, res))  # 断言 ref 和 res 相等

    def test_modules(self):
        class Foo(torch.nn.Module):  # 定义名为 Foo 的神经网络模块类
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)  # 创建一个线性层，输入大小为 4，输出大小为 3

            def forward(self, inp):
                res = torch.zeros(3, 3)  # 创建一个全零张量 res，形状为 (3, 3)
                for mod in self.modules():  # 遍历当前模块及其所有子模块
                    res += self.fc(inp)  # 将输入 inp 传入 self.fc，并累加到 res
                return res  # 返回累加后的 res

        mod = Foo()  # 创建 Foo 类的实例 mod
        args = (torch.ones(3, 4),)  # 创建一个形状为 (3, 4) 的全一张量 args
        cnt = torch._dynamo.testing.CompileCounter()  # 创建一个编译计数器 cnt
        opt_mod = torch._dynamo.optimize(cnt, nopython=True)(mod)  # 对 mod 进行优化
        self.assertTrue(same(mod(*args), opt_mod(*args)))  # 断言 mod 和 opt_mod 对 args 的输出相同
        self.assertEqual(cnt.op_count, 5)  # 断言操作计数器 cnt 的操作数为 5
        self.assertEqual(cnt.frame_count, 1)  # 断言操作计数器 cnt 的帧数为 1

    def test_omegaconf_listconfig_iter(self):
        obj = ListConfig()  # 创建 ListConfig 类的实例 obj
        x = torch.zeros(2)  # 创建一个形状为 (2,) 的全零张量 x

        def fn():
            y = x  # 将 x 赋值给 y
            for i in obj:  # 遍历 obj 的每一个元素
                y += i  # 将 y 与当前元素 i 相加
            return y  # 返回累加后的 y

        expected = fn()  # 调用 fn 函数，获取预期结果 expected
        actual = torch.compile(fn, fullgraph=True, backend="eager")()  # 编译并执行 fn 函数，获取实际结果 actual
        self.assertEqual(actual, expected)  # 断言实际结果与预期结果相等

    def test_user_defined_iter(self):
        class MyIter:  # 定义名为 MyIter 的迭代器类
            def __init__(self):
                self.i = 0  # 初始化计数器 self.i

            def __iter__(self):
                return self  # 返回自身迭代器对象

            def __next__(self):
                if self.i < 3:  # 如果计数器小于 3
                    self.i += 1  # 计数器加 1
                    return self.i  # 返回当前计数器值
                raise StopIteration  # 否则，抛出 StopIteration 异常

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            for i in MyIter():  # 使用 MyIter 的实例遍历
                x += i  # 将 x 与当前迭代器值 i 相加
            return x  # 返回累加后的 x

        self.assertEqual(fn(torch.zeros(1)), torch.full([1], 6.0))  # 断言 fn 函数对形状为 (1,) 的全零张量的输出为全为 6.0 的张量

    def test_stop_iteration_reconstruct(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.sin(), StopIteration(1, 2, 3)  # 返回 x 的正弦值及 StopIteration 异常对象

        _, res = fn(torch.ones(1))  # 调用 fn 函数，获取结果中的第二个返回值 res
        self.assertEqual(str(res), str(StopIteration(1, 2, 3)))  # 断言 res 的字符串表示与指定的 StopIteration 异常对象相同
    def test_tensor_data_kwarg(self):
        # 定义一个嵌套函数f，返回一个包含单个张量的PyTorch张量
        def f():
            return torch.tensor(data=[[1.0, -1.0]])

        # 创建一个编译计数器对象cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用优化器对函数f进行优化，禁用即时编译
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(f)
        # 断言f()和opt_fn()返回相同结果
        self.assertTrue(same(f(), opt_fn()))
        # 断言编译帧数为1
        self.assertEqual(cnt.frame_count, 1)

    @requires_cuda
    def test_norm_dtype(self):
        # 定义一个内部函数foo，接受参数_stack0
        def foo(_stack0):
            # 从_stack0中获取数据项
            getitem = _stack0[(slice(None, None, None), -1)]
            _stack0 = None
            # 对getitem进行正则化，使用L2范数，dim=1
            normalize = torch.nn.functional.normalize(getitem, p=2, dim=1)
            getitem = None
            return (normalize,)

        # 定义测试参数args，包含张量形状、步幅、数据类型、设备类型和梯度需求
        args = [((2, 50, 256), (1, 256, 1), torch.float16, "cuda", False)]
        # 使用rand_strided生成测试数据，并设置requires_grad为True
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        # 使用优化器对foo函数进行优化，传入"aot_eager_decomp_partition"
        opt_foo = torch._dynamo.optimize("aot_eager_decomp_partition")(foo)
        # 启用CUDA混合精度自动混合
        with torch.cuda.amp.autocast(enabled=True):
            # 调用foo函数并获取其第一个返回值作为参考结果ref
            ref = foo(*args)[0]
            # 再次调用foo函数获取其第一个返回值作为结果res
            res = foo(*args)[0]
            # 断言ref和res的数据类型相同
            self.assertEqual(ref.dtype, res.dtype)
            # 断言ref和res在数值上完全相等
            self.assertTrue(same(res, ref))

    def test_for_loop_graph_break(self):
        # 定义一个内部函数inner，返回输入张量的正弦值
        def inner(x):
            return torch.sin(x)

        # 定义函数fn，对输入张量进行100次循环
        def fn(x):
            for _ in range(100):
                # 调用inner函数
                inner(x)
                # 在每次循环中调用torch._dynamo.graph_break()
                torch._dynamo.graph_break()
            return x

        # 创建一个编译计数器对象cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用优化器对fn函数进行优化
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 创建一个大小为4的随机张量x
        x = torch.randn(4)
        # 调用优化后的fn函数并传入张量x
        opt_fn(x)
        # 断言编译帧数为1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作数为1
        self.assertEqual(cnt.op_count, 1)

    def test_for_loop_graph_break_before(self):
        # 检查后向边界计算是否正确
        def inner(x):
            return torch.sin(x)

        # 定义函数fn，在循环之前调用torch._dynamo.graph_break()
        def fn(x):
            torch._dynamo.graph_break()
            for _ in range(100):
                # 调用inner函数
                inner(x)
            return x

        # 创建一个编译计数器对象cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用优化器对fn函数进行优化
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 创建一个大小为4的随机张量x
        x = torch.randn(4)
        # 调用优化后的fn函数并传入张量x
        opt_fn(x)
        # 断言编译帧数为1
        self.assertEqual(cnt.frame_count, 1)
        # 断言操作数为100
        self.assertEqual(cnt.op_count, 100)

    def test_avoid_dupe_specialization(self):
        # 定义一个函数f，接受两个参数x和y，计算它们的和并乘以1
        def f(x, y):
            return (x + y) * 1

        # 使用优化器对函数f进行优化，传入"aot_eager"
        opt_f = torch._dynamo.optimize("aot_eager")(f)

        # 遍历布尔值列表[True, False]
        for b in [True, False]:
            # 创建大小为4的随机张量x，并设置requires_grad为b
            x = torch.randn(4, requires_grad=b)
            # 创建大小为4的随机张量y，并设置requires_grad为b
            y = torch.randn(4, requires_grad=b)
            # 断言调用f(x, x)和opt_f(x, x)返回相同结果
            self.assertEqual(f(x, x), opt_f(x, x))
            # 断言调用f(x, y)和opt_f(x, y)返回相同结果
            self.assertEqual(f(x, y), opt_f(x, y))
    # 测试验证模型关键字参数的功能
    def test_validate_model_kwargs(self):
        # 创建编译计数器对象
        cnt = CompileCounter()

        # 定义一个简单的函数 f1，对给定的参数进行正弦和余弦运算
        def f1(a, b):
            return torch.sin(a) + torch.cos(b)

        # 使用装饰器将函数 f2 编译为 Torch 函数，接受任意关键字参数
        @torch.compile(backend=cnt, fullgraph=True)
        def f2(**kwargs):
            # 验证给定的关键字参数是否符合函数 f1 的参数要求
            _validate_model_kwargs(f1, kwargs)
            # 调用函数 f1，传入验证后的关键字参数，计算结果
            return f1(**kwargs)

        # 创建随机张量 x 和 y
        x = torch.randn(10)
        y = torch.randn(10)

        # 断言编译后的函数 f2 的结果与未编译的函数 f1 相同
        self.assertEqual(f2(a=x, b=y), f1(x, y))
        # 断言编译计数器对象记录的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言编译计数器对象记录的操作数为 3
        self.assertEqual(cnt.op_count, 3)

    # 测试 Swin 模型基础张量属性
    def test_swin_base_tensor_attr(self):
        # 定义一个简单的 PyTorch 模型类 Foo
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个名为 t 的张量，不是模型的参数或缓冲区
                self.t = torch.randn(3)

            # 定义模型的前向传播方法
            def forward(self, x):
                # 返回输入张量 x 加上两个 t 张量的连接
                return x + torch.cat((self.t, self.t))

        # 创建 Foo 类的实例 mod
        mod = Foo()
        # 使用 Torch 内部优化函数对模型进行优化
        opt_mod = torch._dynamo.optimize("eager")(mod)
        # 创建一个随机张量作为输入
        args = [torch.randn(6)]
        # 断言未优化和优化后的模型在相同输入下行为一致
        self.assertTrue(same_two_models(mod, opt_mod, args))
        # 在优化后的模型上调用前向传播方法
        opt_mod(*args)

    # 测试无意义图形的删除
    def test_pointless_graph_removal(self):
        # 创建 Torch 编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()

        # 使用 Torch 编译装饰器对函数 fn 进行编译，使用 cnt 作为后端
        @torch.compile(backend=cnt)
        def fn(x):
            # 使用无梯度上下文管理器执行下列操作
            with torch.no_grad():
                # 手动中断 Torch 动态图
                torch._dynamo.graph_break()
                # 返回输入张量 x 加上 1
                return x + 1

        # 调用优化后的函数 fn，传入随机张量作为参数
        fn(torch.randn(4))
        # 断言编译计数器对象记录的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言编译计数器对象记录的操作数为 3
        self.assertEqual(cnt.op_count, 3)

    # 测试输出别名中间变量
    def test_output_aliases_intermediate(self):
        # 定义一个函数 f，对输入张量 x 执行乘以 2 操作，并返回两个结果
        def f(x):
            intermediate = x.mul(2)
            return intermediate.view(-1), intermediate

        # 使用 Torch 内部优化函数对函数 f 进行优化
        opt_f = torch._dynamo.optimize("aot_eager")(f)

        # 遍历两种情况：requires_grad 为 True 和 False 的随机张量 x
        for b in [True, False]:
            x = torch.randn(4, requires_grad=b)
            # 分别调用原始函数 f 和优化后的函数 opt_f
            out = f(x)
            out_test = opt_f(x)
            # 断言两种函数的输出第一个元素相等
            self.assertEqual(out[0], out_test[0])
            # 断言两种函数的输出第二个元素相等
            self.assertEqual(out[1], out_test[1])
            # 断言两种函数的输出第一个元素的 requires_grad 属性相等
            self.assertEqual(out[0].requires_grad, out_test[0].requires_grad)
            # 断言两种函数的输出第二个元素的 requires_grad 属性相等
            self.assertEqual(out[1].requires_grad, out_test[1].requires_grad)
            # 测试输出的别名关系是否保持不变
            out[0].mul_(2)
            out_test[0].mul_(2)
            # 断言两种函数的输出第一个元素相等
            self.assertEqual(out[0], out_test[0])
            # 断言两种函数的输出第二个元素相等
            self.assertEqual(out[1], out_test[1])

    # 测试 while 循环中的图形中断
    def test_while_loop_graph_break(self):
        # 定义一个内部函数 inner，对输入张量 x 执行正弦操作
        def inner(x):
            return torch.sin(x)

        # 定义函数 fn，接受输入张量 x，进行循环计算直至 i <= 10
        def fn(x):
            i = 20
            while i > 10:
                x = inner(x)
                i -= 1
                # 手动中断 Torch 动态图
                torch._dynamo.graph_break()
            return x

        # 创建 Torch 编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 Torch 编译装饰器对函数 fn 进行编译，使用 cnt 作为后端
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 创建一个随机张量作为输入
        x = torch.randn(4)
        # 在优化后的函数 opt_fn 上调用，传入随机张量作为参数
        opt_fn(x)
        # 断言编译计数器对象记录的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言编译计数器对象记录的操作数为 1
        self.assertEqual(cnt.op_count, 1)
    def test_nested_while_loop_graph_break(self):
        # 定义内部函数 inner_loop，接收参数 x
        def inner_loop(x):
            # 初始化 i 为 3，进入循环，直到 i <= 0 时退出
            i = 3
            while i > 0:
                # 每次循环减少 i，增加 x
                i -= 1
                x += 1
                # 调用 torch._dynamo.graph_break() 中断图优化
                torch._dynamo.graph_break()
            return x

        # 定义内部函数 inner，接收参数 x
        def inner(x):
            # 调用 inner_loop 函数处理 x，然后返回 torch.sin(x) 的结果
            inner_loop(x)
            return torch.sin(x)

        # 定义函数 fn，接收参数 x
        def fn(x):
            # 初始化 i 为 20，进入循环，直到 i <= 10 时退出
            i = 20
            while i > 10:
                # 调用 inner 函数处理 x
                x = inner(x)
                i -= 1
                # 调用 torch._dynamo.graph_break() 中断图优化
                torch._dynamo.graph_break()
            return x

        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 应用优化并赋值给 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 创建张量 x，使用其作为参数调用 opt_fn
        x = torch.randn(4)
        opt_fn(x)
        # 断言 cnt 的 frame_count 为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言 cnt 的 op_count 为 1
        self.assertEqual(cnt.op_count, 1)

    def test_while_loop_graph_break_inside_call_function(self):
        # 重现 huggingface 中在 `get_parameter_dtype` 函数中包含图中断的内部循环
        def inner(x):
            # 循环执行 3 次
            for i in range(3):
                x += 1
                # 调用 torch._dynamo.graph_break() 中断图优化
                torch._dynamo.graph_break()
            return x

        # 定义函数 fn，接收参数 x
        def fn(x):
            # x 增加 2
            x += 2
            # 调用 inner 函数处理 x
            inner(x)
            # x 增加 3
            x += 3
            return x

        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 fn 应用优化并赋值给 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 创建张量 x，使用其作为参数调用 opt_fn
        x = torch.randn(4)
        opt_fn(x)
        # 断言 cnt 的 frame_count 为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言 cnt 的 op_count 为 2
        self.assertEqual(cnt.op_count, 2)

    def test_exception_in_dynamo_handling(self):
        hit_handler = False

        # 参考 https://github.com/pytorch/pytorch/pull/96488
        @contextlib.contextmanager
        def ctx():
            try:
                yield
            except RuntimeError:
                # 设置 hit_handler 为 True 表示捕获到 RuntimeError 异常
                nonlocal hit_handler
                hit_handler = True

        # 使用 @torch._dynamo.optimize("eager") 优化装饰器定义函数 f
        @torch._dynamo.optimize("eager")
        def f():
            # 执行包含异常处理的上下文管理器 ctx
            with ctx():
                # 调用函数 h
                h()

        # 定义函数 h，抛出 RuntimeError 异常
        def h():
            raise RuntimeError("boof")

        # 调用函数 f，期望不会抛出错误
        f()
        # 断言 hit_handler 已被设置为 True，表示成功捕获异常
        self.assertTrue(hit_handler)
    def test_generator_dealloc(self):
        # 这是一个测试函数，用于验证生成器的内存释放机制
        # 参考：https://github.com/pytorch/pytorch/pull/96488
        #
        # 注意：[(...)] 故意这样写，表示包含一个生成器的列表

        # 创建一个包含生成器的列表
        generator_box = [(x for x in [1, 2, 3])]

        # 创建一个编译计数器对象
        counter = torch._dynamo.testing.CompileCounter()

        # 定义一个简单的函数 g，用于测试
        def g(x):
            return x + 2

        # TODO: 这个测试比较细致。为了确认其有效性，需要重新编译 eval_frame.c，并定义 '#define TORCHDYNAMO_DEBUG 1'，
        # 然后查看日志以确认以下信息：
        #
        # TRACE[_custom_eval_frame:650] begin <genexpr> test_repros.py 2276 -1 0 0
        # TRACE[_custom_eval_frame:664] throw <genexpr>
        #
        # 这表示我们确实触发了相关的代码路径

        # 注意：确保不要在这个帧中使用 Dynamo；如果使用 Dynamo，它实际上能理解 list.clear 并且会安排在禁用 eval frame 处理程序时发生生成器释放，
        # 这将阻止错误的发生（我们特别希望在 Dynamo eval frame 处理程序激活时触发生成器释放），因为这会导致生成器耗尽并触发 throw_flag == TRUE 情况。
        @torch._dynamo.disable(recursive=False)
        def f(x):
            generator_box.clear()  # 清空生成器盒子
            return g(x)

        # 断言没有未处理异常
        self.assertNoUnraisable(
            lambda: torch._dynamo.optimize(counter)(f)(torch.randn(3))
        )

        # 确保 x + 2 被捕获（以前修复实现错误会禁用 eval frame 回调，这意味着 g 将不会被跟踪）
        self.assertEqual(counter.op_count, 1)

    def test_error_return_without_exception_set(self):
        # 这个测试用例检查在未设置异常的情况下的错误返回
        # 参考：https://github.com/pytorch/pytorch/issues/93781
        @torch.compile
        def f():
            _generator_type = type(_ for _ in ())

        self.assertNoUnraisable(f)

    def common_merge_criteria_processor_list(self, list_cls, fullgraph):
        # 这个方法定义了一个通用的合并条件处理器列表的测试函数
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=fullgraph)
        def f(x, left, right):
            combined = _merge_criteria_processor_list(left, right)
            return combined(x)

        l1 = list_cls([torch.nn.ReLU(), torch.nn.Sigmoid()])
        l2 = list_cls([])
        input = torch.randn(16)
        result = f(input, l1, l2)
        self.assertEqual(result, l1(input))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        cnt.clear()
        l3 = list_cls([torch.nn.SiLU()])
        expected = l3(l1(input))
        result = f(input, l1, l3)
        self.assertEqual(len(l1), 3)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)
    # 测试方法，用于测试自定义列表处理器函数 common_merge_criteria_processor_list，使用 CustomList1 参数
    def test_merge_criteria_processor_list1(self):
        self.common_merge_criteria_processor_list(CustomList1, False)

    # 测试方法，用于测试自定义列表处理器函数 common_merge_criteria_processor_list，使用 CustomList2 参数
    def test_merge_criteria_processor_list2(self):
        self.common_merge_criteria_processor_list(CustomList2, True)

    # 测试方法，测试自定义列表子类 CustomList2 的功能
    def test_restricted_list_subclass1(self):
        # 创建一个编译计数器实例
        cnt = CompileCounter()

        # 使用 cnt 作为后端编译器，并全图优化编译函数 fn
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # 创建 CustomList2 实例 l，并对其进行操作
            l = CustomList2()
            l.extend([True])  # 扩展列表 l，添加 True
            l.append(a)       # 将 a 添加到列表 l 的末尾
            l.extend([b])     # 扩展列表 l，添加 b
            l.pop(0)          # 移除列表 l 的第一个元素
            l.append(l.length_times_10())  # 将 l.length_times_10() 的结果追加到列表 l
            return sum(l)     # 返回列表 l 中所有元素的和

        # 创建两个随机张量
        x = torch.randn(10)
        y = torch.randn(10)
        # 断言 fn 的结果等于 x + y + 20
        self.assertEqual(fn(x, y), x + y + 20)
        # 断言 cnt.op_count 等于 3
        self.assertEqual(cnt.op_count, 3)

    # 测试方法，测试自定义列表子类 CustomList2 的功能
    def test_restricted_list_subclass2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # 创建包含 a + 1 的 CustomList2 实例 l1
            l1 = CustomList2([a + 1])
            # 创建包含 b + 2 的 CustomList2 实例 l2
            l2 = CustomList2([b + 2])
            l1.extend(l2)  # 将 l2 扩展到 l1
            return l1       # 返回 l1

        x = torch.randn(10)
        y = torch.randn(10)
        z = fn(x, y)
        # 断言 z 的类型为 CustomList2
        self.assertEqual(type(z), CustomList2)
        # 断言 z 的长度为 2
        self.assertEqual(len(z), 2)
        # 断言 z 的 length_times_10() 方法返回值为 20
        self.assertEqual(z.length_times_10(), 20)
        # 断言 z 转换为列表后的值为 [x + 1, y + 2]
        self.assertEqual(list(z), [x + 1, y + 2])

    # 测试方法，测试自定义列表子类 CustomList2 的功能
    def test_restricted_list_subclass3(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a: CustomList2, b: CustomList2):
            a.extend(b)                # 将 b 扩展到 a
            a.append_twice(b[2] + 1)   # 向 a 中追加两次 b[2] + 1
            a.append(b[3] + 2)         # 向 a 中追加 b[3] + 2
            return b                   # 返回 b

        x = torch.randn(10)
        y = torch.randn(10)
        l = CustomList2([x, y])
        # 断言 fn 返回的对象与 l 是同一个对象
        self.assertIs(fn(l, l), l)
        # 断言 l 的长度为 7
        self.assertEqual(len(l), 7)
        # 断言 l 的第一个元素是 x
        self.assertIs(l[0], x)
        # 断言 l 的第二个元素是 y
        self.assertIs(l[1], y)
        # 断言 l 的第三个元素是 x
        self.assertIs(l[2], x)
        # 断言 l 的第四个元素是 y
        self.assertIs(l[3], y)
        # 断言 l 的第五个元素等于 x + 1
        self.assertEqual(l[4], x + 1)
        # 断言 l 的第六个元素与 l[4] 是同一个对象
        self.assertIs(l[5], l[4])
        # 断言 l 的第七个元素等于 y + 2
        self.assertEqual(l[6], y + 2)

    # 测试方法，测试带有消息的断言重写
    def test_rewrite_assert_with_msg(self):
        def f(x):
            b = x.sin()
            # 使用 assert 断言 x[0] 等于 3，如果不满足条件则输出 "First dim need to be 3"
            assert x[0] == 3, "First dim need to be 3"
            return x.cos() + b

        args = (torch.Tensor([3, 4, 5]),)
        cnt = torch._dynamo.testing.CompileCounter()

        # 对函数 f 进行全图优化编译
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)
        # 断言 f 和 opt_f 在给定参数下的输出相同
        self.assertTrue(same(f(*args), opt_f(*args)))
        # 断言 cnt.op_count 等于 6
        self.assertEqual(cnt.op_count, 6)
        # 断言 cnt.frame_count 等于 1
        self.assertEqual(cnt.frame_count, 1)

        # 导出函数 f，并使用导出的函数执行
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        # 断言导出的函数执行结果与原函数 f 的执行结果相同
        self.assertTrue(same(exported(*args), f(*args)))
    # 定义一个测试方法，用于验证列表的别名问题
    def test_list_aliasing(self):
        # 创建一个编译计数器对象
        cnt = CompileCounter()

        # 使用装饰器 torch.compile 对函数 fn 进行编译，指定后端为 cnt，全图模式为 True
        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a):
            # 修改传入列表 a 的内容，向其追加第一个元素的正弦值
            a.append(torch.sin(a[0]))
            return a

        # 生成一个包含随机数的张量 x
        x = torch.randn(10)
        # 创建一个包含张量 x 的列表 l
        l = [x]
        # 断言调用 fn(l) 后返回的列表与 l 是同一对象
        self.assertIs(fn(l), l)
        # 断言列表 l 的长度为 2
        self.assertEqual(len(l), 2)
        # 断言列表 l 的第一个元素仍为 x
        self.assertIs(l[0], x)
        # 断言列表 l 的第二个元素为 x 的正弦值
        self.assertEqual(l[1], torch.sin(x))
        # 断言编译计数器对象 cnt 记录的帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言编译计数器对象 cnt 记录的操作数为 1
        self.assertEqual(cnt.op_count, 1)

    # 定义一个测试方法，验证在其他错误情况下不重写 assert 语句的功能
    def test_not_rewrite_assert_for_other_errors(self):
        # 定义一个函数 f，接收一个参数 x
        def f(x):
            # 对输入张量 x 求正弦，赋值给 b
            b = x.sin()
            # 如果输入张量 x 的总和不小于 3，则抛出值错误异常
            if not x.sum() <= 3:
                raise ValueError("input sum needs to be 3")
            # 返回输入张量 x 的余弦与 b 的和
            return x.cos() + b

        # 准备参数列表 args，包含一个张量 torch.Tensor([3, 4, 5])
        args = (torch.Tensor([3, 4, 5]),)
        # 使用 torch._dynamo.optimize("eager") 对函数 f 进行优化
        opt_fn = torch._dynamo.optimize("eager")(f)
        # 使用断言验证调用优化后的函数 opt_fn(*args) 会抛出包含特定错误信息的值错误异常
        with self.assertRaisesRegex(ValueError, "input sum needs to be 3"):
            opt_fn(*args)

    # 定义一个测试方法，验证重写 assert 语句不会改变字节码
    def test_rewrite_assert_dont_change_bytecode(self):
        # 定义一个函数 fn，接收一个参数 x
        def fn(x):
            # 在不追踪梯度的环境中
            with torch.no_grad():
                # 使用 assert 语句断言张量 x 的最大值小于 5，如果不成立，抛出带有错误信息的异常
                assert x.max() < 5, f"invalid max {x.max()}"
                # 对张量 x 求正弦
                x = torch.sin(x)
            # 返回处理后的张量 x
            return x

        # 创建一个包含全为 1 的张量 x
        x = torch.ones(4)
        # 使用 torch._dynamo.optimize("eager") 对函数 fn 进行优化
        opt_fn = torch._dynamo.optimize("eager")(fn)
        # 使用 self.assertTrue 验证原始函数 fn(x) 和优化后的函数 opt_fn(x) 的结果相同
        self.assertTrue(same(fn(x), opt_fn(x)))

    # 定义一个测试方法，验证重写不带消息的 assert 语句
    def test_rewrite_assert_without_msg(self):
        # 定义一个函数 f，接收一个参数 x
        def f(x):
            # 对张量 x 求正弦，赋值给 b
            b = x.sin()
            # 使用 assert 语句断言张量 x 的第一个元素为 3
            assert x[0] == 3
            # 返回张量 x 的余弦与 b 的和
            return x.cos() + b

        # 准备参数列表 args，包含一个张量 torch.Tensor([3, 4, 5])
        args = (torch.Tensor([3, 4, 5]),)
        # 导出函数 f 的图形表示，以及相应的函数
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        # 使用 self.assertTrue 验证导出函数与原始函数在相同输入下的结果一致
        self.assertTrue(same(exported(*args), f(*args)))

        # 使用 self.assertRaisesRegex 验证调用导出函数时会抛出运行时错误，并包含特定的错误信息
        with self.assertRaisesRegex(RuntimeError, "assertion error"):
            exported(torch.Tensor([5, 6, 7]))

    # 定义一个测试方法，验证重写带非字符串消息的 assert 语句
    def test_rewrite_assert_with_non_string_msg(self):
        # 定义一个函数 f，接收一个参数 x
        def f(x):
            # 对张量 x 求正弦，赋值给 b
            b = x.sin()
            # 使用 assert 语句断言张量 x 的第一个元素为 2，并将张量 x 的大小作为消息
            assert x[0] == 2, x.size()
            # 返回张量 x 的余弦与 b 的和
            return x.cos() + b

        # 清空 torch._dynamo.utils.counters 对象中的计数器
        torch._dynamo.utils.counters.clear()
        # 准备参数 args，包含一个张量 torch.Tensor([3, 4, 5])
        args = torch.Tensor([3, 4, 5])
        # 使用 torch._dynamo.optimize("eager") 对函数 f 进行优化
        opt_f = torch._dynamo.optimize("eager")(f)
        # 使用 self.assertRaisesRegex 验证调用优化函数 opt_f(args) 时会抛出断言错误，并包含特定的错误信息
        with self.assertRaisesRegex(AssertionError, "torch.Size"):
            opt_f(args)
        # 使用 self.assertEqual 验证计数器中记录的“图断点”异常类型为“带有非字符串消息的断言”
        self.assertEqual(
            torch._dynamo.utils.counters["graph_break"][
                "assert with non-string message"
            ],
            1,
        )
    def test_rewrite_assert_noop(self):
        # 定义函数 f，接受参数 x
        def f(x):
            # 对 x 调用 sin() 方法，将结果赋给 b
            b = x.sin()
            # 断言 True，无操作
            assert True
            # 断言 x 的数据类型为 torch.float32
            assert x.dtype == torch.float32
            # 返回 x 调用 cos() 方法的结果加上 b
            return x.cos() + b

        # 设置参数 args 为包含一个 Tensor 的元组
        args = (torch.Tensor([3, 4, 5]),)
        # 使用 torch._dynamo.export 导出函数 f 的结果，并忽略第二个返回值
        exported, _ = torch._dynamo.export(f)(torch.Tensor([3, 4, 5]))
        # 断言导出结果与直接调用 f 返回结果相同
        self.assertTrue(same(exported(*args), f(*args)))

        # 创建 CompileCounter 实例 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 f 进行优化，使用 nopython=True
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)
        # 断言优化后的 f 与未优化的 f 结果相同
        self.assertTrue(same(f(*args), opt_f(*args)))
        # 断言 op_count 属性等于 3，表明 torch._assert 不在图中
        self.assertEqual(cnt.op_count, 3)
        # 断言 frame_count 属性等于 1，表明只有一个帧
        self.assertEqual(cnt.frame_count, 1)

        # 再次导出函数 f 的结果，并忽略第二个返回值
        exported, _ = torch._dynamo.export(f)(torch.Tensor([4, 4, 5]))
        # 断言导出结果与直接调用 f 返回结果相同
        self.assertTrue(same(exported(*args), f(*args)))

    def test_size_typematch(self):
        # 定义函数 f，接受参数 x 和 y
        def f(x, y):
            # 如果 x 是 torch.Size 类型的实例
            if isinstance(x, torch.Size):
                # 返回 y + 1
                return y + 1
            else:
                # 否则返回 y + 2
                return y + 2

        # 创建一个包含单个元素的 Tensor y
        y = torch.zeros(1)
        # 创建一个 torch.Size 类型的实例 x1
        x1 = torch.Size((3,))
        # 创建一个普通元组 x2
        x2 = (3,)

        # 创建 CompileCounter 实例 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 对函数 f 进行优化，使用 nopython=True
        opt_f = torch._dynamo.optimize(cnt, nopython=True)(f)
        # 断言优化后的 f(x1, y) 与未优化的 f(x1, y) 结果相同
        self.assertTrue(same(f(x1, y), opt_f(x1, y)))
        # 断言优化后的 f(x2, y) 与未优化的 f(x2, y) 结果相同
        self.assertTrue(same(f(x2, y), opt_f(x2, y)))
        # 断言 frame_count 属性等于 2，表明有两个帧
        self.assertEqual(cnt.frame_count, 2)

    def test_dict_subclass_contains(self):
        # 定义 ClassInstantier 类，继承自 collections.OrderedDict
        class ClassInstantier(collections.OrderedDict):
            pass

        # 定义函数 f，接受参数 x 和 d
        @torch.compile(fullgraph=True, backend="eager")
        def f(x, d):
            # 如果 d 中包含 "key1"
            if "key1" in d:
                # x 增加 2
                x = x + 2
            # 如果 d 中包含 "key2"
            if "key2" in d:
                # x 增加 4
                x = x + 4
            # x 增加 8
            x = x + 8
            # 返回 x
            return x

        # 使用 ClassInstantier 创建一个包含 "key1" 的实例，并调用 f
        result = f(torch.ones(8), ClassInstantier({"key1": torch.ones(8)}))
        # 断言 result 与 torch.full([8], 11.0) 相同
        self.assertTrue(same(result, torch.full([8], 11.0)))

        # 使用 ClassInstantier 创建一个包含 "key2" 的实例，并调用 f
        result = f(torch.ones(8), ClassInstantier({"key2": torch.ones(8)}))
        # 断言 result 与 torch.full([8], 13.0) 相同
        self.assertTrue(same(result, torch.full([8], 13.0)))

    def test_hf_classinstantier(self):
        # 定义 ClassInstantier 类，继承自 collections.OrderedDict
        class ClassInstantier(collections.OrderedDict):
            # 重写 __getitem__ 方法
            def __getitem__(self, key):
                # 调用父类的 __getitem__ 方法获取内容
                content = super().__getitem__(key)
                # 如果内容是元组，解构为 cls 和 kwargs，否则默认为 (content, {})
                cls, kwargs = content if isinstance(content, tuple) else (content, {})
                # 返回 cls 类的一个实例，使用 kwargs 作为参数
                return cls(**kwargs)

        # 创建 ClassInstantier 的实例 ACT2CLS
        ACT2CLS = ClassInstantier(
            {
                "relu": (nn.ReLU, {"inplace": False}),
                "tanh": nn.Tanh,
            }
        )

        # 定义函数 f，接受参数 x 和 act
        @torch.compile(fullgraph=True, backend="eager")
        def f(x, act):
            # 返回 ACT2CLS 中 act 对应的类的实例，调用该实例的方法 x
            return ACT2CLS[act](x)

        # 创建一个包含 10 个随机数的 Tensor y
        y = torch.randn(10)
        # 断言 f(y, "tanh") 的结果与 torch.tanh(y) 相同
        self.assertTrue(same(f(y, "tanh"), torch.tanh(y)))
        # 断言 f(y, "relu") 的结果与 torch.relu(y) 相同
        self.assertTrue(same(f(y, "relu"), torch.relu(y)))
    # 定义一个测试函数，测试临时模块的功能
    def test_ephemeral_module(self):
        # 创建一个自定义的神经网络模块，使用ReLU激活函数的平方作为正向传播的一部分
        class ReLUSquaredActivation(nn.Module):
            def forward(self, input):
                # 应用ReLU激活函数
                relu_applied = torch.nn.functional.relu(input)
                # 计算ReLU激活后的平方值
                squared = torch.square(relu_applied)
                return squared

        # 使用Torch的即时编译器，将函数f编译为图形化的计算图，全图优化模式
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # 输入张量x增加0.2
            x = x + 0.2
            # 使用ReLUSquaredActivation模块处理x
            x = ReLUSquaredActivation()(x)
            # 将处理后的结果再增加1
            x = x + 1
            return x

        # 生成一个形状为(10,)的随机张量y
        y = torch.randn(10)
        # 断言调用f(y)的结果与使用ReLUSquaredActivation处理y加0.2并加1的结果相同
        self.assertTrue(same(f(y), ReLUSquaredActivation()(y + 0.2) + 1))

    # 定义一个测试函数，测试原位操作unsqueeze_对输入的影响
    def test_inplace_unsqueeze_input(self):
        # 定义一个函数backend，用于返回输入的最后一个示例的大小并将其返回
        def backend(gm, example_inputs):
            self.assertEqual(example_inputs[-1].size(), torch.Size([1, 3, 4]))
            return gm

        # 使用Torch的即时编译器，将函数fn编译并传递给backend函数
        @torch.compile(backend=backend)
        def fn(x):
            # 在x上进行原位操作，增加维度
            x.unsqueeze_(0)
            return x + 1

        # 创建一个长度为1的张量列表inputs，包含一个形状为(3, 4)的随机张量
        inputs = [torch.randn(3, 4)]
        # 断言调用fn(*inputs)的结果的大小为(1, 3, 4)
        self.assertEqual(fn(*inputs).size(), torch.Size([1, 3, 4]))
        # 断言inputs[0]的大小仍为(1, 3, 4)
        self.assertEqual(inputs[0].size(), torch.Size([1, 3, 4]))

    # 定义一个测试函数，测试BatchNorm2d模块的端到端功能
    def test_batchnorm_e2e(self):
        # 定义一个复现用的神经网络模块
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加BatchNorm2d层，设置相关参数
                self.bn = torch.nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                )
                # 添加Conv2d层，设置相关参数
                self.conv1 = torch.nn.Conv2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=False,
                )

            def forward(self, x):
                # 对输入x应用BatchNorm2d
                x1 = self.bn(x)
                # 对应用BatchNorm2d后的结果应用Conv2d
                x2 = self.conv1(x1)
                # 对Conv2d的结果应用ReLU激活函数
                out = torch.nn.functional.relu(x2)
                return (out,)

        # 设定随机种子
        torch.manual_seed(1337)

        # 创建Repro模块的参考和测试副本
        m_ref = Repro()
        m_test = deepcopy(m_ref)

        # 使用Torch的即时编译器，将compiled_fn函数编译为即时计算图
        @torch._dynamo.optimize("aot_eager_decomp_partition")
        def compiled_fn(x):
            return m_test(x)

        # 创建一个形状为(2, 64, 32, 32)的随机张量x_ref，并将其克隆到x_test
        x_ref = torch.randn(2, 64, 32, 32, requires_grad=True)
        x_test = x_ref.clone()

        # 多次循环：每次迭代BatchNorm的running_mean/var将会更新，从而改变下一次迭代的输出
        for _ in range(3):
            # 使用m_ref对x_ref进行前向传播，并保存结果到ref
            ref = m_ref(x_ref)
            # 使用compiled_fn对x_test进行前向传播，并保存结果到res
            res = compiled_fn(x_test)

            # 断言ref与res的输出结果相同
            self.assertTrue(same(ref, res))

            # 对ref中需要梯度的张量进行反向传播求和
            for r in ref:
                if r.requires_grad:
                    r.sum().backward()
            # 对res中需要梯度的张量进行反向传播求和
            for r in res:
                if r.requires_grad:
                    r.sum().backward()

            # 断言m_ref和m_test的参数张量相同
            for param_ref, param_test in zip(m_ref.parameters(), m_test.parameters()):
                self.assertTrue(same(param_ref, param_test))
            # 断言BatchNorm2d层的running_mean/var缓冲区相同
            for buffer_ref, buffer_test in zip(m_ref.buffers(), m_test.buffers()):
                self.assertTrue(same(buffer_ref, buffer_test))

    # 使用Torch的配置管理器，设置assume_static_by_default为False
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_dynamic_shapes_right_side(self):
        # 定义函数 f，返回一个形状为 5*x.shape[0] 的张量
        def f(x):
            return torch.ones(5 * x.shape[0])

        # 创建一个形状为 (6, 5) 的随机张量 inp
        inp = torch.randn(6, 5)

        # 使用 torch._dynamo.export 导出函数 f 的图形表示 gm，其中 aten_graph=True 表示使用 ATen 图形
        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))
        # 断言 gm 在输入 inp 上的形状与 f(inp) 的形状相同
        self.assertEqual(gm(inp).shape, f(inp).shape)

    @torch._dynamo.config.patch("specialize_int", False)
    def test_maybe_multiply_symint(self):
        # 引用来自 torch._functorch.aot_autograd 的简化 AOT 模块
        from torch._functorch.aot_autograd import aot_module_simplified

        # 定义一个 AOT 编译器函数 my_aot_compiler，接受 gm 和示例输入 example_inputs
        def my_aot_compiler(gm, example_inputs):
            # 定义内部编译函数 my_compiler，返回 gm 的 forward 方法
            def my_compiler(gm, example_inputs):
                return gm.forward

            # 调用 aot_module_simplified，使用 my_compiler 编译 gm 和 example_inputs
            return aot_module_simplified(gm, example_inputs, fw_compiler=my_compiler)

        # 定义一个示例函数 my_example，对输入 t1、t2 和标量 d 执行加法操作
        def my_example(t1, t2, d):
            out = torch.add(t1, t2, alpha=d)
            return out

        # 使用 torch.compile 动态编译 my_example 函数，返回编译后的函数 compiled_fn
        compiled_fn = torch.compile(backend=my_aot_compiler, dynamic=True)(my_example)

        # 创建张量 t1 和 t2，要求其梯度跟踪
        t1 = torch.arange(3, dtype=torch.float32).requires_grad_(True)
        t2 = torch.arange(3, dtype=torch.float32).requires_grad_(True)

        # 调用编译后的函数 compiled_fn，分别使用不同的标量值 5 和 6 进行计算
        ra = compiled_fn(t1, t2, 5)
        self.assertEqual(ra, torch.tensor([0.0, 6.0, 12.0]))

        ra = compiled_fn(t1, t2, 6)
        self.assertEqual(ra, torch.tensor([0.0, 7.0, 14.0]))

    def test_build_map_unpack_with_call(self):
        # 定义一个 forward_with_cond_scale 函数，接受多个参数并返回它们的和
        def forward_with_cond_scale(x, t, cond_scale, self_cond, other1, other2):
            return x.sin() + t + cond_scale + self_cond + other1 + other2

        # 使用 torch.compile 编译 fn 函数，该函数接受 x 作为输入
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 创建字典 d1 和 d2，分别包含键 other1 和 other2，并将它们合并为 text_cond 字典
            d1 = dict(other1=5)
            d2 = dict(other2=4)
            text_cond = {**d1, **d2}
            # 调用 forward_with_cond_scale 函数，传入参数 x=1, cond_scale=2, self_cond=3，以及 text_cond 的其余键值对
            return forward_with_cond_scale(x, 1, cond_scale=2, self_cond=3, **text_cond)

        # 断言 fn 函数在输入为 torch.ones(4) 时与 torch.ones(4).sin() + 15 相同
        self.assertTrue(same(fn(torch.ones(4)), torch.ones(4).sin() + 15))

    @torch._dynamo.config.patch(verbose=True)
    def test_graph_break_unsupported_fake(self):
        # 创建一个 CompileCounter 对象 counter
        counter = torch._dynamo.testing.CompileCounter()

        # 使用 torch._dynamo.optimize 优化函数 f，对输入执行 torch.ops.test_sample.foo(x + 1) + 1 操作
        @torch._dynamo.optimize(counter)
        def f(x):
            return torch.ops.test_sample.foo(x + 1) + 1

        # 调用函数 f，输入为一个形状为 (3,) 的随机张量
        f(torch.randn(3))

        # 断言操作计数和帧计数分别为 2
        self.assertEqual(counter.op_count, 2)
        self.assertEqual(counter.frame_count, 2)

    def test_delattr(self):
        # 定义一个类 MyObj，具有属性 a 和 b
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        # 使用 torch.compile 编译 fn 函数，该函数接受 x 和 obj 作为输入
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            # 删除 obj 的属性 a
            del obj.a
            # 设置 obj 的属性 c 为 x + 1
            obj.c = x + 1
            # 删除 obj 的属性 c
            del obj.c
            # 创建临时对象 tmp，其属性 a 和 b 分别为 x + 2 和 x + 3
            tmp = MyObj(x + 2, x + 3)
            # 删除 tmp 的属性 b
            del tmp.b
            # 如果 obj 有属性 a，则返回 x + 1；否则返回 tmp
            if hasattr(obj, "a"):
                return x + 1
            return tmp

        # 创建标量张量 x
        x = torch.zeros([])
        # 创建对象 obj1 和 obj2，均为 MyObj 类的实例
        obj1 = MyObj(x, x)
        obj2 = fn(x, obj1)
        # 断言 obj1 没有属性 a
        self.assertFalse(hasattr(obj1, "a"))
        # 断言 obj1 没有属性 c
        self.assertFalse(hasattr(obj1, "c"))
        # 断言 obj2 没有属性 b
        self.assertFalse(hasattr(obj2, "b"))
        # 断言 obj1 的属性 b 的值为 0
        self.assertEqual(obj1.b.item(), 0)
        # 断言 obj2 的属性 a 的值为 2
        self.assertEqual(obj2.a.item(), 2)
    # 定义一个测试方法，验证删除属性时是否引发异常
    def test_delattr_raises(self):
        # 定义一个简单的类MyObj，带有两个属性a和b
        class MyObj:
            def __init__(self, a, b):
                self.a = a
                self.b = b
        
        # 使用torch.compile装饰器定义一个函数fn，使用"eager"后端编译
        @torch.compile(backend="eager")
        def fn(x, obj):
            # 删除对象obj的属性a
            del obj.a
            # x加1
            x = x + 1
            # 访问已删除的属性a，将会引发异常
            obj.a  # will raise
            return x
        
        # 创建一个torch张量x
        x = torch.zeros([])
        # 实例化MyObj类得到obj1对象
        obj1 = MyObj(x, x)
        # 断言调用fn函数时会引发AttributeError异常
        self.assertRaises(AttributeError, lambda: fn(x, obj1))

    # 定义一个测试方法，验证删除字典元素时的行为
    def test_delsubscr(self):
        # 使用torch.compile装饰器定义一个函数fn，使用"eager"后端编译
        @torch.compile(backend="eager")
        def fn(x):
            # 删除字典x中的键"a"
            del x["a"]
            # 计算字典x中键"b"对应张量的值加1
            y = x["b"] + 1
            return y
        
        # 创建一个字典x，包含两个键值对，值为torch张量
        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        # 调用fn函数得到结果
        result = fn(x)
        # 断言字典x不再包含键"a"
        self.assertFalse(hasattr(x, "a"))
        # 断言fn函数返回值为2
        self.assertEqual(result.item(), 2)

    # 定义一个测试方法，验证删除字典不存在的键时是否引发异常
    def test_delsubscr_raises(self):
        # 使用torch.compile装饰器定义一个函数fn，使用"eager"后端编译
        @torch.compile(backend="eager")
        def fn(x):
            # 删除字典x中的键"a"
            del x["a"]
            # 访问字典x中已删除的键"a"，应当引发KeyError异常
            y = x["a"] + 1  # should raise KeyError
            return y
        
        # 创建一个字典x，包含两个键值对，值为torch张量
        x = {"a": torch.tensor([1]), "b": torch.tensor([1])}
        # 断言调用fn函数时会引发KeyError异常
        self.assertRaises(KeyError, lambda: fn(x))

    # 定义一个测试方法，验证动态添加属性后是否能通过dir()函数访问到
    def test_attached_attribute_in_dir(self):
        # 定义一个继承自torch.nn.Module的类MyModule
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加两个模块属性：一个线性层和ReLU激活函数
                self.linear = torch.nn.Linear(16, 16)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # 前向传播方法，返回ReLU(线性层(x))
                return self.relu(self.linear(x))
        
        # 使用torch.compile编译MyModule类得到模块mod
        mod = torch.compile(MyModule(), backend="eager")
        # 将模块mod的is_compiled属性设置为True
        mod.is_compiled = True
        # 断言"is_compiled"属性存在于模块mod的属性列表中
        self.assertTrue("is_compiled" in dir(mod))

    # 使用torch._dynamo.config.patch修饰测试方法，验证动态形状隐式保护
    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_shapes_implicit_guard(self):
        # 定义一个函数f，接受一个张量x作为参数
        def f(x):
            # y为x的平方，y的形状是x形状的第一个维度大小次幂
            y = x * x.size(x.shape[0])
            # 对y进行按第一个维度求和
            torch.sum(y, [y.shape[0]])
            return y
        
        # 创建一个CompileCounter对象cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用torch._dynamo.optimize装饰cnt对象的f方法，设置nopython=True
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(f)
        # 调用opt_fn函数，传入一个形状为(3, 1, 1, 1, 1)的随机张量
        opt_fn(torch.randn(3, 1, 1, 1, 1))
        # 断言编译计数器cnt的帧计数为1
        self.assertEqual(cnt.frame_count, 1)

    # 定义一个测试方法，验证在函数中使用maybe函数处理图中的条件
    def test_dalle2_maybe(self):
        # 定义一个函数normalize，对输入x执行cos函数
        def normalize(x):
            return x.cos()

        # 使用torch.compile装饰器定义一个函数fn，使用"eager"后端编译，且完全图形化
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, normalize_img):
            # 对输入x执行sin函数得到低分辨率条件图像
            lowres_cond_img = x.sin()
            # 可能地使用normalize_img函数处理低分辨率条件图像
            lowres_cond_img = maybe(normalize_img)(lowres_cond_img)
            return lowres_cond_img
        
        # 断言调用fn函数返回结果与torch.ones([]).sin().cos()相等
        self.assertEqual(fn(torch.ones([]), normalize), torch.ones([]).sin().cos())

    # 定义一个测试方法，验证使用functools.wraps在函数中保留原函数名
    def test_functools_wraps(self):
        # 定义一个函数cool_name，对输入x执行sin函数
        def cool_name(x):
            return x.sin()

        # 使用torch.compile装饰器定义一个函数fn，使用"eager"后端编译，且完全图形化
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 对输入x执行cos函数得到y
            y = x.cos()

            # 使用functools.wraps装饰cool_name函数，保持其名称和功能
            @functools.wraps(cool_name)
            def uncool_name():
                return cool_name(y)

            return uncool_name
        
        # 调用fn函数得到结果
        result = fn(torch.ones([]))
        # 断言结果函数的名称为"cool_name"
        self.assertEqual(result.__name__, "cool_name")
        # 断言结果函数的执行结果与torch.ones([]).cos().sin()相等
        self.assertEqual(result(), torch.ones([]).cos().sin())
    # 定义一个测试方法，用于测试动态形状和浮点数保护
    def test_dynamic_shapes_float_guard(self):
        # 定义一个函数 f，对输入 x 进行 torch.nn.functional.dropout 操作，丢弃率为 x 的形状大小除以 6
        def f(x):
            return torch.nn.functional.dropout(x, x.shape[0] / 6)

        # 创建一个编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用优化器对函数 f 进行优化，禁用 Python 模式
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(f)
        # 对输入为 torch.randn(3) 的 opt_fn 进行调用
        opt_fn(torch.randn(3))
        # 断言编译器帧计数为 1
        self.assertEqual(cnt.frame_count, 1)

    # 标记为使用 torch._dynamo.config.patch 进行配置的测试方法，捕获标量输出
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensor_item(self):
        # 定义一个函数 f，接收两个参数 x 和 y，从 y 中获取其数值并返回 x 的总和加上这个数值
        def f(x, y):
            val = y.item()  # 获取 y 的数值
            return x.sum() + val

        # 导出函数 f 的图形表示，使用 ATen 图形
        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.zeros(6, 4),  # 参数 1: 6x4 的零张量
            torch.tensor(1),    # 参数 2: 值为 1 的张量
        )
        # 断言调用 f 和 gm 的结果相等
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(1)),
            gm(torch.zeros(6, 4), torch.tensor(1)),
        )
        # 再次断言调用 f 和 gm 的结果相等，但参数 2 为 2
        self.assertEqual(
            f(torch.zeros(6, 4), torch.tensor(2)),
            gm(torch.zeros(6, 4), torch.tensor(2)),
        )

    # 定义一个测试方法，用于测试列表的索引操作
    def test_list_index(self):
        # 枚举不同类型的列表
        for i, list_type in enumerate(
            (
                list,  # 普通列表
                tuple,  # 元组
                torch.Size,  # PyTorch 的 Size 类型
                collections.deque,  # collections 模块中的 deque
                namedtuple("FourElems", "one two three four", defaults=[0, 0, 0, 0]),  # 命名元组
            )
        ):
            # 重置 PyTorch 的动态执行环境
            torch._dynamo.reset()
            # 对不同的索引值进行迭代
            for index in ([], [2], [0, 3]):

                # 定义一个函数 f，接收一个参数 t
                def f(t):
                    if i == 4:  # 如果是命名元组
                        xs = list_type(1, 2, 3, 4)  # 创建命名元组类型的实例
                    else:
                        xs = list_type([1, 2, 3, 4])  # 否则创建列表或元组
                    res = xs.index(3, *index)  # 执行索引操作
                    return t + res

                # 对函数 f 进行优化，使用 eager 模式，禁用 Python 模式
                res = torch._dynamo.optimize(backend="eager", nopython=True)(f)(
                    torch.zeros(1)  # 参数为一个大小为 1 的零张量
                )

                # 断言结果与预期的张量 [2.0] 相等
                self.assertEqual(res, torch.tensor([2.0]))

    # 定义一个测试方法，用于测试列表中索引未找到的情况
    def test_list_index_not_found(self):
        # 定义一个函数 f，接收一个参数 t
        def f(t):
            xs = ["bar", "foo", "baz", "buzz"]  # 创建一个字符串列表
            res = xs.index("non-existent")  # 尝试在列表中查找不存在的元素
            return t + res

        # 断言调用函数 f 会抛出 torch._dynamo.exc.Unsupported 异常，因为未找到项不支持抛出 ValueError
        with self.assertRaises(
            torch._dynamo.exc.Unsupported,
        ):
            # 对函数 f 进行优化，使用 eager 模式，禁用 Python 模式
            torch._dynamo.optimize(backend="eager", nopython=True)(f)(torch.zeros(1))

    # 定义一个测试方法，用于测试列表中索引张量不支持的情况
    def test_list_index_tensor_unsupported(self):
        # 对不同的索引值进行迭代
        for index in ([], [2], [0, 3]):

            # 定义一个函数 f，接收一个参数 t
            def f(t):
                xs = [torch.tensor([i]) for i in range(4)]  # 创建包含张量的列表
                res = xs.index(torch.tensor([2]), *index)  # 尝试在列表中查找张量
                return t + res

            # 断言调用函数 f 会抛出 torch._dynamo.exc.UserError 异常，因为动态控制流不受支持
            with self.assertRaisesRegex(
                torch._dynamo.exc.UserError, "Dynamic control flow is not supported"
            ):
                # 对函数 f 进行优化，使用 eager 模式，禁用 Python 模式
                torch._dynamo.optimize(backend="eager", nopython=True)(f)(
                    torch.zeros(1)
                )
    # 定义测试函数，验证在推断模式下 XSoftmax 的行为
    def test_hf_xsoftmax_inference(self):
        # 内部函数 fn 接收输入和掩码，应用 XSoftmax，返回结果加上 2
        def fn(input, mask):
            return XSoftmax.apply(input + 1, mask, 1) + 2

        # 编译 fn 函数为优化后的版本，使用 eager 后端，并开启完整图形分析
        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)

        # 准备输入数据
        inputs = [
            torch.randn(4, 10),
            torch.randn(4, 10) < 0,
        ]
        # 预期输出是 fn 函数的结果
        expected = fn(*inputs)
        # 实际输出是优化后函数 fn_opt 的结果
        actual = fn_opt(*inputs)
        # 断言实际输出与预期输出相同
        self.assertTrue(same(actual, expected))

    # 使用 mock.patch 修饰的测试函数，验证在训练模式下 XSoftmax 的行为
    @mock.patch("torch._dynamo.config.guard_nn_modules", True)
    def test_hf_xsoftmax_training(self):
        # 导入计数器工具类
        from torch._dynamo.utils import counters

        # 清空计数器
        counters.clear()

        # 内部函数 fn 接收输入和掩码，应用 XSoftmax，返回结果
        def fn(input, mask):
            return XSoftmax.apply(input, mask, 1)

        # 创建编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 编译 fn 函数为优化后的版本，使用 cnt 作为后端，关闭完整图形分析
        fn_opt = torch.compile(fn, backend=cnt, fullgraph=False)

        # 设置随机种子
        torch.manual_seed(1234)
        # 准备输入数据 inputs1
        inputs1 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]
        # 重新设置随机种子
        torch.manual_seed(1234)
        # 准备输入数据 inputs2，与 inputs1 保持相同的随机状态
        inputs2 = [
            torch.randn(4, 10, requires_grad=True),
            torch.randn(4, 10) < 0,
        ]

        # 预期输出是 fn 函数在 inputs1 上的结果
        expected = fn(*inputs1)
        # 实际输出是优化后函数 fn_opt 在 inputs2 上的结果
        actual = fn_opt(*inputs2)
        # 断言实际输出与预期输出相同
        self.assertTrue(same(actual, expected))
        # 断言计数器记录的帧数为 1，并且所有帧都成功
        self.assertEqual(dict(counters["frames"]), {"total": 1, "ok": 1})
        # 断言操作计数为 2
        self.assertEqual(cnt.op_count, 2)
        # 断言帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 清空计数器
        cnt.clear()
        counters.clear()

        # 计算预期输出的梯度
        expected.sum().backward()
        # 计算实际输出的梯度
        actual.sum().backward()
        # 断言输入1的梯度与输入2的梯度相同
        self.assertTrue(same(inputs1[0].grad, inputs2[0].grad))

        # 当前不捕获反向传播帧
        self.assertEqual(cnt.frame_count, 0)
        # 操作计数应为 0
        self.assertEqual(cnt.op_count, 0)
        # 断言帧计数为空字典
        self.assertEqual(dict(counters["frames"]), {})
        # 断言图形中断计数为空字典
        self.assertEqual(dict(counters["graph_break"]), {})

    # 测试自动求导函数图形中断
    def test_autograd_function_graph_break(self):
        # 定义自定义的 torch.autograd.Function 子类 MySin
        class MySin(torch.autograd.Function):
            # 前向传播函数，应用 torch._dynamo.graph_break()，保存输入并返回 sin 函数值
            @staticmethod
            def forward(ctx, x):
                torch._dynamo.graph_break()
                ctx.save_for_backward(x)
                return x.sin()

            # 反向传播函数，计算梯度
            @staticmethod
            def backward(ctx, gx):
                (x,) = ctx.saved_tensors
                return gx * x.cos()

        # 创建输入张量 x，要求其梯度
        x = torch.randn([], requires_grad=True)

        # 编译 fn 函数为优化后的版本，使用 eager 后端
        @torch.compile(backend="eager")
        def fn(x):
            return MySin.apply(x)

        # 调用 fn 函数计算输出 y
        y = fn(x)
        # 断言 y 等于 x 的正弦值
        self.assertEqual(y, x.sin())

        # 计算 y 对 x 的梯度
        (gx,) = torch.autograd.grad(y, x)
        # 断言 gx 等于 x 的余弦值
        self.assertEqual(gx, x.cos())

    # 测试 JIT 追踪错误
    def test_jit_trace_errors(self):
        # 定义函数 f，使用 JIT 编译，dynamic=True 表示允许动态输入
        @torch.compile(backend="eager", dynamic=True)
        def f(x):
            return x + 1

        # 断言运行时错误，因为 JIT 不能追踪动态形状的输入
        with self.assertRaises(RuntimeError):
            torch.jit.trace(f, torch.randn(3))

        # 使用配置禁用嵌套 JIT 追踪错误
        with torch._dynamo.config.patch(error_on_nested_jit_trace=False):
            torch.jit.trace(f, torch.randn(3))

    # 使用 assume_static_by_default=False 的配置修饰器
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_tensor_split(self):
        # 定义一个函数 f，用于按照给定维度在张量上进行分割
        def f(x):
            return torch.split(x, x.shape[0] // 2, dim=0)[0]

        # 导出函数 f 的图形表示，并返回导出的模型和图形
        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.zeros(6, 4),
        )

        # 断言导出函数和原函数在给定张量上的输出一致
        self.assertEqual(f(torch.ones(8, 4)), gm(torch.ones(8, 4)))

    def test_optim_state_references_cleared(self):
        # 创建一个线性模型和输入张量
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        state_ref = 0

        # 初始化 Adadelta 优化器
        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        # 定义一个优化步骤函数
        def opt_step():
            optimizer.step()

        # 使用 Torch Dynamo 优化器优化 opt_step 函数
        compiled_opt_step = torch._dynamo.optimize("eager")(opt_step)

        # 定义一个编译后的模型步骤函数
        def compiled_model_step(x):
            optimizer.zero_grad()
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)

        # 弱引用 optimizer 状态中的 "square_avg" 张量，检查其是否被释放
        state_ref = weakref.ref(
            optimizer.state[optimizer.param_groups[0]["params"][0]]["square_avg"]
        )
        optimizer = None

        # 断言 "square_avg" 张量已被释放
        self.assertIsNone(state_ref())

    def test_grad_references_cleared(self):
        # 创建一个线性模型和输入张量
        model = torch.nn.Linear(2048, 2048, bias=False)
        x = torch.ones(2048)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)

        # 定义一个优化步骤函数
        def opt_step():
            optimizer.step()

        # 使用 Torch Dynamo 优化器优化 opt_step 函数
        compiled_opt_step = torch._dynamo.optimize("eager")(opt_step)

        # 定义一个编译后的模型步骤函数
        def compiled_model_step(x):
            optimizer.zero_grad(True)
            y = model(x)
            torch.sum(y).backward()
            compiled_opt_step()

        compiled_model_step(x)

        # 弱引用模型参数的梯度，检查其是否被释放
        param_grad_ref = weakref.ref(next(iter(model.parameters())).grad)
        optimizer.zero_grad(True)

        # 断言模型参数的梯度已被释放
        self.assertIsNone(param_grad_ref())

    def test_batch_encoding_clone_inputs(self):
        class BatchEncoding(dict):
            """
            Copied from test_tokenization
            """

            def __init__(
                self,
                data,
            ):
                super().__init__(data)

            def __getattr__(self, item: str):
                try:
                    return self.data[item]
                except KeyError as e:
                    raise AttributeError from e

        # 创建一个 BatchEncoding 实例
        encoding = BatchEncoding({"key": torch.rand((1, 4))})

        # 克隆输入对象 encoding
        cloned_encoding = torch._dynamo.utils.clone_inputs(encoding)

        # 断言克隆对象的类型不是 dict
        self.assertTrue(type(cloned_encoding) is not dict)

    def test_iadd_graph_break(self):
        # 定义一个函数 fn，执行一系列操作并返回一个包含张量的元组
        def fn(x):
            a = ()
            x = torch.sin(x)
            a += (x,)
            return a

        # 创建一个随机张量
        x = torch.randn(4)
        
        # 调用原始函数 fn，并获取其返回值作为参考值
        ref = fn(x)

        # 使用 Torch Dynamo 优化 fn 函数
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        
        # 调用优化后的函数并获取其返回值
        res = opt_fn(x)

        # 断言优化前后函数的返回值相同
        self.assertTrue(same(ref, res))
    def test_odict_get_item_index_name(self):
        # 创建一个字典，将 float 和 np.float16 映射到对应的 torch 数据类型
        d = {float: torch.float32, np.float16: torch.float16}

        # 使用 eager 模式编译函数 f
        @torch.compile(backend="eager")
        def f(x, y1, y2):
            # 返回两个形状为 (5,) 的零张量，数据类型分别为 d[y1] 和 d[y2]
            return torch.zeros(5, dtype=d[y1]), torch.zeros(5, dtype=d[y2])

        # 调用 f 函数，传入参数 torch.zeros(4), float, np.float16
        f(torch.zeros(4), float, np.float16)

    def test_dedup_global(self):
        # 编译函数 f，未指定后端，默认使用默认后端
        @torch.compile()
        def f():
            # 返回全局常量 _GLOBAL_CPU_TENSOR 两次相加的结果
            return _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR

        # 断言调用 f 函数的返回值等于 _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR
        self.assertEqual(f(), _GLOBAL_CPU_TENSOR + _GLOBAL_CPU_TENSOR)

    def test_randint_out_dynamic(self):
        # 定义函数 randint_fn，接受 high、size 和 out 三个参数，调用 torch.randint 函数
        def randint_fn(high, size, out):
            return torch.randint(high, size, out=out)

        # 编译 randint_fn 函数
        opt_model = torch.compile(randint_fn)

        # 创建一个形状为 (10,) 的空张量 out1，数据类型为 torch.int32
        out1 = torch.empty(10, dtype=torch.int32)
        # 调用 opt_model 函数，传入参数 17, (10,), out1
        opt_model(17, (10,), out1)

        # 创建一个形状为 (12,) 的空张量 out2，数据类型为 torch.int32
        out2 = torch.empty(12, dtype=torch.int32)
        # 调用 opt_model 函数，传入参数 17, (12,), out2
        opt_model(17, (12,), out2)

    @requires_cuda
    def test_guard_default_device(self):
        try:
            # 设置默认设备为 "cuda"
            torch.set_default_device("cuda")

            # 创建 CompileCounter 对象
            counter = torch._dynamo.testing.CompileCounter()

            # 使用 counter 优化函数 f，使其支持动态图模式
            @torch._dynamo.optimize(counter)
            def f():
                # 生成一个形状为 (3,) 的正态分布随机张量 x
                x = torch.randn(3)
                # 返回 x 的每个元素乘以 2 的结果
                return x * 2

            # 断言 f 函数的设备类型为 "cuda"
            self.assertEqual(f().device.type, "cuda")
            # 断言 counter 的帧计数为 1
            self.assertEqual(counter.frame_count, 1)

            # 将默认设备切换回 "cpu"
            torch.set_default_device("cpu")

            # 断言 f 函数的设备类型为 "cpu"
            self.assertEqual(f().device.type, "cpu")
            # 断言 counter 的帧计数为 2
            self.assertEqual(counter.frame_count, 2)

        finally:
            # 最终将默认设备设置为 None
            torch.set_default_device(None)

    def test_list_self_reference(self):
        # 创建一个空列表 root
        root = []
        # 将 root 列表内容替换为 [root, root, None, None]

        # 使用 eager 模式优化函数 test_bug
        @torch._dynamo.optimize("eager")
        def test_bug():
            # 返回列表 root
            return root

        # 调用 test_bug 函数
        test_bug()

    def test_hf_bigbird_unsqueeze(self):
        # 定义函数 torch_bmm_nd，调用 torch.bmm 函数前调用 graph_break
        def torch_bmm_nd(inp_1, inp_2, ndim=None):
            torch._dynamo.graph_break()
            return torch.bmm(inp1, inp2)

        # 定义函数 fn，执行 torch_bmm_nd 函数并对结果进行操作
        def fn(inp1, inp2, inp3, inp4, c):
            # 调用 torch_bmm_nd 函数，将结果 unsqueeze 在第二维
            a = torch_bmm_nd(inp1, inp2, 4)
            a.unsqueeze_(2)
            a = a * 2

            # 调用 torch_bmm_nd 函数，将结果 unsqueeze 在第二维
            b = torch_bmm_nd(inp3, inp4, 4)
            b.unsqueeze_(2)
            l = a + b

            # 在第二维上连接张量 a、b 和 c，形成输出张量 out
            out = torch.cat([a, b, c], dim=2)
            return out, l

        # 创建不同形状的输入张量
        inp1 = torch.rand(1, 64, 448)
        inp2 = torch.rand(1, 448, 64)
        inp3 = torch.rand(1, 64, 448)
        inp4 = torch.rand(1, 448, 64)
        c = torch.rand(1, 64, 1, 64)

        # 创建 CompileCounter 对象 cnt
        cnt = torch._dynamo.testing.CompileCounter()
        # 使用 cnt 优化函数 fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 调用 opt_fn 函数，传入 inp1, inp2, inp3, inp4, c 作为参数
        opt_fn(inp1, inp2, inp3, inp4, c)
        # 断言 cnt 的帧计数为 3
        self.assertEqual(cnt.frame_count, 3)
    # 定义测试函数，用于验证 torch 变量类型的功能
    def test_torch_variable_type(self):
        # 定义一个函数，用于检查对象的类型是否符合给定的类型或检查条件
        def check_type(obj, types_or_checks):
            for type_or_check in types_or_checks:
                # 如果 type_or_check 是类型对象，则检查 obj 是否是这种类型
                # 如果 type_or_check 是可调用对象，则使用它来检查 obj
                if (
                    isinstance(obj, type_or_check)
                    if isinstance(type_or_check, type)
                    else type_or_check(obj)
                ):
                    return True
            return False

        # 通过 torch._dynamo.optimize("eager") 对 check_type 进行优化
        opt_check_type = torch._dynamo.optimize("eager")(check_type)
        # 分别使用原始和优化后的 check_type 函数验证随机生成的张量是否是 torch.Tensor 类型
        ref = check_type(torch.randn(4), [torch.Tensor])
        res = opt_check_type(torch.randn(4), [torch.Tensor])
        # 断言优化前后的结果应当相等
        self.assertEqual(ref, res)

    # 对 https://github.com/pytorch/pytorch/issues/103132 的问题进行测试
    @torch._dynamo.config.patch("assume_static_by_default", False)
    def test_inference_mode_dynamic_shapes(self):
        # 定义一个简单的神经网络模型，用于测试动态形状推断
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, param):
                # 执行矩阵乘法运算
                z = torch.matmul(param, param)
                return z

        # 创建 Repro 类的实例
        model = Repro()
        # 需要一个三维张量来触发错误：
        # 我们会进入 C++ 的矩阵乘法分解路径，调用 sizes() 方法
        inp = torch.randn(4, 4, 4, requires_grad=True)
        # 使用 aot_eager 后端对模型进行编译
        model = torch.compile(model, backend="aot_eager", dynamic=True)
        # 进入推断模式
        with torch.inference_mode():
            model(inp)

    # 测试 kwargs 中的 out 参数列表变量
    def test_kwargs_out_list_variable(self):
        # 定义一个简单的神经网络模型，用于测试参数传递
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, param):
                # 调用 torch.frexp 函数，使用参数中的 input 键和 out 键对应的值
                z = torch.frexp(**param)
                return z

        # 创建 Repro 类的实例
        model = Repro()
        # 准备参数字典，其中 input 键对应一个张量，out 键对应一个包含两个空张量的列表
        params = {"input": torch.tensor([[0.0, 1, 2, 4]])}
        params["out"] = [
            torch.empty(0, dtype=torch.float32),  # mantissa（尾数部分）
            torch.empty(0, dtype=torch.int32),  # exponent（指数部分）
        ]
        # 使用 eager 后端对模型进行编译
        model = torch.compile(model, backend="eager")
        # 调用模型，获取结果 mantissa 和 exponent
        mantissa, exponent = model(params)
        # 预期的结果
        ref_mantissa = torch.tensor([[0.0000, 0.5000, 0.5000, 0.5000]])
        ref_exponent = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        # 断言模型输出与预期结果一致
        self.assertEqual(ref_mantissa, mantissa)
        self.assertEqual(ref_exponent, exponent)

    # 使用 torch._dynamo.config.patch 对函数进行装饰，设置捕获标量输出为 True
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_with_sizes_aot_autograd(self):
        # 定义一个函数，用于执行 torch.ops.aten.split_with_sizes 操作
        def fn(result, split_sizes):
            rs = torch.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        # 准备示例输入
        example_inputs = (
            torch.randn(32, requires_grad=True),
            torch.tensor((7, 16, 9)),
        )
        # 使用 aot_eager 后端对函数进行编译，同时打开 fullgraph 选项
        actual = torch.compile(fn, fullgraph=True, backend="aot_eager")(*example_inputs)
        # 执行原始函数以获取预期结果
        expected = fn(*example_inputs)
        # 断言编译后的结果与原始结果一致
        self.assertEqual(actual, expected)
    def test_unspecialized_nn_module_with_torch_variable_attribute(self):
        """
        In this case self.fn = something that should be a TorchVariable.
        When it's not a TorchVariable, dynamo tries to trace through and fails.
        This makes sure that the self.fn is handled as a TorchVariable.
        """

        # 定义一个继承自torch.nn.Module的用户自定义模块
        class UserModule(torch.nn.Module):
            torchdynamo_force_dynamic = True  # 强制为未专门化的NN模块

            def __init__(self, fn):
                super().__init__()
                self.fn = fn  # 初始化函数参数fn作为模块的一个属性

            def forward(self, **inp):
                return self.fn(**inp)  # 调用self.fn作为模块的前向方法

        # 准备输入数据字典
        inputs = {
            "input": torch.randn([2, 9]).uniform_(0, 1),
            "target": torch.randn([2, 9]).uniform_(0, 1),
            "reduction": "mean",
        }

        # 创建一个UserModule实例，传入torch.nn.functional.binary_cross_entropy作为参数
        mod = UserModule(torch.nn.functional.binary_cross_entropy)
        # 参考结果，调用模块进行前向传播
        ref = mod(**inputs)
        # 优化模块，返回优化后的结果
        res = torch._dynamo.optimize("eager", nopython=True)(mod)(**inputs)
        # 断言优化后的结果与参考结果相等
        self.assertEqual(ref, res)

    def test_call_finally_python_3_8(self):
        # 问题 - https://github.com/pytorch/pytorch/issues/97811
        def make_fn(g):
            def fn():
                while True:
                    try:
                        print(g)
                        break
                    except Exception as _:
                        break

            return torch.compile(fn, backend="eager")

        # 调用make_fn函数并执行返回的函数
        make_fn(None)()

    def test_call_finally_python_3_8_2(self):
        def f(x):
            while x:
                try:
                    pass
                except Exception as _:
                    continue

        # 编译函数f，使用eager后端执行
        torch.compile(f, backend="eager")(0)

    def test_call_finally_opcode_python_3_8(self):
        def fn():
            try:
                return torch.zeros(4)
            finally:
                return torch.ones(4)  # noqa: SIM107, B012

        # 编译函数fn，使用aot_eager后端执行，并返回结果
        result = torch.compile(fn, backend="aot_eager")()
        # 断言结果与torch.ones(4)相等
        self.assertEqual(result, torch.ones(4))

    def test_string_format(self):
        s = "temp{i}"

        # 使用torch.compile装饰器编译函数fn，使用eager后端和完整图模式
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            if s.format(i=4) == "temp4":
                return torch.sin(x)
            return torch.cos(x)

        x = torch.randn(4)
        # 断言fn(x)的结果与torch.sin(x)相等
        self.assertEqual(fn(x), torch.sin(x))

    # 重现torch._dynamo.exc.InternalTorchDynamoError: 'NoneType' object has no attribute 'guards'错误，由于对空列表的处理不当
    def test_empty_list_contains_with_jump(self):
        def fn(x, l):
            if x in l:
                return x.cos()
            return x.sin()

        # 创建CompileCounter实例
        counter = CompileCounter()
        # 优化函数fn，并使用优化后的函数处理输入数据和空列表
        compiled_fn = torch._dynamo.optimize(counter)(fn)(torch.randn([2, 2]), [])
        # 断言帧计数等于1
        self.assertEqual(counter.frame_count, 1)
    # 测试函数，验证在 JIT 编译中是否正确使用 isinstance 进行类型检查
    def test_graph_break_on_jit_isinstance(self):
        # 定义一个使用 torch.compile 装饰器的函数 fn，使用 eager 后端编译
        @torch.compile(backend="eager")
        def fn(x):
            # 如果 x 是 List[str] 类型，则返回 x 的两倍
            if torch.jit.isinstance(x, List[str]):
                return x * 2
            # 否则返回 x 本身
            return x

        # 使用 torch.compile 将 fn 编译为 opt_fn，使用 eager 后端
        opt_fn = torch.compile(fn, backend="eager")
        # 生成一个随机张量 x
        x = torch.rand(4)
        # 断言 fn(x) 和 opt_fn(x) 的结果相同
        self.assertTrue(same(fn(x), opt_fn(x)))

    # 测试函数，验证 torch.add 和 torch.sub 函数在使用 alpha 参数和输出张量时的正确性
    def test_add_sub_alpha_out(self):
        # 生成一个形状为 (2, 3, 4) 的随机输入张量 inp
        inp = torch.randn(2, 3, 4)
        # 设置其他参数为常数 1
        other = 1
        # 设置 alpha 参数为常数 2
        alpha = 2
        # 遍历 torch.add 和 torch.sub 函数
        for op in [torch.add, torch.sub]:
            # 生成一个全零张量 out
            out = torch.zeros(2, 3, 4)
            # 生成一个全零张量 compile_out
            compile_out = torch.zeros(2, 3, 4)
            # 调用 op 函数，对 inp 和 other 进行操作，alpha 参数设置为 alpha，将结果写入 out
            op(inp, other, alpha=alpha, out=out)
            # 使用 torch.compile 动态编译 op 函数，并将结果写入 compile_out
            compiled_fn = torch.compile(op, dynamic=True)
            compiled_fn(inp, other, alpha=alpha, out=compile_out)
            # 断言 out 和 compile_out 的结果相同
            self.assertTrue(same(out, compile_out))

    # 测试函数，验证在编译时对张量形状进行负数限制的功能
    def test_negative_shape_guard(self):
        # 定义一个函数 fn，对输入张量 x 进行形状判断
        def fn(x):
            # 如果 x 的形状不等于 (5, 1, 2, 3)，返回 x 的余弦值
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            # 否则返回 x 的正弦值
            return x.sin()

        # 实例化一个编译计数器对象 counter
        counter = torch._dynamo.testing.CompileCounter()
        # 使用 torch.compile 将 fn 编译为 opt_fn，使用 counter 后端进行动态编译
        opt_fn = torch.compile(fn, backend=counter, dynamic=True)

        # 生成两个张量 x 和 x2，分别形状为 (5, 1, 3, 4) 和 (5, 1, 2, 3)
        x = torch.ones(5, 1, 3, 4)
        x2 = torch.ones(5, 1, 2, 3)
        # 断言 fn(x) 和 opt_fn(x) 的结果相同
        self.assertEqual(fn(x), opt_fn(x))
        # 断言 fn(x2) 和 opt_fn(x2) 的结果相同
        self.assertEqual(fn(x2), opt_fn(x2))
        # 断言编译计数器的 frame_count 属性值为 2
        self.assertEqual(counter.frame_count, 2)

    # 使用 torch._dynamo.config.patch 修饰器，测试延迟运行时断言的功能
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_deferred_runtime_asserts(self):
        # 定义一个使用 torch.compile 装饰器的函数 f，设置 fullgraph 参数为 True
        @torch.compile(fullgraph=True)
        def f(x):
            # 计算张量 x 的标量值 y
            y = x.item()
            # 对 y 进行运行时断言检查
            torch._check_is_size(y)
            # 如果 y 大于等于 0，则返回 x 的两倍，否则返回 x 的三倍
            if y >= 0:
                return x * 2
            else:
                return x * 3

        # 对 torch.tensor([3]) 调用函数 f
        f(torch.tensor([3]))
        # 使用 lambda 表达式断言对 torch.tensor([-2]) 调用 f 时会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: f(torch.tensor([-2])))

    # 测试函数，验证 torch.addr 函数在使用 alpha 和 beta 参数以及输出张量时的正确性
    def test_addr_alpha_beta_out(self):
        # 生成一个形状为 (2, 3) 的随机输入张量 inp
        inp = torch.randn(2, 3)
        # 生成一个形状为 (2,) 的随机向量 vec1
        vec1 = torch.randn(2)
        # 生成一个形状为 (3,) 的随机向量 vec2
        vec2 = torch.randn(3)
        # 设置 alpha 参数为常数 2
        alpha = 2
        # 设置 beta 参数为常数 5
        beta = 5

        # 生成一个全零张量 out
        out = torch.zeros(2, 3)
        # 生成一个全零张量 compile_out
        compile_out = torch.zeros(2, 3)

        # 调用 torch.addr 函数，对 inp、vec1 和 vec2 进行操作，alpha 和 beta 参数分别设置为 alpha 和 beta，将结果写入 out
        torch.addr(inp, vec1, vec2, alpha=alpha, beta=beta, out=out)
        # 使用 torch.compile 动态编译 torch.addr 函数，并将结果写入 compile_out
        compiled_fn = torch.compile(torch.addr, dynamic=True)
        compiled_fn(inp, vec1, vec2, alpha=alpha, beta=beta, out=compile_out)
        # 断言 out 和 compile_out 的结果相同
        self.assertTrue(same(out, compile_out))
    # 测试函数，验证设置 requires_grad 后图形中断的情况
    def test_setattr_requires_grad_graph_breaks(self):
        # 定义一个简单函数 fn，接受参数 x
        def fn(x):
            # 计算 x + 4，并赋值给 z
            z = x + 4
            # 设置 x 的 requires_grad 属性为 True
            x.requires_grad = True
            # 计算 x * z，并赋值给 y
            y = x * z
            # 返回 y
            return y
    
        # 遍历不同的后端选项
        for backend in ["count", "eager", "aot_eager"]:
            # 如果后端选项为 "count"，则使用 CompileCounter 对象
            if backend == "count":
                backend = CompileCounter()
            
            # 使用 torch.compile 函数编译 fn 函数，指定后端为当前 backend
            opt_fn = torch.compile(fn, backend=backend)
    
            # 创建一个全零的 Tensor eager
            eager = torch.zeros(5)
            # 克隆 eager 并赋值给 compiled
            compiled = eager.clone()
    
            # 使用原始函数 fn 对 eager 和 compiled 进行计算
            out_eager = fn(eager)
            out_opt = opt_fn(compiled)
    
            # 断言两种计算方法的结果相等
            self.assertEqual(out_eager, out_opt)
    
            # 对 out_eager 和 out_opt 的和进行反向传播
            out_eager.sum().backward()
            out_opt.sum().backward()
    
            # 断言 eager 和 compiled 保持不变
            self.assertEqual(eager, compiled)
    
            # 如果 backend 是 CompileCounter 类型，则断言 frame_count 为 2（表示图形中断）
            if isinstance(backend, CompileCounter):
                self.assertEqual(backend.frame_count, 2)  # graph breaks
    
    # 测试动态形状下的不等式比较
    def test_dynamic_shapes_double_not_equal(self):
        # 定义一个简单函数 fn，接受参数 x
        def fn(x):
            # 如果 x 的尺寸不等于 (5, 1, 2, 3)，返回 x 的余弦函数
            if x.size() != (5, 1, 2, 3):
                return x.cos()
            # 否则返回 x 的正弦函数
            return x.sin()
    
        # 使用 torch.compile 函数编译 fn 函数，指定后端为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
    
        # 创建两个尺寸不同的全一 Tensor x 和 x2
        x = torch.ones(5, 1, 2, 3)
        x2 = torch.ones(5, 1, 3, 4)
    
        # 断言使用原始函数和编译后的函数对 x 和 x2 的计算结果相等
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x2), opt_fn(x2))
    
    # 测试循环中没有递归错误的情况
    def test_inductor_no_recursionerror_on_for_loops(self):
        # 定义一个简单的前向传播函数 forward，接受参数 x
        def forward(x):
            # 执行 1000 次循环，每次将 x 乘以 1.0
            for _ in range(1000):
                x = 1.0 * x
            # 返回 x
            return x
    
        # 使用 torch.compile 函数编译 forward 函数，并对输入 torch.tensor([1.0]) 进行计算
        self.assertTrue(
            same(torch.compile(forward)(torch.tensor([1.0])), torch.tensor([1.0]))
        )
    
    # 测试用户定义对象可调用的情况
    def test_user_defined_object_callable(self):
        # 定义一个可调用的类 MyCallable
        class MyCallable:
            def __call__(self, x):
                # 返回 x + 1
                return x + 1
    
        # 定义一个简单函数 fn，接受参数 x
        def fn(x):
            # 在图中创建 MyCallable 对象，并对 x 调用该对象
            return MyCallable()(x)
    
        # 使用 torch.compile 函数编译 fn 函数，指定后端为 "eager"，并开启完整图形记录
        fn_opt = torch.compile(fn, backend="eager", fullgraph=True)
    
        # 断言编译后的函数 fn_opt 对 torch.zeros(1) 的计算结果与原始函数 fn 相同
        self.assertEqual(fn_opt(torch.zeros(1)), fn(torch.zeros(1)))
    
    # 使用 torch._dynamo.config.patch 修饰的测试用例
    def test_many_views_with_mutation(self):
        # 当添加符号存储偏移量时（见 #113734），tensors_definitely_do_not_overlap 开始增加形状保护器 - 相对于输入数量的二次量。
        # 测试此配置，并验证添加了合理数量的保护器。
        # 注意，当启用动态形状时，此测试失败，我们仍然得到二次保护器。
        def fn(x):
            # 修改第一个元素的值为其 ReLU（整流线性单元）操作后的结果
            x[0].relu_()
            # 将所有 x 的元素拼接起来并求和
            return torch.cat(x).sum()

        # 创建长度为 16 * (AMT + 1) 的随机张量
        AMT = 32
        src = torch.rand(16 * (AMT + 1))

        # 使用 as_strided 创建长度为 AMT 的 x 列表，每个元素是 src 的不同视图
        x = [src.as_strided((4, 4), (4, 1), 3 + 16 * i) for i in range(AMT)]

        # 重置 torch._dynamo 并清除编译度量
        torch._dynamo.reset()
        torch._dynamo.utils.clear_compilation_metrics()

        # 编译函数 fn 到 AOT eager 后端，使用 x 作为输入
        res = torch.compile(fn, backend="aot_eager")(x)

        # 获取所有编译度量
        all_metrics = torch._dynamo.utils.get_compilation_metrics()

        # 计算所有保护器的总数
        total_guards = sum(metric.guard_count for metric in all_metrics)
        # 断言总保护器数小于 AMT * 8
        self.assertLess(total_guards, AMT * 8)

        # 计算所有形状环境保护器的总数
        total_shape_env_guards = sum(
            metric.shape_env_guard_count for metric in all_metrics
        )
        # 断言总形状环境保护器数小于 AMT * 8
        self.assertLess(total_shape_env_guards, AMT * 8)

    # https://github.com/pytorch/pytorch/issues/118799
    def test_subclass_graph_output_repro(self):
        @torch._dynamo.allow_in_graph
        def to_subclass(x):
            # 返回一个包含两个 x 克隆的 TwoTensor 实例
            return TwoTensor(x.clone(), x.clone())

        def f(x):
            # 通过 to_subclass 函数创建一个子类实例，并对其进行视图变换
            tmp_subclass = to_subclass(x)
            return tmp_subclass.view(-1)

        # 创建一个包含两个元素为 1 的张量 x
        x = torch.ones(2)
        # 使用 AOT eager 后端编译函数 f，并使用 x 作为输入
        out_ref = f(x)
        out_test = torch.compile(f, backend="aot_eager")(x)
        # 断言编译前后输出相等
        self.assertEqual(out_ref, out_test)

    def test_numpy_tobytes_no_error(self):
        def fn(x):
            # 将 x 中的每个元素加一，然后将其转换为字节流
            x += 1
            z = x.tobytes()
            x += 1
            return z

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn 并存储为 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 创建一个包含 [1, 2] 的 numpy 数组，并复制给 opt_arg 和 arg
        opt_arg, arg = np.array([1, 2]), np.array([1, 2])
        # 断言优化后的 fn(opt_arg) 和原始 fn(arg) 的输出相等
        self.assertEqual(opt_fn(opt_arg), fn(arg))
        # 断言帧计数为 2
        self.assertEqual(cnt.frame_count, 2)

    def test_numpy_not_ndarray_recompiles(self):
        import torch

        def fn(x=None):
            if x is None:
                x = np.ones(3)
            elif isinstance(x, int):
                x = np.ones(6)
            elif isinstance(x, str):
                x = np.ones(9)
            return x**2

        # 创建一个编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 优化函数 fn 并存储为 opt_fn
        opt_fn = torch._dynamo.optimize(cnt)(fn)

        # 创建一个 2x2 的全零 numpy 数组 x
        x = np.zeros((2, 2))

        # 断言优化后的 fn(x) 和原始 fn(x) 的输出相等
        self.assertEqual(opt_fn(x), fn(x))
        # 断言帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言优化后的 fn() 和原始 fn() 的输出相等
        self.assertEqual(opt_fn(), fn())
        # 断言帧计数为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言优化后的 fn(10) 和原始 fn(10) 的输出相等
        self.assertEqual(opt_fn(10), fn(10))
        # 断言帧计数为 3
        self.assertEqual(cnt.frame_count, 3)
        # 断言优化后的 fn("10") 和原始 fn("10") 的输出相等
        self.assertEqual(opt_fn("10"), fn("10"))
        # 断言帧计数为 4
        self.assertEqual(cnt.frame_count, 4)

    @parametrize(
        "backend",
        ["eager", "aot_eager", "inductor"],
    )
    @parametrize(
        "func_name",
        ["func1", "func2", "func3"],
    )
    def test_tensor_set_data(self, backend, func_name):
        # 参数化测试，依次测试func1、func2、func3三个函数
        # 详见 https://github.com/pytorch/pytorch/issues/113030
        def func1(x, y):
            # 修改x的数据为y，并原地加1
            x.data = y
            x.add_(1)
            return x

        def func2(x, y):
            # 修改x和y的数据，y的数据变为长度为0的全零张量
            x.data = y
            y.data = torch.zeros([0])
            return x

        def func3(x, y):
            # 复制x给z，修改x和y的数据，y的数据变为长度为0的全零张量，并返回x是否为z的张量
            z = x
            x.data = y
            y.data = torch.zeros([0])
            return torch.tensor(x is z)

        # 将三个函数放入字典中，便于根据名称选择
        funcs = {"func1": func1, "func2": func2, "func3": func3}
        func = funcs[func_name]

        # 如果不是eager模式并且选择的函数是func1，则跳过测试
        if backend != "eager" and func is func1:
            return

        # 重置torch._dynamo
        torch._dynamo.reset()
        # 使用指定的后端和函数编译计数器创建compiled_fn
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
        compiled_fn = torch.compile(func, backend=cnt, fullgraph=True)

        # 确定是否需要梯度
        requires_grad = func is not func1
        for i in range(0, 5):
            # 创建Eager模式和Compiled模式的输入张量
            eager_a = torch.ones([6], requires_grad=requires_grad)
            compiled_a = torch.ones([6], requires_grad=requires_grad)

            eager_b = torch.ones([6], requires_grad=requires_grad)
            compiled_b = torch.ones([6], requires_grad=requires_grad)

            # 在Eager模式下运行函数
            out_eager = func(eager_a, eager_b)
            # 在Compiled模式下运行函数
            out_compiled = compiled_fn(compiled_a, compiled_b)

            # 断言Eager模式和Compiled模式的输出张量相等
            self.assertEqual(eager_a, compiled_a)
            self.assertEqual(eager_b, compiled_b)
            self.assertTrue(torch.equal(out_eager, out_compiled))

            # 如果需要梯度，则进行反向传播并断言梯度相等
            if requires_grad:
                bwd_inp_eager = torch.randn([6])
                bwd_inp_compiled = torch.clone(bwd_inp_eager)
                eager_a.backward(bwd_inp_eager)
                compiled_a.backward(bwd_inp_compiled)
                self.assertEqual(eager_a.grad, compiled_a.grad)

        # 验证保护机制生效 - 编译函数只运行了1次
        self.assertEqual(cnt.frame_count, 1)

    @unittest.skipIf(
        TEST_WITH_ROCM or not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flash_attn_backward_mixed_strides(self):
        # 在这个示例中，"grad_out" 和 "value" 是转置张量，
        # 但 "key" 和 "value" 是连续的张量
        def gen_inputs(device):
            # 生成输入数据，返回元组
            return (
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),  # 转置第一和第二维度的张量
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),  # 转置第一和第二维度的张量
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, device=device),
                None,
                None,
                513,
                513,
                0.0,
                False,
                torch.tensor(1, dtype=torch.int64),
                torch.tensor(1, dtype=torch.int64),
            )

        # 生成 CUDA 设备和 meta 设备的输入数据
        inps_cuda = gen_inputs("cuda")
        inps_meta = gen_inputs("meta")

        # 调用 torch 操作函数处理 CUDA 设备的输入数据
        (out1_ref, out2_ref, out3_ref) = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            *inps_cuda, scale=0.125
        )

        # 导入 meta 环境下的 torch 元数据注册
        from torch._meta_registrations import meta__scaled_dot_product_flash_backward

        # 调用 meta 环境下的函数处理 meta 设备的输入数据
        out1_test, out2_test, out3_test = meta__scaled_dot_product_flash_backward(
            *inps_meta, scale=0.125
        )

        # 断言输出张量的形状和步长相等
        self.assertEqual(out1_ref.shape, out1_test.shape)
        self.assertEqual(out1_ref.stride(), out1_test.stride())
        self.assertEqual(out2_ref.shape, out2_test.shape)
        self.assertEqual(out2_ref.stride(), out2_test.stride())
        self.assertEqual(out3_ref.shape, out3_test.shape)
        self.assertEqual(out3_ref.stride(), out3_test.stride())

    def test_user_ctor_ctx_manager(self):
        # 定义一个用户自定义的上下文管理器
        class UserCtxManager:
            def __enter__(self):
                return 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        # 定义一个函数 fn，使用上下文管理器
        def fn(x, y):
            ucm = UserCtxManager()
            return x * x

        # 使用 torch._dynamo.testing.CompileCounter 对函数进行优化计数
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)

        # 创建随机张量 x
        x = torch.rand([2, 2])

        # 调用优化后的函数
        opt_fn(x, x)

        # 断言编译帧计数为 1
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_arange_in_bounds(self):
        # 这个测试函数用于测试动态形状下的操作，参考：https://github.com/pytorch/pytorch/issues/113002
        class PaddingNet(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, lengths):
                # 计算 lengths 张量中的最大值作为序列的最大长度
                max_seq_len = lengths.max().item()
                # 创建一个从 0 到 max_seq_len-1 的行向量
                row_vector = torch.arange(0, max_seq_len, 1)
                # 创建一个形状为 [lengths.size(0), 1] 的张量，用来和 row_vector 进行比较
                matrix = torch.unsqueeze(lengths, dim=-1)
                # 创建一个掩码张量，标记哪些位置的 row_vector 元素小于对应的 lengths 元素
                mask = row_vector < matrix
                # 将掩码张量转换为 float32 类型
                mask = mask.type(torch.float32)
                # 增加一个维度，形状变为 [lengths.size(0), 1, 1]
                mask_3d_btd = mask[:, :, None]
                return mask_3d_btd

        # 创建 PaddingNet 的实例
        model = PaddingNet()
        # 创建一个长度为 4 的整数张量
        lengths = torch.tensor([5, 4, 4, 4], dtype=torch.int32)

        # 创建一个编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 model 进行优化编译，使用 nopython=True，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(model)
        # 调用优化后的函数，传入 lengths 张量
        opt_fn(lengths)
        # 断言编译帧数为 1
        self.assertEqual(cnt.frame_count, 1)

    def test_overlapping_inputs_with_dynamic_shapes_error(self):
        # 使用 torch.compile 注解，指定后端为 "aot_eager"
        @torch.compile(backend="aot_eager")
        def fn(a, b, c, d, e, f):
            # 在每个输入张量上执行乘以 2 的操作
            a.mul_(2)
            b.mul_(2)
            c.mul_(2)
            d.mul_(2)
            e.mul_(2)
            f.mul_(2)

            # 创建一个 2x20 全 1 的张量 base
            base = torch.ones(2, 20)
            # 重新定义 a, b, c, d, e, f 的值，分别取 base 的不同列范围
            a = base[:, 0:2]
            b = base[:, 2:4]
            c = base[:, 4:6]
            d = base[:, 6:8]
            e = base[:, 8:10]
            f = base[:, 10:12]
            # f2 取 base 的另一列范围
            f2 = base[:, 10:14]
            # 调用 fn 函数，并传入 a, b, c, d, e, f
            out = fn(a, b, c, d, e, f)
            # 使用断言检查是否抛出特定错误信息
            with self.assertRaisesRegex(
                AssertionError, "is being compiled with dynamic shapes"
            ):
                # 再次调用 fn 函数，但传入 f2 作为参数
                out2 = fn(a, b, c, d, e, f2)

    def test_user_ctor_ctx_manager_custom_init(self):
        # 定义一个用户自定义的上下文管理器类
        class UserCtxManager:
            # 初始化方法，接收一个参数 x，将 y[0] 设置为 10
            def __init__(self, x):
                x[0] = 10

            # 进入上下文时调用的方法
            def __enter__(self):
                return 1

            # 退出上下文时调用的方法
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        # 定义一个函数 fn，接收 x 和 y 两个参数
        def fn(x, y):
            # 创建 UserCtxManager 的实例 ucm，传入 y 作为参数
            ucm = UserCtxManager(y)
            # 返回 x 与 y[0] 的乘积
            return x * y[0]

        # 创建一个编译计数器实例
        cnt = torch._dynamo.testing.CompileCounter()
        # 对 fn 函数进行优化编译，使用 nopython=True，返回优化后的函数 opt_fn
        opt_fn = torch._dynamo.optimize(cnt, nopython=True)(fn)
        # 创建一个形状为 [2, 2] 的随机张量 x
        x = torch.rand([2, 2])
        # 使用断言检查优化后的函数 opt_fn 的输出是否与原始函数 fn 的输出相同
        self.assertEqual(opt_fn(x, [5]), fn(x, [5]))
        # 断言编译帧数为 1
        self.assertEqual(cnt.frame_count, 1)
    # 定义一个测试函数，用于测试自定义上下文管理器的初始化、图形打破等情况
    def test_user_ctor_ctx_manager_custom_init_graph_break(self):
        # 定义一个计数器列表，用于记录初始化次数
        counter = [0]

        # 定义一个用户自定义的上下文管理器类
        class UserCtxManager:
            # 初始化方法，接受一个计数器并增加其值
            def __init__(self, k):
                k[0] += 1

            # 进入上下文时返回固定值1
            def __enter__(self):
                return 1

            # 退出上下文时无操作
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        # 定义一个函数，接受参数x和计数器，计算x的平方，并使用用户自定义上下文管理器
        def fn(x, counter):
            x = x * x
            ucm = UserCtxManager(counter)
            return x * x

        # 创建编译计数器对象
        cnt = torch._dynamo.testing.CompileCounter()
        # 对fn函数进行优化编译
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        # 创建一个2x2的随机张量
        x = torch.rand([2, 2])
        # 断言优化后的fn函数与原始fn函数在给定参数x和计数器时的输出相同
        self.assertEqual(opt_fn(x, counter), fn(x, counter))
        # 断言计数器的值为2
        self.assertEqual(counter[0], 2)
        # 多次调用优化后的fn函数，观察计数器的增加情况
        for i in range(0, 10):
            opt_fn(x, counter)
        # 断言计数器的值为12
        self.assertEqual(counter[0], 12)
        # 断言帧计数的值，取决于是否动态静态默认
        self.assertEqual(cnt.frame_count, torch._dynamo.utils.ifdynstaticdefault(3, 2))

    @unittest.expectedFailure
    # 定义一个测试函数，用于验证多个重叠输入不会导致守卫爆炸
    def test_many_overlapping_inputs_does_not_explode_guards(self):
        # 导入必要的模块
        from torch._dynamo.backends.common import aot_autograd

        # 定义初始值，用于记录守卫数量和编译次数
        num_shape_guards = None
        num_aot_guards = None
        num_compiles = 0

        # 定义一个函数，用于统计后端守卫的数量
        def guard_count_backend(gm, *args):
            nonlocal num_shape_guards
            nonlocal num_aot_guards
            nonlocal num_compiles
            num_shape_guards = len(
                torch._guards.TracingContext.try_get().fake_mode.shape_env.guards
            )
            num_aot_guards = len(
                torch._guards.TracingContext.try_get().guards_context.aotautograd_guards
            )
            num_compiles += 1
            return gm

        # 使用aot_autograd装饰器创建aot_guard_counter对象
        aot_guard_counter = aot_autograd(fw_compiler=guard_count_backend)

        # 定义一个使用aot_guard_counter编译的函数
        @torch.compile(backend=aot_guard_counter, dynamic=True)
        def f(*args):
            for a in args:
                a.add_(1)

        # 创建一个需要梯度的1000x1张量x，并将其拆分为长度为10的列表args
        x = torch.ones(1000, requires_grad=True)
        args = x.split(10)

        # 使用torch.no_grad()上下文，调用函数f
        with torch.no_grad():
            f(*args)
        
        # 断言aot守卫的数量小于5000
        self.assertTrue(num_aot_guards < 5000)
        # 断言动态形状守卫的数量为0
        self.assertEqual(num_shape_guards, 0)
        # 再次调用函数f，确认没有重新编译
        with torch.no_grad():
            f(*args)
        # 断言编译次数为1
        self.assertEqual(num_compiles, 1)

    # 定义一个测试函数，用于测试无效的序列解包
    def test_invalid_seq_unpack(self):
        # 定义一个函数myfn，接受一个元组arg，并尝试对其进行解包赋值给a和b
        def myfn(arg):
            (a, b) = arg

        # 定义一个函数fn，调用myfn，并传入一个包含3个元素的元组
        def fn():
            return myfn((1, 2, 3))

        try:
            # 尝试编译并执行fn函数，期望捕获ValueError异常
            torch.compile(fn)()
        except ValueError:
            pass
        else:
            # 如果没有捕获到异常，则测试失败
            self.fail("expected exception")
    # 定义测试方法：测试 megablocks 中的 moe 层
    def test_megablocks_moe(self):
        try:
            # 尝试导入 megablocks 库中的 moe 层和 Arguments 类
            from megablocks.layers import moe
            from megablocks.layers.arguments import Arguments
        except ImportError as e:
            # 如果导入失败，则跳过测试，并抛出 SkipTest 异常
            raise unittest.SkipTest("requires megablocks") from e
        # 设置测试所需的参数：批量大小、序列长度、隐藏层大小、专家数量、top-k 值
        bs, sl, hs, num_experts, top_k = (16, 1024, 512, 1, 1)
        # 创建 Arguments 对象，传入相关参数
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
        )
        # 创建 moe.MoE 对象，使用上述参数
        moe_mlp = moe.MoE(args)
        # 将 moe_mlp 对象移到当前 CUDA 设备上，并设置为半精度运算
        moe_mlp.cuda(torch.cuda.current_device()).half()
        # 生成随机数据张量 x，传输到 CUDA 设备上，并设置为半精度
        x = torch.randn(sl, bs, hs).cuda().half()
        # 使用 moe_mlp 对象处理输入 x，得到输出 out1
        out1, _ = moe_mlp(x)
        # 使用 torch.compile 对象编译 moe_mlp 函数，使用 eager 模式执行，并得到输出 out2
        out2, _ = torch.compile(moe_mlp, backend="eager")(x)
        # 断言 out1 和 out2 相等
        self.assertEqual(out1, out2)

    # 定义测试方法：测试用户定义的函数类重建
    def test_udf_classes_reconstruction(self):
        # 定义一个函数 fn，该函数创建 T(5) 对象并返回其 x 属性与输入 x 相加的结果
        def fn(x):
            o = T(5)
            return o.x + x

        # 使用 torch.compile 对象编译 fn 函数，使用 eager 模式执行
        opt_fn = torch.compile(fn, backend="eager")
        # 将 T 类型指定为 IncByOne 类型
        T = IncByOne

        # 生成随机数据张量 x
        x = torch.randn(4)
        # 断言 fn(x) 和 opt_fn(x) 的结果相等
        self.assertEqual(fn(x), opt_fn(x))

        # 修改 T 类型为 IncByTwo
        T = IncByTwo
        # 再次断言 fn(x) 和 opt_fn(x) 的结果相等，测试是否重新编译了函数
        self.assertEqual(fn(x), opt_fn(x))

    # 定义测试方法：测试包含 range 常量传播的情况
    def test_contains_range_constprop(self):
        # 定义一个函数 fn，根据条件判断是否在 range(0, 10) 内，返回相应的计算结果
        def fn(x):
            # dynamo 应该将 const prop 设置为 False
            if 3 in range(0, 10):
                return x + 1
            else:
                return x + 2

        # 使用 torch.compile 对象编译 fn 函数，使用 eager 模式执行
        opt_fn = torch.compile(fn, backend="eager")
        # 生成零填充的张量 x
        x = torch.zeros(4)
        # 断言 fn(x) 和 opt_fn(x) 的结果相等
        self.assertEqual(fn(x), opt_fn(x))

    # 定义测试方法：测试基于已存在视图的 as_strided 操作
    # 参考链接：https://github.com/pytorch/pytorch/issues/104505
    def test_as_strided_on_base_with_mutation_works(self):
        # 定义一个函数 foo，对输入的张量 a 进行 as_strided 操作并修改其值，返回修改后的张量
        def foo(a):
            f = a.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        # 生成随机张量 a 和其克隆版本 a_ref
        a = torch.randn(2, 4)
        a_ref = a.clone()
        # 使用 foo 函数处理 a_ref，并得到处理后的结果 out_ref
        out_ref = foo(a_ref)
        # 使用 torch.compile 对象编译 foo 函数，使用 aot_eager 模式执行，并得到处理后的结果 out
        f_compiled = torch.compile(foo, backend="aot_eager")
        out = f_compiled(a)
        # 断言处理前后 a_ref 和 a 的值相等
        self.assertEqual(a_ref, a)
        # 断言处理前后 out_ref 和 out 的值相等
        self.assertEqual(out_ref, out)

    # 定义测试方法：测试在已存在视图上使用 as_strided 操作会导致的错误
    # 参考链接：https://github.com/pytorch/pytorch/issues/104505
    def test_as_strided_on_existing_view_banned(self):
        # 定义一个函数 foo，对输入的张量 a 进行 as_strided 操作并修改其值，返回修改后的张量
        def foo(a):
            e = a.diagonal()
            f = e.as_strided((2,), (1,), 0)
            f.add_(1.0)
            return a

        # 生成随机张量 a 和其克隆版本 a_ref
        a = torch.randn(2, 4)
        a_ref = a.clone()
        # 使用 foo 函数处理 a_ref，并得到处理后的结果 out_ref
        out_ref = foo(a_ref)
        # 使用 torch.compile 对象编译 foo 函数，使用 aot_eager 模式执行，并期望引发 RuntimeError 异常
        f_compiled = torch.compile(foo, backend="aot_eager")
        with self.assertRaisesRegex(
            RuntimeError,
            "encountered a mutation on a view chain of length 2, where view 1 was an as_strided",
        ):
            # 对 a 执行编译后的 foo 函数
            out = f_compiled(a)
    # 定义一个测试函数，验证不应过于激进地编写断言
    def test_dont_aggressively_write_assert(self):
        # 创建一个记录图表的实例，用于测试
        record_graph = torch._dynamo.testing.EagerAndRecordGraphs()

        # 使用 torch.compile 装饰器编译函数 f，并使用 record_graph 作为后端
        @torch.compile(dynamic=True, backend=record_graph)
        def f(x):
            # 断言 x 的第一个维度大于 3
            assert x.shape[0] > 3
            # 断言 x 的第一个元素的和大于 0
            assert x[0].sum() > 0
            # 断言 1 对 (x.shape[0] // 2) 取余不等于 0
            assert 1 % (x.shape[0] // 2) != 0
            # 断言一个复杂的数学表达式不等于 0
            assert 32 * (x.shape[0] // 2) ** 2 - 16 * (x.shape[0] // 2) != 0
            # 返回 x 的余弦值
            return x.cos()

        # 调用函数 f，传入一个形状为 (6, 4) 的全为 1 的张量
        f(torch.ones(6, 4))
        
        # 从 record_graph 中获取记录的第一个图表
        graph = record_graph.graphs[0]

        # 断言生成的代码与预期的内联代码匹配，去除首尾空白字符
        # 这里注释说明了即使生成了一些无用的形状检查语句，由于 DCE 应该能移除它们，因为这些检查不会在实际的断言中使用。
        # 这种做法是可接受的，因为 dynamo 只会跳过断言语句，而不会跳过它们之前的指令。
        self.assertExpectedInline(
            str(graph.code).strip(),
            """\
# 定义一个方法 `forward`，接受三个参数 s0、s1 和 L_x_，类型为 torch.SymInt 和 torch.Tensor
def forward(self, s0 : torch.SymInt, s1 : torch.SymInt, L_x_ : torch.Tensor):
    # 复制 L_x_ 到 l_x_
    l_x_ = L_x_
    # 获取 l_x_ 的第一个元素，存储在 getitem_2 中
    getitem_2 = l_x_[0]
    # 计算 getitem_2 的所有元素的和
    sum_1 = getitem_2.sum();  getitem_2 = None
    # 检查 sum_1 是否大于 0，结果存储在 gt_1 中
    gt_1 = sum_1 > 0;  sum_1 = None
    # 断言 gt_1 必须为真，否则抛出 'assertion error' 异常
    _assert_async = torch._assert_async(gt_1, 'assertion error');  gt_1 = None
    # 计算 l_x_ 中每个元素的余弦值，结果存储在 cos 中
    cos = l_x_.cos();  l_x_ = None
    # 返回一个包含 cos 的元组
    return (cos,)
    # super().__init__()
    # self.param = torch.nn.Parameter(torch.randn(4, 4))

    # def forward(self, x):
    #     self.param.untyped_storage().resize_(
    #         self.param.numel() * self.param.itemsize
    #     )
    #     with torch.no_grad():
    #         torch._foreach_copy_([self.param], [x])
    #     out = torch.matmul(self.param, self.param)
    #     self.param.untyped_storage().resize_(0)
    #     return out

    # def post_accumulate_grad_hook(param):
    #     param.untyped_storage().resize_(0)

    # # Beginning of backward, resize and put data into the param
    # def pre_backward_hook(module, grad) -> None:
    #     module.param.untyped_storage().resize_(
    #         self.param.numel() * self.param.itemsize
    #     )
    #     with torch.no_grad():
    #         # simulates loading data into param from allgather
    #         module.param.fill_(2)

    # def post_forward_hook(module, args, output):
    #     output.register_hook(functools.partial(pre_backward_hook, module))

    # x = torch.randn(4, 4)

    # mod_ref = TestModule()
    # mod_test = deepcopy(mod_ref)

    # # Start the param off with zero storage size to mimic fsdp
    # mod_ref.param.untyped_storage().resize_(0)
    # mod_test.param.untyped_storage().resize_(0)

    # # Resize storage at beginning of backward
    # # Free storage at end of backward
    # mod_ref.register_forward_hook(post_forward_hook, prepend=False)
    # mod_ref.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)
    # mod_test.register_forward_hook(post_forward_hook, prepend=False)
    # mod_test.param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)

    # mod_test = torch.compile(mod_test, backend=aot_graph_capture_backend)

    # out_ref = mod_ref(x)
    # out_test = mod_test(x)
    # self.assertExpectedInline(
    #     str(fw_graph[0].code.strip()),
    #     """\
    # def forward(self, primals_1, primals_2):
    #     _foreach_copy = torch.ops.aten._foreach_copy.default([primals_1], [primals_2]);  primals_1 = primals_2 = None
    #     getitem = _foreach_copy[0];  _foreach_copy = None
    #     mm = torch.ops.aten.mm.default(getitem, getitem)
    #     return [mm, getitem]""",
    # )
    # self.assertEqual(out_ref, out_test)


注释：这段代码片段是关于深度学习框架 Torch 中的模型和钩子函数的操作。具体来说：

- `super().__init__()` 和 `self.param = torch.nn.Parameter(torch.randn(4, 4))` 初始化了一个神经网络模块，并创建了一个参数矩阵。
- `forward` 方法定义了模型的前向传播逻辑，包括调整参数大小、数据拷贝操作和矩阵乘法。
- `post_accumulate_grad_hook` 和 `pre_backward_hook` 是用来处理梯度计算过程中的钩子函数，用于释放或调整参数存储。
- `post_forward_hook` 注册了一个前向传播结束后的钩子函数，用来在后续计算中加载数据到参数中。
- 最后部分是模型实例化、钩子函数注册、编译和模型推理的过程，以及一些断言操作来验证预期输出是否正确。
    def test_super_in_staticmethod(self):
        # 定义类 A，其中包含一个静态方法 foo
        class A:
            @staticmethod
            def foo():
                # 调用父类的构造函数，并返回其结果
                return super().__init__()

        # 定义函数 fn，接受一个对象 obj，并调用其 foo 方法
        def fn(obj):
            return obj.foo()

        # 创建类 A 的实例 obj
        obj = A()

        try:
            # 尝试调用 fn 方法，并捕获异常信息到 orig_str
            fn(obj)
        except Exception as e:
            orig_str = str(e)
        
        # 断言异常信息中包含特定字符串 "no arguments"
        self.assertIn("no arguments", orig_str)

        try:
            # 使用 torch.compile 编译 fn 函数，并在此过程中捕获异常信息到 compiled_str
            torch.compile(backend="eager")(fn)(obj)
        except Exception as e:
            compiled_str = str(e)
        
        # 断言编译前后的异常信息应该相同
        self.assertEqual(orig_str, compiled_str)

    def test_nn_module_callable(self):
        # 定义一个继承自 nn.Module 的类 M，重写了 forward 方法
        class M(nn.Module):
            def forward(self, x):
                # 返回输入张量 x 的正弦值
                return x.sin()

        # 定义函数 f，判断其参数是否可调用
        def f(m):
            return callable(m)

        # 使用 torch.compile 编译函数 f，并传入 M 类的实例进行测试，fullgraph=True 表示编译整个图
        res = torch.compile(f, fullgraph=True)(M())
        # 断言 res 应为 True，即 M 的实例是可调用的
        self.assertTrue(res)

    def test_stk_sdd_is_transposed(self):
        # 触发图形破坏的标志位
        trigger_graph_break = False

        # 定义函数 _is_transposed，判断张量 x 是否转置
        def _is_transposed(x):
            return (
                not x.is_contiguous()
                and x.stride()[0] == 1
                and x.stride()[1] == x.size()[0]
            )

        # 定义一个继承自 torch.autograd.Function 的类 SDD，实现其前向和后向传播方法
        class SDD(torch.autograd.Function):
            @staticmethod
            def forward(ctx, lhs, rhs):
                # 保存输入张量 lhs 和 rhs 到上下文对象 ctx
                ctx.save_for_backward(lhs, rhs)
                # 创建一个和 lhs 相同大小和数据类型的全 1 张量 out，并返回
                out = torch.full_like(lhs, 1.0, dtype=lhs.dtype, device=lhs.device)
                return out

            @staticmethod
            def backward(ctx, dy):
                # 从上下文对象中恢复保存的张量
                saved_tensors = ctx.saved_tensors
                lhs, rhs = saved_tensors[:2]
                # 判断 lhs 和 rhs 是否转置
                trans_a = _is_transposed(lhs)
                trans_b = _is_transposed(rhs)
                dlhs = None
                # 如果需要计算 lhs 的梯度
                if ctx.needs_input_grad[0]:
                    # 根据 trans_a 的值创建与 lhs 相同大小的张量 dlhs
                    dlhs = torch.full_like(lhs, 1.0 if trans_a else 2.0)
                drhs = None
                # 如果需要计算 rhs 的梯度
                if ctx.needs_input_grad[1]:
                    # 根据 trans_b 的值创建与 rhs 相同大小的张量 drhs
                    drhs = torch.full_like(rhs, 1.0 if trans_b else 2.0)
                # 如果 trigger_graph_break 为 True，并且 dy 被判断为转置，则返回特定的梯度
                if trigger_graph_break:
                    if _is_transposed(dy):
                        return dlhs + 1, drhs + 1, None, None
                # 否则返回 dlhs 和 drhs
                return dlhs, drhs, None, None

        # 创建两组需要计算梯度的张量 x1, y1 和 x2, y2
        x1 = torch.randn((8, 8), requires_grad=True)
        y1 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)
        x2 = torch.randn((8, 8), requires_grad=True)
        y2 = torch.randn((8, 8)).transpose(0, 1).requires_grad_(True)

        # 调用 SDD 的前向传播并计算其结果的和，然后反向传播
        SDD.apply(x1, y1).sum().backward()

        # 使用 torch.compile 编译 fn 函数，并在此过程中捕获异常信息到 compiled_str
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return SDD.apply(x2, y2)

        # 调用 fn 函数的前向传播并计算其结果的和，然后反向传播
        fn().sum().backward()

        # 断言 x1 和 x2 的梯度应相等
        self.assertEqual(x1.grad, x2.grad)
        # 断言 y1 和 y2 的梯度应相等
        self.assertEqual(y1.grad, y2.grad)

        # 设置 trigger_graph_break 为 True，并断言在此情况下调用 fn 函数会抛出特定异常
        trigger_graph_break = True
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn().sum().backward()
    def test_partially_initialized_module_property(self):
        class Matrix(torch.nn.Module):
            def __init__(self, data):
                super().__init__()
                self._data = data
                # 初始化属性 foo，使用了属性 blocking 的计算结果
                self.foo = 10 * self.blocking

            @property
            def data(self):
                return self._data

            @property
            def blocking(self):
                # 返回数据的第二个维度大小
                return self.data.shape[1]

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # 创建 Matrix 实例并返回
            return Matrix(torch.randn(10, 20))

        v = fn()
        # 断言 foo 的值为 200
        self.assertEqual(v.foo, 200)
        # 断言 data 的形状为 (10, 20)
        self.assertEqual(v.data.shape, (10, 20))
        # 断言 v 的类型为 Matrix
        self.assertEqual(type(v), Matrix)

    def test_nn_parametrize(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(10, 10))

            def forward(self, x):
                return self.param @ x

        class Parametrization(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        m = Module()
        # 注册参数化方式到模块的参数 "param"
        torch.nn.utils.parametrize.register_parametrization(
            m, "param", Parametrization()
        )

        sin_found = False

        def backend(gm, _):
            nonlocal sin_found
            # 检查是否在计算图中找到了 torch.sin 函数的节点
            for node in gm.graph.nodes:
                if node.target is torch.sin:
                    sin_found = True
            return gm

        # 编译模块 m，使用自定义的后端函数 backend
        opt_m = torch.compile(m, backend=backend, fullgraph=True)
        inp = torch.randn(10, 10)
        # 断言模块 m 和优化后的模块 opt_m 在输入 inp 上的输出相等
        self.assertEqual(m(inp), opt_m(inp))
        # 断言已找到 torch.sin 函数的节点
        self.assertTrue(sin_found)

        # 移除参数化方式
        torch.nn.utils.parametrize.remove_parametrizations(m, "param")
        sin_found = False
        # 再次断言模块 m 和优化后的模块 opt_m 在输入 inp 上的输出相等
        self.assertEqual(m(inp), opt_m(inp))
        # 断言未找到 torch.sin 函数的节点
        self.assertFalse(sin_found)

    def test_nn_module_property_closure(self):
        x = torch.randn(10, 10)

        class Mod(torch.nn.Module):
            @property
            def y(self):
                # 返回一个全为 1 的矩阵加上输入 x 的结果
                return torch.ones(10, 10) + x

            def forward(self, x):
                # 返回输入 x 和属性 y 的矩阵乘积
                return x @ self.y

        mod = Mod()

        def fn(x):
            # 返回模块 mod 对输入 x 的前向计算结果
            return mod(x)

        # 编译函数 fn，并使用 "eager" 后端
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        inp = torch.randn(10, 10)
        # 断言函数 fn 和优化后的函数 opt_fn 在输入 inp 上的输出相等
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_global_fn_mutation(self):
        def foo(x, y):
            # 返回全局函数 global_fn(x) 的结果加上输入 y
            return global_fn(x) + y

        x = torch.ones(1)
        y = torch.ones(1)

        # 编译函数 foo，使用 "eager" 后端
        opt = torch.compile(foo, fullgraph=True, backend="eager")
        # 断言优化后的函数 opt 在输入 x 和 y 上的输出与原始函数 foo 的输出相等
        self.assertEqual(opt(x, y), foo(x, y))

        # 更改全局函数 global_fn
        global global_fn

        def new_fn(x):
            # 返回输入 x 的余弦值
            return torch.cos(x)

        global_fn = new_fn
        # 再次断言优化后的函数 opt 在输入 x 和 y 上的输出与修改后的 foo 函数相等
        self.assertEqual(opt(x, y), foo(x, y))

    # ref https://github.com/pytorch/pytorch/issues/123974
    # 定义一个测试函数，用于测试列表反转操作
    def test_list_reverse(self):
        # 定义内部函数 ladder，接受参数 x
        def ladder(x):
            # 获取 x 的最后一个维度的大小
            trail = x.size(-1)
            # 断言最后一个维度的大小大于2
            assert trail > 2
            # 初始化空列表 weights
            weights = []
            # 循环遍历 [trail, trail - 1, trail - 2]，每个元素为 s
            for s in [trail, trail - 1, trail - 2]:
                # 向 weights 中添加一个全为1的大小为 (s, s-1) 的张量
                weights.append(torch.ones(s, s - 1))

            # 遍历 weights 中的张量 w
            for w in weights:
                # 执行矩阵乘法操作 x @ w
                x = x @ w

            # 反转 weights 列表中的张量
            weights.reverse()

            # 再次遍历 weights 中的张量 w
            for w in weights:
                # 执行矩阵乘法操作 x @ w 的转置
                x = x @ w.t()

            # 返回处理后的张量 x
            return x

        # 创建一个形状为 (3, 4) 的随机张量 data
        data = torch.randn(3, 4)
        # 使用 torch.compile 对 ladder 函数进行编译，使用 fullgraph 模式，后端为 "eager"
        opt_ladder = torch.compile(ladder, fullgraph=True, backend="eager")
        # 断言编译后的结果与原始 ladder 函数对相同输入 data 的计算结果相等
        self.assertEqual(opt_ladder(data), ladder(data))

    # 标记为预期失败的测试函数
    @unittest.expectedFailure
    def test_trace_functional_tensor_with_error(self):
        # 导入必要的模块
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        # 定义函数 f，接受参数 a 和 tmp
        def f(a, tmp):
            # 将 a 展开成一个一维张量 a_view
            a_view = a.view(-1)
            # 在无梯度计算环境下，设置 a 的值为 tmp，并将 a_view 中的每个元素乘以2
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            # 返回 a 和 tmp 相加后的结果
            return a + tmp

        # 创建一个 FakeTensorMode 实例
        fake_mode = FakeTensorMode()
        # 进入 FunctionalTensorMode 上下文
        with FunctionalTensorMode():
            # 创建一个形状为 (3, 3) 的需要梯度的张量 inp
            inp = torch.ones(3, 3, requires_grad=True)
            # 使用 fake_mode 将 inp 转换为 FunctionalTensor，启用静态形状
            inp = fake_mode.from_tensor(inp, static_shapes=True)
            # 将 FunctionalTensor 转换为 functional 表示
            inp = FunctionalTensor.to_functional(inp)

            # 创建一个形状为 (3, 3) 的需要梯度的张量 tmp
            tmp = torch.ones(3, 3, requires_grad=True)
            # 使用 fake_mode 将 tmp 转换为 FunctionalTensor，启用静态形状
            tmp = fake_mode.from_tensor(tmp, static_shapes=True)
            # 将 FunctionalTensor 转换为 functional 表示
            tmp = FunctionalTensor.to_functional(tmp)

            # 使用 torch.compile 对函数 f 进行编译，后端为 "eager"
            opt_f = torch.compile(f, backend="eager")
            # 断言调用编译后的函数 opt_f 会抛出 RuntimeError 异常，异常信息包含 "cannot mutate tensors with frozen storage"
            with self.assertRaisesRegex(
                RuntimeError, "cannot mutate tensors with frozen storage"
            ):
                opt_f(inp, tmp)

        # 断言梯度状态是否正确重置
        self.assertTrue(torch.is_grad_enabled())

    # 定义测试函数 test_const_dict_keyerror
    def test_const_dict_keyerror(self):
        # 创建一个空字典 d
        d = {}

        # 定义函数 fn，接受参数 x
        def fn(x):
            # 尝试获取字典 d 中键为 0 的值，捕获 KeyError 异常
            try:
                y = d[0]
            except KeyError:
                y = 1
            # 返回 x 和 y 的和
            return x + y

        # 使用 torch.compile 对函数 fn 进行编译，后端为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        # 创建一个形状为 (3, 3) 的随机张量 inp
        inp = torch.randn(3, 3)
        # 断言调用编译后的函数 opt_fn 和原始函数 fn 在相同输入 inp 下的计算结果相等
        self.assertEqual(fn(inp), opt_fn(inp))

    # 定义测试函数 test_nonconst_issubclass
    def test_nonconst_issubclass(self):
        # 定义函数 fn，接受参数 x
        def fn(x):
            # 如果 x 的类型是 np.ndarray 的子类，则返回 1，否则返回 0
            if issubclass(x.__class__, np.ndarray):
                return 1
            return 0

        # 使用 torch.compile 对函数 fn 进行编译，后端为 "eager"
        opt_fn = torch.compile(fn, backend="eager")
        # 调用编译后的函数 opt_fn，传入一个形状为 [3, 3] 的 numpy 数组 np.ones([3, 3])
        opt_fn(np.ones([3, 3]))

    # 定义测试函数 test_issue126128
    def test_issue126128(self):
        # 定义函数 fn，不接受任何参数
        def fn():
            # 创建一个形状为 (1, 10) 的随机张量 x
            x = torch.randn(1, 10)
            # 创建一个形状为 (10, 1) 的随机张量 y
            y = torch.randn(10, 1)
            # 计算 x 和 y 的矩阵乘积后求和，并返回结果
            return torch.mm(x, y).sum()

        # 定义函数 fn2，不接受任何参数
        def fn2():
            # 创建一个形状为 (10, 100) 的随机张量 x
            x = torch.randn(10, 100)
            # 创建一个形状为 (100, 10) 的随机张量 y
            y = torch.randn(100, 10)
            # 计算 x 和 y 的矩阵乘积后求和，并返回结果
            return torch.mm(x, y).sum()

        # 在 fresh_inductor_cache 上下文中执行以下操作
        with fresh_inductor_cache():
            # 使用 torch.compile 对函数 fn 进行编译并调用
            torch.compile(fn)()

        # 使用 torch.compile 对函数 fn2 进行编译并调用
        torch.compile(fn2)()
    def test_jit_script_defaults(self):
        @torch.jit.script
        def fast_cos(x, c: float = 2.0):
            return torch.cos(x) * c
        # 定义一个使用 Torch JIT 编译的函数 fast_cos，计算输入 x 的余弦值并乘以常数 c

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fast_cos = fast_cos
            # 定义一个继承自 torch.nn.Module 的类 Mod，初始化时将 fast_cos 函数作为成员

            def forward(self, x):
                return self.fast_cos(x)
            # 定义 forward 方法，使用 fast_cos 函数进行前向传播计算

        mod = Mod()
        # 创建 Mod 类的实例 mod

        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        # 使用 Torch 的 JIT 编译器编译 mod 实例，指定后端为 "eager"，完整图模式为 True

        x = torch.randn(4)
        # 创建一个形状为 (4,) 的随机张量 x

        self.assertEqual(mod(x), opt_mod(x))
        # 断言调用 mod 实例和编译后的 opt_mod 实例的结果相等

    def test_enum(self):
        class ExplicitEnum(str, Enum):
            @classmethod
            def _missing_(cls, value):
                raise ValueError(
                    f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
                )
        # 定义一个继承自 Enum 的枚举类 ExplicitEnum，定义了 _missing_ 方法来处理未定义的枚举值错误

        class PaddingStrategy(ExplicitEnum):
            LONGEST = "longest"
            MAX_LENGTH = "max_length"
            DO_NOT_PAD = "do_not_pad"
        # 定义一个继承自 ExplicitEnum 的枚举类 PaddingStrategy，定义了几种填充策略枚举值

        def fn(x):
            a = PaddingStrategy("longest")
            # 创建 PaddingStrategy 枚举类的实例 a，值为 "longest"

            if a == PaddingStrategy.LONGEST:
                return torch.sin(x)
            # 如果 a 的值为 PaddingStrategy.LONGEST，则返回 x 的正弦值

            return torch.cos(x)
            # 否则返回 x 的余弦值

        x = torch.randn(3, 3)
        # 创建一个形状为 (3, 3) 的随机张量 x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # 使用 Torch 的 JIT 编译器编译函数 fn，指定后端为 "eager"，完整图模式为 True

        self.assertEqual(fn(x), opt_fn(x))
        # 断言调用 fn 函数和编译后的 opt_fn 函数的结果相等

    def test_hasattr_builtin(self):
        class MyClass:
            foo: int = 1
        # 定义一个类 MyClass，包含一个属性 foo，值为 1

        def func(x, m):
            if getattr(type(m), "foo", 0):
                return x + MyClass.foo
            # 如果 m 类型的属性中存在 foo，则返回 x 加上 MyClass.foo 的值

            return x
            # 否则返回 x

        opt_func = torch.compile(func, backend="eager", fullgraph=True)
        # 使用 Torch 的 JIT 编译器编译函数 func，指定后端为 "eager"，完整图模式为 True

        m = MyClass()
        # 创建 MyClass 类的实例 m

        x = torch.zeros(())
        # 创建一个形状为空的零张量 x

        self.assertEqual(func(x, m), opt_func(x, m))
        # 断言调用 func 函数和编译后的 opt_func 函数的结果相等
        self.assertEqual(func(x, 0), opt_func(x, 0))
        # 断言调用 func 函数和编译后的 opt_func 函数的结果相等

    def test_grad(self):
        def fn(x, y):
            x._grad = y
            return x.grad.data
        # 定义一个函数 fn，将 y 赋值给 x 的梯度属性 _grad，并返回 x 的梯度数据

        x = torch.randn(4, requires_grad=True)
        # 创建一个形状为 (4,) 的随机张量 x，要求计算其梯度

        y = torch.randn(4)
        # 创建一个形状为 (4,) 的随机张量 y

        opt_fn = torch.compile(fn, backend="eager")
        # 使用 Torch 的 JIT 编译器编译函数 fn，指定后端为 "eager"

        self.assertEqual(fn(x, y), opt_fn(x, y))
        # 断言调用 fn 函数和编译后的 opt_fn 函数的结果相等
    def test_nn_module_stack_bc(self):
        # 导入 GenerationTracker 类，用于跟踪生成信息
        from torch._dynamo.mutation_guard import GenerationTracker
        
        # 定义编译器函数 compiler，接受 gm 和可变参数 args
        def compiler(gm, *args):
            # 从 gm 的图节点中获取 meta 属性为 "nn_module_stack" 的值，组成列表 module_stacks
            module_stacks = [
                node.meta.get("nn_module_stack", None) for node in gm.graph.nodes
            ]
            # 将 module_stacks 展平为一维列表，并丢弃非字符串类型的元素
            module_stacks, _ = pytree.tree_flatten(module_stacks)
            module_stacks = [x for x in module_stacks if isinstance(x, str)]
            # 断言在 module_stacks 中不含有 "_module" 字符串
            for stack in module_stacks:
                self.assertTrue("_module" not in stack)
            # 返回 gm 的 forward 方法
            return gm.forward
        
        # 定义子模块 SubMod，继承自 torch.nn.Module
        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
            
            def forward(self, x):
                return self.linear(x)
        
        # 定义模块 Mod，继承自 torch.nn.Module
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod1 = SubMod()
                self.submod2 = SubMod()
            
            def forward(self, x):
                return self.submod1(x) + self.submod2(x)
        
        # 创建 Mod 实例 mod
        mod = Mod()
        # 使用 compiler 编译 mod，返回优化后的模块 opt_mod
        opt_mod = torch.compile(mod, backend=compiler)
        # 调用 opt_mod 的 forward 方法，传入随机生成的 2x2 张量
        opt_mod(torch.randn(2, 2))
        
        # 在 inline_inbuilt_nn_modules=True 的上下文中
        with torch._dynamo.config.patch(inline_inbuilt_nn_modules=True):
            # 创建 Mod 实例 mod
            mod = Mod()
            # 使用 compiler 编译 mod，返回优化后的模块 opt_mod
            opt_mod = torch.compile(mod, backend=compiler)
            # 调用 opt_mod 的 forward 方法，传入随机生成的 2x2 张量
            opt_mod(torch.randn(2, 2))
        
        # 示例类似于 Pippy 的使用场景
        # 创建 Mod 实例 mod
        mod = Mod()
        # 对 mod.submod1 进行标记，用于生成跟踪
        GenerationTracker.tag(mod.submod1)
        # 标记 mod.submod1 类为动态生成的
        GenerationTracker.mark_class_dynamic(type(mod.submod1))
        # 创建 Mod 实例 mod
        mod = Mod()
        # 使用 compiler 编译 mod，返回优化后的模块 opt_mod
        opt_mod = torch.compile(mod, backend=compiler)
        # 调用 opt_mod 的 forward 方法，传入随机生成的 2x2 张量
        opt_mod(torch.randn(2, 2))
    
    def test_is_make_fx_tracing(self):
        # 定义函数 fn，使用 eager 模式编译，fullgraph=True
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # 调用 torch.nn.modules.activation._is_make_fx_tracing() 方法
            torch.nn.modules.activation._is_make_fx_tracing()
            # 返回输入张量 x 的正弦值
            return torch.sin(x)
        
        # 调用 fn 函数，传入一个形状为 4 的随机张量
        fn(torch.rand(4))
# 实例化参数化测试并传入参数化测试类 ReproTests
instantiate_parametrized_tests(ReproTests)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试
    run_tests()
```