# `.\pytorch\torch\_inductor\utils.py`

```py
# 设置 mypy 选项，允许未标注类型的函数定义
# 导入未来版本的注解特性
from __future__ import annotations

# 导入模块
import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import io
import itertools
import json
import logging
import math
import operator
import os
import platform
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    ValuesView,
)
from typing_extensions import Concatenate, ParamSpec
from unittest import mock

# 导入 sympy 模块
import sympy

# 导入 torch 模块
import torch
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import detect_fake_mode
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils._sympy.functions import (
    CeilDiv,
    CleanDiv,
    FloorDiv,
    Identity,
    ModularIndexing,
)
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from . import config
from .runtime.runtime_utils import cache_dir, ceildiv as runtime_ceildiv

# 获取日志记录器
log = logging.getLogger(__name__)

# 定义类型变量
_T = TypeVar("_T")
VarRanges = Dict[sympy.Expr, sympy.Expr]

# 设置 GPU 对齐字节数
GPU_ALIGN_BYTES = 16

# 设置对齐字节数
ALIGN_BYTES = 64
# 检查对齐字节数是否为 2 的幂且大于等于 8
assert (ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0 and ALIGN_BYTES >= 8, "must be power of 2"


# 定义对齐函数，将字节数向上舍入到最接近的 ALIGN_BYTES 的倍数
def _align(nbytes):
    """Round up to the nearest multiple of ALIGN_BYTES"""
    return (nbytes + ALIGN_BYTES - 1) & -ALIGN_BYTES


# 判断一个表达式是否可以静态证明为 ALIGN_BYTES 的倍数
def _is_aligned(v: sympy.Expr):
    """v can be statically proven to be a multiple of ALIGN_BYTES"""
    if isinstance(v, (sympy.Add, sympy.Max)):
        return all(map(_is_aligned, v.args))
    return isinstance(v, align) or sympy.gcd(v, ALIGN_BYTES) == ALIGN_BYTES


# 定义对齐函数类，用于符号性地将值向上舍入到最接近的 ALIGN_BYTES 的倍数
class align(sympy.Function):
    """Symbolically round up to the nearest multiple of ALIGN_BYTES"""

    nargs = (1,)
    is_integer = True

    @classmethod
    def eval(cls, value):
        if isinstance(value, (int, sympy.Integer)):
            return _align(int(value))
        if _is_aligned(value):
            return value


# 执行使用性能分析的基准测试函数
def do_bench_using_profiling(fn: Callable[[], Any], warmup=25, rep=100) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    # 执行传入的函数
    fn()
    # 同步 CUDA 设备
    torch.cuda.synchronize()
    # 创建一个大小为 256MB 的张量，用于估计函数的运行时间
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # 估计函数的运行时间
    # 创建启动事件对象，用于记录 CUDA 操作的起始时间
    start_event = torch.cuda.Event(enable_timing=True)
    # 创建结束事件对象，用于记录 CUDA 操作的结束时间
    end_event = torch.cuda.Event(enable_timing=True)
    # 记录启动事件的时间点
    start_event.record()
    # 执行五次循环，每次执行前清空缓存
    for _ in range(5):
        cache.zero_()
        fn()
    # 记录结束事件的时间点
    end_event.record()
    # 等待 CUDA 操作完成
    torch.cuda.synchronize()
    # 计算每次循环的平均时间
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # 计算预热和重复次数
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # 预热阶段
    for _ in range(n_warmup):
        fn()

    # 使用 Torch Profiler 进行性能分析
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        # 性能基准测试阶段
        for i in range(n_repeat):
            # 每次运行前清空 L2 缓存
            cache.zero_()
            # 记录 `fn` 函数的运行时间
            fn()
        # 等待 CUDA 操作完成
        torch.cuda.synchronize()

    # 输出调试信息，显示原始事件数据
    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    # 过滤出符合条件的事件列表
    filtered_events = EventList(
        [
            event
            for event in p.events()
            if event.device_type == DeviceType.CUDA and event.name != "Context Sync"
        ]
    )
    # 检查是否成功将所有性能分析事件分成 #repeat 组
    if len(filtered_events) % n_repeat != 0:
        raise RuntimeError(
            "Failed to divide all profiling events into #repeat groups. "
            "#CUDA events: %d, #repeats: %s",
            len(filtered_events),
            n_repeat,
        )
    # 计算每组事件的数量
    num_event_per_group = len(filtered_events) / n_repeat
    # 根据每组事件的数量筛选出实际关键事件
    actual_events = EventList(
        [
            event
            for i, event in enumerate(filtered_events)
            if i % num_event_per_group != 0
        ]
    )
    # 构建事件树结构
    actual_events._build_tree()
    # 获取关键事件的平均统计信息
    actual_events = actual_events.key_averages()

    # 输出调试信息，显示性能分析的时间分布
    log.debug("profiling time breakdown")
    log.debug(actual_events.table(row_limit=-1))

    # 计算每次重复运行的平均设备时间，并转换为毫秒
    res = sum(event.device_time_total for event in actual_events) / 1000.0 / n_repeat
    # 输出调试信息，显示性能分析结果
    log.debug("profiling results: %s ms", res)
    # 返回性能分析结果
    return res
# 使用 functools 模块中的 lru_cache 装饰器来实现缓存功能，不限制缓存大小
@functools.lru_cache(None)
def has_torchvision_roi_align() -> bool:
    try:
        # 尝试导入 torchvision 库中的 roi_align 函数
        from torchvision.ops import roi_align  # noqa: F401
        
        # 检查是否存在指定的 TorchScript 函数
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        
        # 返回是否成功导入 roi_align 并且 torch.ops.torchvision 中有 roi_align 属性
        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except ImportError:
        # 若导入错误，返回 False
        return False
    except RuntimeError as e:
        # 捕获运行时错误，若错误信息包含特定字符串，则断言为 False
        assert "torchvision::nms does not exist" in str(e)
        return False


# 根据输入的 device 参数返回对应的 torch.device 对象
def decode_device(device: Union[Optional[torch.device], str]) -> torch.device:
    if device is None:
        # 若 device 参数为 None，则返回默认设备的 torch.device 对象
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        # 若 device 参数为字符串，则将其转换为 torch.device 对象
        device = torch.device(device)
    if device.type not in ("cpu", "meta") and device.index is None:
        # 若 device 的类型不是 "cpu" 或 "meta"，且未指定索引，则获取对应设备的接口并返回 torch.device 对象
        device_interface = get_interface_for_device(device.type)
        return torch.device(device.type, index=device_interface.Worker.current_device())
    return device


# 计算给定迭代器中所有元素的乘积，使用 functools.reduce 和 operator.mul 实现
def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


# 计算两个序列的点积，要求两个序列长度必须相等
def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


# 返回输入迭代器中唯一值的集合，利用字典的键唯一性来实现
def unique(it: Iterable[_T]) -> ValuesView[_T]:
    return {id(x): x for x in it}.values()


# 计算 numer 除以 denom 的上取整结果，根据参数类型调用不同的实现函数
def ceildiv(
    numer: Union[int, sympy.Expr], denom: Union[int, sympy.Expr]
) -> Union[int, sympy.Expr]:
    if isinstance(numer, sympy.Expr) or isinstance(denom, sympy.Expr):
        return CeilDiv(sympy.sympify(numer), sympy.sympify(denom))
    # TODO: There is a bug in a call to this function, to repro:
    # python benchmarks/dynamo/huggingface.py --inductor -d cuda --accuracy
    # --amp --only YituTechConvBert --dynamic-shapes
    assert isinstance(numer, int) and isinstance(
        denom, int
    ), f"{numer}: {type(numer)}, {denom}: {type(denom)}"
    return runtime_ceildiv(numer, denom)


# 根据输入的 key 返回对应的 Triton 类型字符串，参考 Triton 实现中的类型映射
def _type_of(key):
    # 使用此函数以避免在代码生成期间对 Triton 的依赖
    # 参考 Triton 实现链接：https://github.com/openai/triton/blob/98b5945d2aef679e00ebca8e07c35c3658ec76de/python/triton/runtime/jit.py#L238
    # `None` 是空指针，隐式转换为 *i8 类型
    if key is None:
        return "*i8"
    # 根据 key 的字符串表示选择对应的 Triton 类型字符串
    dtype_str = str(key).split(".")[-1]
    tys = {
        "bool": "i1",
        "float8e4nv": "fp8e4nv",
        "float8e5": "fp8e5",
        "float8e4b15": "fp8e4b15",
        "float8e4b15x4": "fp8e4b15x4",
        "float8_e4m3fn": "fp8e4nv",
        "float8_e5m2": "fp8e5",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    # reinterpret 可以创建 Triton 类型
    for v in list(tys.values()):
        tys[v] = v
    # 如果 key 是字符串，则直接返回 key；否则返回以 '*' 开头的 dtype_str 对应的类型字符串
    return key if isinstance(key, str) else f"*{tys[dtype_str]}"
# 将形状和步长转换为电感器的表达式列表
def convert_shape_to_inductor(
    lst: Iterable[Union[int, torch.SymInt]]
) -> List[sympy.Expr]:
    """
    获取张量的形状和步长。对于非符号张量，这很简单。但对于符号张量，需要将 SymIntNode 映射为 sympy.Expr。
    """
    return [
        i.node.expr if isinstance(i, torch.SymInt) else sympy.Integer(i) for i in lst
    ]


# 将形状转换为 SymInt
def convert_shape_to_symint(
    lst: Iterable[Union[int, sympy.Expr]]
) -> List[Union[int, torch.SymInt]]:
    """
    将来自 Inductor 的形状列表转换为 symint（如果所有形状都是静态，则为 int）。
    """
    from .virtualized import V

    return [
        i
        if isinstance(i, int)
        else int(i)
        if isinstance(i, sympy.Integer)
        else V.graph.sizevars.shape_env.create_symintnode(i, hint=None)
        for i in lst
    ]


# 判断操作是否是视图
def is_view(op: torch._ops.OpOverload):
    """
    这个操作重载是否有别名？
    """
    assert isinstance(op, torch._ops.OpOverload)
    return any(a.alias_info is not None for a in op._schema.arguments)


# 判断是否是逐点使用
def is_pointwise_use(use):
    if not use.op == "call_function":
        return False

    if not (
        isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem
    ):
        return False

    if use.target is operator.getitem or is_view(use.target):
        return all(is_pointwise_use(u) for u in use.users)

    return torch.Tag.pointwise in use.target.tags


# 生成图模型和输入
def gen_gm_and_inputs(target, args, kwargs):
    """
    生成 Torch FX 图和输入参数
    """
    g = torch.fx.Graph()
    g_args = []
    a_args = []
    for n, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            g_args.append(g.placeholder(f"arg{n}"))
            a_args.append(arg)
        else:
            g_args.append(arg)
    assert all(not isinstance(x, torch.Tensor) for x in kwargs.values())
    node = g.call_function(target, tuple(g_args), kwargs)
    if (
        len(target._schema.returns) == 1
        and str(target._schema.returns[0].type) == "Tensor"
    ):
        node = (node,)
    g.output(node)

    gm = torch.fx.GraphModule({}, g)
    return gm, a_args


# 同步设备状态
def synchronize(device: str = "cuda"):
    if device == "cpu":
        return
    device_interface = get_interface_for_device(device)
    if device_interface.is_available():
        device_interface.synchronize()


# 计时函数执行性能
def timed(
    model: Callable[..., Any], example_inputs, times: int = 1, device: str = "cuda"
) -> float:
    """
    计时模型执行时间
    """
    synchronize(device)
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize(device)
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None  # type: ignore[possibly-undefined]
    return t1 - t0


# 打印性能指标
def print_performance(
    fn, args=(), times=10, repeat=10, baseline=1.0, device: str = "cuda"
):
    """
    打印函数执行性能
    """
    timings = torch.tensor([timed(fn, args, times, device) for _ in range(repeat)])
    # 计算timings中元素的中位数，并除以times得到平均值
    took = torch.median(timings) / times
    # 打印took除以baseline的结果，保留6位小数
    print(f"{took / baseline:.6f}")
    # 返回took作为函数的结果
    return took
# 将对象的指定方法替换为返回预计算常量的新方法
def precompute_method(obj: Any, method: str):
    result = getattr(obj, method)()  # 调用原始方法，获取结果
    setattr(obj, method, lambda: result)  # 将方法替换为返回预计算结果的新方法


# 将对象的多个方法替换为返回预计算常量的新方法
def precompute_methods(obj: Any, methods: List[str]):
    for method in methods:
        precompute_method(obj, method)  # 对每个方法调用 precompute_method 进行替换


# 比较函数，返回 a 和 b 的大小关系，1 表示 a > b，-1 表示 a < b，0 表示 a == b
def cmp(a, b) -> int:
    return int(a > b) - int(a < b)


# 如果列表长度为 1，则将其重复扩展为指定 size 大小的列表或类似列表的对象
def pad_listlike(x, size):
    if len(x) == 1:
        return type(x)([x[0]]) * size  # 返回元素重复 size 次后的列表或类似列表的对象
    else:
        return x  # 否则直接返回原始列表或类似列表的对象


# 对集合进行排序，确保迭代集合时的顺序是确定的
def tuple_sorted(x):
    if len(x) == 0:
        return []

    def sort_func(elem):
        if isinstance(elem, str):
            return elem
        else:
            return elem.get_name()  # 对于非字符串元素，调用其 get_name 方法获取排序依据

    return sorted(x, key=sort_func)  # 返回按照 sort_func 函数排序后的列表


# 参数类型规范，定义了一个泛型类型参数 P
P = ParamSpec("P")
# 定义了一个具有协议和泛型的 CachedMethod 类型
RV = TypeVar("RV", covariant=True)


# 缓存方法装饰器，将方法的计算结果缓存到对象的属性中，提高调用效率
def cache_on_self(fn: Callable[Concatenate[Any, P], RV]) -> CachedMethod[P, RV]:
    key = f"__{fn.__name__}_cache"

    @functools.wraps(fn)
    def wrapper(self):
        if not hasattr(self, key):
            setattr(self, key, fn(self))  # 计算并缓存结果
        return getattr(self, key)  # 返回缓存的结果

    def clear_cache(self):
        if hasattr(self, key):
            delattr(self, key)  # 清除缓存

    wrapper.clear_cache = clear_cache  # 将清除缓存方法绑定到 wrapper 对象上
    return wrapper  # 返回修饰后的方法


# 聚合节点调度计划中所有节点的源信息，返回一个集合
def aggregate_origins(node_schedule):
    from . import ir  # 导入模块

    if isinstance(node_schedule, list):
        return functools.reduce(
            operator.or_,
            [
                node.node.origins  # 提取节点的源信息
                for node in node_schedule
                if hasattr(node, "node") and node.node  # 确保节点有效
            ],
            set(),
        )
    elif isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins  # 返回外部内核的源信息
    else:
        return set()  # 其它情况返回空集合


# 获取融合内核的名称，基于描述性名称参数和节点调度计划的源信息
def get_fused_kernel_name(node_schedule, descriptive_names):
    all_origins = aggregate_origins(node_schedule)  # 聚合所有节点的源信息
    if descriptive_names == "original_aten":
        # 基于顶级 aten 操作符生成内核名称（如预分解）
        sources = [
            origin.meta["original_aten"]._overloadpacket.__name__  # 提取每个源的 aten 操作名称
            for origin in all_origins
            if origin.op == "call_function"  # 筛选出函数调用类型的源
            and "original_aten" in origin.meta  # 确保源具有 original_aten 属性
            and origin.meta["original_aten"] is not None
        ]
        sources = sorted(set(sources))  # 对名称进行排序并去重
    elif descriptive_names == "torch":
        # 如果 descriptive_names 参数为 "torch"，则基于顶层的 "torch" 操作符确定核函数名称（即动态图之后的图形）
        sources = []
        # 遍历所有来源
        for origin in all_origins:
            # 检查是否是 "call_function" 操作，并且存在 "source_fn_stack" 元数据
            if origin.op == "call_function" and "source_fn_stack" in origin.meta:
                # 获取源函数栈的最后一个函数名
                source_fn = origin.meta["source_fn_stack"][-1]
                # 如果函数名是字符串，则添加到 sources 列表中
                if isinstance(source_fn[1], str):
                    sources.append(source_fn[1])
                # 否则，将函数对象的名称添加到 sources 列表中
                else:
                    sources.append(source_fn[1].__name__)
        # 对 sources 列表进行去重和排序
        sources = sorted(set(sources))
    elif descriptive_names == "inductor_node":
        # 如果 descriptive_names 参数为 "inductor_node"，则 sources 包含所有 "call_function" 操作符的 origin 名称
        sources = [
            origin.name for origin in all_origins if origin.op == "call_function"
        ]
    else:
        # 如果 descriptive_names 参数既不是 "torch" 也不是 "inductor_node"，则抛出未实现错误
        raise NotImplementedError
    # 返回一个以 "fused" 和 sources 列表中所有元素为部分组成的字符串，用下划线连接
    return "_".join(["fused"] + sources)
def get_kernel_metadata(node_schedule, wrapper):
    # 聚合所有节点的起源信息
    all_origins = aggregate_origins(node_schedule)
    # 筛选出调用函数操作的节点作为感应器节点
    inductor_nodes = [origin for origin in all_origins if origin.op == "call_function"]

    # 创建空的默认字典用于存储来自节点的信息
    from_node_dict = collections.defaultdict(list)
    # 创建空的默认字典用于存储原始 ATen 函数的信息
    original_aten_dict = collections.defaultdict(list)
    # 遍历所有感应器节点
    for node in inductor_nodes:
        # 如果节点的元数据中包含原始 ATen 信息且不为空
        if "original_aten" in node.meta and node.meta["original_aten"] is not None:
            # 将节点的原始 ATen 对象的特定字段转换为字符串作为字典键，将节点名称添加到值列表中
            key = str(node.meta["original_aten"]._overloadpacket)
            original_aten_dict[key].append(node.name)
        # 如果节点的元数据中包含来自节点信息
        if "from_node" in node.meta:
            # 将节点的来自节点信息的特定字段作为字典键，将节点名称添加到值列表中
            key = node.meta["from_node"][0][0]
            from_node_dict[key].append(node.name)
    
    # 构建元数据字符串，包括注释、来源节点和原始 ATen 函数信息
    metadata = (
        f"{wrapper.comment} Source Nodes: [{', '.join(sorted(from_node_dict.keys()))}], "
        f"Original ATen: [{', '.join(sorted(original_aten_dict.keys()))}]"
    )

    # 构建详细的元数据列表，展示每个原始节点及其相关节点
    detailed_metadata = []
    for original_node, nodes in sorted(from_node_dict.items()):
        detailed_metadata.append(
            f"{wrapper.comment} {original_node} => {', '.join(sorted(nodes))}"
        )
    
    # 返回总体元数据和详细元数据的字符串表示
    return metadata, "\n".join(detailed_metadata)


def dominated_nodes(
    initial_queue: Iterable[torch.fx.Node], skip_filter=None
) -> Set[torch.fx.Node]:
    """Returns the set of nodes whose values depend on those within initial_queue"""
    # 将初始队列转换为列表
    initial_queue = list(initial_queue)
    # 创建支配节点集合，并初始化为初始队列的节点集合
    dominated_set = set(initial_queue)

    # 使用广度优先搜索扩展支配节点集合
    while initial_queue:
        # 弹出队列中的一个节点
        node = initial_queue.pop()
        # 遍历该节点的用户节点
        for user in node.users:
            # 如果提供了跳过过滤器并且用户节点被该过滤器跳过，则继续下一个循环
            if skip_filter and skip_filter(user):
                continue
            # 如果用户节点不在支配节点集合中，则将其添加到支配节点集合并添加到初始队列中
            if user not in dominated_set:
                dominated_set.add(user)
                initial_queue.append(user)

    # 返回支配节点集合
    return dominated_set


def gather_origins(args, kwargs):
    import itertools

    from . import ir

    # 判断节点是否为未实现节点的函数
    def is_unrealized_node(n):
        if isinstance(n, ir.TensorBox):
            return is_unrealized_node(n.data)
        if isinstance(n, ir.StorageBox):
            return is_unrealized_node(n.data)
        return isinstance(n, ir.IRNode) and isinstance(n, ir.Pointwise)

    # 获取所有参数和关键字参数中的起源节点，并使用 itertools.chain 连接所有起源节点
    kwarg_origins = [val.origins for val in kwargs.values() if is_unrealized_node(val)]
    arg_origins = [arg.origins for arg in args if is_unrealized_node(arg)]
    return set(itertools.chain(*arg_origins, *kwarg_origins))


def sympy_str(expr: sympy.Expr) -> str:
    """
    Normal sympy str is very slow, this is a lot faster.  The result are
    somewhat worse, as it doesn't do as much simplification.  So don't
    use this for final codegen.
    """
    # 将 sympy 表达式转换为字符串的简单实现，忽略了一些复杂的简化步骤
    if isinstance(expr, sympy.Symbol):
        return expr.name
    if isinstance(expr, sympy.Add):
        return " + ".join(map(sympy_str, expr.args))
    if isinstance(expr, sympy.Mul):
        return " * ".join(map(sympy_str, expr.args))

    # 对于特定的 sympy 函数对象，将其函数名及其参数转换为字符串表示
    if isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv, Identity)):
        return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args)))}"
    # 对于其他情况，直接返回字符串表示
    return str(expr)
def get_bounds_index_expr(index):
    # 导入虚拟化模块中的 V
    from .virtualized import V

    # 如果这个表达式不来自 FX 节点，我们计算其边界
    if (
        config.compute_all_bounds  # 检查是否需要计算所有边界
        and (fx_node := getattr(V.interpreter, "current_node", None))  # 获取当前 FX 节点
        and fx_node.target != "index_expr"  # 确保 FX 节点的目标不是 "index_expr"
    ):
        return bound_sympy(index)  # 返回符号表达式的边界
    else:
        return ValueRanges.unknown()  # 否则返回未知值范围


def sympy_index_symbol_with_prefix(prefix: SymT, idx: int) -> sympy.Symbol:
    """
    用于生成整数非负符号。
    """
    # 这不应用于创建形状/步长符号，因为那些都应该在 Inductor 之前分配。
    assert prefix != SymT.SIZE  # 确保前缀不是 SymT.SIZE
    # 注意：形状符号是正数（> 0），而索引变量只是非负数（>= 0）。
    return make_symbol(prefix, idx, integer=True, nonnegative=True)  # 创建一个整数非负符号


def generate_assert(check):
    # 返回是否需要生成断言的条件：检查条件或者配置中开启了调试索引断言
    return (check or config.debug_index_asserts) and config.assert_indirect_indexing


def sympy_index_symbol(name: str) -> sympy.Symbol:
    """
    用于生成整数非负符号。
    """
    # 这不应用于创建形状/步长符号，因为那些都应该在 Inductor 之前分配。
    assert name[0] != "s"  # 确保名称不以 "s" 开头
    # 注意：形状符号是正数（> 0），而索引变量只是非负数（>= 0）。
    return sympy.Symbol(name, integer=True, nonnegative=True)  # 创建一个整数非负符号


def sympy_subs(expr: sympy.Expr, replacements: Dict[sympy.Expr, Any]) -> sympy.Expr:
    """
    当传递的替换符号 v 是一个字符串时，将其转换为具有相同替换表达式整数和非负属性的符号。
    """

    def to_symbol(replaced, replacement):
        assert isinstance(replaced, sympy.Expr)  # 断言替换前是一个 sympy 表达式
        if isinstance(replacement, str):  # 如果替换是一个字符串
            return sympy.Symbol(
                replacement,
                integer=replaced.is_integer,  # 使用被替换符号的整数属性
                nonnegative=replaced.is_nonnegative,  # 使用被替换符号的非负属性
            )
        else:
            return replacement

    # xreplace 比 subs 更快，但也更挑剔
    return sympy.sympify(expr).xreplace(
        {k: to_symbol(k, v) for k, v in replacements.items()}  # 使用替换字典对表达式进行替换
    )


def is_symbolic(a: Any) -> bool:
    # 判断一个对象是否是 torch.SymInt 类型，或者是 torch.Tensor 且包含符号元素
    return isinstance(a, torch.SymInt) or (
        isinstance(a, torch.Tensor)
        and any(is_symbolic(x) for x in itertools.chain(a.size(), a.stride()))
    )


def any_is_symbolic(*args: Any) -> bool:
    # 判断任意参数中是否存在符号对象
    return any(is_symbolic(a) for a in args)


def get_first_incompatible_cudagraph_node(gm):
    # 导入 torch.fx.experimental.symbolic_shapes 模块中的 free_unbacked_symbols
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
    # 定义禁止集合，包含了一些特定的操作名称
    forbidden_set = {
        "aten._fused_moving_avg_obs_fq_helper.default",
        "aten._fused_moving_avg_obs_fq_helper_functional.default",
        "aten.multinomial.default",
        "fbgemm.dense_to_jagged.default",
        "fbgemm.jagged_to_padded_dense.default",
        "run_and_save_rng_state",
        "run_with_rng_state",
        "aten._local_scalar_dense",
        # Technically, it's not necessary to ban this, because an
        # assert_scalar with constant arguments can be validly run
        # with CUDA graphs, but the operator is also pointless with
        # constant arguments, so might as well ban
        "aten._assert_scalar",
    }
    # 如果启用了确定性算法，则扩展禁止集合以包含更多操作名称
    if torch.are_deterministic_algorithms_enabled():
        forbidden_set.update(
            {
                "aten._unsafe_index_put.default",
                "aten._unsafe_masked_index_put_accumulate.default",
                "aten.index_put.default",
                "aten.index_put_.default",
                "aten.scatter.src",
                "aten.scatter.reduce",
                "aten.scatter.value_reduce",
                "aten.scatter_add_",
                "aten.scatter_add.default",
                "aten.scatter_reduce.two",
                "aten.scatter_reduce_.two",
                "aten.scatter_reduce.two_out",
            }
        )
    # 遍历计算图中的每个节点
    for node in gm.graph.nodes:
        # 如果节点的目标操作在禁止集合中，返回该节点
        if str(node.target) in forbidden_set:
            return node
        # 如果节点的元数据"val"存在，并且"val"中存在未支持的自由符号，返回该节点
        if (val := node.meta.get("val")) is not None and free_unbacked_symbols(val):
            return node
    # 如果未找到符合条件的节点，返回None
    return None
def has_incompatible_cudagraph_ops(gm):
    # 检查给定的 FX 图模块是否有不兼容的 cudagraph 操作节点
    return get_first_incompatible_cudagraph_node(gm) is not None


def output_node(gm: torch.fx.GraphModule):
    """Get the output node from an FX graph"""
    # 获取 FX 图中的输出节点
    last_node = next(iter(reversed(gm.graph.nodes)))
    assert last_node.op == "output"
    return last_node


_registered_caches: List[Any] = []


def clear_on_fresh_inductor_cache(obj: Any):
    """
    Use this decorator to register any caches that should be cache_clear'd
    with fresh_inductor_cache().
    """
    # 检查对象是否具有 cache_clear 方法，如果没有则引发 AttributeError
    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    # 将对象注册到 _registered_caches 列表中
    _registered_caches.append(obj)
    return obj


def clear_inductor_caches():
    """
    Clear all registered caches.
    """
    # 清空所有已注册的缓存
    for obj in _registered_caches:
        obj.cache_clear()


@contextlib.contextmanager
def fresh_inductor_cache(cache_entries=None, dir=None, delete=True):
    """
    Contextmanager that provides a clean tmp cachedir for inductor.

    Optionally, pass a dict as 'cache_entries' to get a list of filenames and sizes
    generated with this cache instance.
    """
    # 清空所有注册的缓存
    clear_inductor_caches()

    # 创建临时的缓存目录
    inductor_cache_dir = tempfile.mkdtemp(dir=dir)
    try:
        with mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_CACHE_DIR": inductor_cache_dir}
        ):
            log.debug("Using inductor cache dir %s", inductor_cache_dir)
            triton_cache_dir = os.path.join(inductor_cache_dir, "triton")
            with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": triton_cache_dir}):
                yield
                # 如果 cache_entries 是字典，则更新其内容为生成的文件名和大小信息
                if isinstance(cache_entries, dict):
                    assert len(cache_entries) == 0, "expected empty cache_entries dict"
                    if os.path.exists(triton_cache_dir):
                        files = os.listdir(triton_cache_dir)
                        cache_entries.update(
                            {
                                f: os.path.getsize(os.path.join(triton_cache_dir, f))
                                for f in files
                                if ".lock" not in f
                            }
                        )
        # 如果 delete 为 True，则删除临时缓存目录
        if delete:
            shutil.rmtree(inductor_cache_dir)
    except Exception:
        log.warning("on error, temporary cache dir kept at %s", inductor_cache_dir)
        raise
    finally:
        # 清空所有注册的缓存
        clear_inductor_caches()


def argsort(seq) -> List[int]:
    # 保持相等步幅时的原始顺序
    getter = seq.__getitem__
    a_r = range(len(seq))
    return list(reversed(sorted(a_r, key=getter, reverse=True)))  # noqa: C413


@functools.lru_cache(8)
def get_dtype_size(dtype):
    # 返回指定数据类型的大小
    return torch.empty((), dtype=dtype).element_size()


class LineContext(NamedTuple):
    # 代表一行代码的上下文
    context: Any


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        # 初始化函数，设置初始缩进和空行列表
        self._lines = []
        self._indent = initial_indent
    def getvaluewithlinemap(self) -> tuple[str, list[tuple[int, LineContext]]]:
        # 创建一个字符串缓冲区
        buf = StringIO()
        # 行号计数器
        p = 1
        # 存储行号和上下文的列表
        linemap = []
        # 遍历所有行
        for line in self._lines:
            # 如果是延迟加载的行对象，需要调用它获取真实的行数据
            if isinstance(line, DeferredLineBase):
                line = line()
                # 如果返回值为空，跳过当前循环
                if line is None:
                    continue
            # 如果是行上下文对象，记录行号和上下文信息，并继续下一次循环
            elif isinstance(line, LineContext):
                linemap.append((p, line.context))
                continue
            # 断言当前行是字符串类型
            assert isinstance(line, str)
            # 将当前行写入缓冲区
            buf.write(line)
            buf.write("\n")
            # 更新行号计数器，考虑多行情况
            p += 1 + line.count("\n")
        # 返回缓冲区中的字符串内容和行号上下文的列表
        return buf.getvalue(), linemap

    def getvalue(self) -> str:
        # 获取值和行号映射，但只返回值部分
        v, _ = self.getvaluewithlinemap()
        return v

    def getrawvalue(self) -> str:
        # 创建一个字符串缓冲区
        buf = StringIO()
        # 遍历所有行
        for line in self._lines:
            # 如果是延迟加载的行对象，需要调用它获取真实的行数据
            if isinstance(line, DeferredLineBase):
                line = line()
                # 如果返回值为空，跳过当前循环
                if line is None:
                    continue
            # 如果是行上下文对象，直接跳过当前循环
            elif isinstance(line, LineContext):
                continue
            # 断言当前行是字符串类型
            assert isinstance(line, str)
            # 如果当前行以反斜杠结尾，表示是多行连接的情况，去除末尾的反斜杠后写入缓冲区
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                # 否则直接将当前行写入缓冲区，并换行
                buf.write(line)
                buf.write("\n")
        # 返回缓冲区中的字符串内容
        return buf.getvalue()

    def clear(self):
        # 清空行列表
        self._lines.clear()

    def __bool__(self):
        # 判断是否有行存在
        return bool(self._lines)

    def prefix(self):
        # 返回当前缩进量对应的空格字符串
        return " " * (self._indent * self.tabwidth)

    def newline(self):
        # 写入一个空行
        self.writeline("\n")

    def writeline(self, line):
        # 如果是行上下文对象，直接添加到行列表
        if isinstance(line, LineContext):
            self._lines.append(line)
        # 如果是延迟加载的行对象，使用当前缩进作为前缀后添加到行列表
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        # 如果是非空白字符串，加上当前缩进后添加到行列表
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        # 否则添加空字符串到行列表
        else:
            self._lines.append("")

    def writelines(self, lines):
        # 遍历所有行，逐行调用 writeline 方法
        for line in lines:
            self.writeline(line)

    def indent(self, offset=1):
        # 定义一个上下文管理器，用于改变当前缩进量
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def do_indent(self, offset=1):
        # 增加当前缩进量
        self._indent += offset

    def do_unindent(self, offset=1):
        # 减少当前缩进量
        self._indent -= offset
    # 定义一个方法，用于将另一个 IndentedBuffer 对象的代码插入到当前对象中
    def splice(self, other_code, strip=False):
        # 如果 other_code 是 IndentedBuffer 类型的对象
        if isinstance(other_code, IndentedBuffer):
            # 初始化需要去除的缩进为无穷大
            dedent = float("inf")
            # 遍历 other_code 对象的每一行
            for line in other_code._lines:
                # 如果该行不是 LineContext 类型且非空
                if not isinstance(line, LineContext) and line:
                    # 计算当前行的缩进，并更新需要去除的最小缩进量
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            # 如果没有找到需要去除的缩进，将 dedent 设置为 0
            if math.isinf(dedent):
                dedent = 0
            # 再次遍历 other_code 对象的每一行
            for line in other_code._lines:
                # 如果该行是 LineContext 类型，直接将其添加到当前对象的 _lines 中
                if isinstance(line, LineContext):
                    self._lines.append(line)
                else:
                    # 否则，将该行去除指定的缩进后，作为新的行添加到当前对象的 _lines 中
                    IndentedBuffer.writeline(self, line[int(dedent):])
        else:
            # 如果 other_code 不是 IndentedBuffer 对象，首先去除其文本的缩进
            other_code = textwrap.dedent(other_code)
            # 如果 strip 参数为 True，则去除文本左侧的空白字符
            if strip:
                other_code = other_code.lstrip()
            # 如果文本为空，则直接返回
            if not other_code:
                return
            # 去除文本右侧的空白字符
            other_code = other_code.rstrip()
            # 将文本按行分割，并逐行添加到当前对象的 _lines 中
            for line in other_code.split("\n"):
                self.writeline(line)

    # 定义一个方法，对当前对象的每一行应用给定的函数 func，并返回一个新的 IndentedBuffer 对象
    def map(self, func: Callable[[Any], Any]) -> IndentedBuffer:
        # 创建一个新的 IndentedBuffer 对象，与当前对象具有相同的初始缩进
        res = IndentedBuffer(initial_indent=self._indent)
        # 对当前对象的每一行应用 func 函数，并将结果添加到 res 对象的 _lines 中
        res._lines = [func(line) for line in self._lines]
        # 返回处理后的结果对象
        return res

    # 定义一个方法，返回当前对象的字符串表示，包括其类型和当前内容
    def __repr__(self):
        return f"{type(self)}({self.getvalue()})"

    # 定义一个方法，实现两个 IndentedBuffer 对象的连接操作
    def __add__(self, other):
        # 断言两个对象的初始缩进相同
        assert self._indent == other._indent
        # 创建一个新的 IndentedBuffer 对象，初始缩进与当前对象相同
        res = IndentedBuffer(initial_indent=self._indent)
        # 将当前对象和 other 对象的所有行依次添加到 res 对象的 _lines 中
        res.writelines(self._lines)
        res.writelines(other._lines)
        # 返回连接后的结果对象
        return res
class FakeIndentedBuffer(IndentedBuffer):
    # FakeIndentedBuffer 类继承自 IndentedBuffer，用于模拟缓冲区功能

    def __init__(self):
        # 初始化方法，调用父类 IndentedBuffer 的初始化方法
        super().__init__()

    def __getattribute__(self, name):
        # 重载 __getattribute__ 方法，允许访问 __class__ 属性以外的属性名
        if name == "__class__":
            return object.__getattribute__(self, name)
        # 抛出运行时异常，禁止在 FakeIndentedBuffer 上调用未显式指定的方法
        raise RuntimeError(
            f"Tried to call self.{name} on FakeIndentedBuffer. This buffer"
            "is currently used on TritonTemplateKernel to prevent actual"
            "writes to the body without explicitly specifying the body with"
            "`TritonTemplateKernel.set_subgraph_body(name)`"
        )


@contextlib.contextmanager
def restore_stdout_stderr(initial_stdout, initial_stderr):
    # 定义一个上下文管理器 restore_stdout_stderr，用于恢复标准输出和标准错误流
    try:
        yield
    finally:
        sys.stdout = initial_stdout
        sys.stderr = initial_stderr


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line):
        # 初始化方法，如果行为空白，则将其置空
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> Optional[str]:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        # 调用对象时返回 self.line 或 None 表示行已被“取消写入”
        raise NotImplementedError

    def _new_line(self, line: str) -> DeferredLineBase:
        """Returns a new deferred line with the same condition"""
        # 返回具有相同条件的新的延迟行对象
        raise NotImplementedError

    def with_prefix(self, prefix):
        # 返回添加了前缀的新行对象
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self):
        # 返回去除左侧空白字符后的新行对象
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index):
        # 返回索引处的字符组成的新行对象
        return self._new_line(self.line[index])

    def __bool__(self):
        # 判断行是否为真（非空）
        return bool(self.line)

    def __len__(self):
        # 返回行的长度
        return len(self.line)


@functools.lru_cache(None)
def is_big_gpu(index) -> bool:
    # 使用 LRU 缓存装饰器缓存结果，判断指定索引的 GPU 是否为大型 GPU
    min_sms = 68  # 3080
    avail_sms = torch.cuda.get_device_properties(index).multi_processor_count
    if avail_sms < min_sms:
        # 如果可用 SM 数量小于最小要求，记录警告信息并返回 False
        log.warning(
            "Not enough SMs to use max_autotune_gemm mode",
            extra={"min_sms": min_sms, "avail_sms": avail_sms},
        )
        return False
    # 否则返回 True
    return True


def use_max_autotune() -> bool:
    # 判断是否启用最大自动调整模式
    return (
        config.max_autotune or config.max_autotune_gemm or config.search_autotune_cache
    )


def _use_template_for_cuda(layout, allowed_layout_dtypes: List[torch.dtype]) -> bool:
    # 判断是否使用模板来优化 CUDA 布局
    return (
        use_max_autotune()
        and layout.device.type == "cuda"
        and layout.dtype in allowed_layout_dtypes
        and is_big_gpu(layout.device.index or 0)
    )


def _use_autotune_backend(backend: str) -> bool:
    # 判断是否使用自动调整后端
    return backend.upper() in [
        x.strip() for x in config.max_autotune_gemm_backends.upper().split(",")
    ]


def use_triton_template(layout, *, enable_int32=False):
    # 使用 Triton 模板进行布局优化
    from .codegen.common import BackendFeature, has_backend_feature

    # 定义支持的布局数据类型列表
    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if enable_int32:
        layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    # 返回一个布尔值，表示以下三个条件是否同时满足：
    # 1. 使用给定布局和布局数据类型进行 CUDA 模板化
    # 2. 使用 TRITON 自动调整后端
    # 3. 布局所在设备支持 TRITON 模板的后端特性
    return (
        _use_template_for_cuda(layout, layout_dtypes)
        and _use_autotune_backend("TRITON")
        and has_backend_feature(layout.device, BackendFeature.TRITON_TEMPLATES)
    )
# 导入模块 V，该模块似乎是从当前目录下的 virtualized 包中导入的
from .virtualized import V

# 计算 GEMM 操作的大小，使用 V.graph.sizevars.size_hint 方法进行估算
gemm_size = V.graph.sizevars.size_hint(m * n * k, fallback=-1)

# 如果无法获取有效的 gemm_size 或者 gemm_size 小于配置中指定的 CUDA 切片最小大小，则返回 False
if gemm_size <= 0 or gemm_size < config.cuda.cutlass_backend_min_gemm_size:
    return False

# 导入 try_import_cutlass 函数用于后续的 CUTLASS 模板引入尝试
from .codegen.cuda.cutlass_utils import try_import_cutlass

# 如果当前运行环境为 ROCm，则不使用 CUTLASS 模板，直接返回 False
if torch.version.hip:
    return False

# 定义支持的数据类型列表，包括 torch.float16, torch.bfloat16, torch.float32, torch.int32
layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]

# 调用 _use_template_for_cuda 函数检查是否可以使用 CUDA 模板，以及 _use_autotune_backend 函数检查是否支持自动调优后端 "CUTLASS"
res = _use_template_for_cuda(layout, layout_dtypes) and _use_autotune_backend("CUTLASS")

# 如果 res 为 True，则尝试导入 CUTLASS 库，如果导入失败则记录警告信息并返回 False
if res:
    if not try_import_cutlass():
        log.warning(
            "Failed to import CUTLASS lib. Please check whether "
            "_inductor.config.cuda.cutlass_dir is set correctly. "
            "Skipping CUTLASS backend for now."
        )
        return False

# 返回 res，表示是否成功使用 CUTLASS 模板
return res
    # 计算 GEMM 操作的预估大小，使用图形变量的大小提示方法来确定
    gemm_size = V.graph.sizevars.size_hint(m * n * k, fallback=-1)
    if gemm_size <= 0:
        # 如果无法确定 GEMM 大小，则返回 False
        return False
    # TBD: 探索是否需要类似 CUTLASS 的后端在小型 GEMM 操作中禁用
    
    # 尝试导入 CK 库并获取其包的目录名
    ck_package_dirname, _, _, _ = try_import_ck_lib()
    
    if not ck_package_dirname:
        # 如果未成功导入 CK 库，则发出警告并返回 False
        log.warning("Please pip install Composable Kernel package")
        return False
    
    if not config.rocm.ck_dir:
        # 如果未设置环境变量 TORCHINDUCTOR_CK_DIR，则发出警告并返回 False
        log.warning("Please set TORCHINDUCTOR_CK_DIR env variable")
        return False
    
    if ck_package_dirname != config.rocm.ck_dir:
        # 如果导入的 CK 库路径与配置文件中的路径不匹配，则发出警告并返回 False
        log.warning("Invalid path to CK library")
        return False
    
    # 所有检查通过，返回 True 表示环境设置正确
    return True
# 检查是否应使用 CPU 布局的模板
def _use_template_for_cpu(layout):
    return use_max_autotune() and layout.device.type == "cpu"

# 使用 CPP 打包 GEMM 模板
def use_cpp_packed_gemm_template(layout, mat1, mat2):
    from . import ir  # 导入 ir 模块
    from .codegen.cpp_micro_gemm import create_micro_gemm  # 导入创建微 GEMM 的函数
    from .kernel.mm_common import mm_args  # 导入 mm_args 函数

    # 如果不应使用 CPU 布局的模板或不使用 CPP 的自动调整，则返回 False
    if not _use_template_for_cpu(layout) or not _use_autotune_backend("CPP"):
        return False

    # 如果不允许 CPP 的权重预打包，则返回 False
    if not config.cpp.weight_prepack:
        return False

    # 定义支持的布局数据类型列表
    layout_dtypes = [torch.float32, torch.bfloat16, torch.half]

    # 调用 mm_args 函数获取矩阵的维度信息和布局
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2)

    # TODO(jgong5): 支持 n 或 k 的动态形状
    if has_free_symbols((n, k)):
        return False

    # 如果 mat2 是 ir.BaseView 的实例，则解包它
    if isinstance(mat2, ir.BaseView):
        mat2 = mat2.unwrap_view()

    # 创建 micro GEMM 并获取其实例
    micro_gemm = create_micro_gemm(
        "micro_gemm",
        m,
        n,
        k,
        input_dtype=layout.dtype,
        output_dtype=torch.float,
        num_threads=parallel_num_threads(),
    )

    # TODO(jgong5): 支持 n % n_block_size != 0
    # 检查布局的数据类型是否在支持列表中，并验证其他条件
    return (
        layout.dtype in layout_dtypes
        and micro_gemm is not None
        and n % micro_gemm.register_blocking[1] == 0
        and mat1.get_stride()[-1] == 1  # TODO(jgong5): 支持转置输入
        and isinstance(mat2, ir.StorageBox)
        and mat2.is_module_buffer()
    )


# 检查是否应使用 ATEN GEMM 内核
def use_aten_gemm_kernels():
    return not use_max_autotune() or _use_autotune_backend("ATEN")


# 调试目录管理器类
class DebugDirManager:
    counter = itertools.count(0)  # 计数器初始化为 0
    prev_debug_name: str  # 前一个调试目录的名称

    # 初始化方法，设置实例的 ID
    def __init__(self):
        self.id = next(DebugDirManager.counter)

    # 进入上下文管理器时调用，保存前一个调试目录的名称，并设置新的临时名称
    def __enter__(self):
        self.prev_debug_name = torch._dynamo.config.debug_dir_root
        self.new_name = f"{self.prev_debug_name}_tmp_{self.id}"
        torch._dynamo.config.debug_dir_root = self.new_name

    # 退出上下文管理器时调用，删除临时目录并恢复前一个调试目录的名称
    def __exit__(self, *args):
        shutil.rmtree(self.new_name)  # 删除临时目录
        torch._dynamo.config.debug_dir_root = self.prev_debug_name


# 运行并获取代码的函数，支持模拟保存输出代码
def run_and_get_code(fn, *args, **kwargs):
    from .graph import GraphLowering  # 导入图形降低模块

    source_codes: List[str] = []  # 初始化存储源代码的列表

    # 定义保存输出代码的函数
    def save_output_code(code: str):
        source_codes.append(code)

    # 使用 mock.patch.object 临时替换 GraphLowering 的 save_output_code 方法
    with mock.patch.object(GraphLowering, "save_output_code", save_output_code):
        torch._dynamo.reset()  # 重置 Torch Dynamo
        result = fn(*args, **kwargs)  # 执行传入的函数并获取结果
    return result, source_codes  # 返回结果和收集到的源代码列表


# 获取通过感应器生成的代码，但跳过任何实际编译或运行的函数
def get_code(fn, *args, **kwargs):
    """Get the inductor-generated code, but skip any actual compilation or running."""
    from .graph import GraphLowering  # 导入图形降低模块

    source_codes: List[str] = []  # 初始化存储源代码的列表

    # 定义保存输出代码的函数
    def save_output_code(code: str):
        source_codes.append(code)
    def patched_compile_to_module(self: GraphLowering):
        # 定义一个虚拟的模块类，用于替换生成的 Triton 模块
        class DummyModule:
            """This is empty to replace the generated triton module"""

            def __init__(self):
                pass

            def call(self, *args, **kwargs):
                # 当被调用时不执行任何操作
                # Don't do anything when called
                pass

        # 生成代码，根据需要使用 C++ 包装器或者直接生成
        code, _ = (
            self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
        )
        # 跳过所有实际的编译过程
        nonlocal save_output_code
        save_output_code(code)

        # 返回虚拟的模块对象
        return DummyModule()

    # 使用 mock.patch.object 对 GraphLowering 类的 compile_to_module 方法进行替换
    # 使用定义好的 patched_compile_to_module 方法替换原方法
    # 同时使用 mock.patch.object 对 GraphLowering 类的 save_output_code 方法进行替换
    # 替换为外部定义的 save_output_code 方法
    with mock.patch.object(
        GraphLowering, "compile_to_module", patched_compile_to_module
    ), mock.patch.object(GraphLowering, "save_output_code", save_output_code):
        # 重置 Torch Dynamo 的状态
        torch._dynamo.reset()
        # 注意这里的返回值是 None
        _ = fn(*args, **kwargs)

    # 返回源代码列表
    return source_codes
# 从指定函数获取代码，并返回代码内容列表
def get_triton_code(fn, *args, **kwargs):
    # 调用get_code函数获取源代码列表
    source_codes = get_code(fn, *args, **kwargs)
    # 如果源代码列表长度不是1或2，则抛出异常
    assert (
        1 <= len(source_codes) <= 2
    ), f"expected one or two code outputs got {len(source_codes)}"
    # 返回第一个源代码
    return source_codes[0]


# 运行指定函数获取代码，并返回代码内容列表的第一个
def run_and_get_triton_code(fn, *args, **kwargs):
    # 调用run_and_get_code函数获取函数运行结果和源代码列表
    _, source_codes = run_and_get_code(fn, *args, **kwargs)
    # 如果源代码列表长度不是1或2，则抛出异常
    assert (
        1 <= len(source_codes) <= 2
    ), f"expected one or two code outputs got {len(source_codes)}"
    # 返回第一个源代码
    return source_codes[0]


# 上下文管理器，用于覆盖aten_op的降低函数
@contextlib.contextmanager
def override_lowering(aten_op, override_fn):
    """
    Override the lowering of aten_op with override_fn.
    The first argument of override_fn is the original lowering fn.
    """
    from torch._inductor import lowering

    # 获取原始的降低函数
    orig_fn = lowering.lowerings[aten_op]
    try:
        # 将aten_op的降低函数替换为override_fn的偏函数
        lowering.lowerings[aten_op] = functools.partial(override_fn, orig_fn)
        # 执行yield，提供上下文
        yield
    finally:
        # 恢复aten_op的原始降低函数
        lowering.lowerings[aten_op] = orig_fn


# 添加调度器初始化钩子函数
def add_scheduler_init_hook(pre_fn, post_fn=None):
    """
    Add hook functions to be called at the beginning and end of Scheduler.__init__.
    Used for unit tests.
    """
    from torch._inductor.scheduler import Scheduler

    # 获取Scheduler类的原始__init__方法
    orig_fn = Scheduler.__init__

    # 定义一个包装函数，用于在调用Scheduler.__init__前后执行钩子函数
    def wrapper(scheduler, nodes):
        pre_fn(scheduler, nodes)
        out = orig_fn(scheduler, nodes)
        if post_fn:
            post_fn(scheduler, nodes)
        return out

    # 使用unittest.mock.patch_object装饰包装函数，替换Scheduler的__init__方法
    return unittest.mock.patch.object(Scheduler, "__init__", wrapper)


# 开发者警告函数，用于向PyTorch开发者发出警告
def developer_warning(msg):
    """
    Warnings that will be actionable for PyTorch developers, but not
    end users.  Allows us to easily disable them in stable releases but
    keep them on for nightly builds.
    """
    # 如果开启了开发者警告，则记录警告信息
    if config.developer_warnings:
        log.warning(msg)
    else:
        log.info(msg)


# 获取基准测试名称的实验性API
def get_benchmark_name():
    """
    An experimental API used only when config.benchmark_kernel is true.

    The benchmark name is only available at codegen time. So we can not
    directly call it in benchmark_all_kernels which is run after codegen.

    The function assumes the argument after --only is the benchmark name.
    It works for torchbench.py/hugginface.py/timm_models.py. But for ad-hoc
    scripts, this function may return None.

    There are 2 flavors of --only argument we need handle:
    1. --only model_name
    2. --only=model_name
    """
    try:
        # 查找命令行参数中的"--only"参数
        idx = sys.argv.index("--only")
        # 如果紧跟在"--only"后的参数不是选项并且长度大于0，则返回该参数作为基准测试名称
        if (
            idx + 1 < len(sys.argv)
            and len(sys.argv[idx + 1]) > 0
            and sys.argv[idx + 1][0] != "-"
        ):
            return sys.argv[idx + 1]
    except ValueError:
        pass

    # 如果未找到"--only"参数，则尝试寻找以"--only="开头的参数，返回其后面的内容作为基准测试名称
    for arg in sys.argv:
        if arg.startswith("--only="):
            return arg[len("--only="):]


# 判断列表中的所有元素是否均为1
def is_ones(items):
    return all(x == 1 for x in items)


# 判断列表中的所有元素是否均为0
def is_zeros(items):
    return all(x == 0 for x in items)


# 判断输入是否为CPU设备
def is_cpu_device(inputs):
    # 返回一个布尔值，判断所有符合条件的输入项是否都位于 CPU 设备上
    return all(
        # 对于每一个输入项，检查其是否为 torch.Tensor 类型，并且设备为 CPU
        item.device == torch.device("cpu")
        for item in inputs  # 遍历输入项列表
        if isinstance(item, torch.Tensor)  # 仅考虑是 torch.Tensor 类型的输入项
    )
# 根据输入值的类型（必须为 sympy.Expr），返回对应的 Torch 数据类型
def get_sympy_Expr_dtype(val: sympy.Expr) -> torch.dtype:
    assert isinstance(
        val, sympy.Expr
    ), "only support sympy.Expr as input to get_sympy_Expr_dtype"
    if val.is_integer:  # 检查是否为整数类型
        return torch.int64
    else:
        return torch.float64


# 根据条件决定是否启用性能分析器，并在需要时创建 Torch 的 profiler 上下文
@contextlib.contextmanager
def maybe_profile(should_profile, *args, **kwargs):
    if should_profile:
        with torch.profiler.profile(*args, **kwargs) as p:
            yield p
    else:
        yield


# 获取并返回用于并行处理的线程数
def parallel_num_threads():
    threads = config.cpp.threads  # 从配置中获取线程数
    if threads < 1:
        threads = torch.get_num_threads()  # 如果配置的线程数小于1，则使用 Torch 的线程数
    return threads


# 根据数据类型获取 GPU 的 TFLOPS
@functools.lru_cache(None)
def get_device_tflops(dtype):
    from triton.testing import get_max_simd_tflops, get_max_tensorcore_tflops

    assert dtype in (torch.float16, torch.bfloat16, torch.float32)  # 断言数据类型为支持的 Torch 浮点数类型

    if inspect.signature(get_max_simd_tflops).parameters.get("clock_rate"):
        # 如果 Triton API 在 https://github.com/openai/triton/pull/2293 中发生了更改
        from torch._utils_internal import max_clock_rate

        sm_clock = max_clock_rate()  # 获取 GPU 的时钟频率
        if dtype in (torch.float16, torch.bfloat16):
            return get_max_tensorcore_tflops(dtype, sm_clock)  # 返回 TensorCore 加速的 TFLOPS

        if torch.backends.cuda.matmul.allow_tf32:
            return get_max_tensorcore_tflops(torch.float32, sm_clock)  # 返回 TensorCore 加速的 TFLOPS
        else:
            return get_max_simd_tflops(torch.float32, sm_clock)  # 返回 SIMD 加速的 TFLOPS
    else:
        if dtype in (torch.float16, torch.bfloat16):
            return get_max_tensorcore_tflops(dtype)  # 返回 TensorCore 加速的 TFLOPS

        if torch.backends.cuda.matmul.allow_tf32:
            return get_max_tensorcore_tflops(torch.float32)  # 返回 TensorCore 加速的 TFLOPS
        else:
            return get_max_simd_tflops(torch.float32)  # 返回 SIMD 加速的 TFLOPS


# 获取 GPU 的 DRAM 带宽（吉比特每秒）
@functools.lru_cache(None)
def get_gpu_dram_gbps():
    from triton.testing import get_dram_gbps

    return get_dram_gbps()


# 获取 GPU 的共享内存大小
def get_gpu_shared_memory():
    from triton.runtime import driver

    return driver.active.utils.get_device_properties(0).get("max_shared_mem", 0)  # 获取指定设备的最大共享内存大小


# 判断给定的规约类型是否为 Welford 形式的规约
def is_welford_reduction(reduction_type):
    return reduction_type.startswith("welford")  # 检查规约类型是否以 "welford" 开头


# 返回特定规约类型的输出数量（Welford 规约为 3，其他为 1）
def reduction_num_outputs(reduction_type):
    return 3 if is_welford_reduction(reduction_type) else 1  # 如果是 Welford 规约返回 3，否则返回 1


# 判断当前操作系统是否为 Linux
def is_linux() -> bool:
    return platform.system() == "Linux"


# 检查给定的可迭代对象中是否有包含 sympy.Expr 类型但不是数值的元素
def has_free_symbols(itr: Iterable[Any]):
    return any(isinstance(x, sympy.Expr) and not x.is_number for x in itr)


# 当前函数是否与动态相关，需要从 ir 模块中导入相关内容
def is_dynamic(*args):
    from . import ir
    # 遍历传入的参数列表 args
    for t in args:
        # 检查当前元素 t 是否为 ir.TensorBox 类型的实例
        if isinstance(t, ir.TensorBox):
            # 如果 t.data 的大小具有自由符号或者 t.data 具有 get_stride 方法且其步长具有自由符号
            if has_free_symbols(t.data.get_size()) or (
                hasattr(t.data, "get_stride") and has_free_symbols(t.data.get_stride())
            ):
                # 如果满足条件，返回 True 表示动态特性存在
                return True
        # 如果当前元素 t 是 ir.StorageBox, ir.BaseView 或 ir.ComputedBuffer 类型的实例
        elif isinstance(t, (ir.StorageBox, ir.BaseView, ir.ComputedBuffer)):
            # 断言当前元素 t 具有 get_size 和 get_stride 方法
            assert hasattr(t, "get_size") and hasattr(t, "get_stride")
            # 如果 t 的大小或步长具有自由符号
            if has_free_symbols(t.get_size()) or has_free_symbols(t.get_stride()):
                # 如果满足条件，返回 True 表示动态特性存在
                return True
        # 如果当前元素 t 不是 ir.IRNode 的实例
        elif not isinstance(t, ir.IRNode):
            # 继续下一个元素的检查
            continue
        # 如果当前元素 t 是意料之外的类型
        else:
            # 抛出类型错误，指明未预期的类型，使用 f-string 给出具体类型信息
            raise TypeError(f"unexpected type for is_dynamic {type(t)}")

    # 若所有元素检查完毕没有发现动态特性，返回 False
    return False
# 定义枚举类，包含用于 Triton 代码生成的占位符字符串常量
class Placeholder(enum.Enum):
    # triton 核函数的实际名称的占位符
    # 例如，对于 "def triton_"，它将是 "triton_"
    KERNEL_NAME = "KERNEL_NAME"

    # triton 核函数的描述性名称；当 unique_kernel_names = False 时，此占位符将被替换为包含更多信息的字符串
    DESCRIPTIVE_NAME = "DESCRIPTIVE_NAME"


# 执行函数并保存结果
def pass_execution_and_save(func, gm, inp, msg):
    from .pattern_matcher import stable_topological_sort

    # 使用临时文件保存执行前后的图形状态
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as f:
        # 创建两个字符串 IO 对象来保存执行前后的图形状态
        before_io = io.StringIO()
        after_io = io.StringIO()

        # 根据输入进行形状属性传播
        ShapeProp(gm=gm, fake_mode=detect_fake_mode(inp)).propagate(*inp)
        # 将执行前的图形状态写入临时文件和字符串 IO 对象
        print(f"Before:\n{gm.graph}", file=f)
        print(gm.graph, file=before_io)

        # 记录开始时间
        start_time = datetime.now()
        # 使用图形变换观察器观察图形变换过程
        with GraphTransformObserver(gm, msg, config.trace.log_url_for_graph_xform):
            func(gm.graph)
        # 计算执行时间
        time_elapsed = datetime.now() - start_time

        # 对图形进行稳定拓扑排序和语法检查
        stable_topological_sort(gm.graph)
        gm.graph.lint()
        gm.recompile()

        # 将执行后的图形状态写入临时文件和字符串 IO 对象
        print(f"After:\n{gm.graph}", file=f)
        print(gm.graph, file=after_io)

        # 检查执行前后的图形状态是否相同
        t = before_io.getvalue() == after_io.getvalue()
        # 记录日志信息
        log.info(
            "%s, save before/after graph to %s, graph before/after are the same = %s, time elapsed = %s",
            msg,
            f.name,
            t,
            time_elapsed,
        )


# 判断节点是否为 Collective 类型
def is_collective(node):
    from . import ir

    return type(node) == ir._CollectiveKernel


# 判断节点是否为 Wait 类型
def is_wait(node):
    from . import ir

    return type(node) == ir._WaitKernel


# 计算前向固定参数数量
def num_fw_fixed_arguments(dynamo_gm_num_inputs: int, aot_fw_gm_num_inputs: int):
    "计算 AOT 前向图的固定地址输入数量（参数和缓冲区）"
    num_rng_seed_offset_inputs = (
        2 if torch._functorch.config.functionalize_rng_ops else 0
    )
    return aot_fw_gm_num_inputs - dynamo_gm_num_inputs - num_rng_seed_offset_inputs


# 统计反向图中的切线数量
def count_tangents(fx_g: torch.fx.GraphModule):
    """
    推断反向图中的静态输入
    """

    def is_saved_tensor(x):
        return (
            "tangents" not in x.name
            and "bwd_seed" not in x.name
            and "bwd_base_offset" not in x.name
        )

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_saved_tensor(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)


@dataclasses.dataclass
class BoxedBool:
    value: bool

    def __bool__(self):
        return self.value

    @staticmethod
    def disable(obj):
        # 禁用对象的静态方法
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False
@contextlib.contextmanager
def collect_defined_kernels(kernel_list):
    # 引入代码生成器的包装类
    from .codegen.wrapper import WrapperCodeGen

    # 保存原始的定义内核函数
    orig_define_kernel = WrapperCodeGen.define_kernel

    # 定义新的内核函数定义方法
    def new_define_kernel(wrapper, name, kernel_code, metadata, *args, **kwargs):
        nonlocal kernel_list
        # 将新定义的内核代码添加到 kernel_list 中
        kernel_list.append(kernel_code)
        return orig_define_kernel(wrapper, name, kernel_code, metadata, *args, **kwargs)

    # 使用 unittest.mock.patch.object 修改 WrapperCodeGen 的 define_kernel 方法
    with unittest.mock.patch.object(WrapperCodeGen, "define_kernel", new_define_kernel):
        yield


def get_cloned_parameter_buffer_name(name: str):
    # 返回参数名后加上 "__original__" 的新名称
    return name + "__original__"


def is_gpu(device: str):
    # 断言设备参数为字符串或者为 None
    assert isinstance(device, str) or device is None, device
    # 判断设备是否为 "cuda" 或 "xpu"
    return device in ["cuda", "xpu"]


def device_need_guard(device: str):
    # 断言设备参数为字符串类型
    assert isinstance(device, str)
    # 判断设备是否需要 GPU 保护
    return is_gpu(device)


def needs_fallback_due_to_atomic_add_limitations(dtype):
    # tl.atomic_add 不支持以下类型
    return dtype in {torch.int64, torch.bool, torch.bfloat16}


def use_scatter_fallback(
    op_overload: torch._ops.OpOverload,
    reduction_type,
    self_dtype,
    src_dtype,
    src_device_type,
    src_is_tensor,
):
    # 如果操作重载的包是 scatter_reduce_ 或 scatter_reduce，且没有指定 reduction_type，则返回 False
    if (
        op_overload.overloadpacket
        in (torch.ops.aten.scatter_reduce_, torch.ops.aten.scatter_reduce)
        and reduction_type is None
    ):
        return False

    # 如果操作重载的包是 scatter_，则 reduce_ty 为 "add"，否则为 "sum"
    reduce_ty = (
        "add" if op_overload.overloadpacket == torch.ops.aten.scatter_ else "sum"
    )

    # 判断是否需要使用 scatter 的回退逻辑
    return (
        reduction_type not in {None, reduce_ty}
        or (
            src_is_tensor
            and is_gpu(src_device_type)
            and needs_fallback_due_to_atomic_add_limitations(src_dtype)
        )
        or (
            op_overload.overloadpacket == torch.ops.aten.scatter_reduce_
            and reduction_type == "sum"
            and src_is_tensor
            and src_device_type == "cpu"
            and config.cpp.fallback_scatter_reduce_sum
            and (config.cpp.dynamic_threads or parallel_num_threads() != 1)
        )
        or (reduction_type == reduce_ty and self_dtype in {torch.bool, torch.int64})
        or torch.are_deterministic_algorithms_enabled()
    )


def dump_node_schedule(node_schedule):
    """
    An API that can be used in pdb to dump a node_schedule.
    Right mainly dump the read/write dependencies but can add more as needed.
    """
    # 引入需要的模块
    from torch._inductor.codegen.simd import DisableReduction, EnableReduction
    from torch._inductor.scheduler import SchedulerNode

    # 打印节点调度的信息，包括节点数目
    print(f"Node schedule with {len(node_schedule)} nodes")
    # 对于给定的节点调度列表，使用索引和节点进行迭代
    for idx, node in enumerate(node_schedule):
        # 打印节点的索引，使用格式化字符串保证输出占据三个字符的宽度
        print(f" {idx:3}:")
        
        # 检查节点类型并输出相应信息
        if node is EnableReduction:
            # 如果节点是 EnableReduction 类型，则打印启用减少的消息
            print("enable reduction")
        elif node is DisableReduction:
            # 如果节点是 DisableReduction 类型，则打印禁用减少的消息
            print("disable reduction")
        elif isinstance(node, SchedulerNode):
            # 如果节点是 SchedulerNode 类型，则进一步处理
            is_red = node.is_reduction()
            # 检查节点是否标记为减少操作，并据此打印相应消息
            print(f"{'red' if is_red else 'pw'} scheduler node")
            
            if is_red:
                # 如果节点是减少操作，则断言节点不为空，并打印原始的减少提示信息
                assert node.node is not None
                print(f"original reduction hint {node.node.data.reduction_hint}")  # type: ignore[attr-defined]
            
            # 打印节点的读依赖信息
            print("ReadDep:")
            for dep in node.read_writes.reads:
                print(dep)
            
            # 打印节点的写依赖信息
            print("WriteDep:")
            for dep in node.read_writes.writes:
                print(dep)
        else:
            # 如果遇到未识别的节点类型，则引发运行时错误
            raise RuntimeError(f"Unrecognized node type: {type(node)}")
# 根据输入的张量判断其是否对齐于 GPU 的要求
def tensor_is_aligned(tensor: torch.Tensor):
    # 查看注释：[Inductor 中的输入对齐处理]
    # 目前我们不尝试对存储偏移进行对齐保护。
    # 当编写这条注释时，非符号化的存储偏移没有被保护，
    # 但符号化的存储偏移是被保护的。为了保持一致性，
    # 我们在执行此检查时禁止保护的创建：这确保我们在添加此逻辑时不会添加重新编译。
    return (
        tensor.storage_offset() * get_dtype_size(tensor.dtype)
    ) % GPU_ALIGN_BYTES == 0


def should_assume_input_aligned(example_input: torch.Tensor):
    # 查看注释：[Inductor 中的输入对齐处理]

    # 目前，我们只关心 CUDA 张量的对齐情况。
    if not is_gpu(example_input.device.type):
        return False
    return config.assume_aligned_inputs or tensor_is_aligned(example_input)


def maybe_get_suppress_shape_guards_ctx():
    # 尝试获取 TracingContext.try_get().fake_mode.shape_env.suppress_guards()
    # 如果不可用，则返回一个 nullcontext。

    # 如果我们处理 cudagraphs，可能没有 tracing_context
    tracing_context = torch._guards.TracingContext.try_get()
    if not tracing_context:
        return contextlib.nullcontext()

    # 在独立的 Inductor 编译模式下，我们可能没有附加到 fake mode 的 shape_env
    shape_env = tracing_context.fake_mode.shape_env
    if not shape_env:
        return contextlib.nullcontext()

    return shape_env.suppress_guards()


def aoti_eager_cache_dir(namespace: str, device: str):
    # 返回 AOTI eager 操作的缓存目录路径
    return Path(cache_dir()) / "aoti_eager" / namespace / device


def aoti_eager_op_conf_lock(op_func_name_with_overload: str):
    # 避免循环导入
    from filelock import FileLock

    # 获取锁文件目录和超时时间
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT

    op_conf_lock_file = f"{op_func_name_with_overload}.lock"
    lock_dir = get_lock_dir()
    return FileLock(os.path.join(lock_dir, op_conf_lock_file), timeout=LOCK_TIMEOUT)


def load_aoti_eager_cache(ns: str, op_func_name_with_overload: str, device_type: str):
    # 获取 AOTI eager 操作的缓存目录
    device_kernel_cache = aoti_eager_cache_dir(ns, device_type)
    # 构建操作配置文件的路径
    op_conf = device_kernel_cache / f"{op_func_name_with_overload}.json"
    if not op_conf.exists():
        return []
    # 使用 `aoti_eager_op_conf_lock` 函数锁定操作函数名称及其重载信息
    with aoti_eager_op_conf_lock(op_func_name_with_overload):
        # 打开指定路径的 JSON 配置文件
        with open(op_conf) as f:
            # 加载 JSON 文件内容为 Python 对象
            json_data = json.load(f)
            # 遍历 JSON 数据中的每个项目
            for item in json_data:
                # 构建内核库的绝对路径
                kernel_lib_abs_path = device_kernel_cache / item["kernel_path"]
                # 更新项目中的内核路径为 POSIX 形式的字符串
                item["kernel_path"] = kernel_lib_abs_path.as_posix()

                # 检查内核库是否存在
                if not kernel_lib_abs_path.exists():
                    return []  # 如果不存在，返回空列表并终止函数

                # 遍历项目中的每个元数据信息
                for metadata in item["meta_info"]:
                    # 断言元数据中的 is_dynamic 字段为 False
                    assert not metadata["is_dynamic"], "Only support static shape for now"
                    # 如果设备类型为 CPU，则设置设备索引为 -1
                    if metadata["device_type"] == "cpu":
                        metadata["device_index"] = -1
                    # 将元数据的 dtype 字段转换为对应的 Torch 数据类型
                    metadata["dtype"] = getattr(torch, metadata["dtype"].split(".")[-1])

            return json_data  # 返回处理后的 JSON 数据
def aoti_compile_with_persistent_cache(
    ns: str,
    op_func_name_with_overload: str,
    device_type: str,
    dynamic: bool,
    f: Callable[..., Any],
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
):
    """
    Compile the given function with persistent cache for AOTI eager mode.
    """
    # 检查是否为静态形状，目前仅支持静态形状
    assert not dynamic, "Only support static shape for now"

    # 导入 torch 的 AOT 编译函数
    from torch._export import aot_compile

    # 定义标量类型到 Torch 数据类型的映射
    type_to_torch_dtype = {int: torch.int32, float: torch.float, bool: torch.bool}
    # 支持的标量类型元组
    supported_scalar_types = tuple(type_to_torch_dtype.keys())
    # 展开输入参数
    flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)
    # 检查所有输入参数是否为支持的标量类型或者 Torch 张量
    if not all(
        isinstance(input, (supported_scalar_types, torch.Tensor))
        for input in flattened_inputs
    ):
        raise NotImplementedError("Only support tensor, int, float, bool for now")

    # 获取 AOTI eager 模式的持久化缓存目录
    persistent_cache = aoti_eager_cache_dir(ns, device_type)
    if not persistent_cache.exists():
        persistent_cache.mkdir(parents=True)

    # 创建持久化缓存库目录
    persistent_cache_lib = persistent_cache / "lib"
    if not persistent_cache_lib.exists():
        persistent_cache_lib.mkdir()

    # 使用 mock.patch.dict 上下文管理器，设置环境变量 TORCHINDUCTOR_CACHE_DIR
    with mock.patch.dict(
        os.environ,
        {"TORCHINDUCTOR_CACHE_DIR": persistent_cache_lib.absolute().as_posix()},
    ):
```