# `.\pytorch\torch\_inductor\fx_passes\replace_random.py`

```py
# 引入必要的模块和库
import collections  # 导入collections模块，用于创建默认字典等数据结构
import logging  # 导入logging模块，用于记录日志信息

import torch  # 导入PyTorch库
from torch.fx.passes.graph_transform_observer import GraphTransformObserver  # 从PyTorch的FX passes中导入图形转换观察器
from torch.fx.passes.shape_prop import _extract_tensor_metadata  # 从PyTorch的FX passes中导入提取张量元数据的函数
from .. import config, inductor_prims  # 导入上级包中的config和inductor_prims模块
from ..pattern_matcher import (  # 从上级包中导入模式匹配器相关内容
    CallFunctionVarArgs,  # 导入CallFunctionVarArgs类，用于函数调用的变长参数匹配
    Match,  # 导入Match类，用于模式匹配
    PatternMatcherPass,  # 导入PatternMatcherPass类，用于模式匹配器的通行
    register_graph_pattern  # 导入register_graph_pattern函数，用于注册图形模式
)
from ..virtualized import V  # 从上级包中导入V对象

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
patterns = PatternMatcherPass()  # 创建模式匹配器对象
aten = torch.ops.aten  # 获取PyTorch的aten操作集

def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0  # 如果配置中指定了回退随机数生成方式，则直接返回0

    count = patterns.apply(gm)  # 应用模式匹配器到给定的FX图中
    with GraphTransformObserver(
        gm, "fuse_seed_creation_pass", config.trace.log_url_for_graph_xform
    ):
        count += fuse_seed_creation_pass(gm.graph)  # 调用seed融合传递函数，并记录融合操作数

    return count  # 返回模式匹配器应用的计数

def fuse_seed_creation_pass(graph: torch.fx.Graph):
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_seeds = collections.defaultdict(list)  # 创建一个默认字典，用于按设备存储种子生成节点

    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.seed).match(node):
            device_seeds[node.args[0]].append(node)  # 如果节点匹配种子生成函数，则将其添加到对应设备的种子列表中

    if not device_seeds:
        return 0  # 如果没有找到设备种子节点，直接返回0

    for device, seeds in device_seeds.items():
        with graph.inserting_before(seeds[0]):
            combined = graph.call_function(inductor_prims.seeds, (len(seeds), device))  # 在第一个种子节点前插入组合种子生成函数调用节点
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(seeds)], device=device, dtype=torch.int64
                )  # 使用虚拟模式创建新种子张量，并将其存储在组合节点的元数据中
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )  # 提取组合种子张量的元数据信息

        for idx, seed in enumerate(seeds):
            with graph.inserting_before(seed):
                new_seed = graph.call_function(
                    inductor_prims.lookup_seed, (combined, idx)
                )  # 在每个种子生成节点前插入查找新种子函数调用节点
            seed.replace_all_uses_with(new_seed)  # 将原种子节点的所有使用替换为新种子节点
            new_seed.meta.update(seed.meta)  # 更新新种子节点的元数据
            graph.erase_node(seed)  # 删除原种子节点

    return len(device_seeds)  # 返回处理的设备种子数目

def default_kwargs(device):
    return {}  # 返回一个空字典作为默认参数

def get_device(device):
    if device is not None:
        return device  # 如果设备参数不为空，则直接返回设备参数
    return torch.empty([]).device  # 否则返回一个空张量的设备作为默认设备

@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.rand.generator), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.randn.generator), pass_dict=patterns)
def replace_random(
    match: Match,
    size,
    *,
    generator=None,
    dtype=None,
    device=None,
    # 设备参数，用于指定数据加载到的设备，默认为None，即使用主机内存
    layout=None,
    # 布局参数，用于指定数据的布局方式，默认为None，通常在处理图像或张量数据时指定
    pin_memory=None,
    # 是否使用可固定内存，通常在数据加载到GPU时使用，以提高性能，默认为None，由系统自动决定
# 如果 generator 不为 None，则返回空，不执行后续操作
):
    if generator is not None:
        return

    # 定义一个替换函数，生成指定大小的随机数据
    def replacement(size):
        # 调用 inductor_prims.random 函数生成随机数据，使用设备种子和模式参数
        result = inductor_prims.random(
            size, inductor_prims.seed(device), mode, **default_kwargs(device)
        )
        # 如果指定了数据类型 dtype，则将结果转换为该类型
        if dtype is not None:
            result = result.to(dtype)
        return result

    # 根据输出节点的重载包确定模式，这里使用了类型忽略注释，表示不检查 union-attr 属性
    mode = {
        aten.rand: "rand",
        aten.randn: "randn",
    }[
        match.output_node().target.overloadpacket  # type: ignore[union-attr]
    ]  # type: ignore[union-attr]

    # 获取设备对象
    device = get_device(device)

    # 使用 match 对象的 replace_by_example 方法，替换匹配的模式为替换函数的输出结果
    match.replace_by_example(replacement, [size])


# 注册一个图模式匹配，匹配 aten.randint.low 函数调用，通过字典传递模式
@register_graph_pattern(CallFunctionVarArgs(aten.randint.low), pass_dict=patterns)
def replace_randint(
    match: Match,
    low,
    high,
    size,
    *,
    dtype=torch.int64,
    device=None,
    layout=None,
    pin_memory=None,
):
    # 定义替换函数，生成指定范围和大小的随机整数数据
    def replacement(low, high, size):
        # 调用 inductor_prims.randint 函数生成随机整数数据，使用设备种子
        result = inductor_prims.randint(low, high, size, inductor_prims.seed(device))
        # 将结果转换为指定的数据类型 dtype
        return result.to(dtype)

    # 获取设备对象
    device = get_device(device)

    # 使用 match 对象的 replace_by_example 方法，替换匹配的模式为替换函数的输出结果，传递参数 low, high, size
    match.replace_by_example(replacement, [low, high, size])
```